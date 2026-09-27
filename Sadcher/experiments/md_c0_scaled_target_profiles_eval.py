"""Scale two 'best-for-C0' profiles up beyond training range and check whether
MILP-IL v2 keeps (a) its small gap to MILP and (b) its advantage over greedy
as instance size grows.

Selection rationale (from the 6-profile MILP-IL v2 results):
  - dependency_deep: smallest gap to MILP (+6.4%), beats greedy by 3.7% —
    the "best absolute quality" profile for C0.
  - process_scarce: beats greedy by 13.7pp, largest C0-vs-greedy advantage —
    the "best relative advantage" profile.

Both profiles preserve their structural signature under scaling: the
transport_ratio / precedence_density / capacity_slack / speed_ratio /
skill_count values are held fixed, and only task_count + robot counts +
critical_path_length are scaled proportionally.

For each profile x task_count x seed we solve pure MILP (300s time limit,
4 threads), the MILP-IL v2 C0 online scheduler (with MIP fallback), and
three greedy baselines, then aggregate makespan / gaps / MILP timeout
counts.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from baselines.md_oracle_dispatch import solve_md_oracle
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_instance_profiles import INSTANCE_PROFILES
from experiments.protocol import DatasetSplit
from imitation_learning.md_train import load_md_policy_checkpoint
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.md_constrained_decoder import LearnedConstrainedDecoder
from schedulers.md_greedy_baselines import (
    run_greedy_distance,
    run_greedy_eta,
    run_greedy_unlock,
)
from schedulers.online_md_scheduler import (
    ExplicitMIPFallback,
    OnlineMDScheduler,
    OnlineNeuralScoreProvider,
)
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator

TARGET_PROFILES = ("dependency_deep", "process_scarce")
TASK_COUNTS = (12, 24, 42, 60)
SEEDS = (301, 302, 303, 304)
MAX_ROLLOUT_STEPS = 20_000
MILP_TIME_LIMIT_SECONDS = 300.0
MILP_THREADS = 4

GREEDY_BASELINES = {
    "greedy_distance": run_greedy_distance,
    "greedy_eta": run_greedy_eta,
    "greedy_unlock": run_greedy_unlock,
}


def _scaled_profile_config(profile_name: str, task_count: int, seed: int) -> MDGeneratorConfig:
    """Scale a profile's task_count while preserving its structural ratios."""
    base = dict(INSTANCE_PROFILES[profile_name].parameters)
    base_task_count = base["task_count"]
    scale = task_count / base_task_count
    process_tasks = int(task_count * (1 - base["transport_ratio"]))
    critical_path = min(
        max(base["critical_path_length"], round(base["critical_path_length"] * scale)),
        process_tasks,
    )
    return MDGeneratorConfig(
        seed=seed,
        task_count=task_count,
        transport_ratio=base["transport_ratio"],
        precedence_density=base["precedence_density"],
        critical_path_length=critical_path,
        capacity_slack=base["capacity_slack"],
        speed_ratio=base["speed_ratio"],
        process_robot_count=max(1, round(base["process_robot_count"] * scale)),
        transport_robot_count=max(base.get("transport_robot_count", 1), round(base["transport_robot_count"] * scale)),
        skill_count=base["skill_count"],
        scarce_skill_count=base.get("scarce_skill_count", 0),
        material_downstream_stratified=base.get("material_downstream_stratified", False),
        process_duration_range=base.get("process_duration_range", None),
    )


def _run_c0(model, domain, *, run_id, instance_id, seed, device, confidence_threshold=0.0):
    scheduler = OnlineMDScheduler(
        scorer=OnlineNeuralScoreProvider(
            model, build_md_policy_inputs_from_simulator, device=device
        ),
        decoder=LearnedConstrainedDecoder(),
        fallback=ExplicitMIPFallback(threads=1),
        confidence_threshold=confidence_threshold,
        max_steps=MAX_ROLLOUT_STEPS,
    )
    started = time.perf_counter()
    result = scheduler.run(
        domain,
        run_id=run_id,
        instance_id=instance_id,
        seed=seed,
        split=DatasetSplit.TEST,
    )
    elapsed = time.perf_counter() - started
    exp = result.experiment
    return {
        "success": exp.success,
        "makespan": exp.makespan,
        "solver_calls": len(result.fallback_records),
        "solver_time_seconds": sum(r.solver_time_seconds for r in result.fallback_records),
        "wall_time_seconds": elapsed,
    }


def _run_milp(domain, time_limit_seconds: float = MILP_TIME_LIMIT_SECONDS):
    r = solve_md_oracle(
        domain, time_limit_seconds=time_limit_seconds, threads=MILP_THREADS
    )
    return {
        "status": r.status.value,
        "makespan": r.objective,
        "solve_time_seconds": r.solve_time_seconds,
        "timeout": r.timeout,
        "optimality_gap": r.optimality_gap,
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def _apply_config_overrides(
    cfg: MDGeneratorConfig,
    *,
    process_robot_count: int | None,
    transport_robot_count: int | None,
    process_duration_range: tuple[int, int] | None,
    skill_count: int | None,
    scarce_skill_count: int | None,
) -> MDGeneratorConfig:
    updates: dict = {}
    if process_robot_count is not None:
        updates["process_robot_count"] = process_robot_count
    if transport_robot_count is not None:
        updates["transport_robot_count"] = transport_robot_count
    if process_duration_range is not None:
        updates["process_duration_range"] = process_duration_range
    if skill_count is not None:
        updates["skill_count"] = skill_count
    if scarce_skill_count is not None:
        updates["scarce_skill_count"] = scarce_skill_count
    if not updates:
        return cfg
    return replace(cfg, **updates)


def run(
    checkpoint: Path,
    output: Path,
    device_name: str,
    profiles: tuple[str, ...] | None = None,
    task_counts: tuple[int, ...] | None = None,
    override_process_robot_count: int | None = None,
    override_transport_robot_count: int | None = None,
    override_process_duration_range: tuple[int, int] | None = None,
    override_skill_count: int | None = None,
    override_scarce_skill_count: int | None = None,
    milp_time_limit_seconds: float = MILP_TIME_LIMIT_SECONDS,
    skip_greedies: bool = False,
    confidence_threshold: float = 0.0,
) -> dict:
    device = torch.device(device_name)
    model, _ = load_md_policy_checkpoint(checkpoint, device=device)
    model.eval()

    active_profiles = tuple(profiles) if profiles else TARGET_PROFILES
    active_task_counts = tuple(task_counts) if task_counts else TASK_COUNTS

    rows = []
    for profile in active_profiles:
        for tc in active_task_counts:
            for seed in SEEDS:
                cfg = _scaled_profile_config(profile, tc, seed)
                cfg = _apply_config_overrides(
                    cfg,
                    process_robot_count=override_process_robot_count,
                    transport_robot_count=override_transport_robot_count,
                    process_duration_range=override_process_duration_range,
                    skill_count=override_skill_count,
                    scarce_skill_count=override_scarce_skill_count,
                )
                domain = generate_md_instance(cfg).domain

                milp = _run_milp(domain, time_limit_seconds=milp_time_limit_seconds)
                c0 = _run_c0(
                    model,
                    domain,
                    run_id=f"scaled-{profile}-c0-{tc}-{seed}",
                    instance_id=f"scaled-{profile}-{tc}-{seed}",
                    seed=seed,
                    device=device,
                    confidence_threshold=confidence_threshold,
                )
                greedies = {}
                if not skip_greedies:
                  for name, fn in GREEDY_BASELINES.items():
                    r = fn(
                        MDDiscreteSimulator(domain),
                        run_id=f"scaled-{profile}-{name}-{tc}-{seed}",
                        instance_id=f"scaled-{profile}-{tc}-{seed}",
                        seed=seed,
                        split=DatasetSplit.TEST,
                        max_steps=MAX_ROLLOUT_STEPS,
                    )
                    greedies[name] = {"success": r.success, "makespan": r.makespan}

                row = {
                    "profile": profile,
                    "task_count": tc,
                    "seed": seed,
                    "config": {slot: getattr(cfg, slot) for slot in cfg.__slots__},
                    "milp": milp,
                    "c0_milp_il_v2": c0,
                    "greedies": greedies,
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "profile": profile,
                            "tc": tc,
                            "seed": seed,
                            "milp_status": milp["status"],
                            "milp_ms": milp["makespan"],
                            "c0_ms": c0["makespan"],
                            "gd_ms": greedies.get("greedy_distance", {}).get("makespan"),
                            "ge_ms": greedies.get("greedy_eta", {}).get("makespan"),
                            "gu_ms": greedies.get("greedy_unlock", {}).get("makespan"),
                        }
                    ),
                    flush=True,
                )

    output.mkdir(parents=True, exist_ok=True)
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    def gap(a, b):
        if a is None or b is None or b == 0:
            return None
        return (a - b) / b * 100.0

    def fmt(v):
        return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"

    def fmt_pct(v):
        return f"{v:+.1f}%" if isinstance(v, (int, float)) else "n/a"

    lines = [
        f"# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, {len(SEEDS)} seeds/tier)",
        "",
        "**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP "
        "(+6.4%); process_scarce = largest advantage over greedy (-13.7pp). "
        "Task_count scaled 12 → 60, other structural params preserved. MILP time "
        "limit 300s, 4 threads.",
        "",
    ]

    summary_rows = []
    _iter_task_counts = active_task_counts
    for profile in active_profiles:
        milp_by_tc = defaultdict(list)
        c0_by_tc = defaultdict(list)
        greedy_by_tc = defaultdict(lambda: defaultdict(list))
        milp_timeout_by_tc = defaultdict(int)
        for r in rows:
            if r["profile"] != profile:
                continue
            tc = r["task_count"]
            if r["milp"]["makespan"] is not None:
                milp_by_tc[tc].append(r["milp"]["makespan"])
            if r["milp"]["timeout"]:
                milp_timeout_by_tc[tc] += 1
            if r["c0_milp_il_v2"]["success"]:
                c0_by_tc[tc].append(r["c0_milp_il_v2"]["makespan"])
            for name, g in r["greedies"].items():
                if g["success"]:
                    greedy_by_tc[tc][name].append(g["makespan"])

        lines.append(f"## profile: `{profile}`")
        lines.append("")
        lines.append(
            "| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |"
        )
        lines.append("|---:|---|---:|---:|---:|---|---:|---:|")
        for tc in _iter_task_counts:
            milp_mean = _mean(milp_by_tc[tc])
            c0_mean = _mean(c0_by_tc[tc])
            gm = {name: _mean(v) for name, v in greedy_by_tc[tc].items() if v}
            best_name = min(gm, key=gm.get) if gm else "n/a"
            best_mean = gm.get(best_name)
            timeouts = milp_timeout_by_tc[tc]
            status_str = f"{len(SEEDS) - timeouts}/{len(SEEDS)} optimal, {timeouts} timeout"
            summary_rows.append(
                {
                    "profile": profile,
                    "task_count": tc,
                    "milp_incumbent": milp_mean,
                    "milp_timeout_count": timeouts,
                    "c0_makespan": c0_mean,
                    "best_greedy_name": best_name,
                    "best_greedy_makespan": best_mean,
                    "c0_gap_vs_milp_percent": gap(c0_mean, milp_mean),
                    "best_greedy_gap_vs_milp_percent": gap(best_mean, milp_mean),
                    "c0_vs_best_greedy_percent": gap(c0_mean, best_mean),
                }
            )
            lines.append(
                f"| {tc} | {status_str} | {fmt(milp_mean)} | {fmt(c0_mean)} | {fmt_pct(gap(c0_mean, milp_mean))} | "
                f"{fmt(best_mean)} ({best_name}) | {fmt_pct(gap(best_mean, milp_mean))} | "
                f"{fmt_pct(gap(c0_mean, best_mean))} |"
            )
        lines.append("")

    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "schema": "md-c0-scaled-target-profiles-eval-1.0",
        "checkpoint": str(checkpoint),
        "profiles": list(active_profiles),
        "task_counts": list(active_task_counts),
        "seeds": list(SEEDS),
        "override_process_robot_count": override_process_robot_count,
        "override_transport_robot_count": override_transport_robot_count,
        "override_process_duration_range": (
            list(override_process_duration_range)
            if override_process_duration_range is not None
            else None
        ),
        "milp_time_limit_seconds": milp_time_limit_seconds,
        "per_tier": summary_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/md_c0_scaled_target_profiles_eval_2026-09-14"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Subset of profile names to evaluate. Defaults to TARGET_PROFILES.",
    )
    parser.add_argument(
        "--tc-list",
        type=str,
        default=None,
        help="Comma-separated task_count values to evaluate (defaults to 12,24,42,60).",
    )
    parser.add_argument(
        "--override-process-robot-count",
        type=int,
        default=None,
        help="Override scaled process_robot_count for every generated instance.",
    )
    parser.add_argument(
        "--override-transport-robot-count",
        type=int,
        default=None,
        help="Override scaled transport_robot_count for every generated instance.",
    )
    parser.add_argument(
        "--override-process-duration-min",
        type=int,
        default=None,
        help="Override process duration lower bound; requires the max override too.",
    )
    parser.add_argument(
        "--override-process-duration-max",
        type=int,
        default=None,
        help="Override process duration upper bound; requires the min override too.",
    )
    parser.add_argument(
        "--override-skill-count",
        type=int,
        default=None,
        help="Override skill_count (default from profile).",
    )
    parser.add_argument(
        "--override-scarce-skill-count",
        type=int,
        default=None,
        help="Override scarce_skill_count (default from profile).",
    )
    parser.add_argument(
        "--milp-time-limit",
        type=float,
        default=MILP_TIME_LIMIT_SECONDS,
        help="MILP wall-clock time limit in seconds (default 300).",
    )
    parser.add_argument(
        "--skip-greedies",
        action="store_true",
        help="Skip greedy baselines to save wallclock (large tc runs).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Fallback to MILP when C0 assignment margin < threshold (default 0.0).",
    )
    args = parser.parse_args()
    profiles = tuple(args.profiles) if args.profiles else None
    task_counts = (
        tuple(int(x) for x in args.tc_list.split(","))
        if args.tc_list
        else None
    )
    override_range: tuple[int, int] | None
    if (args.override_process_duration_min is None) != (
        args.override_process_duration_max is None
    ):
        parser.error(
            "--override-process-duration-min and --override-process-duration-max "
            "must be provided together."
        )
    if args.override_process_duration_min is not None:
        override_range = (
            args.override_process_duration_min,
            args.override_process_duration_max,
        )
    else:
        override_range = None
    print(
        json.dumps(
            run(
                args.checkpoint,
                args.output,
                args.device,
                profiles=profiles,
                task_counts=task_counts,
                override_process_robot_count=args.override_process_robot_count,
                override_transport_robot_count=args.override_transport_robot_count,
                override_process_duration_range=override_range,
                override_skill_count=args.override_skill_count,
                override_scarce_skill_count=args.override_scarce_skill_count,
                milp_time_limit_seconds=args.milp_time_limit,
                skip_greedies=args.skip_greedies,
                confidence_threshold=args.confidence_threshold,
            ),
            indent=2,
        )
    )
