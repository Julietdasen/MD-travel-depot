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
from pathlib import Path

import torch

from baselines.gurobi_md_oracle import solve_gurobi_md_oracle
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
    )


def _run_c0(model, domain, *, run_id, instance_id, seed, device):
    scheduler = OnlineMDScheduler(
        scorer=OnlineNeuralScoreProvider(
            model, build_md_policy_inputs_from_simulator, device=device
        ),
        decoder=LearnedConstrainedDecoder(),
        fallback=ExplicitMIPFallback(threads=1),
        confidence_threshold=0.0,
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


def _run_milp(domain):
    r = solve_gurobi_md_oracle(
        domain, time_limit_seconds=MILP_TIME_LIMIT_SECONDS, threads=MILP_THREADS
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


def run(checkpoint: Path, output: Path, device_name: str) -> dict:
    device = torch.device(device_name)
    model, _ = load_md_policy_checkpoint(checkpoint, device=device)
    model.eval()

    rows = []
    for profile in TARGET_PROFILES:
        for tc in TASK_COUNTS:
            for seed in SEEDS:
                cfg = _scaled_profile_config(profile, tc, seed)
                domain = generate_md_instance(cfg).domain

                milp = _run_milp(domain)
                c0 = _run_c0(
                    model,
                    domain,
                    run_id=f"scaled-{profile}-c0-{tc}-{seed}",
                    instance_id=f"scaled-{profile}-{tc}-{seed}",
                    seed=seed,
                    device=device,
                )
                greedies = {}
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
                            "gd_ms": greedies["greedy_distance"]["makespan"],
                            "ge_ms": greedies["greedy_eta"]["makespan"],
                            "gu_ms": greedies["greedy_unlock"]["makespan"],
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
    for profile in TARGET_PROFILES:
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
        for tc in TASK_COUNTS:
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
        "profiles": list(TARGET_PROFILES),
        "task_counts": list(TASK_COUNTS),
        "seeds": list(SEEDS),
        "milp_time_limit_seconds": MILP_TIME_LIMIT_SECONDS,
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
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.output, args.device), indent=2))
