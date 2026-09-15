"""Evaluate a MILP-supervised C0 checkpoint across the six MD profiles.

Mirrors experiments.md_c0_profile_eval_pilot but loads the checkpoint via
imitation_learning.md_train.load_md_policy_checkpoint (the payload schema
differs from the ticket 46 C0 training script). Produces a comparison
report against the six-profile MILP optimum (already computed in
reports/md_pure_milp_baseline_pilot_2026-09-13/) and the previously trained
synthetic-relational C0 (reports/md_c0_profile_eval_pilot_2026-09-13/).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from data_generation.md_instance_generator import generate_md_instance
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

PROFILES = tuple(INSTANCE_PROFILES)
EVAL_SEEDS = (101, 102, 201, 301)
MAX_ROLLOUT_STEPS = 10_000
FALLBACK_CONFIDENCE_THRESHOLD = 0.0
GREEDY_BASELINES = {
    "greedy_distance": run_greedy_distance,
    "greedy_eta": run_greedy_eta,
    "greedy_unlock": run_greedy_unlock,
}
DEFAULT_MILP_OPT = Path(
    "reports/md_pure_milp_baseline_pilot_2026-09-13/profile_rows.json"
)
DEFAULT_OLD_C0 = Path(
    "reports/md_c0_profile_eval_pilot_2026-09-13/c0_rows.json"
)


def _run_new_c0(model, domain, *, run_id: str, instance_id: str, seed: int, device):
    scheduler = OnlineMDScheduler(
        scorer=OnlineNeuralScoreProvider(
            model, build_md_policy_inputs_from_simulator, device=device
        ),
        decoder=LearnedConstrainedDecoder(),
        fallback=ExplicitMIPFallback(threads=1),
        confidence_threshold=FALLBACK_CONFIDENCE_THRESHOLD,
        max_steps=MAX_ROLLOUT_STEPS,
    )
    result = scheduler.run(
        domain,
        run_id=run_id,
        instance_id=instance_id,
        seed=seed,
        split=DatasetSplit.TEST,
    )
    experiment = result.experiment
    return {
        "success": experiment.success,
        "makespan": experiment.makespan,
        "solver_calls": len(result.fallback_records),
        "solver_time_seconds": sum(r.solver_time_seconds for r in result.fallback_records),
        "wall_time_seconds": experiment.wall_time_seconds,
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def run(checkpoint: Path, output: Path, device_name: str, milp_opt: Path, old_c0: Path) -> dict:
    device = torch.device(device_name)
    model, _ = load_md_policy_checkpoint(checkpoint, device=device)
    model.eval()

    rows_new = []
    rows_baselines = []
    for profile in PROFILES:
        for seed in EVAL_SEEDS:
            domain = generate_md_instance(INSTANCE_PROFILES[profile].config(seed)).domain
            new_row = _run_new_c0(
                model,
                domain,
                run_id=f"c0-milp-il-{profile}-{seed}",
                instance_id=f"{profile}-{seed}",
                seed=seed,
                device=device,
            )
            rows_new.append({"method": "C0_milp_il", "profile": profile, "seed": seed, **new_row})
            for name, fn in GREEDY_BASELINES.items():
                r = fn(
                    MDDiscreteSimulator(domain),
                    run_id=f"c0-milp-il-{name}-{profile}-{seed}",
                    instance_id=f"{profile}-{seed}",
                    seed=seed,
                    split=DatasetSplit.TEST,
                    max_steps=MAX_ROLLOUT_STEPS,
                )
                rows_baselines.append(
                    {
                        "method": name,
                        "profile": profile,
                        "seed": seed,
                        "success": r.success,
                        "makespan": r.makespan,
                    }
                )

    output.mkdir(parents=True, exist_ok=True)
    (output / "new_c0_rows.json").write_text(json.dumps(rows_new, indent=2) + "\n")
    (output / "baseline_rows.json").write_text(json.dumps(rows_baselines, indent=2) + "\n")

    milp_rows = json.loads(milp_opt.read_text()) if milp_opt.exists() else []
    old_rows = json.loads(old_c0.read_text()) if old_c0.exists() else []

    milp_by_profile = defaultdict(list)
    for r in milp_rows:
        if r.get("objective") is not None:
            milp_by_profile[r["profile"]].append(r["objective"])

    old_by_profile = defaultdict(list)
    for r in old_rows:
        if r.get("success"):
            old_by_profile[r["profile"]].append(r["makespan"])

    new_by_profile = defaultdict(list)
    for r in rows_new:
        if r["success"]:
            new_by_profile[r["profile"]].append(r["makespan"])

    best_greedy_by_profile = defaultdict(list)
    for profile in PROFILES:
        by_method = defaultdict(list)
        for r in rows_baselines:
            if r["profile"] == profile and r["success"]:
                by_method[r["method"]].append(r["makespan"])
        method_means = {m: _mean(v) for m, v in by_method.items() if v}
        if method_means:
            best_greedy_by_profile[profile] = min(method_means.values())

    lines = [
        "# MILP-supervised C0 vs synthetic-relational C0 vs MILP optimum (six profiles)",
        "",
        "| profile | MILP optimum | old C0 (synthetic) | old C0 gap | **new C0 (MILP-IL)** | **new C0 gap** | best greedy | best greedy gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary_rows = []
    for profile in PROFILES:
        milp = _mean(milp_by_profile.get(profile, []))
        new = _mean(new_by_profile.get(profile, []))
        old = _mean(old_by_profile.get(profile, []))
        best_gr = best_greedy_by_profile.get(profile)
        def gap(value):
            if value is None or milp is None or milp == 0:
                return None
            return (value - milp) / milp * 100.0
        summary_rows.append(
            {
                "profile": profile,
                "milp_optimum": milp,
                "old_c0_makespan": old,
                "new_c0_makespan": new,
                "best_greedy_makespan": best_gr,
                "old_c0_gap_percent": gap(old),
                "new_c0_gap_percent": gap(new),
                "best_greedy_gap_percent": gap(best_gr),
            }
        )
        def fmt(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"
        def fmt_pct(v):
            return f"{v:+.1f}%" if isinstance(v, (int, float)) else "n/a"
        lines.append(
            f"| {profile} | {fmt(milp)} | {fmt(old)} | {fmt_pct(gap(old))} | "
            f"{fmt(new)} | {fmt_pct(gap(new))} | {fmt(best_gr)} | {fmt_pct(gap(best_gr))} |"
        )
    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "schema": "md-c0-milp-supervised-profile-eval-1.0",
        "checkpoint": str(checkpoint),
        "profiles": list(PROFILES),
        "eval_seeds": list(EVAL_SEEDS),
        "row_count_new": len(rows_new),
        "row_count_baselines": len(rows_baselines),
        "per_profile": summary_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/md_c0_milp_supervised_profile_eval_2026-09-13"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--milp-opt", type=Path, default=DEFAULT_MILP_OPT)
    parser.add_argument("--old-c0", type=Path, default=DEFAULT_OLD_C0)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.output, args.device, args.milp_opt, args.old_c0), indent=2))
