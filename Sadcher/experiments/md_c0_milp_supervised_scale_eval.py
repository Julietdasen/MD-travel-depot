"""Scale-ladder evaluation for a MILP-supervised C0 checkpoint.

Mirrors md_c0_scale_baseline_comparison_pilot.py but loads the checkpoint
via imitation_learning.md_train.load_md_policy_checkpoint (matching the
payload schema written by train_md_policy). Produces a scale-ladder
comparison table that also folds in the pure-MILP baseline (from
reports/md_pure_milp_baseline_pilot_2026-09-13/) so the report answers the
direct question: at each task_count, does the new C0 beat the best
greedy? does it approach MILP?
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
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

TASK_COUNTS = (12, 24, 42, 60, 90, 114, 150)
SEEDS = (9601, 9602, 9603)  # matches md_c0_scale_baseline_comparison_pilot's grid
MAX_ROLLOUT_STEPS = 20_000
GREEDY_BASELINES = {
    "greedy_distance": run_greedy_distance,
    "greedy_eta": run_greedy_eta,
    "greedy_unlock": run_greedy_unlock,
}
DEFAULT_MILP_SCALE = Path(
    "reports/md_pure_milp_baseline_pilot_2026-09-13/scale_rows.json"
)


def _scaled_config(task_count: int, seed: int) -> MDGeneratorConfig:
    scale = task_count / 18
    return MDGeneratorConfig(
        seed=seed,
        task_count=task_count,
        transport_ratio=1.0 / 3.0,
        precedence_density=0.30,
        critical_path_length=6,
        capacity_slack=0.10,
        speed_ratio=0.70,
        process_robot_count=max(1, round(4 * scale)),
        transport_robot_count=max(1, round(3 * scale)),
        skill_count=3,
    )


def _run_new_c0(model, domain, *, run_id, instance_id, seed, device):
    scheduler = OnlineMDScheduler(
        scorer=OnlineNeuralScoreProvider(
            model, build_md_policy_inputs_from_simulator, device=device
        ),
        decoder=LearnedConstrainedDecoder(),
        fallback=ExplicitMIPFallback(threads=1),
        confidence_threshold=0.0,
        max_steps=MAX_ROLLOUT_STEPS,
    )
    result = scheduler.run(
        domain, run_id=run_id, instance_id=instance_id, seed=seed, split=DatasetSplit.TEST
    )
    exp = result.experiment
    return {
        "success": exp.success,
        "makespan": exp.makespan,
        "solver_calls": len(result.fallback_records),
        "solver_time_seconds": sum(r.solver_time_seconds for r in result.fallback_records),
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def run(checkpoint: Path, output: Path, device_name: str, milp_scale: Path) -> dict:
    device = torch.device(device_name)
    model, _ = load_md_policy_checkpoint(checkpoint, device=device)
    model.eval()

    rows_new = []
    rows_baselines = []
    for tc in TASK_COUNTS:
        for seed in SEEDS:
            domain = generate_md_instance(_scaled_config(tc, seed)).domain
            new = _run_new_c0(
                model,
                domain,
                run_id=f"c0-milp-il-scale-{tc}-{seed}",
                instance_id=f"scale{tc}-{seed}",
                seed=seed,
                device=device,
            )
            rows_new.append({"method": "C0_milp_il", "task_count": tc, "seed": seed, **new})
            for name, fn in GREEDY_BASELINES.items():
                r = fn(
                    MDDiscreteSimulator(domain),
                    run_id=f"c0-milp-il-scale-{name}-{tc}-{seed}",
                    instance_id=f"scale{tc}-{seed}",
                    seed=seed,
                    split=DatasetSplit.TEST,
                    max_steps=MAX_ROLLOUT_STEPS,
                )
                rows_baselines.append(
                    {
                        "method": name,
                        "task_count": tc,
                        "seed": seed,
                        "success": r.success,
                        "makespan": r.makespan,
                    }
                )
            print(json.dumps({"task_count": tc, "seed": seed, "done": True}), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    (output / "new_c0_rows.json").write_text(json.dumps(rows_new, indent=2) + "\n")
    (output / "baseline_rows.json").write_text(json.dumps(rows_baselines, indent=2) + "\n")

    milp_rows = json.loads(milp_scale.read_text()) if milp_scale.exists() else []
    milp_by_tc = defaultdict(list)
    for r in milp_rows:
        if r.get("objective") is not None:
            milp_by_tc[r["task_count"]].append(r["objective"])

    new_by_tc = defaultdict(list)
    for r in rows_new:
        if r["success"]:
            new_by_tc[r["task_count"]].append(r["makespan"])

    best_greedy_by_tc = {}
    best_greedy_name_by_tc = {}
    for tc in TASK_COUNTS:
        by_method = defaultdict(list)
        for r in rows_baselines:
            if r["task_count"] == tc and r["success"]:
                by_method[r["method"]].append(r["makespan"])
        method_means = {m: _mean(v) for m, v in by_method.items() if v}
        if method_means:
            best_name = min(method_means, key=method_means.get)
            best_greedy_by_tc[tc] = method_means[best_name]
            best_greedy_name_by_tc[tc] = best_name

    lines = [
        "# MILP-supervised C0 vs greedy vs MILP on the scale ladder",
        "",
        "MILP status: 12/24 all optimal, 42 partial timeout, 60+ mostly timeout. "
        "Timed-out MILP rows contribute the best incumbent it found; the true "
        "optimum can only be lower, so new-C0 / greedy gaps against MILP are "
        "**lower bounds** at large scale.",
        "",
        "| task_count | MILP incumbent | new C0 (MILP-IL) | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 beats greedy by** |",
        "|---:|---:|---:|---:|---|---:|---:|",
    ]
    summary_rows = []
    for tc in TASK_COUNTS:
        milp = _mean(milp_by_tc.get(tc, []))
        new = _mean(new_by_tc.get(tc, []))
        best_g = best_greedy_by_tc.get(tc)
        best_g_name = best_greedy_name_by_tc.get(tc, "n/a")
        def gap_vs(val, ref):
            if val is None or ref is None or ref == 0:
                return None
            return (val - ref) / ref * 100.0
        def diff_pct(a, b):
            if a is None or b is None or b == 0:
                return None
            return (a - b) / b * 100.0
        summary_rows.append(
            {
                "task_count": tc,
                "milp_incumbent": milp,
                "new_c0_makespan": new,
                "best_greedy_makespan": best_g,
                "best_greedy_name": best_g_name,
                "new_c0_gap_vs_milp_percent": gap_vs(new, milp),
                "best_greedy_gap_vs_milp_percent": gap_vs(best_g, milp),
                "new_c0_vs_best_greedy_percent": diff_pct(new, best_g),
            }
        )
        def fmt(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"
        def fmt_pct(v):
            return f"{v:+.1f}%" if isinstance(v, (int, float)) else "n/a"
        lines.append(
            f"| {tc} | {fmt(milp)} | {fmt(new)} | {fmt_pct(gap_vs(new, milp))} | "
            f"{fmt(best_g)} ({best_g_name}) | {fmt_pct(gap_vs(best_g, milp))} | "
            f"{fmt_pct(diff_pct(new, best_g))} |"
        )

    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "schema": "md-c0-milp-supervised-scale-eval-1.0",
        "checkpoint": str(checkpoint),
        "task_counts": list(TASK_COUNTS),
        "seeds": list(SEEDS),
        "row_count_new": len(rows_new),
        "row_count_baselines": len(rows_baselines),
        "per_task_count": summary_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/md_c0_milp_supervised_scale_eval_2026-09-13"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--milp-scale", type=Path, default=DEFAULT_MILP_SCALE)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.output, args.device, args.milp_scale), indent=2))
