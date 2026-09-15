"""Confirm the C0-vs-greedy quality gap found at out-of-range scales with
more seeds.

md_c0_scale_baseline_comparison_pilot found C0's makespan trailing the best
greedy baseline by 12%-24% at task_count 42+, using only 3 seeds per size.
This reruns the two task_counts with the largest observed gap (42 and 114)
with 6 additional fresh seeds each, to check whether the gap holds up or
was partly seed noise from a small sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from experiments.protocol import DatasetSplit
from models.md_online_features import build_md_policy_inputs_from_simulator
from experiments.md_task_process_context_models import build_context_ablation_model
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

TASK_COUNTS = (42, 114)
SEEDS = (9701, 9702, 9703, 9704, 9705, 9706)
MAX_ROLLOUT_STEPS = 20_000
GREEDY_BASELINES = {
    "greedy_distance": run_greedy_distance,
    "greedy_eta": run_greedy_eta,
    "greedy_unlock": run_greedy_unlock,
}


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


def run(checkpoint_path: Path, model_seed: int, output: Path, device_name: str) -> dict:
    device = torch.device(device_name)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_context_ablation_model(
        "current_pair_aware", seed=model_seed, device=device
    ).eval()
    model.load_state_dict(payload["model_state_dict"], strict=True)

    rows: list[dict] = []
    for task_count in TASK_COUNTS:
        for seed in SEEDS:
            domain = generate_md_instance(_scaled_config(task_count, seed)).domain

            scheduler = OnlineMDScheduler(
                scorer=OnlineNeuralScoreProvider(
                    model, build_md_policy_inputs_from_simulator, device=device
                ),
                decoder=LearnedConstrainedDecoder(),
                fallback=ExplicitMIPFallback(threads=1),
                confidence_threshold=0.0,
                max_steps=MAX_ROLLOUT_STEPS,
            )
            c0_result = scheduler.run(
                domain,
                run_id=f"c0-scale-baseline-confirm-c0-{task_count}-{seed}",
                instance_id=f"scaleconfirm{task_count}-{seed}",
                seed=seed,
                split=DatasetSplit.TEST,
            )
            c0_experiment = c0_result.experiment
            rows.append(
                {
                    "task_count": task_count,
                    "seed": seed,
                    "method": "C0",
                    "success": c0_experiment.success,
                    "makespan": c0_experiment.makespan,
                }
            )

            for name, fn in GREEDY_BASELINES.items():
                result = fn(
                    MDDiscreteSimulator(domain),
                    run_id=f"c0-scale-baseline-confirm-{name}-{task_count}-{seed}",
                    instance_id=f"scaleconfirm{task_count}-{seed}",
                    seed=seed,
                    split=DatasetSplit.TEST,
                    max_steps=MAX_ROLLOUT_STEPS,
                )
                rows.append(
                    {
                        "task_count": task_count,
                        "seed": seed,
                        "method": name,
                        "success": result.success,
                        "makespan": result.makespan,
                    }
                )
            print(json.dumps({"task_count": task_count, "seed": seed, "done": True}), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    (output / "confirm_rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# C0 vs greedy baseline gap confirmation (extra seeds)",
        "",
        "| task_count | method | n | success | mean makespan |",
        "|---:|---|---:|---:|---:|",
    ]
    for task_count in TASK_COUNTS:
        for method in ("C0", *GREEDY_BASELINES):
            subset = [r for r in rows if r["task_count"] == task_count and r["method"] == method]
            ok = [r["makespan"] for r in subset if r["success"] and r["makespan"] is not None]
            mean_makespan = (sum(ok) / len(ok)) if ok else "NA"
            lines.append(
                f"| {task_count} | {method} | {len(subset)} | {len(ok)}/{len(subset)} | {mean_makespan} |"
            )
    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "schema": "md-c0-scale-baseline-confirm-1.0",
        "model_seed": model_seed,
        "task_counts": list(TASK_COUNTS),
        "row_count": len(rows),
        "all_success": all(r["success"] for r in rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/md_c0_scale_baseline_confirm_pilot_2026-09-13"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.model_seed, args.output, args.device), indent=2))
