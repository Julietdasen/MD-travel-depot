"""Combined scale + structural extreme: where do the two axes compound?

md_c0_scale_stress_extreme_pilot pushed task_count to 150 with
scale_medium-like ratios and found no failures, only growing solver time.
md_c0_structural_stress_pilot pushed precedence_density/capacity_slack/
transport_ratio to their bounds at a fixed task_count=24 and also found no
failures. Neither axis alone broke the exact MIP fallback. This combines
both: real scale (task_count=90, comparable to the extreme pilot's middle
tier) together with all three structural knobs maxed out at once, to see
whether the combination is where the safety net actually strains.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from experiments.protocol import DatasetSplit
from models.md_online_features import build_md_policy_inputs_from_simulator
from experiments.md_task_process_context_models import build_context_ablation_model
from schedulers.md_constrained_decoder import LearnedConstrainedDecoder
from schedulers.online_md_scheduler import (
    ExplicitMIPFallback,
    OnlineMDScheduler,
    OnlineNeuralScoreProvider,
)

TASK_COUNT = 90
SEEDS = (9501, 9502, 9503)
MAX_ROLLOUT_STEPS = 20_000

VARIANTS = {
    "scale_only": dict(
        transport_ratio=1.0 / 3.0,
        precedence_density=0.30,
        capacity_slack=0.10,
    ),
    "scale_plus_combined_hard": dict(
        transport_ratio=0.5,
        precedence_density=1.0,
        capacity_slack=0.0,
    ),
}


def _config(overrides: dict, seed: int) -> MDGeneratorConfig:
    scale = TASK_COUNT / 18
    return MDGeneratorConfig(
        seed=seed,
        task_count=TASK_COUNT,
        critical_path_length=6,
        speed_ratio=0.70,
        process_robot_count=max(1, round(4 * scale)),
        transport_robot_count=max(1, round(3 * scale)),
        skill_count=3,
        **overrides,
    )


def run(checkpoint_path: Path, model_seed: int, output: Path, device_name: str) -> dict:
    device = torch.device(device_name)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_context_ablation_model(
        "current_pair_aware", seed=model_seed, device=device
    ).eval()
    model.load_state_dict(payload["model_state_dict"], strict=True)

    rows: list[dict] = []
    for variant_name, overrides in VARIANTS.items():
        for seed in SEEDS:
            domain = generate_md_instance(_config(overrides, seed)).domain
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
                run_id=f"c0-combined-extreme-{variant_name}-{seed}",
                instance_id=f"combined-{variant_name}-{seed}",
                seed=seed,
                split=DatasetSplit.TEST,
            )
            wall_elapsed = time.perf_counter() - started
            experiment = result.experiment
            solver_times = [r.solver_time_seconds for r in result.fallback_records]
            row = {
                "variant": variant_name,
                "seed": seed,
                "robot_count": len(domain.robots),
                "success": experiment.success,
                "failure_reason": None if experiment.failure_reason is None else experiment.failure_reason.value,
                "makespan": experiment.makespan,
                "decision_count": len(result.decision_records),
                "solver_calls": len(result.fallback_records),
                "solver_time_total_seconds": sum(solver_times),
                "solver_time_max_seconds": max(solver_times) if solver_times else 0.0,
                "wall_time_seconds": wall_elapsed,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    (output / "combined_rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        f"# Combined scale + structural extreme stress test (task_count={TASK_COUNT})",
        "",
        "| variant | n | success | mean solver calls | mean solver time (s) | max single solver call (s) | mean wall time (s) | mean makespan |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_name in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant_name]
        successes = [r for r in subset if r["success"]]
        mean_calls = sum(r["solver_calls"] for r in subset) / len(subset)
        mean_solver_time = sum(r["solver_time_total_seconds"] for r in subset) / len(subset)
        max_solver_call = max((r["solver_time_max_seconds"] for r in subset), default=0.0)
        mean_wall = sum(r["wall_time_seconds"] for r in subset) / len(subset)
        mean_makespan = (
            sum(r["makespan"] for r in successes) / len(successes) if successes else None
        )
        lines.append(
            f"| {variant_name} | {len(subset)} | {len(successes)}/{len(subset)} | "
            f"{mean_calls:.2f} | {mean_solver_time:.3f} | {max_solver_call:.3f} | {mean_wall:.3f} | {mean_makespan} |"
        )
    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "schema": "md-c0-combined-extreme-1.0",
        "model_seed": model_seed,
        "task_count": TASK_COUNT,
        "variants": list(VARIANTS),
        "row_count": len(rows),
        "all_success": all(r["success"] for r in rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/md_c0_combined_extreme_pilot_2026-09-13"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.model_seed, args.output, args.device), indent=2))
