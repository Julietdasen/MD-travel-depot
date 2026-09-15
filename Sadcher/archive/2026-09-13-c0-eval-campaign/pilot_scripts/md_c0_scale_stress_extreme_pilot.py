"""Extreme-scale continuation of md_c0_scale_stress_pilot.

The first stress pass (task_count 12-42) showed exact MIP fallback solver
time growing but staying well under a second even at 42 tasks / 16 robots.
This pushes further (task_count 60-150) to find where solver time or
rollout wall time actually starts to strain, using the same scaled-profile
construction and the same trained C0 checkpoint.
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

TASK_COUNTS = (60, 78, 96, 114, 132, 150)
SEEDS = (9301, 9302)
MAX_ROLLOUT_STEPS = 20_000


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
            started = time.perf_counter()
            result = scheduler.run(
                domain,
                run_id=f"c0-scale-stress-extreme-{task_count}-{seed}",
                instance_id=f"scalex{task_count}-{seed}",
                seed=seed,
                split=DatasetSplit.TEST,
            )
            wall_elapsed = time.perf_counter() - started
            experiment = result.experiment
            solver_times = [r.solver_time_seconds for r in result.fallback_records]
            row = {
                "task_count": task_count,
                "robot_count": len(domain.robots),
                "seed": seed,
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
    (output / "stress_rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# C0 + exact MIP fallback extreme scale-up stress test",
        "",
        "| task_count | robots | n | success | mean solver calls | mean solver time (s) | max single solver call (s) | mean wall time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task_count in TASK_COUNTS:
        subset = [r for r in rows if r["task_count"] == task_count]
        successes = [r for r in subset if r["success"]]
        mean_calls = sum(r["solver_calls"] for r in subset) / len(subset)
        mean_solver_time = sum(r["solver_time_total_seconds"] for r in subset) / len(subset)
        max_solver_call = max((r["solver_time_max_seconds"] for r in subset), default=0.0)
        mean_wall = sum(r["wall_time_seconds"] for r in subset) / len(subset)
        robots = subset[0]["robot_count"] if subset else None
        lines.append(
            f"| {task_count} | {robots} | {len(subset)} | {len(successes)}/{len(subset)} | "
            f"{mean_calls:.2f} | {mean_solver_time:.3f} | {max_solver_call:.3f} | {mean_wall:.3f} |"
        )
    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "schema": "md-c0-scale-stress-extreme-1.0",
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
    parser.add_argument("--output", type=Path, default=Path("reports/md_c0_scale_stress_extreme_pilot_2026-09-13"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.model_seed, args.output, args.device), indent=2))
