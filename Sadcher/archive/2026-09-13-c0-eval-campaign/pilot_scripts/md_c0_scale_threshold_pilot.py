"""Does confidence_threshold trade off differently at large scale?

md_c0_mip_threads_pilot found that adding solver threads barely reduces
total MIP fallback time at task_count=150 and suggested the real lever is
triggering the exact solver less often. md_c0_confidence_threshold_sweep
already showed raising confidence_threshold *raises* fallback rate (it
makes the scheduler less trusting of the network, not more), so that
particular knob does not help. This checks the sweep directly at
task_count=150 to confirm that conclusion holds at scale, not just in the
six-profile size range.
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

TASK_COUNT = 150
SEEDS = (9301, 9302)
THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 5.0)
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
    for threshold in THRESHOLDS:
        for seed in SEEDS:
            domain = generate_md_instance(_scaled_config(TASK_COUNT, seed)).domain
            scheduler = OnlineMDScheduler(
                scorer=OnlineNeuralScoreProvider(
                    model, build_md_policy_inputs_from_simulator, device=device
                ),
                decoder=LearnedConstrainedDecoder(),
                fallback=ExplicitMIPFallback(threads=1),
                confidence_threshold=threshold,
                max_steps=MAX_ROLLOUT_STEPS,
            )
            started = time.perf_counter()
            result = scheduler.run(
                domain,
                run_id=f"c0-scale-threshold-{threshold}-{seed}",
                instance_id=f"scalethresh{threshold}-{seed}",
                seed=seed,
                split=DatasetSplit.TEST,
            )
            wall_elapsed = time.perf_counter() - started
            experiment = result.experiment
            solver_times = [r.solver_time_seconds for r in result.fallback_records]
            row = {
                "confidence_threshold": threshold,
                "seed": seed,
                "success": experiment.success,
                "makespan": experiment.makespan,
                "decision_count": len(result.decision_records),
                "solver_calls": len(result.fallback_records),
                "solver_time_total_seconds": sum(solver_times),
                "wall_time_seconds": wall_elapsed,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    (output / "scale_threshold_rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        f"# Confidence threshold sweep at scale (task_count={TASK_COUNT}, C0 model_seed={model_seed})",
        "",
        "| threshold | n | success | mean fallback rate | mean solver calls | mean solver time (s) | mean wall time (s) | mean makespan |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in THRESHOLDS:
        subset = [r for r in rows if r["confidence_threshold"] == threshold]
        successes = [r for r in subset if r["success"]]
        total_decisions = sum(r["decision_count"] for r in subset)
        total_calls = sum(r["solver_calls"] for r in subset)
        fallback_rate = total_calls / total_decisions if total_decisions else 0.0
        mean_calls = total_calls / len(subset)
        mean_solver_time = sum(r["solver_time_total_seconds"] for r in subset) / len(subset)
        mean_wall = sum(r["wall_time_seconds"] for r in subset) / len(subset)
        mean_makespan = (
            sum(r["makespan"] for r in successes) / len(successes) if successes else None
        )
        lines.append(
            f"| {threshold} | {len(subset)} | {len(successes)}/{len(subset)} | {fallback_rate:.2%} | "
            f"{mean_calls:.2f} | {mean_solver_time:.3f} | {mean_wall:.3f} | {mean_makespan} |"
        )
    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "schema": "md-c0-scale-threshold-1.0",
        "model_seed": model_seed,
        "task_count": TASK_COUNT,
        "thresholds": list(THRESHOLDS),
        "row_count": len(rows),
        "all_success": all(r["success"] for r in rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/md_c0_scale_threshold_pilot_2026-09-13"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.model_seed, args.output, args.device), indent=2))
