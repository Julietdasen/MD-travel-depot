"""Sweep the online scheduler's confidence_threshold for a trained C0
checkpoint across the six MD instance profiles.

Confidence is a score margin: the OnlineMDScheduler falls back to the exact
ExplicitMIPFallback MIP whenever the neural top choice's margin over the next
best alternative is below confidence_threshold. This sweep quantifies the
tradeoff between MIP fallback rate/solver time and makespan as that
threshold rises from "trust the network almost always" to "verify almost
every decision with the exact solver."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_generation.md_instance_profiles import INSTANCE_PROFILES
from data_generation.md_instance_generator import generate_md_instance
from experiments.protocol import DatasetSplit
from models.md_online_features import build_md_policy_inputs_from_simulator
from experiments.md_task_process_context_models import build_context_ablation_model
from schedulers.md_constrained_decoder import LearnedConstrainedDecoder
from schedulers.online_md_scheduler import (
    ExplicitMIPFallback,
    OnlineMDScheduler,
    OnlineNeuralScoreProvider,
)

PROFILES = tuple(INSTANCE_PROFILES)
EVAL_SEEDS = (101, 102, 201, 301)
THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 5.0)
MAX_ROLLOUT_STEPS = 10_000


def run(checkpoint_path: Path, model_seed: int, output: Path, device_name: str) -> dict:
    device = torch.device(device_name)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_context_ablation_model(
        "current_pair_aware", seed=model_seed, device=device
    ).eval()
    model.load_state_dict(payload["model_state_dict"], strict=True)

    rows: list[dict] = []
    for threshold in THRESHOLDS:
        for profile in PROFILES:
            for seed in EVAL_SEEDS:
                domain = generate_md_instance(INSTANCE_PROFILES[profile].config(seed)).domain
                scheduler = OnlineMDScheduler(
                    scorer=OnlineNeuralScoreProvider(
                        model, build_md_policy_inputs_from_simulator, device=device
                    ),
                    decoder=LearnedConstrainedDecoder(),
                    fallback=ExplicitMIPFallback(threads=1),
                    confidence_threshold=threshold,
                    max_steps=MAX_ROLLOUT_STEPS,
                )
                result = scheduler.run(
                    domain,
                    run_id=f"c0-threshold-sweep-{threshold}-{profile}-{seed}",
                    instance_id=f"{profile}-{seed}",
                    seed=seed,
                    split=DatasetSplit.TEST,
                )
                experiment = result.experiment
                rows.append(
                    {
                        "confidence_threshold": threshold,
                        "profile": profile,
                        "seed": seed,
                        "success": experiment.success,
                        "makespan": experiment.makespan,
                        "solver_calls": len(result.fallback_records),
                        "solver_time_seconds": sum(
                            r.solver_time_seconds for r in result.fallback_records
                        ),
                        "decision_count": len(result.decision_records),
                    }
                )

    output.mkdir(parents=True, exist_ok=True)
    (output / "sweep_rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# Confidence threshold sweep (C0, model_seed={})".format(model_seed),
        "",
        "| threshold | mean fallback rate | mean solver time (s) | mean makespan | success |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threshold in THRESHOLDS:
        subset = [r for r in rows if r["confidence_threshold"] == threshold]
        total_decisions = sum(r["decision_count"] for r in subset)
        total_calls = sum(r["solver_calls"] for r in subset)
        total_solver_time = sum(r["solver_time_seconds"] for r in subset)
        successes = [r for r in subset if r["success"]]
        mean_makespan = (
            sum(r["makespan"] for r in successes) / len(successes) if successes else None
        )
        fallback_rate = total_calls / total_decisions if total_decisions else 0.0
        lines.append(
            f"| {threshold} | {fallback_rate:.2%} | {total_solver_time:.3f} | "
            f"{mean_makespan} | {len(successes)}/{len(subset)} |"
        )
    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "schema": "md-c0-confidence-sweep-1.0",
        "model_seed": model_seed,
        "thresholds": list(THRESHOLDS),
        "row_count": len(rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/md_c0_confidence_threshold_sweep_2026-09-13"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.model_seed, args.output, args.device), indent=2))
