"""Evaluate a trained C0 checkpoint against greedy baselines across the six
scale profiles from data_generation.md_instance_profiles.

This does not train a new model. It loads an existing C0 checkpoint
(produced by experiments.md_c0_end_to_end_diagnostic_pilot train-c0) and
rolls it out, with the exact MIP fallback enabled, on the same profile/seed
grid used by experiments/md_profile_pilot.py, so C0 and the three greedy
baselines are compared on identical instances.
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
MODEL_SEEDS = (3101, 3102, 3103)
FALLBACK_CONFIDENCE_THRESHOLD = 0.0
MAX_ROLLOUT_STEPS = 10_000
GREEDY_BASELINES = {
    "greedy_distance": run_greedy_distance,
    "greedy_eta": run_greedy_eta,
    "greedy_unlock": run_greedy_unlock,
}


def _load_c0_checkpoint(checkpoint_path: Path, model_seed: int, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_context_ablation_model(
        "current_pair_aware", seed=model_seed, device=device
    ).eval()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def _run_c0(model, domain, *, run_id: str, instance_id: str, seed: int, device: torch.device) -> dict:
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
        "failure_reason": None if experiment.failure_reason is None else experiment.failure_reason.value,
        "solver_calls": len(result.fallback_records),
        "solver_time_seconds": sum(r.solver_time_seconds for r in result.fallback_records),
        "decision_count": len(result.decision_records),
        "wall_time_seconds": experiment.wall_time_seconds,
    }


def run(checkpoint_root: Path, output: Path, device_name: str, eval_seeds: tuple[int, ...] = EVAL_SEEDS) -> dict:
    device = torch.device(device_name)
    rows: list[dict] = []
    for model_seed in MODEL_SEEDS:
        checkpoint_path = checkpoint_root / f"C0_seed{model_seed}" / "checkpoints" / "best_checkpoint.pt"
        model = _load_c0_checkpoint(checkpoint_path, model_seed, device)
        for profile in PROFILES:
            for seed in eval_seeds:
                domain = generate_md_instance(INSTANCE_PROFILES[profile].config(seed)).domain
                c0_row = _run_c0(
                    model,
                    domain,
                    run_id=f"c0-profile-eval-{profile}-{seed}-{model_seed}",
                    instance_id=f"{profile}-{seed}",
                    seed=seed,
                    device=device,
                )
                rows.append(
                    {
                        "method": "C0",
                        "model_seed": model_seed,
                        "profile": profile,
                        "seed": seed,
                        **c0_row,
                    }
                )
        del model
        torch.cuda.empty_cache()

    baseline_rows: list[dict] = []
    for profile in PROFILES:
        for seed in eval_seeds:
            domain = generate_md_instance(INSTANCE_PROFILES[profile].config(seed)).domain
            for name, fn in GREEDY_BASELINES.items():
                result = fn(
                    MDDiscreteSimulator(domain),
                    run_id=f"c0-profile-eval-{name}-{profile}-{seed}",
                    instance_id=f"{profile}-{seed}",
                    seed=seed,
                    split=DatasetSplit.TEST,
                    max_steps=MAX_ROLLOUT_STEPS,
                )
                baseline_rows.append(
                    {
                        "method": name,
                        "profile": profile,
                        "seed": seed,
                        "success": result.success,
                        "makespan": result.makespan,
                        "wall_time_seconds": result.wall_time_seconds,
                    }
                )

    output.mkdir(parents=True, exist_ok=True)
    (output / "c0_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "baseline_rows.json").write_text(json.dumps(baseline_rows, indent=2) + "\n")

    summary_lines = [
        "# C0 vs greedy baselines across the six MD instance profiles",
        "",
        "| profile | method | n | success | mean makespan |",
        "|---|---|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for method in ("C0", *GREEDY_BASELINES):
            if method == "C0":
                subset = [r for r in rows if r["profile"] == profile]
            else:
                subset = [r for r in baseline_rows if r["profile"] == profile and r["method"] == method]
            ok = [r["makespan"] for r in subset if r["success"] and r["makespan"] is not None]
            mean_makespan = (sum(ok) / len(ok)) if ok else "NA"
            summary_lines.append(
                f"| {profile} | {method} | {len(subset)} | {len(ok)}/{len(subset)} | {mean_makespan} |"
            )
    (output / "final_report.md").write_text("\n".join(summary_lines) + "\n")

    summary = {
        "schema": "md-c0-profile-eval-1.0",
        "profiles": list(PROFILES),
        "model_seeds": list(MODEL_SEEDS),
        "c0_row_count": len(rows),
        "baseline_row_count": len(baseline_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/md_c0_profile_eval_pilot_2026-09-13"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint_root, args.output, args.device, tuple(args.seeds)), indent=2))
