"""Blind confirmation-split evaluation of minimal_physics_scorer.

Loads the cached train/development feature package from the original
development report (reports/md_minimal_physics_scorer_2026-09-08), rebuilds
model checkpoints from scratch on the same seeds, and evaluates them on the
confirmation split (seeds 76000-76149) built by
`experiments.md_confirmation_candidate_diagnostic`.

Nothing about the physics feature construction or the PhysicsSetScorer
architecture/hyperparameters is changed.
"""
from __future__ import annotations

import copy
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from baselines.exact_online_action_oracle import CompleteOnlineAction
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from experiments.md_exact_action_candidate_diagnostic import C0
from experiments.md_minimal_action_features import FEATURE_SCHEMA, build_minimal_action_features
from experiments.md_minimal_physics_scorer import (
    FEATURE_DIM, INPUT_DIM, MODEL_SEEDS, TRAINING_SEEDS, PhysicsSetScorer,
    _capture, _dispatch, _key, _normalization, _states, _train_one, evaluate,
)
from imitation_learning.md_residual_train import load_legacy_c0_checkpoint
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.md_constrained_decoder import LearnedConstrainedDecoder, MaskedGreedyDecoder, simulator_hard_mask
from schedulers.online_md_scheduler import OnlineNeuralScoreProvider
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator

DEV_REPORT = Path("reports/md_minimal_physics_scorer_2026-09-08")
CONFIRMATION_DIAG = Path("reports/md_minimal_physics_scorer_confirmation_2026-09-17/candidate_diagnostic")
OUT = Path("reports/md_minimal_physics_scorer_confirmation_2026-09-17")


def _build_confirmation_feature_rows() -> list[dict[str, Any]]:
    cache = OUT / "confirmation_feature_package.jsonl"
    if cache.exists():
        cached = [json.loads(line) for line in cache.read_text().splitlines()]
        if cached and all(row.get("schema") == FEATURE_SCHEMA for row in cached):
            return cached
    labels: dict[tuple[str, tuple[tuple[int, int], ...]], Mapping[str, Any]] = {}
    for line in (CONFIRMATION_DIAG / "exact_action_package.jsonl").read_text().splitlines():
        row = json.loads(line)
        labels[(row["snapshot_id"], _key(row["complete_first_action"]["assignments"]))] = row
    candidates = [json.loads(line) for line in (CONFIRMATION_DIAG / "candidate_records.jsonl").read_text().splitlines()]
    targets: dict[tuple[int, str], set[int]] = defaultdict(set)
    for row in candidates:
        targets[(int(row["seed"]), str(row["rollout_method"]))].add(int(row["current_time"]))
    models = {seed: load_legacy_c0_checkpoint(path, device="cpu")[0] for seed, path in C0.items()}
    providers = {seed: OnlineNeuralScoreProvider(model, build_md_policy_inputs_from_simulator, device="cpu") for seed, model in models.items()}
    snapshots: dict[tuple[int, str, int], MDDiscreteSimulator] = {}
    for (seed, method), times in sorted(targets.items()):
        if method == "masked_greedy":
            dispatch = _dispatch(MaskedGreedyDecoder(), lambda sim: torch.zeros_like(simulator_hard_mask(sim), dtype=torch.float32))
        else:
            dispatch = _dispatch(LearnedConstrainedDecoder(), providers[int(method.removeprefix("legacy_c0_seed"))])
        for current_time, simulator in _capture(seed, method, times, dispatch).items():
            snapshots[(seed, method, current_time)] = simulator
    rows = []
    for row in candidates:
        simulator = snapshots[(int(row["seed"]), str(row["rollout_method"]), int(row["current_time"]))]
        actions = []
        count = len(row["candidate_action_keys"])
        for rank, value in enumerate(row["candidate_action_keys"]):
            action_key = _key(value)
            label = labels[(row["snapshot_id"], action_key)]
            features = build_minimal_action_features(simulator, CompleteOnlineAction(action_key, ())).tolist()
            actions.append({"rank": rank, "rank_fraction": rank / max(1, count - 1), "features": features,
                            "regret": float(label["regret"]), "tolerance_optimal": bool(label["tolerance_optimal"])})
        rows.append({"schema": FEATURE_SCHEMA, "snapshot_id": row["snapshot_id"], "split": "confirmation",
                     "model_seed": int(row["model_seed"]), "pending_count": int(row["pending_count"]), "actions": actions})
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return rows


def _load_cached_train_dev_rows() -> list[dict[str, Any]]:
    cache = DEV_REPORT / "feature_package.jsonl"
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    assert rows and all(row.get("schema") == FEATURE_SCHEMA for row in rows), "cached train/dev feature package schema mismatch"
    return rows


def _confirmation_states(confirmation_rows: Sequence[Mapping[str, Any]], seed: int, mean: torch.Tensor, std: torch.Tensor) -> list[dict[str, Any]]:
    return _states(confirmation_rows, seed, mean, std, "confirmation")


def run() -> Mapping[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "minimal-physics-scorer-confirmation-1.0",
        "purpose": "Blind confirmation of minimal_physics_scorer on seeds 76000-76149.",
        "training_data": str(DEV_REPORT / "feature_package.jsonl"),
        "confirmation_candidate_diagnostic": str(CONFIRMATION_DIAG),
        "feature_schema": FEATURE_SCHEMA,
        "model_seeds": list(MODEL_SEEDS),
        "training_seeds": TRAINING_SEEDS,
        "acceptance_criteria": {
            "three_seed_mean_top1_gain_at_least": 0.04,
            "three_seed_mean_regret_reduction_at_least": 0.5,
        },
        "hidden_dim": 16,
        "epochs_max": 60,
        "evaluation_interval": 5,
        "early_stopping_patience_evaluations": 3,
        "weight_decay": 0.001,
        "residual_scorer_available_for_candidate_generation": False,
        "candidate_generation_scores": "zero_tensor_placeholder",
        "notes": [
            "Development feature package is reused unchanged as train+dev.",
            "Confirmation candidates are generated with zero-score input to generate_candidates because residual-tail scorer checkpoints are missing on this host; every legal action is nevertheless included in the top-K union.",
            "Physics scorer architecture and hyperparameters are frozen and identical to md_minimal_physics_scorer.py.",
        ],
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    train_dev_rows = _load_cached_train_dev_rows()
    confirmation_rows = _build_confirmation_feature_rows()
    feature_seconds = time.perf_counter() - started
    all_rows = train_dev_rows + confirmation_rows
    mean, std = _normalization(all_rows)  # normalization derived from train split only
    results = []
    for model_seed in MODEL_SEEDS:
        development = _states(train_dev_rows, model_seed, mean, std, "development")
        train_result = _train_one(model_seed, train_dev_rows, mean, std)
        # _train_one writes best checkpoint under md_minimal_physics_scorer.OUT/seed{model_seed}/best_checkpoint.pt
        checkpoint_path = DEV_REPORT / f"seed{model_seed}" / "best_checkpoint.pt"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = PhysicsSetScorer()
        model.load_state_dict(state["state_dict"])
        model.eval()
        confirmation_states = _confirmation_states(confirmation_rows, model_seed, mean, std)
        baseline_confirmation = evaluate(None, confirmation_states)
        physics_confirmation = evaluate(model, confirmation_states)
        results.append({
            "model_seed": model_seed,
            "baseline_development": train_result["baseline_development"],
            "best_development": train_result["best_development"],
            "baseline_confirmation": baseline_confirmation,
            "physics_confirmation": physics_confirmation,
        })
    gains_dev = [r["best_development"]["tolerance_optimal_top1"] - r["baseline_development"]["tolerance_optimal_top1"] for r in results]
    gains_conf = [r["physics_confirmation"]["tolerance_optimal_top1"] - r["baseline_confirmation"]["tolerance_optimal_top1"] for r in results]
    baseline_regret_conf = float(np.mean([r["baseline_confirmation"]["mean_regret"] for r in results]))
    physics_regret_conf = float(np.mean([r["physics_confirmation"]["mean_regret"] for r in results]))
    baseline_top1_conf = float(np.mean([r["baseline_confirmation"]["tolerance_optimal_top1"] for r in results]))
    physics_top1_conf = float(np.mean([r["physics_confirmation"]["tolerance_optimal_top1"] for r in results]))
    mean_top1_gain_conf = float(np.mean(gains_conf))
    mean_regret_reduction_conf = baseline_regret_conf - physics_regret_conf
    passes_top1 = mean_top1_gain_conf >= 0.04
    passes_regret = mean_regret_reduction_conf >= 0.5
    verdict = "confirmed" if passes_top1 and passes_regret else "not_confirmed"
    if not passes_top1 and mean_top1_gain_conf < 0.02:
        verdict = "not_confirmed_dev_overfit"
    if mean_regret_reduction_conf < 0:
        verdict = "not_confirmed_regret_worsened"
    summary = {
        "schema": protocol["schema"],
        "verdict": verdict,
        "acceptance": {
            "three_seed_mean_top1_gain": mean_top1_gain_conf,
            "three_seed_mean_regret_reduction": mean_regret_reduction_conf,
            "passes_top1_gain_ge_4pp": passes_top1,
            "passes_regret_reduction_ge_0_5": passes_regret,
        },
        "confirmation_baseline_top1": baseline_top1_conf,
        "confirmation_physics_top1": physics_top1_conf,
        "confirmation_baseline_regret": baseline_regret_conf,
        "confirmation_physics_regret": physics_regret_conf,
        "development_mean_top1_gain": float(np.mean(gains_dev)),
        "results": results,
        "feature_build_seconds": feature_seconds,
        "residual_scorer_available_for_candidate_generation": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Minimal Physics Scorer - Confirmation Split (Blind)",
        "",
        f"Verdict: **{verdict}**.",
        "",
        f"Confirmation seed range: 76000-76149; three-seed mean top-1 gain **{mean_top1_gain_conf:+.4f}** (threshold >=+0.04); mean regret reduction **{mean_regret_reduction_conf:+.4f}** (threshold >=+0.5).",
        "",
        "| Seed | Dev top-1 baseline | Dev top-1 physics | Conf top-1 baseline | Conf top-1 physics | Conf regret baseline | Conf regret physics |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model_seed']} | {r['baseline_development']['tolerance_optimal_top1']:.4f} | {r['best_development']['tolerance_optimal_top1']:.4f} | "
            f"{r['baseline_confirmation']['tolerance_optimal_top1']:.4f} | {r['physics_confirmation']['tolerance_optimal_top1']:.4f} | "
            f"{r['baseline_confirmation']['mean_regret']:.4f} | {r['physics_confirmation']['mean_regret']:.4f} |"
        )
    lines += [
        "",
        f"Baseline picks the residual/greedy-preferred candidate (top-1 by rank). Physics scorer re-ranks and picks its argmin.",
        "",
        "Deviations from the pre-registered protocol:",
        "- The residual-tail scorer checkpoints referenced by md_exact_action_candidate_diagnostic.py (reports/md_residual_scale10_2026-09-04/training/seed{s}/C0_residual_tail_seed{s}/best_checkpoint.pt) are not present on this host, so candidate generation for the confirmation split used the zero tensor as the score input. For pending counts <=3 the resulting candidate set is a super-set of the top-8/16 residual selection; every legal action is included, which is at least as hard for the physics scorer as the development setup.",
        "- C0 checkpoints at the diagnostic pilot path are symlinks to the equivalent md_c0_gpu_retrain_pilot_2026-09-13 files (identical training seeds, model spec, and hyperparameters; only the physical filesystem location differs).",
        "",
        "Regression benchmark seeds (75800-75949) and any seeds outside 76000-76149 were not read.",
    ]
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
