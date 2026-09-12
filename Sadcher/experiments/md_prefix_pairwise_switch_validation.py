"""Train and validate a pairwise classifier that approves gate replacements."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.md_prefix_autoregressive_pilot import MODEL_SEEDS, _candidate, _label_state
from experiments.md_prefix_cost_gate_large_validation import _load_frozen_gate, _metrics_for_rows
from experiments.md_prefix_cost_gate_pilot import (
    INPUT_DIM,
    GateState,
    _action_features,
    _ensemble_cost,
    _label_from_saved_row,
    _load_existing_split,
    _load_prefix,
)
from reinforcement_learning.md_joint_action import oracle_aligned_action_is_legal


OUTPUT = Path("reports/md_prefix_pairwise_switch_validation_2026-09-10")
CLASSIFIER_SEEDS = (201, 202, 203)
STATE_POOL = range(79250, 80500)
STATE_COUNT = 50
MIN_ACTIONS = 24
MAX_ACTIONS = 96
PAIR_DIM = INPUT_DIM * 3 + 2
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0)
MAX_EPOCHS = 250
PATIENCE = 30
LEARNING_RATE = 1e-3
BOOTSTRAP_REPEATS = 10_000


@dataclass(frozen=True, slots=True)
class BeamPairSet:
    pair_features: torch.Tensor
    improvements: torch.Tensor


class MDPairwiseSwitchClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PAIR_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.network(pair_features).squeeze(-1)


def _pair_feature(
    base: torch.Tensor,
    candidate: torch.Tensor,
    beam_rank: int,
    log_probability_gap: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    base_norm = (base - mean) / std
    candidate_norm = (candidate - mean) / std
    return torch.cat(
        (
            base_norm,
            candidate_norm,
            candidate_norm - base_norm,
            torch.tensor([beam_rank / 15.0, log_probability_gap / 20.0]),
        )
    )


@torch.no_grad()
def _beam_data(model, gates, state: GateState, mean, std) -> dict[str, Any]:
    candidates = model.beam_candidates(state.observation, beam_width=16)
    actions = [tuple(int(value) for value in row) for row in candidates.actions.tolist()]
    features = torch.stack(
        [_action_features(state.observation, state.simulator, action) for action in actions]
    )
    predicted_cost = _ensemble_cost(gates, features, mean, std)
    gate_index = int(torch.argmin(predicted_cost))
    pairs = torch.stack(
        [
            _pair_feature(
                features[0],
                features[index],
                index,
                float(candidates.log_probabilities[0] - candidates.log_probabilities[index]),
                mean,
                std,
            )
            for index in range(1, len(actions))
        ]
    )
    regrets = torch.tensor(
        [state.regret_by_action.get(action, float("inf")) for action in actions],
        dtype=torch.float32,
    )
    return {
        "actions": actions,
        "features": features,
        "pairs": pairs,
        "regrets": regrets,
        "gate_index": gate_index,
    }


def _training_examples(states, prefixes, gates, mean, std) -> BeamPairSet:
    pair_features = []
    improvements = []
    for model in prefixes.values():
        for state in states:
            data = _beam_data(model, gates, state, mean, std)
            pair_features.append(data["pairs"])
            improvements.append(data["regrets"][0] > data["regrets"][1:])
    return BeamPairSet(torch.cat(pair_features), torch.cat(improvements).float())


def _train_classifier(
    seed: int,
    train: BeamPairSet,
    development: BeamPairSet,
) -> tuple[MDPairwiseSwitchClassifier, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MDPairwiseSwitchClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    positives = train.improvements.sum()
    negatives = train.improvements.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).detach()
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(train.pair_features.shape[0], generator=generator)
        for batch_indices in order.split(128):
            logits = model(train.pair_features[batch_indices])
            loss = F.binary_cross_entropy_with_logits(
                logits,
                train.improvements[batch_indices],
                pos_weight=positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.grad is not None
            ):
                raise FloatingPointError("non-finite switch-classifier gradient")
            optimizer.step()
        model.eval()
        with torch.no_grad():
            development_loss = float(
                F.binary_cross_entropy_with_logits(
                    model(development.pair_features),
                    development.improvements,
                    pos_weight=positive_weight,
                )
            )
        if development_loss < best_loss - 1e-8:
            best_loss = development_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("switch classifier did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    destination = OUTPUT / "classifier" / f"seed{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "md-prefix-pairwise-switch-classifier-1.0",
            "state_dict": best_state,
            "classifier_seed": seed,
            "best_epoch": best_epoch,
            "development_bce": best_loss,
        },
        destination / "best_checkpoint.pt",
    )
    return model, {
        "classifier_seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "development_bce": best_loss,
    }


@torch.no_grad()
def _switch_probability(classifiers, pair_feature: torch.Tensor) -> float:
    probabilities = torch.stack(
        [torch.sigmoid(classifier(pair_feature.unsqueeze(0))).squeeze(0) for classifier in classifiers]
    )
    return float(probabilities.median())


def _policy_records(states, prefixes, gates, classifiers, mean, std) -> list[dict[str, Any]]:
    records = []
    with torch.no_grad():
        for model_seed, model in prefixes.items():
            for state in states:
                data = _beam_data(model, gates, state, mean, std)
                gate_index = data["gate_index"]
                probability = (
                    0.0
                    if gate_index == 0
                    else _switch_probability(classifiers, data["pairs"][gate_index - 1])
                )
                records.append(
                    {
                        "model_seed": model_seed,
                        "state_seed": state.seed,
                        "state": state,
                        "actions": data["actions"],
                        "regrets": data["regrets"].tolist(),
                        "gate_index": gate_index,
                        "switch_probability": probability,
                    }
                )
    return records


def _threshold_metrics(records, threshold: float) -> dict[str, Any]:
    regrets = []
    switches = 0
    for record in records:
        selected = (
            record["gate_index"]
            if record["gate_index"] != 0 and record["switch_probability"] >= threshold
            else 0
        )
        switches += int(selected != 0)
        regrets.append(float(record["regrets"][selected]))
    values = np.asarray(regrets)
    return {
        "threshold": threshold,
        "mean_regret": float(values.mean()),
        "top1_optimal": float((values <= 1e-6).mean()),
        "max_regret": float(values.max()),
        "catastrophic_regret_at_least_20": int((values >= 20.0).sum()),
        "switches": switches,
    }


def _select_threshold(records) -> tuple[float, list[dict[str, Any]]]:
    results = [_threshold_metrics(records, threshold) for threshold in THRESHOLDS]
    baseline = results[-1]
    safe = [
        row
        for row in results
        if row["top1_optimal"] >= baseline["top1_optimal"]
        and row["max_regret"] <= baseline["max_regret"]
        and row["catastrophic_regret_at_least_20"] <= baseline["catastrophic_regret_at_least_20"]
    ]
    selected = min(safe, key=lambda row: (row["mean_regret"], row["threshold"]))
    return float(selected["threshold"]), results


def _manifest() -> list[dict[str, int]]:
    selected = []
    for seed in STATE_POOL:
        _candidate_data, state, legal = _candidate(seed)
        if MIN_ACTIONS <= len(legal) <= MAX_ACTIONS:
            selected.append(
                {
                    "seed": seed,
                    "pending_count": len(state.pending_task_ids),
                    "legal_action_count": len(legal),
                }
            )
        if len(selected) == STATE_COUNT:
            return selected
    raise RuntimeError(f"only found {len(selected)} eligible validation states")


def _label_states(manifest: Sequence[dict[str, int]]) -> tuple[list[GateState], int]:
    states = []
    rows = []
    action_count = 0
    for entry in manifest:
        labeled, row = _label_state("pairwise_validation", int(entry["seed"]))
        if labeled.legal_action_count != int(entry["legal_action_count"]):
            raise AssertionError("pairwise-validation action count changed")
        candidate, _residual, _legal = _candidate(int(entry["seed"]))
        states.append(_label_from_saved_row("pairwise_validation", row, candidate))
        rows.append(row)
        action_count += labeled.legal_action_count
    (OUTPUT / "labels.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return states, action_count


@torch.no_grad()
def _evaluate(records, zeros, threshold: float) -> dict[str, Any]:
    decisions = []
    timings = []
    for record in records:
        started = time.perf_counter()
        selected = (
            record["gate_index"]
            if record["gate_index"] != 0 and record["switch_probability"] >= threshold
            else 0
        )
        state = record["state"]
        model_seed = record["model_seed"]
        base_action = record["actions"][0]
        pairwise_action = record["actions"][selected]
        unrestricted_action = record["actions"][record["gate_index"]]
        zero_action = tuple(
            int(value)
            for value in zeros[model_seed]
            .beam_decode(state.observation, beam_width=16)
            .actions[0]
            .tolist()
        )
        timings.append((time.perf_counter() - started) * 1000.0)
        decisions.append(
            {
                "model_seed": model_seed,
                "state_seed": state.seed,
                "base_action": list(base_action),
                "pairwise_action": list(pairwise_action),
                "unrestricted_action": list(unrestricted_action),
                "prefix_zero_action": list(zero_action),
                "base_regret": state.regret_by_action.get(base_action, float("inf")),
                "pairwise_regret": state.regret_by_action.get(pairwise_action, float("inf")),
                "unrestricted_regret": state.regret_by_action.get(unrestricted_action, float("inf")),
                "prefix_zero_regret": state.regret_by_action.get(zero_action, float("inf")),
                "minimum_continuation_cost": state.minimum_continuation_cost,
                "switched": selected != 0,
                "switch_probability": record["switch_probability"],
                "gate_beam_rank": record["gate_index"] + 1,
                "legal": oracle_aligned_action_is_legal(
                    state.observation, torch.tensor([pairwise_action], dtype=torch.long)
                ),
            }
        )
    return {
        "prefix": _metrics_for_rows(decisions, "base_regret"),
        "pairwise_gate": _metrics_for_rows(decisions, "pairwise_regret"),
        "unrestricted_gate": _metrics_for_rows(decisions, "unrestricted_regret"),
        "prefix_zero": _metrics_for_rows(decisions, "prefix_zero_regret"),
        "switches": sum(row["switched"] for row in decisions),
        "illegal_actions": sum(not row["legal"] for row in decisions),
        "unknown_actions": sum(not np.isfinite(row["pairwise_regret"]) for row in decisions),
        "fallback_only_latency_ms": {
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
        },
        "decisions": decisions,
    }


def _paired(decisions) -> dict[str, Any]:
    delta = np.asarray(
        [float(row["base_regret"]) - float(row["pairwise_regret"]) for row in decisions]
    )
    state_delta = np.asarray(
        [
            np.mean(
                [
                    float(row["base_regret"]) - float(row["pairwise_regret"])
                    for row in decisions
                    if int(row["state_seed"]) == state_seed
                ]
            )
            for state_seed in sorted({int(row["state_seed"]) for row in decisions})
        ]
    )
    generator = np.random.default_rng(20260910)
    samples = generator.choice(
        state_delta, size=(BOOTSTRAP_REPEATS, len(state_delta)), replace=True
    ).mean(axis=1)
    return {
        "mean_regret_reduction": float(delta.mean()),
        "improved": int((delta > 0).sum()),
        "equal": int((delta == 0).sum()),
        "worse": int((delta < 0).sum()),
        "state_clustered_95pct_bootstrap_ci": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
    }


def _per_seed(decisions) -> dict[str, Any]:
    result = {}
    for seed in MODEL_SEEDS:
        rows = [row for row in decisions if int(row["model_seed"]) == seed]
        base = _metrics_for_rows(rows, "base_regret")
        gate = _metrics_for_rows(rows, "pairwise_regret")
        result[str(seed)] = {
            "prefix": base,
            "pairwise_gate": gate,
            "mean_regret_reduction": float(base["mean_regret"] - gate["mean_regret"]),
        }
    return result


def run() -> dict[str, Any]:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(f"completed output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    gates, mean, std, _margin = _load_frozen_gate()
    prefixes = {seed: _load_prefix("prefix", seed) for seed in MODEL_SEEDS}
    zeros = {seed: _load_prefix("prefix-zero", seed) for seed in MODEL_SEEDS}
    train_states = _load_existing_split("train")
    development_states = _load_existing_split("development")
    train_examples = _training_examples(train_states, prefixes, gates, mean, std)
    development_examples = _training_examples(development_states, prefixes, gates, mean, std)
    classifiers = []
    training = []
    for seed in CLASSIFIER_SEEDS:
        classifier, metrics = _train_classifier(
            seed, train_examples, development_examples
        )
        classifiers.append(classifier)
        training.append(metrics)
    development_records = _policy_records(
        development_states, prefixes, gates, classifiers, mean, std
    )
    threshold, threshold_results = _select_threshold(development_records)
    manifest = _manifest()
    protocol = {
        "schema": "md-prefix-pairwise-switch-protocol-1.0",
        "created_before_validation_labels": True,
        "training_split": "original train",
        "selection_split": "original development",
        "classifier_seeds": list(CLASSIFIER_SEEDS),
        "threshold_candidates": list(THRESHOLDS),
        "selected_threshold": threshold,
        "development_results": threshold_results,
        "state_pool": [STATE_POOL.start, STATE_POOL.stop - 1],
        "state_count": STATE_COUNT,
        "manifest": manifest,
        "acceptance": {
            "mean_regret_lower": True,
            "top1_not_lower": True,
            "maximum_regret_not_higher": True,
            "catastrophic_count_not_higher": True,
            "at_least_two_model_seeds_not_worse": True,
            "illegal_and_unknown_zero": True,
        },
        "production_decoder_modified": False,
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    states, action_count = _label_states(manifest)
    records = _policy_records(states, prefixes, gates, classifiers, mean, std)
    evaluation = _evaluate(records, zeros, threshold)
    per_seed = _per_seed(evaluation["decisions"])
    paired = _paired(evaluation["decisions"])
    base = evaluation["prefix"]
    pairwise = evaluation["pairwise_gate"]
    checks = {
        "mean_regret_lower": pairwise["mean_regret"] < base["mean_regret"],
        "top1_not_lower": pairwise["top1_optimal"] >= base["top1_optimal"],
        "maximum_regret_not_higher": pairwise["max_regret"] <= base["max_regret"],
        "catastrophic_count_not_higher": (
            pairwise["catastrophic_regret_at_least_20"]
            <= base["catastrophic_regret_at_least_20"]
        ),
        "at_least_two_model_seeds_not_worse": sum(
            row["mean_regret_reduction"] >= 0 for row in per_seed.values()
        ) >= 2,
        "illegal_actions_zero": evaluation["illegal_actions"] == 0,
        "unknown_actions_zero": evaluation["unknown_actions"] == 0,
    }
    summary = {
        "schema": "md-prefix-pairwise-switch-validation-1.0",
        "status": (
            "pairwise_switch_validated"
            if all(checks.values())
            else "pairwise_switch_not_validated"
        ),
        "production_readiness": (
            "eligible_for_separate_integration_review"
            if all(checks.values())
            else "experimental_only"
        ),
        "selected_threshold": threshold,
        "training": training,
        "state_count": len(states),
        "exact_action_label_count": action_count,
        "decision_count": len(evaluation["decisions"]),
        "evaluation": evaluation,
        "per_model_seed": per_seed,
        "paired_analysis": paired,
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "production_decoder_modified": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ci = paired["state_clustered_95pct_bootstrap_ci"]
    (OUTPUT / "final_report.md").write_text(
        "\n".join(
            [
                "# MD Prefix Pairwise Switch Validation",
                "",
                f"Status: **{summary['status']}**.",
                f"Frozen switch probability threshold: {threshold:.3f}.",
                f"Validation: {len(states)} new states, {action_count} exact actions, {len(evaluation['decisions'])} decisions.",
                "",
                "| Decoder | Mean regret | Mean duration | Top-1 | Max regret | Regret >= 20 |",
                "|---|---:|---:|---:|---:|---:|",
                *[
                    f"| {name} | {evaluation[name]['mean_regret']:.3f} | {evaluation[name]['mean_continuation_duration']:.3f} | {evaluation[name]['top1_optimal']:.3f} | {evaluation[name]['max_regret']:.1f} | {evaluation[name]['catastrophic_regret_at_least_20']} |"
                    for name in ("prefix", "pairwise_gate", "unrestricted_gate", "prefix_zero")
                ],
                "",
                f"Approved switches: {evaluation['switches']}.",
                f"Improved/equal/worse: {paired['improved']}/{paired['equal']}/{paired['worse']}.",
                f"Mean paired regret reduction: {paired['mean_regret_reduction']:.3f}.",
                f"State-clustered 95% bootstrap interval: [{ci[0]:.3f}, {ci[1]:.3f}].",
                "",
                "Classifier and threshold were frozen before validation labels were generated. Production scheduling remains unchanged.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))

