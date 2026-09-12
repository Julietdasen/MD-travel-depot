"""Train preregistered exact-action rerankers on the Ticket 48 package."""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DATA_DIR = Path("reports/md_exact_action_candidate_diagnostic_2026-09-06")
OUTPUT_DIR = Path("reports/md_exact_action_scorer_ablation_2026-09-07")
MODEL_SEEDS = (3101, 3102, 3103)
TRAINING_SEEDS = {3101: 4101, 3102: 4102, 3103: 4103}
K_VALUES = (1, 2, 4, 8, 16)
EDGE_DIM = 7


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 200
    evaluation_interval: int = 10
    learning_rate: float = 0.003
    hidden_dim: int = 32
    regression_weight: float = 0.25
    classification_weight: float = 0.25


@dataclass(frozen=True, slots=True)
class ActionExample:
    key: tuple[tuple[int, int], ...]
    rank: int
    edge_features: torch.Tensor
    regret: float
    tolerance_optimal: bool


@dataclass(frozen=True, slots=True)
class StateExample:
    snapshot_id: str
    pending_count: int
    actions: tuple[ActionExample, ...]


class LinearActionScorer(nn.Module):
    """Linear ablation over a pooled action representation."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(EDGE_DIM, 2)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        return self.head(edges.mean(dim=0))


class DeepSetsActionScorer(nn.Module):
    """Permutation-invariant scorer over whole atomic assignments."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        return self.head(self.edge_encoder(edges).sum(dim=0))


def _key(value: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(pair[0]), int(pair[1])) for pair in value))


def load_examples(model_seed: int) -> tuple[tuple[StateExample, ...], tuple[StateExample, ...]]:
    label_rows = [json.loads(line) for line in (DATA_DIR / "exact_action_package.jsonl").read_text().splitlines()]
    candidate_rows = [json.loads(line) for line in (DATA_DIR / "candidate_records.jsonl").read_text().splitlines()]
    labels = {
        (row["snapshot_id"], _key(row["complete_first_action"]["assignments"])): row
        for row in label_rows
    }
    examples: list[tuple[str, StateExample]] = []
    for row in candidate_rows:
        if int(row["model_seed"]) != model_seed:
            continue
        candidate_keys = tuple(_key(value) for value in row["candidate_action_keys"])
        actions = []
        denominator = max(1, len(candidate_keys) - 1)
        for rank, action_key in enumerate(candidate_keys):
            label = labels[row["snapshot_id"], action_key]
            cardinality = len(action_key)
            edge_features = torch.tensor(
                [
                    [
                        robot_id / 4.0,
                        (task_id - 1) / 11.0,
                        rank / denominator,
                        cardinality / 5.0,
                        int(row["pending_count"]) / 3.0,
                        int(row["candidate_count"]) / 20.0,
                        int(row["complete_action_count"]) / 20.0,
                    ]
                    for robot_id, task_id in action_key
                ],
                dtype=torch.float32,
            )
            actions.append(
                ActionExample(
                    action_key,
                    rank,
                    edge_features,
                    float(label["regret"]),
                    bool(label["tolerance_optimal"]),
                )
            )
        examples.append((str(row["split"]), StateExample(row["snapshot_id"], int(row["pending_count"]), tuple(actions))))
    return (
        tuple(example for split, example in examples if split == "train"),
        tuple(example for split, example in examples if split == "development"),
    )


def _outputs(model: nn.Module, state: StateExample) -> torch.Tensor:
    return torch.stack([model(action.edge_features) for action in state.actions])


def state_loss(model: nn.Module, state: StateExample, config: TrainingConfig) -> torch.Tensor:
    outputs = _outputs(model, state)
    predicted_cost = outputs[:, 0]
    predicted_optimal = outputs[:, 1]
    regrets = torch.tensor([action.regret for action in state.actions], dtype=torch.float32)
    optimal = torch.tensor([action.tolerance_optimal for action in state.actions], dtype=torch.float32)
    target_distribution = optimal / optimal.sum().clamp_min(1.0)
    listwise = -(target_distribution * F.log_softmax(-predicted_cost, dim=0)).sum()
    scale = regrets.max().clamp_min(1.0)
    regression = F.smooth_l1_loss(predicted_cost, regrets / scale)
    classification = F.binary_cross_entropy_with_logits(predicted_optimal, optimal)
    return listwise + config.regression_weight * regression + config.classification_weight * classification


def evaluate(model: nn.Module | None, states: Sequence[StateExample]) -> dict[str, Any]:
    regrets = []
    exact = []
    tolerance = []
    recalls: dict[str, list[float]] = {str(k): [] for k in K_VALUES}
    by_pending: dict[int, list[float]] = {1: [], 2: [], 3: []}
    with torch.no_grad():
        for state in states:
            if model is None:
                ordered = sorted(state.actions, key=lambda action: action.rank)
            else:
                outputs = _outputs(model, state)[:, 0]
                ordered = [state.actions[index] for index in torch.argsort(outputs).tolist()]
            chosen = ordered[0]
            regrets.append(chosen.regret)
            exact.append(float(chosen.regret == 0.0))
            tolerance.append(float(chosen.tolerance_optimal))
            by_pending[state.pending_count].append(float(chosen.tolerance_optimal))
            for k in K_VALUES:
                recalls[str(k)].append(float(any(action.tolerance_optimal for action in ordered[:k])))
    return {
        "state_count": len(states),
        "tolerance_optimal_top1": sum(tolerance) / len(tolerance),
        "exact_optimal_top1": sum(exact) / len(exact),
        "mean_regret": sum(regrets) / len(regrets),
        "recall_at_k": {key: sum(values) / len(values) for key, values in recalls.items()},
        "tolerance_optimal_top1_by_pending": {
            str(key): sum(values) / len(values)
            for key, values in by_pending.items()
            if values
        },
    }


def train_one(kind: str, model_seed: int, config: TrainingConfig) -> dict[str, Any]:
    training_seed = TRAINING_SEEDS[model_seed]
    random.seed(training_seed); np.random.seed(training_seed); torch.manual_seed(training_seed)
    train, development = load_examples(model_seed)
    model: nn.Module = LinearActionScorer() if kind == "linear" else DeepSetsActionScorer(config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    evaluations = []
    best_state = None
    order = list(range(len(train)))
    for epoch in range(1, config.epochs + 1):
        random.Random(training_seed + epoch).shuffle(order)
        model.train()
        losses = []
        for index in order:
            loss = state_loss(model, train[index], config)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % config.evaluation_interval == 0:
            model.eval()
            metrics = {"epoch": epoch, "train_loss": sum(losses) / len(losses), **evaluate(model, development)}
            evaluations.append(metrics)
            selected = max(evaluations, key=lambda row: (row["tolerance_optimal_top1"], -row["mean_regret"], -row["epoch"]))
            if selected is metrics:
                best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    best = max(evaluations, key=lambda row: (row["tolerance_optimal_top1"], -row["mean_regret"], -row["epoch"]))
    destination = OUTPUT_DIR / f"seed{model_seed}" / kind
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "exact-action-scorer-1.0", "kind": kind, "model_seed": model_seed,
                "training_seed": training_seed, "config": asdict(config), "state_dict": best_state,
                "best_development": best}, destination / "best_checkpoint.pt")
    summary = {"kind": kind, "model_seed": model_seed, "training_seed": training_seed,
               "baseline_development": evaluate(None, development), "best_development": best,
               "evaluations": evaluations, "train_snapshot_count": len(train),
               "development_snapshot_count": len(development), "confirmation_read": False}
    (destination / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _write_protocol(config: TrainingConfig) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "exact-action-scorer-ablation-1.0",
        "created_before_training": True,
        "data": str(DATA_DIR),
        "train_snapshot_count": 150,
        "development_snapshot_count": 60,
        "confirmation_forbidden": [76000, 76149],
        "model_seeds": list(MODEL_SEEDS),
        "training_seeds": TRAINING_SEEDS,
        "models": ["residual_additive_baseline", "linear_action", "deepsets_action"],
        "inputs": ["residual_candidate_rank", "canonical_robot_task_ids", "action_cardinality", "pending_count", "candidate_counts"],
        "forbidden_inputs": ["exact_oracle_value", "old_forced_oracle_value", "regret"],
        "loss": "listwise_tie_distribution + 0.25 smooth_l1_regret + 0.25 tie_bce",
        "checkpoint_selection": "max development tolerance-optimal top1, then min mean regret, then earliest epoch",
        "acceptance": {
            "each_seed_not_below_additive_top1": True,
            "mean_top1_gain_at_least": 0.02,
            "mean_regret_strictly_lower": True,
        },
        "config": asdict(config),
        "production_decoder_modified": False,
        "dagger_run": False,
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


def run(config: TrainingConfig) -> Mapping[str, Any]:
    _write_protocol(config)
    runs = [train_one(kind, seed, config) for kind in ("linear", "deepsets") for seed in MODEL_SEEDS]
    baseline = {str(seed): next(row["baseline_development"] for row in runs if row["model_seed"] == seed) for seed in MODEL_SEEDS}
    grouped = {kind: {str(seed): next(row["best_development"] for row in runs if row["kind"] == kind and row["model_seed"] == seed) for seed in MODEL_SEEDS} for kind in ("linear", "deepsets")}
    comparisons = {}
    for kind, values in grouped.items():
        gains = [values[str(seed)]["tolerance_optimal_top1"] - baseline[str(seed)]["tolerance_optimal_top1"] for seed in MODEL_SEEDS]
        comparisons[kind] = {
            "top1_gain_by_seed": {str(seed): gain for seed, gain in zip(MODEL_SEEDS, gains, strict=True)},
            "mean_top1_gain": sum(gains) / len(gains),
            "mean_regret": sum(values[str(seed)]["mean_regret"] for seed in MODEL_SEEDS) / len(MODEL_SEEDS),
            "accepted": all(gain >= 0 for gain in gains) and sum(gains) / len(gains) >= 0.02
            and sum(values[str(seed)]["mean_regret"] for seed in MODEL_SEEDS) < sum(baseline[str(seed)]["mean_regret"] for seed in MODEL_SEEDS),
        }
    selected = max((kind for kind in comparisons), key=lambda kind: (comparisons[kind]["accepted"], comparisons[kind]["mean_top1_gain"], -comparisons[kind]["mean_regret"]))
    summary = {"schema": "exact-action-scorer-ablation-1.0", "baseline": baseline,
               "models": grouped, "comparisons": comparisons, "selected_model": selected if comparisons[selected]["accepted"] else None,
               "status": "scorer_supported_for_confirmation" if comparisons[selected]["accepted"] else "scorer_not_supported",
               "confirmation_read": False, "production_decoder_modified": False}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# Exact Action Scorer Ablation", "", f"Status: **{summary['status']}**.", "",
             "| Model | Seed | Tolerance top-1 | Mean regret |", "|---|---:|---:|---:|"]
    for seed in MODEL_SEEDS: lines.append(f"| Additive baseline | {seed} | {baseline[str(seed)]['tolerance_optimal_top1']:.4f} | {baseline[str(seed)]['mean_regret']:.4f} |")
    for kind in ("linear", "deepsets"):
        for seed in MODEL_SEEDS: lines.append(f"| {kind} | {seed} | {grouped[kind][str(seed)]['tolerance_optimal_top1']:.4f} | {grouped[kind][str(seed)]['mean_regret']:.4f} |")
    lines += ["", "Training used only train snapshots. Development selected checkpoints. Confirmation was not read and production decoding was not changed."]
    (OUTPUT_DIR / "final_report.md").write_text("\n".join(lines) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args(argv); print(json.dumps(run(TrainingConfig(epochs=args.epochs)), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
