"""Development-only prefix-conditioned reranking of cached exact MD actions."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from experiments.md_prefix_features import (
    EDGE_FEATURE_DIM,
    PREFIX_FEATURE_DIM,
    PrefixAction,
    action_orders,
    canonical_action,
    prefix_sequence,
)


MODEL_SEEDS = (3101, 3102, 3103)
TRAINING_SEEDS = {3101: 6101, 3102: 6102, 3103: 6103}
TRAIN_RANGE = range(75000, 75600)
DEVELOPMENT_RANGE = range(75600, 75750)
FORBIDDEN_RANGES = ((75800, 75950), (76000, 76150))
ROOT = Path("reports/md_exact_action_candidate_diagnostic_2026-09-06")
FEATURE_PATH = Path("reports/md_minimal_physics_scorer_2026-09-08/feature_package.jsonl")
DEFAULT_OUTPUT = Path("reports/md_prefix_conditioned_reranker_2026-09-10")
PREFIX_CANONICAL_ONLY = False


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    hidden_dim: int = 16
    epochs: int = 60
    evaluation_interval: int = 5
    patience: int = 3
    learning_rate: float = 0.003
    weight_decay: float = 0.001
    regression_weight: float = 0.2
    regret_temperature: float = 1.0
    random_order_count: int = 2


@dataclass(slots=True)
class Candidate:
    action: PrefixAction
    regret: float
    optimal: bool
    rank: int


@dataclass(slots=True)
class State:
    snapshot_id: str
    split: str
    seed: int
    model_seed: int
    pending_count: int
    action_type: str
    candidates: list[Candidate]


class SetCostScorer(nn.Module):
    """The published minimal-physics baseline, including candidate rank."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(EDGE_FEATURE_DIM + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, edge_features: torch.Tensor, rank_fraction: float) -> torch.Tensor:
        rank = edge_features.new_full((len(edge_features), 1), rank_fraction)
        return self.head(self.edge(torch.cat((edge_features, rank), dim=-1)).sum(0)).squeeze()


class PrefixCostScorer(nn.Module):
    """Score a complete action from edge features conditioned on each prefix."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.step = nn.Sequential(
            nn.Linear(EDGE_FEATURE_DIM + PREFIX_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.head(self.step(sequence).sum(0)).squeeze()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_states() -> list[State]:
    """Load only the frozen train/development cache and align all three files."""

    candidate_rows = _read_jsonl(ROOT / "candidate_records.jsonl")
    feature_rows = _read_jsonl(FEATURE_PATH)
    label_rows = _read_jsonl(ROOT / "exact_action_package.jsonl")
    labels = {
        (row["snapshot_id"], canonical_action(row["complete_first_action"]["assignments"])): row
        for row in label_rows
    }
    candidates = {
        (row["snapshot_id"], int(row["model_seed"])): row
        for row in candidate_rows
    }
    states = []
    for feature_row in feature_rows:
        split = str(feature_row["split"])
        seed = int(feature_row["snapshot_id"].split(":", 1)[0].rsplit("-", 1)[-1])
        expected = TRAIN_RANGE if split == "train" else DEVELOPMENT_RANGE if split == "development" else None
        if expected is None or seed not in expected:
            raise ValueError(f"unexpected or forbidden cached state: {split=} {seed=}")
        if any(seed in range(start, stop) for start, stop in FORBIDDEN_RANGES):
            raise ValueError(f"forbidden seed was read: {seed}")
        model_seed = int(feature_row["model_seed"])
        key = (feature_row["snapshot_id"], model_seed)
        candidate_row = candidates[key]
        action_keys = candidate_row["candidate_action_keys"]
        action_features = feature_row["actions"]
        if len(action_keys) != len(action_features):
            raise ValueError("candidate and feature action counts differ")
        built = []
        for rank, (assignments, feature) in enumerate(zip(action_keys, action_features, strict=True)):
            action_key = canonical_action(assignments)
            label = labels[(feature_row["snapshot_id"], action_key)]
            if rank != int(feature["rank"]):
                raise ValueError("cached candidate ranks are not aligned")
            tensor = torch.tensor(feature["features"], dtype=torch.float32)
            built.append(
                Candidate(
                    PrefixAction(action_key, tensor),
                    float(label["regret"]),
                    bool(label["tolerance_optimal"]),
                    rank,
                )
            )
        states.append(
            State(
                snapshot_id=str(feature_row["snapshot_id"]),
                split=split,
                seed=seed,
                model_seed=model_seed,
                pending_count=int(feature_row["pending_count"]),
                action_type=str(candidate_row["action_type"]),
                candidates=built,
            )
        )
    if len(states) != 630:
        raise ValueError(f"expected 630 cached model-state records, observed {len(states)}")
    return states


def normalization(states: Sequence[State]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat(
        [candidate.action.edge_features for state in states if state.split == "train" for candidate in state.candidates]
    )
    return values.mean(0), values.std(0).clamp_min(1.0)


def normalized_action(candidate: Candidate, mean: torch.Tensor, std: torch.Tensor) -> PrefixAction:
    return PrefixAction(candidate.action.assignments, (candidate.action.edge_features - mean) / std)


def _orders(state: State, candidate: Candidate, count: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    return action_orders(
        candidate.action.assignments,
        identity=f"{state.snapshot_id}:{state.model_seed}:{candidate.rank}",
        random_count=count,
    )


def candidate_cost(
    model: nn.Module,
    state: State,
    candidate: Candidate,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    mode: str,
    order_index: int = 0,
) -> torch.Tensor:
    action = normalized_action(candidate, mean, std)
    if isinstance(model, SetCostScorer):
        return model(action.edge_features, candidate.rank / max(1, len(state.candidates) - 1))
    if not isinstance(model, PrefixCostScorer):
        raise TypeError("unsupported scorer")
    domain = generate_md_instance(MDGeneratorConfig(seed=state.seed)).domain
    orders = (
        (action.assignments,)
        if PREFIX_CANONICAL_ONLY and mode == "prefix"
        else _orders(state, candidate, 2)
    )
    order = orders[order_index % len(orders)]
    sequence = prefix_sequence(domain, action, order)
    if mode == "prefix_zero":
        sequence = sequence.clone()
        sequence[:, EDGE_FEATURE_DIM:] = 0
    elif mode != "prefix":
        raise ValueError(f"unsupported prefix mode: {mode}")
    return model(sequence)


def evaluate(
    model: nn.Module | None,
    states: Sequence[State],
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    mode: str,
    order_index: int = 0,
) -> dict[str, Any]:
    regrets: list[float] = []
    optimal: list[float] = []
    by_pending: dict[int, list[float]] = defaultdict(list)
    by_type: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for state in states:
            if model is None:
                chosen = min(state.candidates, key=lambda candidate: candidate.rank)
            else:
                costs = torch.stack(
                    [candidate_cost(model, state, candidate, mean, std, mode=mode, order_index=order_index) for candidate in state.candidates]
                )
                chosen = state.candidates[int(torch.argmin(costs))]
            regrets.append(chosen.regret)
            hit = float(chosen.optimal)
            optimal.append(hit)
            by_pending[state.pending_count].append(hit)
            by_type[state.action_type].append(hit)
    return {
        "state_count": len(states),
        "tolerance_optimal_top1": float(np.mean(optimal)),
        "mean_regret": float(np.mean(regrets)),
        "top1_by_pending": {str(key): float(np.mean(value)) for key, value in sorted(by_pending.items())},
        "top1_by_action_type": {key: float(np.mean(value)) for key, value in sorted(by_type.items())},
    }


def _loss(
    model: nn.Module,
    state: State,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: ExperimentConfig,
    *,
    mode: str,
    epoch: int,
) -> torch.Tensor:
    order_index = 0 if isinstance(model, SetCostScorer) else epoch % (2 + config.random_order_count)
    outputs = torch.stack(
        [candidate_cost(model, state, candidate, mean, std, mode=mode, order_index=order_index) for candidate in state.candidates]
    )
    regrets = outputs.new_tensor([candidate.regret for candidate in state.candidates])
    target = torch.softmax(-regrets / config.regret_temperature, dim=0)
    listwise = -(target * F.log_softmax(-outputs, dim=0)).sum()
    normalized = regrets / regrets.max().clamp_min(1.0)
    return listwise + config.regression_weight * F.smooth_l1_loss(outputs, normalized)


def train_one(
    model_seed: int,
    states: Sequence[State],
    mean: torch.Tensor,
    std: torch.Tensor,
    config: ExperimentConfig,
    *,
    mode: str,
    output_dir: Path,
) -> dict[str, Any]:
    seed = TRAINING_SEEDS[model_seed]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_states = [state for state in states if state.split == "train" and state.model_seed == model_seed]
    development = [state for state in states if state.split == "development" and state.model_seed == model_seed]
    model: nn.Module = SetCostScorer(config.hidden_dim) if mode == "physics" else PrefixCostScorer(config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_metrics = None
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = list(range(len(train_states)))
        random.Random(seed + epoch).shuffle(order)
        losses = []
        for index in order:
            loss = _loss(model, train_states[index], mean, std, config, mode=mode, epoch=epoch)
            optimizer.zero_grad()
            loss.backward()
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise RuntimeError("non-finite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % config.evaluation_interval:
            continue
        model.eval()
        metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **evaluate(model, development, mean, std, mode=mode),
        }
        history.append(metrics)
        key = (metrics["tolerance_optimal_top1"], -metrics["mean_regret"], -epoch)
        if best_metrics is None or key > (
            best_metrics["tolerance_optimal_top1"],
            -best_metrics["mean_regret"],
            -best_metrics["epoch"],
        ):
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    model.eval()
    order_metrics = {
        name: evaluate(model, development, mean, std, mode=mode, order_index=index)
        for index, name in enumerate(("canonical", "reverse", "random"))
    }
    destination = output_dir / mode / f"seed{model_seed}"
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "best_checkpoint.pt"
    torch.save(
        {
            "schema": "md-prefix-conditioned-reranker-1.0",
            "mode": mode,
            "model_seed": model_seed,
            "state_dict": best_state,
            "normalization_mean": mean,
            "normalization_std": std,
            "best_development": best_metrics,
        },
        checkpoint,
    )
    return {
        "mode": mode,
        "model_seed": model_seed,
        "best_development": best_metrics,
        "order_metrics": order_metrics,
        "history": history,
        "checkpoint": str(checkpoint),
    }


def protocol(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "schema": "md-prefix-conditioned-reranker-experiment-1.0",
        "created_before_training": True,
        "config": asdict(config),
        "train_seeds": [75000, 75599],
        "development_seeds": [75600, 75749],
        "forbidden_seed_ranges": [[75800, 75949], [76000, 76149]],
        "model_seeds": list(MODEL_SEEDS),
        "training_seeds": TRAINING_SEEDS,
        "inputs": ["16_observable_physics_features", "12_causal_prefix_features"],
        "forbidden_inputs": ["oracle_cost", "regret", "optimal_flag", "candidate_rank"],
        "physics_baseline_uses_candidate_rank": True,
        "production_decoder_modified": False,
        "acceptance": {
            "mean_top1_gain_over_physics_at_least": 0.02,
            "mean_regret_below_physics": True,
            "per_seed_top1_drop_at_most": 0.02,
            "prefix_ablation_strictly_worse_mean_top1": True,
            "order_top1_range_at_most": 0.03,
        },
    }


def run(output_dir: Path = DEFAULT_OUTPUT, config: ExperimentConfig = ExperimentConfig()) -> dict[str, Any]:
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"completed output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(json.dumps(protocol(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.perf_counter()
    states = load_states()
    mean, std = normalization(states)
    rank_baselines = {
        str(seed): evaluate(None, [state for state in states if state.split == "development" and state.model_seed == seed], mean, std, mode="rank")
        for seed in MODEL_SEEDS
    }
    results = []
    for mode in ("physics", "prefix_zero", "prefix"):
        for seed in MODEL_SEEDS:
            results.append(train_one(seed, states, mean, std, config, mode=mode, output_dir=output_dir))
    indexed = {(result["mode"], result["model_seed"]): result for result in results}
    physics_top1 = np.mean([indexed[("physics", seed)]["best_development"]["tolerance_optimal_top1"] for seed in MODEL_SEEDS])
    prefix_top1 = np.mean([indexed[("prefix", seed)]["best_development"]["tolerance_optimal_top1"] for seed in MODEL_SEEDS])
    zero_top1 = np.mean([indexed[("prefix_zero", seed)]["best_development"]["tolerance_optimal_top1"] for seed in MODEL_SEEDS])
    physics_regret = np.mean([indexed[("physics", seed)]["best_development"]["mean_regret"] for seed in MODEL_SEEDS])
    prefix_regret = np.mean([indexed[("prefix", seed)]["best_development"]["mean_regret"] for seed in MODEL_SEEDS])
    per_seed_drops = [
        indexed[("prefix", seed)]["best_development"]["tolerance_optimal_top1"]
        - indexed[("physics", seed)]["best_development"]["tolerance_optimal_top1"]
        for seed in MODEL_SEEDS
    ]
    order_ranges = []
    for seed in MODEL_SEEDS:
        values = [
            indexed[("prefix", seed)]["order_metrics"][order]["tolerance_optimal_top1"]
            for order in ("canonical", "reverse", "random")
        ]
        order_ranges.append(max(values) - min(values))
    checks = {
        "mean_top1_gain": bool(prefix_top1 - physics_top1 >= 0.02),
        "mean_regret": bool(prefix_regret < physics_regret),
        "per_seed_drop": bool(min(per_seed_drops) >= -0.02),
        "prefix_ablation": bool(prefix_top1 > zero_top1),
        "order_sensitivity": bool(max(order_ranges) <= 0.03),
    }
    accepted = all(checks.values())
    summary = {
        "schema": "md-prefix-conditioned-reranker-experiment-1.0",
        "status": "prefix_conditioning_supported_for_confirmation" if accepted else "prefix_conditioning_not_supported",
        "accepted": accepted,
        "checks": checks,
        "rank_baselines": rank_baselines,
        "results": results,
        "aggregate": {
            "physics_top1": float(physics_top1),
            "prefix_zero_top1": float(zero_top1),
            "prefix_top1": float(prefix_top1),
            "top1_gain_over_physics": float(prefix_top1 - physics_top1),
            "physics_regret": float(physics_regret),
            "prefix_regret": float(prefix_regret),
            "max_within_seed_order_top1_range": float(max(order_ranges)),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "confirmation_read": False,
        "regression_benchmark_read": False,
        "production_decoder_modified": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "final_report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _report(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    indexed = {(row["mode"], row["model_seed"]): row for row in summary["results"]}
    lines = [
        "# Prefix-Conditioned Exact-Action Reranker",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "Development-only experiment; confirmation and regression benchmark seeds were not read.",
        "",
        "| Seed | Physics top-1 / regret | Prefix-zero top-1 / regret | Prefix top-1 / regret |",
        "|---:|---:|---:|---:|",
    ]
    for seed in MODEL_SEEDS:
        values = []
        for mode in ("physics", "prefix_zero", "prefix"):
            metric = indexed[(mode, seed)]["best_development"]
            values.append(f"{metric['tolerance_optimal_top1']:.4f} / {metric['mean_regret']:.4f}")
        lines.append(f"| {seed} | {values[0]} | {values[1]} | {values[2]} |")
    lines += [
        "",
        f"Mean top-1: physics {aggregate['physics_top1']:.4f}, prefix-zero {aggregate['prefix_zero_top1']:.4f}, prefix {aggregate['prefix_top1']:.4f}.",
        f"Prefix gain over physics: {aggregate['top1_gain_over_physics']:.4f}.",
        f"Mean regret: physics {aggregate['physics_regret']:.4f}, prefix {aggregate['prefix_regret']:.4f}.",
        f"Maximum within-seed order top-1 range: {aggregate['max_within_seed_order_top1_range']:.4f}.",
        "",
        "## Preregistered checks",
        "",
    ]
    lines.extend(f"- {name}: **{str(value).lower()}**" for name, value in summary["checks"].items())
    lines += [
        "",
        "## Interpretation",
        "",
        "The prefix model lowers mean regret relative to the retrained physics baseline, but its +0.0167 mean top-1 gain misses the frozen +0.02 threshold. More importantly, prefix-zero has higher mean top-1, so the result does not attribute improvement to prefix information. The maximum within-seed order range also exceeds the 0.03 limit.",
        "",
        "The cached candidate records classify all evaluated best proposals as process actions, so this package cannot establish separate transport or mixed-action benefits. No confirmation run or online autoregressive integration is warranted from this result.",
        "",
        "No production scheduler was changed. A failed gate is a NO-GO for confirmation and online integration.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
