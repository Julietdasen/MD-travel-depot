"""Test a learned quality gate over complete Beam-16 prefix actions.

The prefix decoder remains responsible for proposing legal complete actions.
This experimental module scores only those completed candidates from observable
state/action features, removes candidates whose predicted regret is too high,
and then keeps the highest-probability survivor.  It is deliberately separate
from PPO, RLlib, and the production scheduler.
"""

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

from experiments.md_minimal_action_features import FEATURE_DIM, build_minimal_action_features
from experiments.md_prefix_autoregressive_pilot import (
    BEAM_WIDTH,
    MODEL_SEEDS,
    OUTPUT as PREFIX_OUTPUT,
    _candidate,
    _label_state,
    _new_model,
    _observation,
)
from models.md_prefix_autoregressive import MDPrefixAutoregressivePolicy
from reinforcement_learning.md_joint_action import oracle_aligned_action_is_legal


OUTPUT = Path("reports/md_prefix_cost_gate_pilot_2026-09-10")
GATE_SEEDS = (101, 102, 103)
MARGINS = (0.0, 2.0, 5.0, 10.0, 20.0)
CONFIRMATION_POOL = range(77250, 77350)
CONFIRMATION_STATES = 6
MIN_ACTIONS = 24
MAX_ACTIONS = 96
MAX_EPOCHS = 200
PATIENCE = 25
LEARNING_RATE = 1e-3
REGRET_TEMPERATURE = 3.0
REGRET_SCALE = 40.0
STATE_FEATURE_DIM = 66
ACTION_FEATURE_DIM = FEATURE_DIM * 4
INPUT_DIM = STATE_FEATURE_DIM + ACTION_FEATURE_DIM


@dataclass(frozen=True, slots=True)
class GateState:
    split: str
    seed: int
    observation: dict[str, torch.Tensor]
    simulator: Any
    vectors: torch.Tensor
    features: torch.Tensor
    regrets: torch.Tensor
    regret_by_action: dict[tuple[int, ...], float]
    minimum_continuation_cost: float


class MDCompleteActionCostGate(nn.Module):
    """Small MLP predicting complete-action continuation regret."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _state_summary(observation: dict[str, torch.Tensor]) -> torch.Tensor:
    values = []
    for key in ("robot_features", "task_features", "robot_metadata", "task_metadata"):
        tensor = observation[key][0].float()
        values.extend((tensor.mean(dim=0), tensor.amax(dim=0)))
    opportunity = observation["opportunity_context"][0].float().reshape(-1, 5)
    values.extend((opportunity.mean(dim=0), opportunity.amax(dim=0)))
    result = torch.cat(values)
    if result.numel() != STATE_FEATURE_DIM:
        raise AssertionError(f"unexpected state summary width: {result.numel()}")
    return result


def _assignments_from_vector(simulator: Any, vector: Sequence[int]) -> tuple[tuple[int, int], ...]:
    robot_ids = sorted(robot.robot_id for robot in simulator.domain.robots)
    task_ids = sorted(task.task_id for task in simulator.domain.tasks)
    return tuple(
        (robot_ids[index], task_ids[int(choice) - 1])
        for index, choice in enumerate(vector)
        if int(choice) > 0
    )


def _action_features(
    observation: dict[str, torch.Tensor], simulator: Any, vector: Sequence[int]
) -> torch.Tensor:
    assignments = _assignments_from_vector(simulator, vector)
    edges = build_minimal_action_features(simulator, assignments)
    if edges.shape[0] == 0:
        raise ValueError("the quality gate does not accept an all-idle action")
    pooled = torch.cat(
        (edges.mean(dim=0), edges.sum(dim=0), edges.amax(dim=0), edges.amin(dim=0))
    )
    result = torch.cat((_state_summary(observation), pooled))
    if result.numel() != INPUT_DIM or not bool(torch.isfinite(result).all()):
        raise AssertionError("invalid complete-action gate feature vector")
    return result


def _gate_state_from_row(split: str, row: dict[str, Any]) -> GateState:
    seed = int(row["seed"])
    candidate, _residual, _legal = _candidate(seed)
    return _label_from_saved_row(split, row, candidate)


def _label_from_saved_row(split: str, row: dict[str, Any], candidate: Any) -> GateState:
    vectors = torch.tensor([action["vector"] for action in row["actions"]], dtype=torch.long)
    regrets = torch.tensor([action["regret"] for action in row["actions"]], dtype=torch.float32)
    observation = _observation(candidate.domain, candidate.simulator)
    features = torch.stack(
        [_action_features(observation, candidate.simulator, vector) for vector in vectors.tolist()]
    )
    return GateState(
        split=split,
        seed=int(row["seed"]),
        observation=observation,
        simulator=candidate.simulator,
        vectors=vectors,
        features=features,
        regrets=regrets,
        regret_by_action={
            tuple(int(value) for value in action["vector"]): float(action["regret"])
            for action in row["actions"]
        },
        minimum_continuation_cost=float(row["minimum_continuation_cost"]),
    )


def _load_existing_split(split: str) -> list[GateState]:
    return [
        _gate_state_from_row(split, row)
        for row in _read_jsonl(PREFIX_OUTPUT / f"{split}_labels.jsonl")
    ]


def _normalization(states: Sequence[GateState]) -> tuple[torch.Tensor, torch.Tensor]:
    joined = torch.cat([state.features for state in states])
    mean = joined.mean(dim=0)
    std = joined.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def _predict(
    model: MDCompleteActionCostGate,
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return model((features - mean) / std)


def _state_loss(
    model: MDCompleteActionCostGate,
    state: GateState,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    predicted = _predict(model, state.features, mean, std)
    target = F.softmax(-state.regrets / REGRET_TEMPERATURE, dim=0)
    listwise = -(target * F.log_softmax(-predicted / REGRET_TEMPERATURE, dim=0)).sum()
    regression = F.smooth_l1_loss(predicted / REGRET_SCALE, state.regrets / REGRET_SCALE)
    return listwise + 0.5 * regression


@torch.no_grad()
def _direct_metrics(
    model: MDCompleteActionCostGate,
    states: Sequence[GateState],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, float]:
    regrets = []
    for state in states:
        selected = int(torch.argmin(_predict(model, state.features, mean, std)))
        regrets.append(float(state.regrets[selected]))
    return {
        "mean_regret": float(np.mean(regrets)),
        "top1_optimal": float(np.mean(np.asarray(regrets) <= 1e-6)),
    }


def _train_gate(
    gate_seed: int,
    train_states: Sequence[GateState],
    development_states: Sequence[GateState],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[MDCompleteActionCostGate, dict[str, Any]]:
    random.seed(gate_seed)
    np.random.seed(gate_seed)
    torch.manual_seed(gate_seed)
    model = MDCompleteActionCostGate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_state = None
    best_metric = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = list(range(len(train_states)))
        random.Random(gate_seed * 1000 + epoch).shuffle(order)
        model.train()
        for index in order:
            loss = _state_loss(model, train_states[index], mean, std)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.grad is not None
            ):
                raise FloatingPointError("non-finite cost-gate gradient")
            optimizer.step()
        model.eval()
        metric = _direct_metrics(model, development_states, mean, std)["mean_regret"]
        if metric < best_metric - 1e-8:
            best_metric = metric
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("cost gate did not produce a finite development checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    destination = OUTPUT / "cost_gate" / f"seed{gate_seed}"
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "md-prefix-complete-action-cost-gate-1.0",
            "state_dict": best_state,
            "feature_mean": mean,
            "feature_std": std,
            "gate_seed": gate_seed,
            "best_epoch": best_epoch,
        },
        destination / "best_checkpoint.pt",
    )
    return model, {
        "gate_seed": gate_seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_development_mean_regret": best_metric,
        "checkpoint": str(destination / "best_checkpoint.pt"),
    }


def _load_prefix(mode: str, seed: int) -> MDPrefixAutoregressivePolicy:
    checkpoint = torch.load(
        PREFIX_OUTPUT / mode / f"seed{seed}" / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = _new_model(mode == "prefix")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _ensemble_cost(
    gates: Sequence[MDCompleteActionCostGate],
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return torch.stack([_predict(gate, features, mean, std) for gate in gates]).median(dim=0).values


@torch.no_grad()
def _choose_from_beam(
    prefix: MDPrefixAutoregressivePolicy,
    gates: Sequence[MDCompleteActionCostGate],
    state: GateState,
    mean: torch.Tensor,
    std: torch.Tensor,
    margin: float,
) -> tuple[tuple[int, ...], tuple[int, ...], int, float]:
    started = time.perf_counter()
    candidates = prefix.beam_candidates(state.observation, beam_width=BEAM_WIDTH)
    candidate_features = torch.stack(
        [_action_features(state.observation, state.simulator, row) for row in candidates.actions.tolist()]
    )
    predicted = _ensemble_cost(gates, candidate_features, mean, std)
    retained = predicted <= predicted.min() + margin
    selected_index = int(torch.nonzero(retained, as_tuple=False)[0])
    base = tuple(int(value) for value in candidates.actions[0].tolist())
    gated = tuple(int(value) for value in candidates.actions[selected_index].tolist())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return base, gated, int((~retained).sum()), elapsed_ms


def _tune_margin(
    prefixes: Sequence[MDPrefixAutoregressivePolicy],
    gates: Sequence[MDCompleteActionCostGate],
    states: Sequence[GateState],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for margin in MARGINS:
        regrets = []
        masked = []
        for prefix in prefixes:
            for state in states:
                _base, gated, removed, _elapsed = _choose_from_beam(
                    prefix, gates, state, mean, std, margin
                )
                regrets.append(state.regret_by_action.get(gated, float("inf")))
                masked.append(removed)
        rows.append(
            {
                "margin": margin,
                "mean_regret": float(np.mean(regrets)),
                "top1_optimal": float(np.mean(np.asarray(regrets) <= 1e-6)),
                "mean_candidates_screened": float(np.mean(masked)),
            }
        )
    best = min(rows, key=lambda row: (row["mean_regret"], -row["top1_optimal"], row["margin"]))
    return float(best["margin"]), rows


def _confirmation_manifest() -> list[dict[str, int]]:
    selected = []
    for seed in CONFIRMATION_POOL:
        _candidate_data, state, legal = _candidate(seed)
        if MIN_ACTIONS <= len(legal) <= MAX_ACTIONS:
            selected.append(
                {"seed": seed, "pending_count": len(state.pending_task_ids), "legal_action_count": len(legal)}
            )
        if len(selected) == CONFIRMATION_STATES:
            return selected
    raise RuntimeError("not enough confirmation states in the preregistered pool")


def _label_confirmation(manifest: Sequence[dict[str, int]]) -> list[GateState]:
    states = []
    rows = []
    for entry in manifest:
        labeled, row = _label_state("confirmation", int(entry["seed"]))
        if labeled.legal_action_count != int(entry["legal_action_count"]):
            raise AssertionError("confirmation action count changed")
        candidate, _state, _legal = _candidate(int(entry["seed"]))
        states.append(_label_from_saved_row("confirmation", row, candidate))
        rows.append(row)
    (OUTPUT / "confirmation_labels.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return states


def _evaluate_confirmation(
    prefix_models: dict[int, MDPrefixAutoregressivePolicy],
    zero_models: dict[int, MDPrefixAutoregressivePolicy],
    gates: Sequence[MDCompleteActionCostGate],
    states: Sequence[GateState],
    mean: torch.Tensor,
    std: torch.Tensor,
    margin: float,
) -> dict[str, Any]:
    decisions = []
    timings = []
    for seed in MODEL_SEEDS:
        prefix = prefix_models[seed]
        zero = zero_models[seed]
        for state in states:
            base, gated, removed, elapsed = _choose_from_beam(prefix, gates, state, mean, std, margin)
            zero_action = tuple(int(value) for value in zero.beam_decode(state.observation, beam_width=BEAM_WIDTH).actions[0])
            timings.append(elapsed)
            decisions.append(
                {
                    "model_seed": seed,
                    "state_seed": state.seed,
                    "base_action": list(base),
                    "gated_action": list(gated),
                    "prefix_zero_action": list(zero_action),
                    "base_regret": state.regret_by_action.get(base, float("inf")),
                    "gated_regret": state.regret_by_action.get(gated, float("inf")),
                    "prefix_zero_regret": state.regret_by_action.get(zero_action, float("inf")),
                    "minimum_continuation_cost": state.minimum_continuation_cost,
                    "candidates_screened": removed,
                    "gated_legal": oracle_aligned_action_is_legal(
                        state.observation, torch.tensor([gated], dtype=torch.long)
                    ),
                    "online_latency_ms": elapsed,
                }
            )

    def metrics(key: str) -> dict[str, float]:
        regrets = np.asarray([float(row[key]) for row in decisions])
        minimums = np.asarray([float(row["minimum_continuation_cost"]) for row in decisions])
        return {
            "mean_regret": float(regrets.mean()),
            "mean_continuation_duration": float((minimums + regrets).mean()),
            "top1_optimal": float((regrets <= 1e-6).mean()),
            "max_regret": float(regrets.max()),
            "catastrophic_regret_at_least_20": int((regrets >= 20.0).sum()),
        }

    return {
        "state_count": len(states),
        "decision_count": len(decisions),
        "prefix": metrics("base_regret"),
        "prefix_gated": metrics("gated_regret"),
        "prefix_zero": metrics("prefix_zero_regret"),
        "mean_candidates_screened": float(np.mean([row["candidates_screened"] for row in decisions])),
        "illegal_gated_actions": sum(not row["gated_legal"] for row in decisions),
        "unknown_gated_actions": sum(not np.isfinite(row["gated_regret"]) for row in decisions),
        "online_latency_ms": {
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
        },
        "decisions": decisions,
    }


def run() -> dict[str, Any]:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(f"completed output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    train_states = _load_existing_split("train")
    development_states = _load_existing_split("development")
    mean, std = _normalization(train_states)
    gates = []
    training = []
    for seed in GATE_SEEDS:
        gate, result = _train_gate(seed, train_states, development_states, mean, std)
        gates.append(gate)
        training.append(result)

    prefix_models = {seed: _load_prefix("prefix", seed) for seed in MODEL_SEEDS}
    zero_models = {seed: _load_prefix("prefix-zero", seed) for seed in MODEL_SEEDS}
    margin, margin_results = _tune_margin(
        list(prefix_models.values()), gates, development_states, mean, std
    )
    manifest = _confirmation_manifest()
    lock = {
        "schema": "md-prefix-cost-gate-lock-1.0",
        "gate_seeds": list(GATE_SEEDS),
        "margin_candidates": list(MARGINS),
        "selected_margin": margin,
        "margin_development_results": margin_results,
        "confirmation_labels_read": False,
        "confirmation_manifest": manifest,
        "production_decoder_modified": False,
    }
    (OUTPUT / "training_locked.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    confirmation_states = _label_confirmation(manifest)
    evaluation = _evaluate_confirmation(
        prefix_models, zero_models, gates, confirmation_states, mean, std, margin
    )
    checks = {
        "gated_mean_regret_below_prefix": (
            evaluation["prefix_gated"]["mean_regret"] < evaluation["prefix"]["mean_regret"]
        ),
        "gated_mean_regret_not_above_prefix_zero": (
            evaluation["prefix_gated"]["mean_regret"] <= evaluation["prefix_zero"]["mean_regret"]
        ),
        "gated_catastrophic_count_not_above_prefix": (
            evaluation["prefix_gated"]["catastrophic_regret_at_least_20"]
            <= evaluation["prefix"]["catastrophic_regret_at_least_20"]
        ),
        "illegal_gated_actions_zero": evaluation["illegal_gated_actions"] == 0,
        "unknown_gated_actions_zero": evaluation["unknown_gated_actions"] == 0,
        "mac_cpu_online_p95_ms_at_most_25": evaluation["online_latency_ms"]["p95"] <= 25.0,
    }
    summary = {
        "schema": "md-prefix-cost-gate-pilot-1.0",
        "status": "promising_experimental_gate" if all(checks.values()) else "experimental_gate_not_validated",
        "training": training,
        "selected_margin": margin,
        "confirmation_manifest": manifest,
        "evaluation": evaluation,
        "checks": checks,
        "confirmation_labels_read_after_lock": True,
        "elapsed_seconds": time.perf_counter() - started,
        "ppo_or_rllib_connected": False,
        "production_decoder_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "final_report.md").write_text(
        "\n".join(
            [
                "# MD Prefix Complete-Action Quality Gate Pilot",
                "",
                f"Status: **{summary['status']}**.",
                f"Development-selected screening margin: {margin:.1f} predicted regret units.",
                "",
                "| Decoder | Mean regret | Mean continuation duration | Top-1 | Max regret | Regret >= 20 |",
                "|---|---:|---:|---:|---:|---:|",
                *[
                    f"| {name} | {evaluation[name]['mean_regret']:.3f} | {evaluation[name]['mean_continuation_duration']:.3f} | {evaluation[name]['top1_optimal']:.3f} | {evaluation[name]['max_regret']:.1f} | {evaluation[name]['catastrophic_regret_at_least_20']} |"
                    for name in ("prefix", "prefix_gated", "prefix_zero")
                ],
                "",
                f"Mac CPU complete Beam-16 + gate p95: {evaluation['online_latency_ms']['p95']:.3f} ms.",
                f"Mean candidates screened: {evaluation['mean_candidates_screened']:.2f} of {BEAM_WIDTH}.",
                "This remains an offline supervised experiment and is not connected to production scheduling.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
