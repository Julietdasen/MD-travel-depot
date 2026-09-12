"""Independent JSON schema and helpers for residual counterfactual labels."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from baselines.gurobi_md_residual_oracle import ForcedAssignmentBatch, ResidualMDState, ResidualOracleResult


@dataclass(frozen=True, slots=True)
class ResidualBatchLabel:
    batch: ForcedAssignmentBatch
    makespan: float
    remaining_task_horizon: float
    terminal_return_tail: float
    latest_return_robot_id: int | None
    regret: float = 0.0
    tie_optimal: bool = False


@dataclass(frozen=True, slots=True)
class ResidualEdgeLabel:
    robot_id: int
    task_id: int
    best_batch: ForcedAssignmentBatch
    total_cost: float
    task_cost: float
    tail_cost: float


@dataclass(frozen=True, slots=True)
class ResidualDecisionSample:
    instance_id: str
    seed: int
    split: str
    state: ResidualMDState
    legal_batches: tuple[ResidualBatchLabel, ...]
    edge_labels: tuple[ResidualEdgeLabel, ...]
    simulator_snapshot: Mapping[str, Any]
    legal_mask: tuple[tuple[bool, ...], ...]
    joint_projection_error: float = 0.0
    batch_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"]["robot_locations"] = {str(k): list(v) for k, v in self.state.robot_locations.items()}
        value["legal_batches"] = [dict(item, batch=list(item["batch"]["assignments"])) for item in value["legal_batches"]]
        value["edge_labels"] = [dict(item, best_batch=list(item["best_batch"]["assignments"])) for item in value["edge_labels"]]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResidualDecisionSample":
        raw = dict(value)
        state = dict(raw["state"])
        state["robot_locations"] = {int(k): tuple(v) for k, v in state["robot_locations"].items()}
        state_obj = ResidualMDState(**state)
        batches = tuple(ResidualBatchLabel(ForcedAssignmentBatch(tuple(tuple(x) for x in b["batch"])), b["makespan"], b["remaining_task_horizon"], b["terminal_return_tail"], b.get("latest_return_robot_id"), b.get("regret", 0.0), b.get("tie_optimal", False)) for b in raw["legal_batches"])
        edges = tuple(ResidualEdgeLabel(int(e["robot_id"]), int(e["task_id"]), ForcedAssignmentBatch(tuple(tuple(x) for x in e["best_batch"])), e["total_cost"], e["task_cost"], e["tail_cost"]) for e in raw["edge_labels"])
        return cls(str(raw["instance_id"]), int(raw["seed"]), str(raw["split"]), state_obj, batches, edges, raw.get("simulator_snapshot", {}), tuple(tuple(row) for row in raw["legal_mask"]), float(raw.get("joint_projection_error", 0.0)), int(raw.get("batch_count", len(batches))))


def build_batch_label(batch: ForcedAssignmentBatch, result: ResidualOracleResult, best_makespan: float, *, tie_timestep: float = 1.0) -> ResidualBatchLabel:
    if result.makespan is None or result.remaining_task_horizon is None or result.terminal_return_tail is None:
        raise ValueError("cannot label an unsolved residual result")
    total = result.remaining_task_horizon + result.terminal_return_tail
    regret = total - best_makespan
    return ResidualBatchLabel(batch, result.makespan, result.remaining_task_horizon, result.terminal_return_tail, result.latest_return_robot_id, regret, regret <= tie_timestep)


def project_edge_labels(labels: tuple[ResidualBatchLabel, ...]) -> tuple[ResidualEdgeLabel, ...]:
    """Project joint labels to edges, averaging tied batch decompositions."""
    grouped: dict[tuple[int, int], list[ResidualBatchLabel]] = {}
    for label in labels:
        total = label.remaining_task_horizon + label.terminal_return_tail
        for edge in label.batch.assignments:
            grouped.setdefault(edge, []).append(label)
    result = []
    for (robot_id, task_id), candidates in sorted(grouped.items()):
        best_total = min(x.remaining_task_horizon + x.terminal_return_tail for x in candidates)
        tied = [x for x in candidates if abs((x.remaining_task_horizon + x.terminal_return_tail) - best_total) <= 1.0]
        representative = min(tied, key=lambda x: x.batch.assignments)
        result.append(ResidualEdgeLabel(robot_id, task_id, representative.batch,
            sum(x.remaining_task_horizon + x.terminal_return_tail for x in tied) / len(tied),
            sum(x.remaining_task_horizon for x in tied) / len(tied),
            sum(x.terminal_return_tail for x in tied) / len(tied)))
    return tuple(result)


def joint_projection_error(
    labels: tuple[ResidualBatchLabel, ...],
    edges: tuple[ResidualEdgeLabel, ...],
) -> float:
    """Return mean relative error induced by additive edge projection."""
    if not labels:
        return 0.0
    projected = {(edge.robot_id, edge.task_id): edge.total_cost for edge in edges}
    errors = []
    for label in labels:
        actual = label.remaining_task_horizon + label.terminal_return_tail
        estimate = sum(projected[edge] for edge in label.batch.assignments)
        errors.append(abs(estimate - actual) / max(1.0, abs(actual)))
    return sum(errors) / len(errors)

def robust_scale(values: tuple[float, ...], *, lower: float = 0.05, upper: float = 0.95) -> tuple[float, ...]:
    """Normalize a state-local target; constant states map to zero."""
    if not values:
        return ()
    ordered = sorted(values)
    def quantile(q: float) -> float:
        pos = q * (len(ordered) - 1); lo = int(pos); hi = min(lo + 1, len(ordered) - 1); return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    lo, hi = quantile(lower), quantile(upper)
    if hi <= lo:
        return tuple(0.0 for _ in values)
    return tuple(max(0.0, min(1.0, (x - lo) / (hi - lo))) for x in values)


def dump_samples(samples: tuple[ResidualDecisionSample, ...], path: str | Path) -> None:
    Path(path).write_text("\n".join(json.dumps(s.to_dict(), sort_keys=True) for s in samples) + ("\n" if samples else ""), encoding="utf-8")


def load_samples(path: str | Path) -> tuple[ResidualDecisionSample, ...]:
    return tuple(ResidualDecisionSample.from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


@dataclass(frozen=True, slots=True)
class ResidualSplitPlan:
    train_seeds: tuple[int, ...]
    development_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = (self.train_seeds, self.development_seeds, self.test_seeds)
        if any(not group for group in groups):
            raise ValueError("every residual split must contain at least one seed")
        flattened = tuple(seed for group in groups for seed in group)
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in flattened):
            raise ValueError("residual split seeds must be non-negative integers")
        if len(flattened) != len(set(flattened)):
            raise ValueError("residual split seeds must be disjoint")

    def split_for_seed(self, seed: int) -> str:
        for name, seeds in self.as_mapping().items():
            if seed in seeds:
                return name
        raise ValueError(f"seed {seed} is outside the residual split plan")

    def as_mapping(self) -> dict[str, tuple[int, ...]]:
        return {
            "train": self.train_seeds,
            "development": self.development_seeds,
            "test": self.test_seeds,
        }

    def to_dict(self) -> dict[str, list[int]]:
        return {name: list(seeds) for name, seeds in self.as_mapping().items()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Sequence[int]]) -> "ResidualSplitPlan":
        return cls(
            tuple(int(seed) for seed in value["train"]),
            tuple(int(seed) for seed in value["development"]),
            tuple(int(seed) for seed in value["test"]),
        )


LEGACY_RESIDUAL_SPLIT_PLAN = ResidualSplitPlan(
    tuple(range(74000, 74030)),
    tuple(range(74100, 74110)),
    tuple(range(74200, 74230)),
)
FORMAL_RESIDUAL_SPLIT_PLAN = ResidualSplitPlan(
    tuple(range(75000, 75600)),
    tuple(range(75600, 75750)),
    tuple(range(75800, 75950)),
)
SPLIT_SEEDS = LEGACY_RESIDUAL_SPLIT_PLAN.as_mapping()

def split_for_seed(
    seed: int,
    split_plan: ResidualSplitPlan = LEGACY_RESIDUAL_SPLIT_PLAN,
) -> str:
    return split_plan.split_for_seed(seed)
