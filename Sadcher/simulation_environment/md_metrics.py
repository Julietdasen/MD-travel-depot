"""Structured execution records and terminal metrics for MD rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, TypedDict

from experiments.protocol import FailureReason


class TransportExecutionRecord(TypedDict):
    task_id: int
    robot_id: int
    assigned_at: int
    arrived_pickup_at: int | None
    loading_started_at: int | None
    loading_completed_at: int | None
    arrived_delivery_at: int | None
    unloading_started_at: int | None
    unloading_completed_at: int | None
    completed_at: int | None
    empty_travel_duration: int
    loading_duration: int
    loaded_travel_duration: int
    unloading_duration: int
    service_duration: int
    occupied_duration: int | None


class ProcessExecutionRecord(TypedDict):
    task_id: int
    robot_ids: tuple[int, ...]
    started_at: int
    completed_at: int | None
    waiting_duration: int
    travel_duration: int
    service_duration: float
    occupied_duration: int | None


@dataclass(frozen=True, slots=True)
class MDMetrics:
    """A validated terminal snapshot shared by all MD rollout methods."""

    success: bool
    failure_reason: FailureReason | None
    makespan: float | None
    material_starvation: Mapping[str, float]
    robot_utilization: Mapping[str, float]
    inference_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError("success must be boolean")
        if self.success:
            if self.failure_reason is not None:
                raise ValueError("successful metrics cannot have a failure reason")
            _nonnegative_finite("makespan", self.makespan)
        else:
            if not isinstance(self.failure_reason, FailureReason):
                raise ValueError("failed metrics require a stable failure reason")
            if self.makespan is not None:
                raise ValueError("failed metrics require a null makespan")
        _nonnegative_finite("inference_time_seconds", self.inference_time_seconds)
        starvation = _metric_map("material starvation", self.material_starvation)
        utilization = _metric_map(
            "robot utilization", self.robot_utilization, maximum=1.0
        )
        object.__setattr__(self, "material_starvation", MappingProxyType(starvation))
        object.__setattr__(self, "robot_utilization", MappingProxyType(utilization))

    @property
    def total_material_starvation(self) -> float:
        return sum(self.material_starvation.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "failure_reason": (
                self.failure_reason.value if self.failure_reason is not None else None
            ),
            "makespan": self.makespan,
            "material_starvation": dict(self.material_starvation),
            "robot_utilization": dict(self.robot_utilization),
            "inference_time_seconds": self.inference_time_seconds,
        }


def _metric_map(
    name: str, values: Mapping[str, float], *, maximum: float | None = None
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        _nonnegative_finite(f"{name}[{key!r}]", value)
        numeric = float(value)
        if maximum is not None and numeric > maximum:
            raise ValueError(f"{name}[{key!r}] must be at most {maximum}")
        result[key] = numeric
    return result


def _nonnegative_finite(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")


__all__ = [
    "MDMetrics",
    "ProcessExecutionRecord",
    "TransportExecutionRecord",
]
