"""Canonical integer-timestep transport timing rules."""

from __future__ import annotations

import math
from dataclasses import dataclass

from simulation_environment.domain_model import TransportRobot, TransportTask


@dataclass(frozen=True, slots=True)
class TransportDurations:
    empty_travel: int
    loading: int
    loaded_travel: int
    unloading: int

    @property
    def service(self) -> int:
        return self.loading + self.loaded_travel + self.unloading

    @property
    def occupied(self) -> int:
        return self.empty_travel + self.service


def discrete_duration(value: float) -> int:
    return max(0, math.ceil(value))


def travel_duration(
    start: tuple[float, float], end: tuple[float, float], speed: float
) -> int:
    return max(0, math.ceil(math.dist(start, end) / speed))


def transport_durations(
    robot_location: tuple[float, float],
    robot: TransportRobot,
    task: TransportTask,
) -> TransportDurations:
    if robot.loaded_speed is None:
        raise ValueError(f"TRANSPORT_ROBOT {robot.robot_id} has no loaded speed")
    return TransportDurations(
        empty_travel=travel_duration(
            robot_location, task.pickup_location, robot.unloaded_speed
        ),
        loading=discrete_duration(task.loading_duration),
        loaded_travel=travel_duration(
            task.pickup_location, task.delivery_location, robot.loaded_speed
        ),
        unloading=discrete_duration(task.unloading_duration),
    )


__all__ = [
    "TransportDurations",
    "discrete_duration",
    "transport_durations",
    "travel_duration",
]
