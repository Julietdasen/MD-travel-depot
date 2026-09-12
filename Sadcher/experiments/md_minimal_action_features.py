"""Compact, oracle-independent physical features for exact-action scoring."""
from __future__ import annotations

import math
from typing import Sequence

import torch

from baselines.exact_online_action_oracle import CompleteOnlineAction
from simulation_environment.domain_model import ProcessTask, TransportRobot, TransportTask
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from simulation_environment.md_discrete_simulator import RobotActivity
from simulation_environment.hard_feasibility import TaskStatus
from simulation_environment.transport_timing import transport_durations


FEATURE_SCHEMA = "exact-action-minimal-physics-1.1"
FEATURE_DIM = 16


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.dist(left, right)


def _task_start(task: ProcessTask | TransportTask) -> tuple[float, float]:
    return task.location if isinstance(task, ProcessTask) else task.pickup_location


def build_minimal_action_features(
    simulator: MDDiscreteSimulator,
    action: CompleteOnlineAction | Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Return one physical feature row per assignment in ``action``.

    Values are derived only from the present simulator/domain state.  Assignment
    rows are sorted canonically, so permuting an action does not change its
    representation.  Robot/task identifiers are used only for lookup.
    """
    if not isinstance(simulator, MDDiscreteSimulator):
        raise TypeError("simulator must be an MDDiscreteSimulator")
    assignments = action.assignments if isinstance(action, CompleteOnlineAction) else tuple(action)
    assignments = tuple(sorted((int(robot), int(task)) for robot, task in assignments))
    robots = {robot.robot_id: robot for robot in simulator.domain.robots}
    tasks = {task.task_id: task for task in simulator.domain.tasks}
    rows: list[list[float]] = []
    for robot_id, task_id in assignments:
        if robot_id not in robots or task_id not in tasks:
            raise ValueError("action references an unknown robot or task")
        robot = robots[robot_id]
        task = tasks[task_id]
        runtime_robot = simulator.robot_states[robot_id]
        runtime_task = simulator.task_states[task_id]
        start = _task_start(task)
        speed = float(getattr(robot, "speed", getattr(robot, "unloaded_speed", 1.0)))
        if isinstance(robot, TransportRobot) and isinstance(task, TransportTask):
            durations = transport_durations(runtime_robot.location, robot, task)
            travel = float(durations.empty_travel)
            service = float(durations.service)
        else:
            travel = _distance(runtime_robot.location, start) / max(speed, 1e-9)
            service = float(task.duration) if isinstance(task, ProcessTask) else 0.0
        requirements = tuple(getattr(task, "requirements", ()))
        capabilities = tuple(getattr(robot, "capabilities", ()))
        capability_match = sum(bool(a and b) for a, b in zip(requirements, capabilities))
        required_count = max(1, sum(bool(value) for value in requirements))
        capacity = float(getattr(robot, "capacity", 0.0) or 0.0)
        load = float(getattr(task, "load", 0.0) or 0.0)
        predecessor_count = len(getattr(task, "normal_predecessors", ()))
        material_predecessor = (
            task.material_predecessor if isinstance(task, ProcessTask) else None
        )
        material_pending = float(material_predecessor is not None)
        normal_ready = all(
            simulator.task_states[predecessor].status is TaskStatus.COMPLETE
            for predecessor in getattr(task, "normal_predecessors", ())
        )
        material_ready = (
            material_predecessor is None
            or simulator.task_states[material_predecessor].status is TaskStatus.COMPLETE
        )
        return_time = _distance(
            task.delivery_location if isinstance(task, TransportTask) else task.location,
            simulator.exit_location,
        ) / max(speed, 1e-9)
        rows.append([
            float(hasattr(robot, "capacity")),
            float(isinstance(task, TransportTask)),
            float(runtime_robot.activity is RobotActivity.AVAILABLE),
            float(travel),
            float(service),
            float(travel + service),
            float(capability_match / required_count),
            float(capacity - load),
            float(load),
            float(predecessor_count),
            material_pending,
            float(runtime_task.status is TaskStatus.PENDING and normal_ready and material_ready),
            float(runtime_task.phase_remaining),
            float(return_time),
            float(len(assignments)),
            float(simulator.time),
        ])
    if not rows:
        return torch.empty((0, FEATURE_DIM), dtype=torch.float32)
    result = torch.tensor(rows, dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("minimal action features must be finite")
    return result


__all__ = ["FEATURE_DIM", "FEATURE_SCHEMA", "build_minimal_action_features"]
