"""Oracle-independent prefix features for complete MD joint actions."""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from simulation_environment.domain_model import (
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportTask,
)


Assignment = tuple[int, int]
EDGE_FEATURE_DIM = 16
PREFIX_FEATURE_DIM = 12
PREFIX_FEATURE_NAMES = (
    "prefix_fraction",
    "remaining_fraction",
    "same_task_prefix_fraction",
    "task_open",
    "required_skill_remaining_before",
    "required_skill_contribution",
    "required_skill_remaining_after",
    "coalition_completed_by_step",
    "selected_transport_fraction",
    "selected_process_fraction",
    "opened_task_fraction",
    "previous_assignment_same_task",
)


@dataclass(frozen=True, slots=True)
class PrefixAction:
    assignments: tuple[Assignment, ...]
    edge_features: torch.Tensor

    def __post_init__(self) -> None:
        if self.edge_features.shape != (len(self.assignments), EDGE_FEATURE_DIM):
            raise ValueError("edge feature rows must align with assignments")
        if not bool(torch.isfinite(self.edge_features).all()):
            raise ValueError("edge features must be finite")


def canonical_action(assignments: Iterable[Sequence[int]]) -> tuple[Assignment, ...]:
    return tuple(sorted((int(pair[0]), int(pair[1])) for pair in assignments))


def validate_complete_action(
    domain: SchedulingDomain, assignments: Sequence[Assignment]
) -> None:
    """Validate the structural joint-action rules used by the cached package."""

    robots = {robot.robot_id: robot for robot in domain.robots}
    tasks = {task.task_id: task for task in domain.tasks}
    robot_ids = [robot_id for robot_id, _ in assignments]
    if len(robot_ids) != len(set(robot_ids)):
        raise ValueError("a robot may appear only once in a complete action")
    by_task: dict[int, list[int]] = {}
    for robot_id, task_id in assignments:
        if robot_id not in robots or task_id not in tasks:
            raise ValueError("action references an unknown robot or task")
        by_task.setdefault(task_id, []).append(robot_id)
    for task_id, assigned in by_task.items():
        task = tasks[task_id]
        if isinstance(task, TransportTask):
            if len(assigned) != 1:
                raise ValueError("transport actions must be singleton")
            continue
        assert isinstance(task, ProcessTask)
        selected = [robots[robot_id] for robot_id in assigned]
        if not all(isinstance(robot, ProcessRobot) for robot in selected):
            raise ValueError("process coalitions contain only process robots")
        for skill, required in enumerate(task.requirements):
            if required and not any(robot.capabilities[skill] for robot in selected):
                raise ValueError("process action does not complete its coalition")


def action_orders(
    assignments: Sequence[Assignment], *, identity: str, random_count: int
) -> tuple[tuple[Assignment, ...], ...]:
    """Return canonical, reverse, and reproducible random action orders."""

    canonical = canonical_action(assignments)
    if not canonical:
        raise ValueError("complete action must be non-empty")
    candidates = [canonical, tuple(reversed(canonical))]
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    for _ in range(random_count):
        shuffled = list(canonical)
        generator.shuffle(shuffled)
        candidates.append(tuple(shuffled))
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def prefix_sequence(
    domain: SchedulingDomain,
    action: PrefixAction,
    order: Sequence[Assignment],
) -> torch.Tensor:
    """Append causal, current-event prefix state to every ordered edge row."""

    validate_complete_action(domain, action.assignments)
    if set(order) != set(action.assignments) or len(order) != len(action.assignments):
        raise ValueError("order must be a permutation of the complete action")
    edge_by_assignment = {
        assignment: action.edge_features[index]
        for index, assignment in enumerate(action.assignments)
    }
    robots = {robot.robot_id: robot for robot in domain.robots}
    tasks = {task.task_id: task for task in domain.tasks}
    task_totals: dict[int, int] = {}
    for _, task_id in action.assignments:
        task_totals[task_id] = task_totals.get(task_id, 0) + 1

    selected_by_task: dict[int, list[int]] = {}
    selected_transport = 0
    selected_process = 0
    rows = []
    total = len(order)
    for index, assignment in enumerate(order):
        robot_id, task_id = assignment
        task = tasks[task_id]
        previous = selected_by_task.get(task_id, [])
        task_fraction = len(previous) / task_totals[task_id]
        required_before = contribution = required_after = completed = 0.0
        if isinstance(task, ProcessTask):
            robot = robots[robot_id]
            assert isinstance(robot, ProcessRobot)
            required_indices = [
                skill for skill, required in enumerate(task.requirements) if required
            ]
            covered = {
                skill
                for previous_robot_id in previous
                for skill in required_indices
                if isinstance(robots[previous_robot_id], ProcessRobot)
                and robots[previous_robot_id].capabilities[skill]
            }
            remaining = set(required_indices) - covered
            added = {
                skill for skill in remaining if robot.capabilities[skill]
            }
            scale = max(1, len(required_indices))
            required_before = len(remaining) / scale
            contribution = len(added) / scale
            required_after = len(remaining - added) / scale
            completed = float(bool(required_indices) and not (remaining - added))

        prefix = torch.tensor(
            (
                index / total,
                (total - index) / total,
                task_fraction,
                float(bool(previous)),
                required_before,
                contribution,
                required_after,
                completed,
                selected_transport / total,
                selected_process / total,
                len(selected_by_task) / max(1, len(task_totals)),
                float(index > 0 and order[index - 1][1] == task_id),
            ),
            dtype=action.edge_features.dtype,
        )
        rows.append(torch.cat((edge_by_assignment[assignment], prefix)))
        selected_by_task.setdefault(task_id, []).append(robot_id)
        if isinstance(task, TransportTask):
            selected_transport += 1
        else:
            selected_process += 1
    result = torch.stack(rows)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("prefix sequence must be finite")
    return result


__all__ = [
    "EDGE_FEATURE_DIM",
    "PREFIX_FEATURE_DIM",
    "PREFIX_FEATURE_NAMES",
    "PrefixAction",
    "action_orders",
    "canonical_action",
    "prefix_sequence",
    "validate_complete_action",
]
