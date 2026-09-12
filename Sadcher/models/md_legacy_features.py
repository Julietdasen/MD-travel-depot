"""Legacy 7/9 feature tensors derived from typed MD decision state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from data_generation.md_expert_dataset import MDDecisionSample, TaskDecisionState
from models.md_policy_features import MDPolicyStateView
from simulation_environment.domain_model import TaskType
from simulation_environment.hard_feasibility import TaskStatus


LEGACY_ROBOT_FEATURE_DIM = 7
LEGACY_TASK_FEATURE_DIM = 9
LEGACY_SKILL_DIM = 3


@dataclass(frozen=True, slots=True)
class LegacyPolicyInputs:
    robot_features: torch.Tensor
    task_features: torch.Tensor
    task_adjacency: torch.Tensor


def legacy_policy_inputs_from_samples(
    samples: Sequence[MDDecisionSample | MDPolicyStateView],
) -> LegacyPolicyInputs:
    batch = tuple(samples)
    if not batch:
        raise ValueError("samples must not be empty")
    first = batch[0]
    robot_batches = []
    task_batches = []
    adjacency_batches = []

    for sample in batch:
        if sample.robot_ids != first.robot_ids or sample.task_ids != first.task_ids:
            raise ValueError("one legacy policy batch requires identical IDs")
        task_by_id = {task.task_id: task for task in sample.tasks}
        location_scale = max(
            (
                abs(coordinate)
                for state in (*sample.robots, *sample.tasks)
                for coordinate in state.location
            ),
            default=1.0,
        )
        location_scale = max(location_scale, 1.0)
        duration_scale = max(
            (task.duration for task in sample.tasks), default=1.0
        )
        duration_scale = max(duration_scale, 1.0)
        robot_batches.append(
            [
                _robot_features(robot, task_by_id, location_scale, duration_scale)
                for robot in sample.robots
            ]
        )
        task_batches.append(
            [
                _task_features(task, location_scale, duration_scale)
                for task in sample.tasks
            ]
        )
        task_index = {
            task_id: index for index, task_id in enumerate(sample.task_ids)
        }
        adjacency = torch.zeros(len(sample.task_ids), len(sample.task_ids))
        for source, target in sample.typed_graph.normal_edges:
            adjacency[task_index[source], task_index[target]] = 1.0
        adjacency_batches.append(adjacency)

    return LegacyPolicyInputs(
        robot_features=torch.tensor(robot_batches, dtype=torch.float32),
        task_features=torch.tensor(task_batches, dtype=torch.float32),
        task_adjacency=torch.stack(adjacency_batches),
    )


def legacy_policy_inputs_from_state_views(
    states: Sequence[MDPolicyStateView],
) -> LegacyPolicyInputs:
    """Build legacy 7/9 tensors from present simulator state views."""

    return legacy_policy_inputs_from_samples(tuple(states))


def _robot_features(robot, tasks, location_scale, duration_scale):
    capabilities = _skills(robot.capabilities)
    workload = 0.0
    if not robot.available and robot.task_id is not None:
        workload = max(robot.remaining, tasks[robot.task_id].phase_remaining)
    return (
        robot.location[0] / location_scale,
        robot.location[1] / location_scale,
        workload / duration_scale,
        *capabilities,
        float(robot.available),
    )


def _task_features(task: TaskDecisionState, location_scale, duration_scale):
    requirements = _skills(task.requirements)
    duration = (
        task.duration
        if task.task_type == TaskType.PROCESS.value
        else task.loading_duration + task.unloading_duration
    )
    return (
        task.location[0] / location_scale,
        task.location[1] / location_scale,
        duration / duration_scale,
        *requirements,
        float(task.ready),
        float(bool(task.assigned_robot_ids)),
        float(task.status != TaskStatus.COMPLETE.value),
    )


def _skills(values: tuple[bool, ...]) -> tuple[float, float, float]:
    if len(values) > LEGACY_SKILL_DIM:
        raise ValueError("legacy 7/9 contract supports at most three skills")
    padded = (*values, *(False for _ in range(LEGACY_SKILL_DIM - len(values))))
    return tuple(float(value) for value in padded)


__all__ = [
    "LEGACY_ROBOT_FEATURE_DIM",
    "LEGACY_TASK_FEATURE_DIM",
    "LegacyPolicyInputs",
    "legacy_policy_inputs_from_samples",
    "legacy_policy_inputs_from_state_views",
]
