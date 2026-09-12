"""Build auditable MD policy tensors from offline decision samples."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from data_generation.md_expert_dataset import (
    MDDecisionSample,
    RobotDecisionState,
    TaskDecisionState,
    TypedTaskGraph,
)
from models.md_policy import MDOpportunityContext, MDPolicyInputs
from simulation_environment.domain_model import RobotType, TaskType
from simulation_environment.hard_feasibility import TaskStatus


@dataclass(frozen=True, slots=True)
class MDPolicyStateView:
    """Present simulator state in the same shape consumed by offline features."""

    robot_ids: tuple[int, ...]
    task_ids: tuple[int, ...]
    typed_graph: TypedTaskGraph
    robots: tuple[RobotDecisionState, ...]
    tasks: tuple[TaskDecisionState, ...]
    hard_feasibility_mask: tuple[tuple[bool, ...], ...]


def md_policy_inputs_from_samples(
    samples: Sequence[MDDecisionSample | MDPolicyStateView],
) -> MDPolicyInputs:
    """Convert a same-shape sample batch without recomputing legality."""

    batch = tuple(samples)
    if not batch:
        raise ValueError("samples must not be empty")
    first = batch[0]
    for sample in batch:
        if not isinstance(sample, (MDDecisionSample, MDPolicyStateView)):
            raise TypeError(
                "samples must contain MDDecisionSample or MDPolicyStateView values"
            )
        if sample.robot_ids != first.robot_ids or sample.task_ids != first.task_ids:
            raise ValueError("one MD policy batch requires identical robot/task IDs")

    robot_metadata = []
    task_metadata = []
    pair_metadata = []
    task_is_transport = []
    typed_adjacency = []
    downstream_indices = []
    hard_masks = []
    opportunity_context = []
    task_index = {task_id: index for index, task_id in enumerate(first.task_ids)}

    for sample in batch:
        robots = {state.robot_id: state for state in sample.robots}
        tasks = {state.task_id: state for state in sample.tasks}
        task_opportunity = _task_opportunity_contexts(robots, tasks)
        robot_metadata.append(
            [_robot_metadata(robots[robot_id], tasks) for robot_id in sample.robot_ids]
        )
        task_metadata.append(
            [_task_metadata(tasks[task_id], tasks) for task_id in sample.task_ids]
        )
        pair_metadata.append(
            [
                [
                    _pair_metadata(robots[robot_id], tasks[task_id], tasks)
                    for task_id in sample.task_ids
                ]
                for robot_id in sample.robot_ids
            ]
        )
        opportunity_context.append(
            [
                [task_opportunity[task_id].values for task_id in sample.task_ids]
                for _robot_id in sample.robot_ids
            ]
        )
        task_is_transport.append(
            [
                tasks[task_id].task_type == TaskType.TRANSPORT.value
                for task_id in sample.task_ids
            ]
        )
        relations = torch.zeros(2, len(sample.task_ids), len(sample.task_ids))
        for source, target in sample.typed_graph.normal_edges:
            relations[0, task_index[source], task_index[target]] = 1.0
        for source, target in sample.typed_graph.material_edges:
            relations[1, task_index[source], task_index[target]] = 1.0
        typed_adjacency.append(relations)
        downstream_indices.append(
            [
                _downstream_index(tasks[task_id], task_index)
                for task_id in sample.task_ids
            ]
        )
        hard_masks.append(sample.hard_feasibility_mask)

    return MDPolicyInputs(
        robot_metadata=torch.tensor(robot_metadata, dtype=torch.float32),
        task_metadata=torch.tensor(task_metadata, dtype=torch.float32),
        pair_metadata=torch.tensor(pair_metadata, dtype=torch.float32),
        task_is_transport=torch.tensor(task_is_transport, dtype=torch.bool),
        typed_adjacency=torch.stack(typed_adjacency),
        downstream_task_index=torch.tensor(downstream_indices, dtype=torch.long),
        hard_feasibility_mask=torch.tensor(hard_masks, dtype=torch.bool),
        opportunity_context=torch.tensor(
            opportunity_context, dtype=torch.float32
        ),
    )


def md_policy_inputs_from_state_views(
    states: Sequence[MDPolicyStateView],
) -> MDPolicyInputs:
    """Build policy metadata from present simulator state views."""

    return md_policy_inputs_from_samples(tuple(states))


def _robot_metadata(
    robot: RobotDecisionState, tasks: dict[int, TaskDecisionState]
) -> tuple[float, float, float, float]:
    is_transport = robot.robot_type == RobotType.TRANSPORT_ROBOT.value
    available_in = 0.0
    if not robot.available:
        available_in = float(robot.remaining)
        if robot.task_id is not None:
            available_in = max(
                available_in, float(tasks[robot.task_id].phase_remaining)
            )
    return (
        float(is_transport),
        available_in,
        0.0 if robot.capacity is None else robot.capacity,
        0.0 if robot.loaded_speed is None else robot.loaded_speed,
    )


def _task_metadata(
    task: TaskDecisionState, tasks: dict[int, TaskDecisionState]
) -> tuple[float, float, float, float, float, float, float, float]:
    unfinished_normal = sum(
        tasks[task_id].status != TaskStatus.COMPLETE.value
        for task_id in task.normal_predecessors
    )
    unfinished_material = int(
        task.material_predecessor is not None
        and tasks[task.material_predecessor].status != TaskStatus.COMPLETE.value
    )
    return (
        float(task.task_type == TaskType.TRANSPORT.value),
        task.load,
        task.loading_duration,
        task.unloading_duration,
        float(unfinished_normal),
        float(unfinished_material),
        task.duration,
        float(task.status == TaskStatus.PENDING.value),
    )


def _pair_metadata(
    robot: RobotDecisionState,
    task: TaskDecisionState,
    tasks: dict[int, TaskDecisionState],
) -> tuple[float, float, float, float, float]:
    if (
        robot.robot_type != RobotType.TRANSPORT_ROBOT.value
        or task.task_type != TaskType.TRANSPORT.value
    ):
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    if (
        task.pickup_location is None
        or task.delivery_location is None
        or robot.loaded_speed is None
        or robot.capacity is None
        or robot.speed <= 0
        or robot.loaded_speed <= 0
    ):
        raise ValueError("transport pair requires complete positive physics metadata")
    empty_distance = math.dist(robot.location, task.pickup_location)
    loaded_distance = math.dist(task.pickup_location, task.delivery_location)
    available_in = _robot_metadata(robot, tasks)[1]
    eta = (
        available_in
        + empty_distance / robot.speed
        + task.loading_duration
        + loaded_distance / robot.loaded_speed
        + task.unloading_duration
    )
    return (
        eta,
        task.load,
        robot.capacity - task.load,
        robot.loaded_speed,
        loaded_distance,
    )


def _downstream_index(
    task: TaskDecisionState, task_index: dict[int, int]
) -> int:
    downstream_id = task.downstream_process_task_id
    if downstream_id is None:
        return -1
    return task_index[downstream_id]


def _task_opportunity_contexts(
    robots: dict[int, RobotDecisionState],
    tasks: dict[int, TaskDecisionState],
) -> dict[int, MDOpportunityContext]:
    """Return present-state proxies without consulting an expert future schedule."""

    successors = {
        task_id: tuple(
            candidate.task_id
            for candidate in tasks.values()
            if task_id in candidate.normal_predecessors
        )
        for task_id in tasks
    }
    path_cache: dict[int, float] = {}

    def remaining_critical_path(
        task_id: int, visiting: frozenset[int] = frozenset()
    ) -> float:
        if task_id in path_cache:
            return path_cache[task_id]
        if task_id in visiting:
            raise ValueError("normal precedence graph must be acyclic")
        task = tasks[task_id]
        if task.status == TaskStatus.COMPLETE.value:
            return 0.0
        own_work = max(
            float(task.phase_remaining) if task.phase_remaining > 0 else task.duration,
            0.0,
        )
        tail = max(
            (
                remaining_critical_path(successor, visiting | {task_id})
                for successor in successors[task_id]
            ),
            default=0.0,
        )
        path_cache[task_id] = own_work + tail
        return path_cache[task_id]

    process_paths = [
        remaining_critical_path(task.task_id)
        for task in tasks.values()
        if task.task_type == TaskType.PROCESS.value
        and task.status != TaskStatus.COMPLETE.value
    ]
    path_scale = max(process_paths, default=1.0) or 1.0
    available_process = tuple(
        robot
        for robot in robots.values()
        if robot.robot_type == RobotType.PROCESS_ROBOT.value and robot.available
    )
    available_transport = tuple(
        robot
        for robot in robots.values()
        if robot.robot_type == RobotType.TRANSPORT_ROBOT.value and robot.available
    )
    contexts: dict[int, MDOpportunityContext] = {}
    for task in tasks.values():
        downstream_id = task.downstream_process_task_id
        if task.task_type != TaskType.TRANSPORT.value or downstream_id is None:
            contexts[task.task_id] = MDOpportunityContext()
            continue
        downstream = tasks[downstream_id]
        criticality = remaining_critical_path(downstream_id) / path_scale
        last_blocker = float(
            task.status != TaskStatus.COMPLETE.value
            and downstream.status != TaskStatus.COMPLETE.value
            and downstream.material_predecessor == task.task_id
            and all(
                tasks[predecessor].status == TaskStatus.COMPLETE.value
                for predecessor in downstream.normal_predecessors
            )
        )
        required_skills = tuple(
            index for index, required in enumerate(downstream.requirements) if required
        )
        skill_supply = tuple(
            sum(robot.capabilities[index] for robot in available_process)
            for index in required_skills
        )
        coalition_availability = (
            1.0
            if not skill_supply
            else sum(supply > 0 for supply in skill_supply) / len(skill_supply)
        )
        coalition_scarcity = max(
            (1.0 / supply if supply > 0 else 1.0 for supply in skill_supply),
            default=0.0,
        )
        capable_transport = sum(
            robot.capacity is not None and robot.capacity >= task.load
            for robot in available_transport
        )
        capacity_scarcity = 1.0 / capable_transport if capable_transport else 1.0
        contexts[task.task_id] = MDOpportunityContext(
            critical_path_proxy=criticality,
            last_material_blocker=last_blocker,
            coalition_availability=coalition_availability,
            coalition_scarcity=coalition_scarcity,
            capacity_scarcity=capacity_scarcity,
        )
    return contexts


__all__ = [
    "MDPolicyStateView",
    "md_policy_inputs_from_samples",
    "md_policy_inputs_from_state_views",
]
