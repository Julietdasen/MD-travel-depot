"""Original skill-coverage Greedy adapted to the typed MD simulator."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from experiments.protocol import DatasetSplit, ExperimentResult, FailureReason
from simulation_environment.domain_model import ProcessRobot, ProcessTask
from simulation_environment.hard_feasibility import TaskStatus
from simulation_environment.md_discrete_simulator import (
    MDDiscreteSimulator,
    RobotActivity,
)


def _has_pending_precursor(
    task: ProcessTask, simulator: MDDiscreteSimulator
) -> bool:
    """True iff any precursor is PENDING with no coalition committed yet.

    Mirrors ``schedulers/md_constrained_decoder._apply_ready_preference_guard``:
    a PENDING precursor whose coalition is already traveling (non-empty
    ``assigned_robot_ids``) is safe to overlap with — the successor's robot
    can depart in parallel. Only PENDING+uncommitted precursors risk a
    coalition-starvation deadlock.
    """
    precursors = list(task.normal_predecessors)
    if task.material_predecessor is not None:
        precursors.append(task.material_predecessor)
    task_states = simulator.task_states
    return any(
        task_states[predecessor].status is TaskStatus.PENDING
        and not task_states[predecessor].assigned_robot_ids
        for predecessor in precursors
    )


@dataclass(frozen=True, slots=True)
class ProcessGreedyAssignment:
    """One auditable process assignment selected by the Greedy rule."""

    robot_id: int
    task_id: int
    remaining_required_skills: int
    travel_time: float


class ProcessGreedyMDAdapter:
    """Assign available ProcessRobots using the legacy skill-coverage rule.

    Readiness, type checks, and uncovered-skill contribution are delegated to
    the simulator's centralized hard-feasibility entry point. Assignments are
    applied immediately so later robots in the same decision observe the
    coalition assembled by earlier robots, as in the legacy Greedy scheduler.
    """

    def assign(
        self, simulator: MDDiscreteSimulator
    ) -> tuple[ProcessGreedyAssignment, ...]:
        assignments: list[ProcessGreedyAssignment] = []
        process_robots = sorted(
            (
                robot
                for robot in simulator.domain.robots
                if isinstance(robot, ProcessRobot)
            ),
            key=lambda robot: robot.robot_id,
        )
        process_tasks = sorted(
            (
                task
                for task in simulator.domain.tasks
                if isinstance(task, ProcessTask)
            ),
            key=lambda task: task.task_id,
        )
        robots_by_id = {robot.robot_id: robot for robot in process_robots}

        for robot in process_robots:
            robot_state = simulator.robot_state(robot.robot_id)
            if robot_state.activity is not RobotActivity.AVAILABLE:
                continue

            candidates: list[
                tuple[tuple[int, float, int], ProcessTask, float]
            ] = []
            for task in process_tasks:
                if not simulator.assignment_feasibility(
                    robot_id=robot.robot_id, task_id=task.task_id
                ).is_feasible:
                    continue
                if _has_pending_precursor(task, simulator):
                    continue
                remaining = _remaining_required_skills(
                    task,
                    robot,
                    simulator.task_state(task.task_id).assigned_robot_ids,
                    robots_by_id,
                )
                travel_time = (
                    math.dist(robot_state.location, task.location) / robot.speed
                )
                candidates.append(
                    ((remaining, travel_time, task.task_id), task, travel_time)
                )

            if not candidates:
                continue
            score, selected_task, travel_time = min(
                candidates, key=lambda item: item[0]
            )
            simulator.assign(
                robot_id=robot.robot_id, task_id=selected_task.task_id
            )
            assignments.append(
                ProcessGreedyAssignment(
                    robot_id=robot.robot_id,
                    task_id=selected_task.task_id,
                    remaining_required_skills=score[0],
                    travel_time=travel_time,
                )
            )

        return tuple(assignments)


AuxiliaryPolicy = Callable[[MDDiscreteSimulator], object]


def run_process_greedy_md(
    simulator: MDDiscreteSimulator,
    *,
    run_id: str,
    instance_id: str,
    seed: int,
    split: DatasetSplit,
    max_steps: int,
    auxiliary_policy: AuxiliaryPolicy | None = None,
    method: str = "process_greedy_md",
    metadata: Mapping[str, Any] | None = None,
) -> ExperimentResult:
    """Run Process Greedy and emit the simulator's canonical result envelope.

    ``auxiliary_policy`` is the composition point for a transport baseline. It
    may assign non-process work through the same simulator before Process
    Greedy runs at each decision point.
    """

    if (
        not isinstance(max_steps, int)
        or isinstance(max_steps, bool)
        or max_steps < 0
    ):
        raise ValueError("max_steps must be a non-negative integer")

    scheduler = ProcessGreedyMDAdapter()
    inference_time_seconds = 0.0
    started_at = time.perf_counter()
    steps = 0
    failure_reason: FailureReason | None = None

    while not simulator.done and steps < max_steps:
        decision_started_at = time.perf_counter()
        if auxiliary_policy is not None:
            auxiliary_policy(simulator)
        scheduler.assign(simulator)
        inference_time_seconds += time.perf_counter() - decision_started_at

        if not simulator.has_advancing_work:
            failure_reason = FailureReason.DEADLOCK
            break
        simulator.step()
        steps += 1

    if not simulator.done and failure_reason is None:
        failure_reason = FailureReason.TIMEOUT

    wall_time_seconds = time.perf_counter() - started_at
    return simulator.build_experiment_result(
        run_id=run_id,
        method=method,
        instance_id=instance_id,
        seed=seed,
        split=split,
        failure_reason=failure_reason,
        inference_time_seconds=inference_time_seconds,
        wall_time_seconds=wall_time_seconds,
        metadata={} if metadata is None else metadata,
    )


def _remaining_required_skills(
    task: ProcessTask,
    candidate: ProcessRobot,
    assigned_robot_ids: set[int],
    robots_by_id: dict[int, ProcessRobot],
) -> int:
    covered = list(candidate.capabilities)
    for robot_id in assigned_robot_ids:
        assigned = robots_by_id[robot_id]
        covered = [
            left or right
            for left, right in zip(
                covered, assigned.capabilities, strict=True
            )
        ]
    return sum(
        required and not provided
        for required, provided in zip(
            task.requirements, covered, strict=True
        )
    )




__all__ = [
    "AuxiliaryPolicy",
    "ProcessGreedyAssignment",
    "ProcessGreedyMDAdapter",
    "run_process_greedy_md",
]
