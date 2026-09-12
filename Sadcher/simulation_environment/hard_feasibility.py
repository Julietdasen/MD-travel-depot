"""Centralized hard feasibility checks for MD scheduling actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from simulation_environment.domain_model import (
    ProcessRobot,
    ProcessTask,
    RobotEntity,
    TaskEntity,
    TransportRobot,
    TransportTask,
)


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"


class FeasibilityCode(str, Enum):
    FEASIBLE = "feasible"
    INVALID_ROBOT = "invalid_robot"
    INVALID_TASK = "invalid_task"
    TASK_NOT_PENDING = "task_not_pending"
    TASK_NOT_READY = "task_not_ready"
    ROBOT_OCCUPIED = "robot_occupied"
    ROBOT_TYPE_MISMATCH = "robot_type_mismatch"
    TRANSPORT_CAPABILITY_REQUIRED = "transport_capability_required"
    INSUFFICIENT_CAPACITY = "insufficient_capacity"
    NO_SKILL_CONTRIBUTION = "no_skill_contribution"
    COALITION_INCOMPLETE = "coalition_incomplete"
    INVALID_GRAPH = "invalid_graph"
    INVALID_LOCATION = "invalid_location"
    STATIC_INFEASIBLE = "static_infeasible"
    INVALID_SPEED = "invalid_speed"


@dataclass(frozen=True, slots=True)
class FeasibilityResult:
    is_feasible: bool
    reason_code: FeasibilityCode | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.is_feasible:
            if self.reason_code not in (None, FeasibilityCode.FEASIBLE):
                raise ValueError("feasible result cannot contain a failure code")
            if self.message is not None:
                raise ValueError("feasible result cannot contain a failure message")
            return
        if not isinstance(self.reason_code, FeasibilityCode):
            raise ValueError("infeasible result requires a reason_code")
        if self.reason_code is FeasibilityCode.FEASIBLE:
            raise ValueError("infeasible result cannot use FEASIBLE code")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("infeasible result requires a message")

    @classmethod
    def feasible(cls) -> "FeasibilityResult":
        return cls(True, FeasibilityCode.FEASIBLE)

    @classmethod
    def rejected(cls, code: FeasibilityCode, message: str) -> "FeasibilityResult":
        return cls(False, code, message)


@dataclass(frozen=True, slots=True)
class RobotFeasibilityContext:
    robot: RobotEntity
    available: bool = True


@dataclass(frozen=True, slots=True)
class TaskFeasibilityContext:
    task: TaskEntity
    ready: bool = True
    status: TaskStatus = TaskStatus.PENDING
    covered_skills: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, TaskStatus):
            raise ValueError("status must be a TaskStatus")
        object.__setattr__(self, "covered_skills", tuple(self.covered_skills))


def is_assignment_feasible(
    robot: RobotEntity | RobotFeasibilityContext,
    task: TaskEntity | TaskFeasibilityContext,
) -> FeasibilityResult:
    """Check one atomic robot-task assignment.

    A process robot can contribute an uncovered skill without being the whole
    coalition. Starting that coalition is checked separately below.
    """

    robot_context = (
        robot
        if isinstance(robot, RobotFeasibilityContext)
        else RobotFeasibilityContext(robot)
    )
    task_context = (
        task
        if isinstance(task, TaskFeasibilityContext)
        else TaskFeasibilityContext(task)
    )
    entity_robot = robot_context.robot
    entity_task = task_context.task
    if not isinstance(entity_robot, (ProcessRobot, TransportRobot)):
        return FeasibilityResult.rejected(
            FeasibilityCode.INVALID_ROBOT, "robot must be a typed robot entity"
        )
    if not isinstance(entity_task, (ProcessTask, TransportTask)):
        return FeasibilityResult.rejected(
            FeasibilityCode.INVALID_TASK, "task must be a typed task entity"
        )
    location_result = _locations_feasible(entity_robot, entity_task)
    if location_result is not None:
        return location_result
    if not robot_context.available:
        return FeasibilityResult.rejected(
            FeasibilityCode.ROBOT_OCCUPIED,
            f"robot {entity_robot.robot_id} is occupied",
        )
    if task_context.status is not TaskStatus.PENDING:
        return FeasibilityResult.rejected(
            FeasibilityCode.TASK_NOT_PENDING,
            f"task {entity_task.task_id} is not pending",
        )
    if not task_context.ready:
        return FeasibilityResult.rejected(
            FeasibilityCode.TASK_NOT_READY,
            f"task {entity_task.task_id} is not ready",
        )

    if isinstance(entity_task, ProcessTask):
        if not isinstance(entity_robot, ProcessRobot):
            return FeasibilityResult.rejected(
                FeasibilityCode.ROBOT_TYPE_MISMATCH,
                "PROCESS tasks require PROCESS_ROBOT",
            )
        covered = _skill_vector(entity_robot.capabilities, entity_task.requirements)
        existing = _skill_vector(task_context.covered_skills, entity_task.requirements)
        if not any(
            required and capability and not already
            for required, capability, already in zip(
                entity_task.requirements, covered, existing, strict=True
            )
        ) and any(entity_task.requirements):
            return FeasibilityResult.rejected(
                FeasibilityCode.NO_SKILL_CONTRIBUTION,
                f"PROCESS_ROBOT {entity_robot.robot_id} adds no uncovered skill",
            )
        return FeasibilityResult.feasible()

    if not isinstance(entity_robot, TransportRobot):
        return FeasibilityResult.rejected(
            FeasibilityCode.ROBOT_TYPE_MISMATCH,
            "TRANSPORT tasks require TRANSPORT_ROBOT",
        )
    if entity_robot.loaded_speed is None or entity_robot.loaded_speed <= 0:
        return FeasibilityResult.rejected(
            FeasibilityCode.INVALID_SPEED,
            f"TRANSPORT_ROBOT {entity_robot.robot_id} requires a positive loaded speed",
        )
    if not entity_robot.transport_capable:
        return FeasibilityResult.rejected(
            FeasibilityCode.TRANSPORT_CAPABILITY_REQUIRED,
            f"TRANSPORT_ROBOT {entity_robot.robot_id} is not transport capable",
        )
    if entity_robot.capacity < entity_task.load:
        return FeasibilityResult.rejected(
            FeasibilityCode.INSUFFICIENT_CAPACITY,
            f"robot capacity {entity_robot.capacity} is below load {entity_task.load}",
        )
    return FeasibilityResult.feasible()


def is_process_coalition_start_feasible(
    task: ProcessTask | TaskFeasibilityContext,
    robots: Sequence[ProcessRobot | RobotFeasibilityContext],
) -> FeasibilityResult:
    """Check whether a process coalition covers every required skill."""

    task_context = (
        task if isinstance(task, TaskFeasibilityContext) else TaskFeasibilityContext(task)
    )
    if not isinstance(task_context.task, ProcessTask):
        return FeasibilityResult.rejected(
            FeasibilityCode.ROBOT_TYPE_MISMATCH,
            "coalition start requires a PROCESS task",
        )
    if task_context.status is not TaskStatus.PENDING:
        return FeasibilityResult.rejected(
            FeasibilityCode.TASK_NOT_PENDING,
            f"task {task_context.task.task_id} is not pending",
        )
    if not task_context.ready:
        return FeasibilityResult.rejected(
            FeasibilityCode.TASK_NOT_READY,
            f"task {task_context.task.task_id} is not ready",
        )
    capabilities = list(task_context.covered_skills)
    if len(capabilities) < len(task_context.task.requirements):
        capabilities.extend([False] * (len(task_context.task.requirements) - len(capabilities)))
    for candidate in robots:
        robot = candidate.robot if isinstance(candidate, RobotFeasibilityContext) else candidate
        if not isinstance(robot, ProcessRobot):
            return FeasibilityResult.rejected(
                FeasibilityCode.ROBOT_TYPE_MISMATCH,
                "process coalitions contain only PROCESS_ROBOT entities",
            )
        if isinstance(candidate, RobotFeasibilityContext) and not candidate.available:
            continue
        capabilities = [left or right for left, right in zip(capabilities, robot.capabilities)]
    if all(not required or covered for required, covered in zip(task_context.task.requirements, capabilities, strict=True)):
        return FeasibilityResult.feasible()
    return FeasibilityResult.rejected(
        FeasibilityCode.COALITION_INCOMPLETE,
        f"PROCESS task {task_context.task.task_id} coalition does not cover requirements",
    )


def hard_feasibility_mask(
    robots: Sequence[RobotEntity | RobotFeasibilityContext],
    tasks: Sequence[TaskEntity | TaskFeasibilityContext],
) -> tuple[tuple[bool, ...], ...]:
    """Return a deterministic robot-by-task boolean hard mask."""

    return tuple(
        tuple(is_assignment_feasible(robot, task).is_feasible for task in tasks)
        for robot in robots
    )


def _locations_feasible(
    robot: RobotEntity, task: TaskEntity
) -> FeasibilityResult | None:
    locations = [robot.location]
    if isinstance(task, ProcessTask):
        locations.append(task.location)
    else:
        locations.extend((task.pickup_location, task.delivery_location))
    if all(
        len(location) == 2
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in location)
        for location in locations
    ):
        return None
    return FeasibilityResult.rejected(
        FeasibilityCode.INVALID_LOCATION, "robot and task locations must be finite 2D coordinates"
    )


def _skill_vector(values: Sequence[bool], requirements: Sequence[bool]) -> tuple[bool, ...]:
    return tuple(
        bool(values[index]) if index < len(values) else False
        for index in range(len(requirements))
    )


__all__ = [
    "FeasibilityCode",
    "FeasibilityResult",
    "RobotFeasibilityContext",
    "TaskFeasibilityContext",
    "TaskStatus",
    "hard_feasibility_mask",
    "is_assignment_feasible",
    "is_process_coalition_start_feasible",
]
