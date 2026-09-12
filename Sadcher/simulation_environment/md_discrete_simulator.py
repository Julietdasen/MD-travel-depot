"""Integer-timestep simulator for validated material-delivery domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from experiments.protocol import DatasetSplit, ExperimentResult, FailureReason

from simulation_environment.domain_model import (
    ProcessRobot,
    ProcessTask,
    RobotEntity,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)
from simulation_environment.hard_feasibility import (
    FeasibilityCode,
    FeasibilityResult,
    RobotFeasibilityContext,
    TaskFeasibilityContext,
    TaskStatus,
    is_assignment_feasible,
    is_process_coalition_start_feasible,
)
from simulation_environment.md_static_validator import require_valid_md_domain
from simulation_environment.md_metrics import (
    MDMetrics,
    ProcessExecutionRecord,
    TransportExecutionRecord,
)
from simulation_environment.transport_timing import (
    discrete_duration,
    transport_durations,
    travel_duration,
)


class TransportPhase(str, Enum):
    WAITING = "WAITING"
    TO_PICKUP = "TO_PICKUP"
    LOADING = "LOADING"
    TO_DELIVERY = "TO_DELIVERY"
    UNLOADING = "UNLOADING"
    COMPLETE = "COMPLETE"


class RobotActivity(str, Enum):
    AVAILABLE = "AVAILABLE"
    TRANSPORT = "TRANSPORT"
    PROCESS = "PROCESS"
    RETURNING = "RETURNING"
    AT_EXIT = "AT_EXIT"


@dataclass(slots=True)
class TaskRuntimeState:
    task_id: int
    status: TaskStatus = TaskStatus.PENDING
    transport_phase: TransportPhase | None = None
    assigned_robot_ids: set[int] = field(default_factory=set)
    phase_remaining: int = 0
    started_at: int | None = None
    completed_at: int | None = None


@dataclass(slots=True)
class RobotRuntimeState:
    robot_id: int
    location: tuple[float, float]
    activity: RobotActivity = RobotActivity.AVAILABLE
    task_id: int | None = None
    remaining: int = 0
    busy_since: int | None = None


class MDDiscreteSimulator:
    """Run validated domains through explicit transport and process semantics."""

    def __init__(
        self,
        domain: SchedulingDomain,
        *,
        exit_location: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        require_valid_md_domain(domain)
        self.domain = domain
        self.exit_location = _location(exit_location)
        self.time = 0
        self.done = False
        self.all_real_tasks_completed = False
        self.all_robots_at_exit = False
        self._tasks = {task.task_id: task for task in domain.tasks}
        self._robots = {robot.robot_id: robot for robot in domain.robots}
        self.task_states = {
            task.task_id: TaskRuntimeState(
                task.task_id,
                transport_phase=(
                    TransportPhase.WAITING if isinstance(task, TransportTask) else None
                ),
            )
            for task in domain.tasks
        }
        self.robot_states = {
            robot.robot_id: RobotRuntimeState(robot.robot_id, robot.location)
            for robot in domain.robots
        }
        self._transport_records: dict[int, TransportExecutionRecord] = {}
        self._process_records: dict[int, ProcessExecutionRecord] = {}
        self._robot_occupied_durations = dict.fromkeys(self._robots, 0)
        self._refresh_terminal_flags()

    def task_state(self, task_id: int) -> TaskRuntimeState:
        try:
            return self.task_states[task_id]
        except KeyError as error:
            raise KeyError(f"unknown task {task_id}") from error

    def robot_state(self, robot_id: int) -> RobotRuntimeState:
        try:
            return self.robot_states[robot_id]
        except KeyError as error:
            raise KeyError(f"unknown robot {robot_id}") from error

    @property
    def transport_execution_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for _, record in sorted(self._transport_records.items()))

    @property
    def process_execution_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(record) for _, record in sorted(self._process_records.items())
        )

    @property
    def has_advancing_work(self) -> bool:
        """Whether stepping can advance active work toward termination."""

        if self.done:
            return True
        if any(
            state.status is TaskStatus.IN_PROGRESS
            for state in self.task_states.values()
        ):
            return True
        return any(
            state.activity in (RobotActivity.TRANSPORT, RobotActivity.RETURNING)
            for state in self.robot_states.values()
        )

    def metrics(
        self,
        *,
        inference_time_seconds: float = 0.0,
        failure_reason: FailureReason | None = None,
    ) -> MDMetrics:
        """Return the canonical terminal snapshot for a success or explicit failure."""
        if self.done:
            if failure_reason is not None:
                raise ValueError("completed simulator cannot have a failure reason")
        elif failure_reason is None:
            raise ValueError("unfinished simulator metrics require a failure reason")
        return MDMetrics(
            success=self.done,
            failure_reason=failure_reason,
            makespan=float(self.time) if self.done else None,
            material_starvation=self._material_starvation(),
            robot_utilization=self._robot_utilization(),
            inference_time_seconds=inference_time_seconds,
        )

    def build_experiment_result(
        self,
        *,
        run_id: str,
        method: str,
        instance_id: str,
        seed: int,
        split: DatasetSplit,
        failure_reason: FailureReason | None = None,
        inference_time_seconds: float = 0.0,
        wall_time_seconds: float = 0.0,
        illegal_assignment_count: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentResult:
        """Adapt a terminal simulator snapshot to the frozen experiment protocol."""
        metrics = self.metrics(
            inference_time_seconds=inference_time_seconds,
            failure_reason=failure_reason,
        )
        common: dict[str, Any] = {
            "run_id": run_id,
            "method": method,
            "instance_id": instance_id,
            "seed": seed,
            "split": split,
            "all_real_tasks_completed": self.all_real_tasks_completed,
            "all_robots_at_exit": self.all_robots_at_exit,
            "illegal_assignment_count": illegal_assignment_count,
            "material_starvation": metrics.material_starvation,
            "robot_utilization": metrics.robot_utilization,
            "inference_time_seconds": metrics.inference_time_seconds,
            "wall_time_seconds": wall_time_seconds,
            "process_execution_records": self.process_execution_records,
            "transport_execution_records": self.transport_execution_records,
            "metadata": {} if metadata is None else metadata,
        }
        if metrics.success:
            assert metrics.makespan is not None
            return ExperimentResult.succeeded(makespan=metrics.makespan, **common)
        assert metrics.failure_reason is not None
        return ExperimentResult.failed(reason=metrics.failure_reason, **common)

    def is_task_ready(self, task_id: int) -> bool:
        task = self._tasks[task_id]
        state = self.task_states[task_id]
        if state.status is not TaskStatus.PENDING:
            return False
        if isinstance(task, TransportTask):
            return state.transport_phase is TransportPhase.WAITING
        if any(self.task_states[predecessor].status is not TaskStatus.COMPLETE for predecessor in task.normal_predecessors):
            return False
        if task.material_predecessor is not None:
            if self.task_states[task.material_predecessor].status is not TaskStatus.COMPLETE:
                return False
        return True

    def assignment_feasibility(self, *, robot_id: int, task_id: int) -> FeasibilityResult:
        robot = self._robots.get(robot_id)
        if robot is None:
            return FeasibilityResult.rejected(
                FeasibilityCode.INVALID_ROBOT, f"unknown robot {robot_id}"
            )
        task = self._tasks.get(task_id)
        if task is None:
            return FeasibilityResult.rejected(
                FeasibilityCode.INVALID_TASK, f"unknown task {task_id}"
            )
        robot_state = self.robot_states[robot_id]
        task_state = self.task_states[task_id]
        covered: tuple[bool, ...] = ()
        if isinstance(task, ProcessTask):
            covered = _coalition_capabilities(task, task_state, self._robots)
        return is_assignment_feasible(
            RobotFeasibilityContext(
                robot,
                available=robot_state.activity is RobotActivity.AVAILABLE,
            ),
            TaskFeasibilityContext(
                task,
                ready=self.is_task_ready(task_id),
                status=task_state.status,
                covered_skills=covered,
            ),
        )

    def assign(self, *, robot_id: int, task_id: int) -> FeasibilityResult:
        result = self.assignment_feasibility(robot_id=robot_id, task_id=task_id)
        if not result.is_feasible:
            assert result.reason_code is not None
            raise ValueError(f"{result.reason_code.value}: {result.message}")
        robot = self._robots[robot_id]
        task = self._tasks[task_id]
        task_state = self.task_states[task_id]
        robot_state = self.robot_states[robot_id]
        task_state.assigned_robot_ids.add(robot_id)
        robot_state.task_id = task_id
        robot_state.busy_since = self.time
        if isinstance(task, TransportTask):
            assert isinstance(robot, TransportRobot)
            assert robot.loaded_speed is not None
            durations = transport_durations(robot_state.location, robot, task)
            task_state.transport_phase = TransportPhase.TO_PICKUP
            task_state.phase_remaining = durations.empty_travel
            robot_state.activity = RobotActivity.TRANSPORT
            self._transport_records[task_id] = {
                "task_id": task_id,
                "robot_id": robot_id,
                "assigned_at": self.time,
                "arrived_pickup_at": None,
                "loading_started_at": None,
                "loading_completed_at": None,
                "arrived_delivery_at": None,
                "unloading_started_at": None,
                "unloading_completed_at": None,
                "completed_at": None,
                "empty_travel_duration": durations.empty_travel,
                "loading_duration": durations.loading,
                "loaded_travel_duration": durations.loaded_travel,
                "unloading_duration": durations.unloading,
                "service_duration": durations.service,
                "occupied_duration": None,
            }
        else:
            robot_state.activity = RobotActivity.PROCESS
            if self._can_start_process(task_id):
                self._start_process(task_id)
        return result

    def step(self) -> None:
        if self.done:
            return
        process_task_ids: set[int] = set()
        for robot_id in sorted(self.robot_states):
            state = self.robot_states[robot_id]
            if state.activity is RobotActivity.TRANSPORT:
                self._advance_transport(robot_id)
            elif state.activity is RobotActivity.PROCESS:
                assert state.task_id is not None
                process_task_ids.add(state.task_id)
            elif state.activity is RobotActivity.RETURNING:
                self._advance_return(robot_id)
        for task_id in sorted(process_task_ids):
            self._advance_process(task_id)
        self.time += 1
        self._refresh_terminal_flags()

    def _advance_transport(self, robot_id: int) -> None:
        robot_state = self.robot_states[robot_id]
        task_id = robot_state.task_id
        assert task_id is not None
        task = self._tasks[task_id]
        assert isinstance(task, TransportTask)
        task_state = self.task_states[task_id]
        if task_state.phase_remaining > 0:
            task_state.phase_remaining -= 1
        while task_state.phase_remaining <= 0:
            phase = task_state.transport_phase
            timestamp = self.time + 1
            record = self._transport_records[task_id]
            if phase is TransportPhase.TO_PICKUP:
                robot_state.location = task.pickup_location
                task_state.transport_phase = TransportPhase.LOADING
                task_state.phase_remaining = int(record["loading_duration"])
                record["arrived_pickup_at"] = timestamp
                record["loading_started_at"] = timestamp
                task_state.started_at = timestamp
            elif phase is TransportPhase.LOADING:
                task_state.transport_phase = TransportPhase.TO_DELIVERY
                task_state.phase_remaining = int(record["loaded_travel_duration"])
                record["loading_completed_at"] = timestamp
            elif phase is TransportPhase.TO_DELIVERY:
                robot_state.location = task.delivery_location
                task_state.transport_phase = TransportPhase.UNLOADING
                task_state.phase_remaining = int(record["unloading_duration"])
                record["arrived_delivery_at"] = timestamp
                record["unloading_started_at"] = timestamp
            elif phase is TransportPhase.UNLOADING:
                task_state.transport_phase = TransportPhase.COMPLETE
                task_state.status = TaskStatus.COMPLETE
                task_state.completed_at = timestamp
                record["unloading_completed_at"] = timestamp
                record["completed_at"] = timestamp
                record["occupied_duration"] = timestamp - record["assigned_at"]
                robot_state.activity = RobotActivity.AVAILABLE
                self._release_robot(robot_state, timestamp)
                robot_state.task_id = None
                return
            else:
                return
            if task_state.phase_remaining > 0:
                return

    def _advance_process(self, task_id: int) -> None:
        task_state = self.task_states[task_id]
        if task_state.status is not TaskStatus.IN_PROGRESS:
            return
        task = self._tasks[task_id]
        assert isinstance(task, ProcessTask)
        task_state.phase_remaining -= 1
        if task_state.phase_remaining <= 0:
            completed_at = self.time + 1
            task_state.status = TaskStatus.COMPLETE
            task_state.completed_at = completed_at
            record = self._process_records[task_id]
            record["completed_at"] = completed_at
            record["occupied_duration"] = completed_at - record["started_at"]
            for assigned_id in tuple(task_state.assigned_robot_ids):
                assigned = self.robot_states[assigned_id]
                assigned.activity = RobotActivity.AVAILABLE
                self._release_robot(assigned, completed_at)
                assigned.task_id = None
            task_state.assigned_robot_ids.clear()

    def _advance_return(self, robot_id: int) -> None:
        state = self.robot_states[robot_id]
        if state.remaining > 0:
            state.remaining -= 1
        if state.remaining <= 0:
            state.location = self.exit_location
            state.activity = RobotActivity.AT_EXIT

    def _can_start_process(self, task_id: int) -> bool:
        task = self._tasks[task_id]
        state = self.task_states[task_id]
        if not isinstance(task, ProcessTask):
            return False
        robots = tuple(
            robot
            for robot_id in state.assigned_robot_ids
            for robot in (self._robots[robot_id],)
            if isinstance(robot, ProcessRobot)
        )
        return is_process_coalition_start_feasible(
            TaskFeasibilityContext(
                task,
                ready=self.is_task_ready(task_id),
                status=state.status,
                covered_skills=(),
            ),
            robots,
        ).is_feasible

    def _start_process(self, task_id: int) -> None:
        task = self._tasks[task_id]
        state = self.task_states[task_id]
        assert isinstance(task, ProcessTask)
        state.status = TaskStatus.IN_PROGRESS
        state.started_at = self.time
        state.phase_remaining = discrete_duration(task.duration)
        assignment_times = (
            self.robot_states[robot_id].busy_since
            for robot_id in state.assigned_robot_ids
        )
        first_assignment = min(
            (timestamp for timestamp in assignment_times if timestamp is not None),
            default=self.time,
        )
        self._process_records[task_id] = {
            "task_id": task_id,
            "robot_ids": tuple(sorted(state.assigned_robot_ids)),
            "started_at": self.time,
            "completed_at": None,
            "waiting_duration": (
                self.time - first_assignment if self.domain.config.enabled else 0
            ),
            "service_duration": discrete_duration(task.duration),
            "occupied_duration": None,
        }
        for robot_id in state.assigned_robot_ids:
            self.robot_states[robot_id].location = task.location

    def _refresh_terminal_flags(self) -> None:
        self.all_real_tasks_completed = all(
            state.status is TaskStatus.COMPLETE for state in self.task_states.values()
        )
        if self.all_real_tasks_completed:
            for robot_id, state in self.robot_states.items():
                if state.activity is not RobotActivity.AVAILABLE:
                    continue
                robot = self._robots[robot_id]
                speed = (
                    robot.speed
                    if isinstance(robot, ProcessRobot)
                    else robot.unloaded_speed
                )
                state.activity = RobotActivity.RETURNING
                state.remaining = travel_duration(
                    state.location, self.exit_location, speed
                )
                if state.remaining == 0:
                    state.activity = RobotActivity.AT_EXIT
        self.all_robots_at_exit = all(
            state.activity is RobotActivity.AT_EXIT for state in self.robot_states.values()
        )
        self.done = self.all_real_tasks_completed and self.all_robots_at_exit

    def _release_robot(self, state: RobotRuntimeState, timestamp: int) -> None:
        assert state.busy_since is not None
        self._robot_occupied_durations[state.robot_id] += timestamp - state.busy_since
        state.busy_since = None

    def _material_starvation(self) -> dict[str, float]:
        starvation: dict[str, float] = {}
        for task in self.domain.tasks:
            if not isinstance(task, ProcessTask):
                continue
            if task.material_predecessor is None:
                starvation[str(task.task_id)] = 0.0
                continue
            material_completion = self.task_states[
                task.material_predecessor
            ].completed_at
            if material_completion is None:
                continue
            normal_completions = [
                self.task_states[predecessor].completed_at
                for predecessor in task.normal_predecessors
            ]
            if any(completion is None for completion in normal_completions):
                continue
            last_normal_completion = max(
                (completion for completion in normal_completions if completion is not None),
                default=0,
            )
            starvation[str(task.task_id)] = float(
                max(0, material_completion - last_normal_completion)
            )
        return starvation

    def _robot_utilization(self) -> dict[str, float]:
        elapsed = self.time
        utilization: dict[str, float] = {}
        for robot_id, state in sorted(self.robot_states.items()):
            occupied = self._robot_occupied_durations[robot_id]
            if state.busy_since is not None:
                occupied += elapsed - state.busy_since
            utilization[str(robot_id)] = occupied / elapsed if elapsed else 0.0
        return utilization


def _coalition_capabilities(
    task: ProcessTask,
    task_state: TaskRuntimeState,
    robots: dict[int, RobotEntity],
) -> tuple[bool, ...]:
    capabilities = [False] * len(task.requirements)
    for robot_id in task_state.assigned_robot_ids:
        robot = robots[robot_id]
        if isinstance(robot, ProcessRobot):
            capabilities = [left or right for left, right in zip(capabilities, robot.capabilities)]
    return tuple(capabilities)


def _location(value: tuple[float, float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("exit_location must contain exactly two coordinates")
    return (float(value[0]), float(value[1]))


__all__ = [
    "MDDiscreteSimulator",
    "MDMetrics",
    "ProcessExecutionRecord",
    "TransportExecutionRecord",
    "RobotActivity",
    "RobotRuntimeState",
    "TaskRuntimeState",
    "TaskStatus",
    "TransportPhase",
]
