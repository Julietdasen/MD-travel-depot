"""Shared dataclasses, enums, and pure-Python helpers for the MD MILP oracle.

Historically these lived alongside the Gurobi backend in
``baselines.gurobi_md_oracle`` and ``baselines.gurobi_md_residual_oracle``.
The Gurobi backend was removed on 2026-09-21 (WLS license expired months
earlier and the OR-Tools CP-SAT twin has been the production path since
2026-09-17).  The types and formulation helpers stay in this
solver-agnostic module so both OR-Tools entry points and downstream
callers (data generation, exact-action pipeline, joint decoder) can
consume them without touching a solver backend.

Class names retain the historical ``Gurobi`` prefix (``GurobiOracleResult``,
``GurobiOracleStatus``, ``GurobiMDModel``) to keep the migration diff
mechanical.  A rename is a follow-up.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from experiments.protocol import DatasetSplit, ExperimentResult, FailureReason
from simulation_environment.domain_model import (
    ProcessRobot,
    ProcessTask,
    RobotEntity,
    SchedulingDomain,
    TaskEntity,
    TransportRobot,
    TransportTask,
)
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from simulation_environment.md_static_validator import require_valid_md_domain
from simulation_environment.transport_timing import (
    discrete_duration,
    transport_durations,
    travel_duration,
)


class GurobiOracleStatus(str, Enum):
    MODEL_READY = "model_ready"
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class OracleAction:
    order: int
    robot_id: int
    task_id: int
    planned_assignment: float
    planned_start: float


@dataclass(frozen=True, slots=True)
class OracleScheduleEntry:
    robot_id: int
    task_id: int
    start: float
    completion: float


@dataclass(frozen=True, slots=True)
class GurobiMDModel:
    """Solver-neutral MD MILP specification."""

    domain: SchedulingDomain
    exit_location: tuple[float, float]
    normal_precedence: tuple[tuple[int, int], ...]
    material_precedence: tuple[tuple[int, int], ...]
    transport_eligible_robots: Mapping[int, tuple[int, ...]]
    process_eligible_robots: Mapping[int, tuple[int, ...]]
    process_skill_coverage: Mapping[int, Mapping[int, tuple[int, ...]]]
    objective: str
    big_m: float


@dataclass(frozen=True, slots=True)
class GurobiOracleResult:
    status: GurobiOracleStatus
    feasible: bool | None
    solve_time_seconds: float
    timeout: bool
    optimality_gap: float | None
    objective: float | None
    action_order: tuple[OracleAction, ...]
    schedule: tuple[OracleScheduleEntry, ...]
    model: GurobiMDModel
    message: str


class ResidualOracleStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ResidualMDState:
    current_time: int
    robot_locations: Mapping[int, tuple[float, float]]
    completed_task_ids: tuple[int, ...]
    pending_task_ids: tuple[int, ...]
    satisfied_material_dependencies: tuple[tuple[int, int], ...] = ()
    exit_location: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if (
            isinstance(self.current_time, bool)
            or not isinstance(self.current_time, int)
            or self.current_time < 0
        ):
            raise ValueError("current_time must be a non-negative integer")
        object.__setattr__(
            self,
            "robot_locations",
            {int(k): _location(v) for k, v in self.robot_locations.items()},
        )
        object.__setattr__(
            self,
            "completed_task_ids",
            tuple(int(x) for x in self.completed_task_ids),
        )
        object.__setattr__(
            self,
            "pending_task_ids",
            tuple(int(x) for x in self.pending_task_ids),
        )
        object.__setattr__(self, "exit_location", _location(self.exit_location))


@dataclass(frozen=True, slots=True)
class ForcedAssignmentBatch:
    assignments: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        pairs = tuple((int(r), int(t)) for r, t in self.assignments)
        if not pairs:
            raise ValueError("forced batch must be non-empty")
        if len({r for r, _ in pairs}) != len(pairs):
            raise ValueError("forced batch may not duplicate robots")
        object.__setattr__(self, "assignments", pairs)


@dataclass(frozen=True, slots=True)
class ResidualOracleResult:
    status: ResidualOracleStatus
    makespan: float | None
    remaining_task_horizon: float | None
    terminal_return_tail: float | None
    latest_return_robot_id: int | None
    optimality_gap: float | None
    solve_time_seconds: float
    schedule: tuple[OracleScheduleEntry, ...] = ()
    actions: tuple[OracleAction, ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class ResidualTaskMap:
    original_to_compact: Mapping[int, int]
    compact_to_original: Mapping[int, int]


def build_gurobi_md_model(
    domain: SchedulingDomain,
    *,
    exit_location: tuple[float, float] = (0.0, 0.0),
) -> GurobiMDModel:
    """Build a solver-neutral MILP specification without requiring any solver."""

    require_valid_md_domain(domain)
    exit_point = _location(exit_location)
    transport_eligible: dict[int, tuple[int, ...]] = {}
    process_eligible: dict[int, tuple[int, ...]] = {}
    skill_coverage: dict[int, Mapping[int, tuple[int, ...]]] = {}

    for task in domain.tasks:
        if isinstance(task, TransportTask):
            transport_eligible[task.task_id] = tuple(
                robot.robot_id
                for robot in domain.robots
                if isinstance(robot, TransportRobot)
                and robot.transport_capable
                and robot.capacity >= task.load
                and robot.loaded_speed is not None
            )
            continue
        eligible = tuple(
            robot.robot_id
            for robot in domain.robots
            if isinstance(robot, ProcessRobot)
            and (
                not any(task.requirements)
                or any(
                    required and capability
                    for required, capability in zip(
                        task.requirements, robot.capabilities, strict=True
                    )
                )
            )
        )
        process_eligible[task.task_id] = eligible
        skill_coverage[task.task_id] = MappingProxyType(
            {
                skill: tuple(
                    robot.robot_id
                    for robot in domain.robots
                    if isinstance(robot, ProcessRobot)
                    and robot.capabilities[skill]
                )
                for skill, required in enumerate(task.requirements)
                if required
            }
        )

    return GurobiMDModel(
        domain=domain,
        exit_location=exit_point,
        normal_precedence=domain.normal_edges,
        material_precedence=domain.material_edges,
        transport_eligible_robots=MappingProxyType(transport_eligible),
        process_eligible_robots=MappingProxyType(process_eligible),
        process_skill_coverage=MappingProxyType(skill_coverage),
        objective="minimize_latest_robot_exit_return",
        big_m=_horizon_bound(domain, exit_point),
    )


def replay_oracle_actions(
    domain: SchedulingDomain,
    actions: tuple[OracleAction, ...],
    *,
    exit_location: tuple[float, float] = (0.0, 0.0),
    run_id: str,
    instance_id: str,
    seed: int,
    split: DatasetSplit,
    max_steps: int,
) -> ExperimentResult:
    """Replay timed oracle assignments through the canonical simulator."""

    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps < 0
    ):
        raise ValueError("max_steps must be a non-negative integer")

    simulator = MDDiscreteSimulator(domain, exit_location=exit_location)
    pending = list(sorted(actions, key=lambda action: action.order))
    failure_reason: FailureReason | None = None
    replay_error: dict[str, object] | None = None
    steps = 0
    while not simulator.done and steps < max_steps:
        for action in tuple(pending):
            if action.planned_assignment > simulator.time:
                continue
            feasibility = simulator.assignment_feasibility(
                robot_id=action.robot_id, task_id=action.task_id
            )
            if not feasibility.is_feasible:
                replay_error = {
                    "order": action.order,
                    "robot_id": action.robot_id,
                    "task_id": action.task_id,
                    "planned_assignment": action.planned_assignment,
                    "observed_time": simulator.time,
                    "reason_code": (
                        None
                        if feasibility.reason_code is None
                        else feasibility.reason_code.value
                    ),
                    "message": feasibility.message,
                }
                failure_reason = FailureReason.SCHEDULER_FAILURE
                break
            simulator.assign(robot_id=action.robot_id, task_id=action.task_id)
            pending.remove(action)

        if replay_error is not None:
            break
        if simulator.done:
            break
        if not simulator.has_advancing_work:
            future_action = any(
                action.planned_assignment > simulator.time for action in pending
            )
            if not future_action:
                failure_reason = FailureReason.DEADLOCK
                break
        simulator.step()
        steps += 1

    if not simulator.done and failure_reason is None:
        failure_reason = FailureReason.TIMEOUT
    return simulator.build_experiment_result(
        run_id=run_id,
        method="md_oracle_replay",
        instance_id=instance_id,
        seed=seed,
        split=split,
        failure_reason=failure_reason,
        metadata={
            "oracle_actions": tuple(asdict(action) for action in actions),
            "pending_action_count": len(pending),
            "replay_error": replay_error,
        },
    )


def _build_replay_actions(
    spec: GurobiMDModel,
    schedule: tuple[OracleScheduleEntry, ...],
) -> tuple[OracleAction, ...]:
    tasks = {task.task_id: task for task in spec.domain.tasks}
    robots = {robot.robot_id: robot for robot in spec.domain.robots}
    provisional: list[OracleAction] = []
    for robot_id, robot in robots.items():
        robot_schedule = sorted(
            (entry for entry in schedule if entry.robot_id == robot_id),
            key=lambda entry: (entry.start, entry.task_id),
        )
        previous_completion = 0.0
        previous_location = robot.location
        for entry in robot_schedule:
            task = tasks[entry.task_id]
            departure = max(
                previous_completion,
                entry.start
                - _assignment_travel(previous_location, task, _speed(robot)),
            )
            provisional.append(
                OracleAction(
                    order=-1,
                    robot_id=robot_id,
                    task_id=entry.task_id,
                    planned_assignment=float(departure),
                    planned_start=entry.start,
                )
            )
            previous_completion = entry.completion
            previous_location = _task_end(task)

    coalition_rank = _process_coalition_replay_rank(spec, schedule)
    ordered = sorted(
        provisional,
        key=lambda action: (
            action.planned_assignment,
            action.planned_start,
            action.task_id,
            coalition_rank.get((action.task_id, action.robot_id), action.robot_id),
        ),
    )
    return tuple(
        OracleAction(
            order=index,
            robot_id=action.robot_id,
            task_id=action.task_id,
            planned_assignment=action.planned_assignment,
            planned_start=action.planned_start,
        )
        for index, action in enumerate(ordered)
    )


def _process_coalition_replay_rank(
    spec: GurobiMDModel,
    schedule: tuple[OracleScheduleEntry, ...],
) -> dict[tuple[int, int], int]:
    robots = {robot.robot_id: robot for robot in spec.domain.robots}
    tasks = {task.task_id: task for task in spec.domain.tasks}
    rank: dict[tuple[int, int], int] = {}
    for task_id, task in tasks.items():
        if not isinstance(task, ProcessTask):
            continue
        assigned = tuple(
            sorted(
                entry.robot_id
                for entry in schedule
                if entry.task_id == task_id
            )
        )
        minimal_cover: tuple[int, ...] = ()
        for size in range(1, len(assigned) + 1):
            for candidate in itertools.combinations(assigned, size):
                if all(
                    not required
                    or any(
                        isinstance(robots[robot_id], ProcessRobot)
                        and robots[robot_id].capabilities[skill]
                        for robot_id in candidate
                    )
                    for skill, required in enumerate(task.requirements)
                ):
                    minimal_cover = candidate
                    break
            if minimal_cover:
                break
        minimal_ids = set(minimal_cover)
        ordered = tuple(
            robot_id for robot_id in assigned if robot_id not in minimal_ids
        ) + minimal_cover
        rank.update(
            ((task_id, robot_id), index)
            for index, robot_id in enumerate(ordered)
        )
    return rank


def _canonical_time(value: float) -> float:
    """Snap solver noise to the integer simulator time grid."""
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-5):
        return float(nearest)
    return float(value)


def _transport_service(task: TransportTask, robot: RobotEntity) -> float:
    assert isinstance(robot, TransportRobot)
    return float(transport_durations(robot.location, robot, task).service)


def _assignment_travel(
    origin: tuple[float, float], task: TaskEntity, speed: float
) -> int:
    travel = travel_duration(origin, _task_start(task), speed)
    if isinstance(task, TransportTask):
        return max(1, travel)
    return travel


def _task_start(task: TaskEntity) -> tuple[float, float]:
    return task.location if isinstance(task, ProcessTask) else task.pickup_location


def _task_end(task: TaskEntity) -> tuple[float, float]:
    return task.location if isinstance(task, ProcessTask) else task.delivery_location


def _speed(robot: RobotEntity) -> float:
    return robot.speed if isinstance(robot, ProcessRobot) else robot.unloaded_speed


def _horizon_bound(
    domain: SchedulingDomain, exit_location: tuple[float, float]
) -> float:
    points = [exit_location]
    points.extend(_task_start(task) for task in domain.tasks)
    points.extend(_task_end(task) for task in domain.tasks)
    points.extend(robot.location for robot in domain.robots)
    diameter = max(
        (math.dist(left, right) for left in points for right in points),
        default=0.0,
    )
    slowest_speed = min(_speed(robot) for robot in domain.robots)
    max_travel = math.ceil(diameter / slowest_speed)
    service = 0.0
    for task in domain.tasks:
        if isinstance(task, ProcessTask):
            service += discrete_duration(task.duration)
        else:
            service += max(
                _transport_service(task, robot)
                for robot in domain.robots
                if isinstance(robot, TransportRobot)
                and robot.transport_capable
                and robot.capacity >= task.load
            )
    return max(1.0, service + max_travel * (len(domain.tasks) + 1) + 1.0)


def _location(value: tuple[float, float]) -> tuple[float, float]:
    if len(value) != 2 or not all(math.isfinite(float(item)) for item in value):
        raise ValueError("exit_location must be a finite 2D location")
    return float(value[0]), float(value[1])


def residual_task_map(
    domain: SchedulingDomain, state: ResidualMDState
) -> ResidualTaskMap:
    pending = set(state.pending_task_ids)
    ordered = tuple(task.task_id for task in domain.tasks if task.task_id in pending)
    if set(ordered) != pending:
        missing = sorted(pending - set(ordered))
        raise ValueError(f"pending tasks are absent from domain: {missing}")
    forward = {task_id: index + 1 for index, task_id in enumerate(ordered)}
    return ResidualTaskMap(forward, {value: key for key, value in forward.items()})


def _latest_return_robot(
    domain: SchedulingDomain,
    schedule: tuple[OracleScheduleEntry, ...],
    exit_location: tuple[float, float],
) -> int | None:
    tasks = {task.task_id: task for task in domain.tasks}
    result = []
    for robot in domain.robots:
        assigned = [entry for entry in schedule if entry.robot_id == robot.robot_id]
        if assigned:
            last = max(assigned, key=lambda entry: (entry.completion, entry.task_id))
            task = tasks[last.task_id]
            origin = (
                task.location
                if isinstance(task, ProcessTask)
                else task.delivery_location
            )
        else:
            origin = robot.location
        speed = robot.speed if isinstance(robot, ProcessRobot) else robot.unloaded_speed
        target = (
            robot.home_location if isinstance(robot, ProcessRobot) else exit_location
        )
        result.append((
            travel_duration(origin, target, speed),
            robot.robot_id,
        ))
    return max(result)[1] if result else None


def _compact_task(task: TaskEntity, task_ids: Mapping[int, int]) -> TaskEntity:
    if isinstance(task, ProcessTask):
        return replace(
            task,
            task_id=task_ids[task.task_id],
            normal_predecessors=tuple(
                task_ids[x] for x in task.normal_predecessors if x in task_ids
            ),
            material_predecessor=None
            if task.material_predecessor is None
            or task.material_predecessor not in task_ids
            else task_ids[task.material_predecessor],
        )
    return replace(
        task,
        task_id=task_ids[task.task_id],
        downstream_process_task_id=None
        if task.downstream_process_task_id is None
        or task.downstream_process_task_id not in task_ids
        else task_ids[task.downstream_process_task_id],
    )


def residual_domain(
    domain: SchedulingDomain, state: ResidualMDState
) -> SchedulingDomain:
    """Create a zero-based domain containing only pending tasks and snapshot robots."""
    pending = set(state.pending_task_ids)
    original_tasks = tuple(task for task in domain.tasks if task.task_id in pending)
    task_ids = residual_task_map(domain, state).original_to_compact
    tasks = tuple(_compact_task(task, task_ids) for task in original_tasks)
    normal = tuple(
        (task_ids[a], task_ids[b])
        for a, b in domain.normal_edges
        if a in pending and b in pending
    )
    satisfied_material = set(state.satisfied_material_dependencies)
    material = tuple(
        (task_ids[a], task_ids[b])
        for a, b in domain.material_edges
        if a in pending and b in pending and (a, b) not in satisfied_material
    )
    rebuilt: list[TaskEntity] = []
    for task in tasks:
        if isinstance(task, ProcessTask):
            rebuilt.append(
                ProcessTask(
                    task.task_id,
                    task.location,
                    task.duration,
                    task.requirements,
                    tuple(a for a, b in normal if b == task.task_id),
                    next((a for a, b in material if b == task.task_id), None),
                )
            )
        else:
            rebuilt.append(task)
    robots: list[RobotEntity] = []
    for robot in domain.robots:
        location = state.robot_locations.get(robot.robot_id, robot.location)
        if isinstance(robot, ProcessRobot):
            robots.append(
                ProcessRobot(
                    robot.robot_id,
                    location,
                    robot.capabilities,
                    robot.speed,
                    home_location=robot.home_location,
                )
            )
        else:
            robots.append(
                TransportRobot(
                    robot.robot_id,
                    location,
                    robot.transport_capable,
                    robot.capacity,
                    robot.unloaded_speed,
                    robot.loaded_speed,
                )
            )
    return SchedulingDomain.create(
        config=domain.config,
        tasks=rebuilt,
        robots=robots,
        normal_edges=normal,
        material_edges=material,
    )


def enumerate_forced_batches(
    domain: SchedulingDomain, state: ResidualMDState
) -> tuple[ForcedAssignmentBatch, ...]:
    """Enumerate non-empty combinations of legal, disjoint first actions."""
    rd = residual_domain(domain, state)
    tasks = {task.task_id: task for task in rd.tasks}
    robots = {robot.robot_id: robot for robot in rd.robots}
    spec = build_gurobi_md_model(rd)
    atomic: list[tuple[tuple[int, int], ...]] = []
    for task_id, task in sorted(tasks.items()):
        if isinstance(task, TransportTask):
            atomic.extend(
                ((rid, task_id),)
                for rid in spec.transport_eligible_robots.get(task_id, ())
            )
            continue
        if task.normal_predecessors or task.material_predecessor is not None:
            continue
        eligible = tuple(spec.process_eligible_robots.get(task_id, ()))
        minimal: list[tuple[int, ...]] = []
        for size in range(1, len(eligible) + 1):
            for coalition in itertools.combinations(eligible, size):
                if not all(
                    not required
                    or any(robots[r].capabilities[i] for r in coalition)
                    for i, required in enumerate(task.requirements)
                ):
                    continue
                if not any(
                    set(previous).issubset(coalition) for previous in minimal
                ):
                    minimal.append(coalition)
        atomic.extend(
            tuple((rid, task_id) for rid in coalition) for coalition in minimal
        )
    batches: list[ForcedAssignmentBatch] = []
    for size in range(1, len(atomic) + 1):
        for actions in itertools.combinations(atomic, size):
            assignments = tuple(edge for action in actions for edge in action)
            action_task_ids = tuple(action[0][1] for action in actions)
            if len(action_task_ids) != len(set(action_task_ids)):
                continue
            if len({rid for rid, _ in assignments}) == len(assignments):
                batches.append(ForcedAssignmentBatch(tuple(sorted(assignments))))
    compact_to_original = residual_task_map(domain, state).compact_to_original
    return tuple(
        dict.fromkeys(
            ForcedAssignmentBatch(
                tuple(
                    (robot_id, compact_to_original[task_id])
                    for robot_id, task_id in batch.assignments
                )
            )
            for batch in batches
        )
    )


__all__ = [
    "ForcedAssignmentBatch",
    "GurobiMDModel",
    "GurobiOracleResult",
    "GurobiOracleStatus",
    "OracleAction",
    "OracleScheduleEntry",
    "ResidualMDState",
    "ResidualOracleResult",
    "ResidualOracleStatus",
    "ResidualTaskMap",
    "build_gurobi_md_model",
    "enumerate_forced_batches",
    "replay_oracle_actions",
    "residual_domain",
    "residual_task_map",
]
