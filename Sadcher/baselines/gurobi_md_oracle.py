"""Optional Gurobi oracle for small material-delivery instances."""

from __future__ import annotations

import importlib
import itertools
import math
import time
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

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
    """License-free, inspectable formulation consumed by the Gurobi adapter."""

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


def build_gurobi_md_model(
    domain: SchedulingDomain,
    *,
    exit_location: tuple[float, float] = (0.0, 0.0),
) -> GurobiMDModel:
    """Build a solver-neutral MILP specification without requiring a license."""

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


def solve_gurobi_md_oracle(
    domain: SchedulingDomain,
    *,
    exit_location: tuple[float, float] = (0.0, 0.0),
    time_limit_seconds: float = 60.0,
    threads: int = 1,
) -> GurobiOracleResult:
    """Materialize and solve the formulation when gurobipy and a license exist."""

    if (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, (int, float))
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds <= 0
    ):
        raise ValueError("time_limit_seconds must be positive and finite")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    spec = build_gurobi_md_model(domain, exit_location=exit_location)
    started = time.perf_counter()
    try:
        gp = importlib.import_module("gurobipy")
    except ImportError:
        return _result(
            spec,
            GurobiOracleStatus.UNAVAILABLE,
            None,
            started,
            message="gurobipy is not installed; model specification remains available",
        )

    try:
        model, assignment, start, completion = _materialize(spec, gp)
        model.Params.TimeLimit = float(time_limit_seconds)
        model.Params.Threads = threads
        model.Params.OutputFlag = 0
        model.optimize()
        status = int(model.Status)
        grb = gp.GRB
        if status == int(grb.INFEASIBLE):
            return _result(
                spec,
                GurobiOracleStatus.INFEASIBLE,
                False,
                started,
                message="Gurobi proved the MD model infeasible",
            )
        has_solution = int(model.SolCount) > 0
        timed_out = status == int(grb.TIME_LIMIT)
        if not has_solution:
            return _result(
                spec,
                GurobiOracleStatus.TIMEOUT if timed_out else GurobiOracleStatus.ERROR,
                None,
                started,
                timeout=timed_out,
                message="Gurobi produced no feasible incumbent",
            )

        schedule = tuple(
            sorted(
                (
                    OracleScheduleEntry(
                        robot_id,
                        task_id,
                        _canonical_time(float(start[task_id].X)),
                        _canonical_time(float(completion[task_id].X)),
                    )
                    for (robot_id, task_id), variable in assignment.items()
                    if float(variable.X) > 0.5
                ),
                key=lambda item: (item.start, item.task_id, item.robot_id),
            )
        )
        actions = _build_replay_actions(spec, schedule)
        gap = float(model.MIPGap) if hasattr(model, "MIPGap") else None
        return _result(
            spec,
            GurobiOracleStatus.OPTIMAL
            if status == int(grb.OPTIMAL)
            else GurobiOracleStatus.FEASIBLE,
            True,
            started,
            timeout=timed_out,
            gap=gap,
            objective=_canonical_time(
                float(model.getVarByName("latest_exit_return").X)
            ),
            actions=actions,
            schedule=schedule,
            message="Gurobi returned a replayable assignment order",
        )
    except Exception as error:
        gurobi_error = getattr(gp, "GurobiError", None)
        license_failure = (
            isinstance(gurobi_error, type)
            and isinstance(error, gurobi_error)
            and (
                getattr(error, "errno", None) == 10009
                or "license" in str(error).lower()
            )
        )
        result_status = (
            GurobiOracleStatus.UNAVAILABLE
            if license_failure
            else GurobiOracleStatus.ERROR
        )
        prefix = (
            "gurobipy license/runtime is unavailable"
            if license_failure
            else "Gurobi model construction or solve failed"
        )
        return _result(
            spec,
            result_status,
            None,
            started,
            message=f"{prefix}: {type(error).__name__}: {error}",
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
        method="gurobi_md_oracle_replay",
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
            if isinstance(task, TransportTask):
                departure = max(
                    previous_completion,
                    entry.start
                    - _assignment_travel(
                        previous_location, task, _speed(robot)
                    ),
                )
            else:
                # The current simulator models process positioning atomically.
                departure = entry.start
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


def _materialize(
    spec: GurobiMDModel, gp: Any
) -> tuple[Any, dict[tuple[int, int], Any], dict[int, Any], dict[int, Any]]:
    model = gp.Model("md_sadcher_oracle")
    tasks = {task.task_id: task for task in spec.domain.tasks}
    robots = {robot.robot_id: robot for robot in spec.domain.robots}
    eligible = {
        **spec.transport_eligible_robots,
        **spec.process_eligible_robots,
    }
    start = {
        task_id: model.addVar(vtype=gp.GRB.INTEGER, lb=0.0, name=f"start_{task_id}")
        for task_id in tasks
    }
    completion = {
        task_id: model.addVar(vtype=gp.GRB.INTEGER, lb=0.0, name=f"completion_{task_id}")
        for task_id in tasks
    }
    assignment = {
        (robot_id, task_id): model.addVar(
            vtype=gp.GRB.BINARY, name=f"a_{robot_id}_{task_id}"
        )
        for task_id, robot_ids in eligible.items()
        for robot_id in robot_ids
    }
    last_assignment = {
        pair: model.addVar(
            vtype=gp.GRB.BINARY, name=f"last_{pair[0]}_{pair[1]}"
        )
        for pair in assignment
    }
    unassigned = {
        robot_id: model.addVar(
            vtype=gp.GRB.BINARY, name=f"unassigned_{robot_id}"
        )
        for robot_id in robots
    }
    task_horizon = model.addVar(
        vtype=gp.GRB.INTEGER, lb=0.0, name="all_tasks_complete"
    )
    return_horizon = model.addVar(
        vtype=gp.GRB.INTEGER, lb=0.0, name="longest_exit_return"
    )
    makespan = model.addVar(
        vtype=gp.GRB.INTEGER, lb=0.0, name="latest_exit_return"
    )

    for task_id, task in tasks.items():
        if isinstance(task, TransportTask):
            model.addConstr(
                gp.quicksum(
                    assignment[robot_id, task_id]
                    for robot_id in eligible[task_id]
                )
                == 1,
                name=f"transport_one_robot_{task_id}",
            )
            duration = gp.quicksum(
                _transport_service(task, robots[robot_id])
                * assignment[robot_id, task_id]
                for robot_id in eligible[task_id]
            )
        else:
            if not spec.process_skill_coverage[task_id]:
                model.addConstr(
                    gp.quicksum(
                        assignment[robot_id, task_id]
                        for robot_id in eligible[task_id]
                    )
                    == 1,
                    name=f"empty_skill_coalition_{task_id}",
                )
            for skill, robot_ids in spec.process_skill_coverage[task_id].items():
                model.addConstr(
                    gp.quicksum(
                        assignment[robot_id, task_id] for robot_id in robot_ids
                    )
                    >= 1,
                    name=f"skill_{task_id}_{skill}",
                )

            # Every selected process robot must own a required skill.
            for skill, robot_ids in spec.process_skill_coverage[task_id].items():
                owner_variables = tuple(
                    (
                        robot_id,
                        model.addVar(
                            vtype=gp.GRB.BINARY,
                            name=f"skill_owner_{task_id}_{skill}_{robot_id}",
                        ),
                    )
                    for robot_id in robot_ids
                )
                model.addConstr(
                    gp.quicksum(variable for _, variable in owner_variables) == 1,
                    name=f"skill_owner_one_{task_id}_{skill}",
                )
                for robot_id, variable in owner_variables:
                    model.addConstr(
                        assignment[robot_id, task_id] >= variable,
                        name=f"skill_owner_requires_assignment_{task_id}_{skill}_{robot_id}",
                    )
            # A coalition is inclusion-minimal only when every selected robot
            # has a required skill that no other selected robot covers. The
            # owner variables above ensure coverage; these variables prevent
            # a fully covering robot from being paired with a redundant member.
            unique_skill: dict[tuple[int, int], Any] = {}
            for skill, robot_ids in spec.process_skill_coverage[task_id].items():
                for robot_id in robot_ids:
                    variable = model.addVar(
                        vtype=gp.GRB.BINARY,
                        name=f"unique_skill_{task_id}_{skill}_{robot_id}",
                    )
                    unique_skill[skill, robot_id] = variable
                    model.addConstr(
                        assignment[robot_id, task_id] >= variable,
                        name=(
                            f"unique_skill_requires_assignment_"
                            f"{task_id}_{skill}_{robot_id}"
                        ),
                    )
                    for other_robot_id in robot_ids:
                        if other_robot_id == robot_id:
                            continue
                        model.addConstr(
                            1 - assignment[other_robot_id, task_id] >= variable,
                            name=(
                                f"unique_skill_exclusive_"
                                f"{task_id}_{skill}_{robot_id}_{other_robot_id}"
                            ),
                        )
            if unique_skill:
                for robot_id in eligible[task_id]:
                    covering_unique = tuple(
                        variable
                        for skill in spec.process_skill_coverage[task_id]
                        if (variable := unique_skill.get((skill, robot_id))) is not None
                    )
                    model.addConstr(
                        gp.quicksum(covering_unique) >= assignment[robot_id, task_id],
                        name=f"minimal_coalition_{task_id}_{robot_id}",
                    )
            duration = discrete_duration(task.duration)
        model.addConstr(
            completion[task_id] == start[task_id] + duration,
            name=f"duration_{task_id}",
        )
        model.addConstr(
            task_horizon >= completion[task_id],
            name=f"task_horizon_{task_id}",
        )
        for robot_id in eligible[task_id]:
            robot = robots[robot_id]
            initial_travel = _assignment_travel(
                robot.location, task, _speed(robot)
            )
            model.addConstr(
                start[task_id]
                >= initial_travel
                - spec.big_m * (1 - assignment[robot_id, task_id]),
                name=f"initial_travel_{robot_id}_{task_id}",
            )
            return_travel = travel_duration(
                _task_end(task), spec.exit_location, _speed(robot)
            )
            model.addConstr(
                return_horizon
                >= return_travel * last_assignment[robot_id, task_id],
                name=f"last_return_{robot_id}_{task_id}",
            )

    for source, target in (*spec.normal_precedence, *spec.material_precedence):
        model.addConstr(
            start[target] >= completion[source],
            name=f"precedence_{source}_{target}",
        )

    for robot_id, robot in robots.items():
        robot_tasks = sorted(
            task_id
            for task_id, robot_ids in eligible.items()
            if robot_id in robot_ids
        )
        robot_assignments = tuple(
            assignment[robot_id, task_id] for task_id in robot_tasks
        )
        robot_lasts = tuple(
            last_assignment[robot_id, task_id] for task_id in robot_tasks
        )
        model.addConstr(
            gp.quicksum(robot_lasts) + unassigned[robot_id] == 1,
            name=f"one_last_location_{robot_id}",
        )
        model.addConstr(
            unassigned[robot_id] >= 1 - gp.quicksum(robot_assignments),
            name=f"unassigned_lower_{robot_id}",
        )
        for task_id in robot_tasks:
            model.addConstr(
                assignment[robot_id, task_id]
                >= last_assignment[robot_id, task_id],
                name=f"last_requires_assignment_{robot_id}_{task_id}",
            )
            model.addConstr(
                1 - assignment[robot_id, task_id] >= unassigned[robot_id],
                name=f"unassigned_upper_{robot_id}_{task_id}",
            )
        initial_return = travel_duration(
            robot.location, spec.exit_location, _speed(robot)
        )
        model.addConstr(
            return_horizon >= initial_return * unassigned[robot_id],
            name=f"initial_return_{robot_id}",
        )

        for index, left in enumerate(robot_tasks):
            for right in robot_tasks[index + 1 :]:
                order = model.addVar(
                    vtype=gp.GRB.BINARY,
                    name=f"before_{robot_id}_{left}_{right}",
                )
                both_left_first = (
                    3
                    - assignment[robot_id, left]
                    - assignment[robot_id, right]
                    - order
                )
                both_right_first = (
                    2
                    - assignment[robot_id, left]
                    - assignment[robot_id, right]
                    + order
                )
                model.addConstr(
                    start[right]
                    >= completion[left]
                    + _assignment_travel(
                        _task_end(tasks[left]), tasks[right], _speed(robot)
                    )
                    - spec.big_m * both_left_first,
                    name=f"sequence_{robot_id}_{left}_{right}",
                )
                model.addConstr(
                    start[left]
                    >= completion[right]
                    + _assignment_travel(
                        _task_end(tasks[right]), tasks[left], _speed(robot)
                    )
                    - spec.big_m * both_right_first,
                    name=f"sequence_{robot_id}_{right}_{left}",
                )
                model.addConstr(
                    2 - assignment[robot_id, right] - order
                    >= last_assignment[robot_id, left],
                    name=f"last_order_{robot_id}_{left}_{right}",
                )
                model.addConstr(
                    1 - assignment[robot_id, left] + order
                    >= last_assignment[robot_id, right],
                    name=f"last_order_{robot_id}_{right}_{left}",
                )

    model.addConstr(
        makespan == task_horizon + return_horizon,
        name="canonical_terminal_makespan",
    )
    assignment_count = gp.quicksum(assignment.values())
    makespan_weight = len(assignment) + 1
    model.setObjective(
        makespan_weight * makespan + assignment_count, gp.GRB.MINIMIZE
    )
    return model, assignment, start, completion


def _canonical_time(value: float) -> float:
    """Snap solver noise to the integer simulator time grid."""
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-5):
        return float(nearest)
    return float(value)


def _result(
    model: GurobiMDModel,
    status: GurobiOracleStatus,
    feasible: bool | None,
    started: float,
    *,
    timeout: bool = False,
    gap: float | None = None,
    objective: float | None = None,
    actions: tuple[OracleAction, ...] = (),
    schedule: tuple[OracleScheduleEntry, ...] = (),
    message: str,
) -> GurobiOracleResult:
    return GurobiOracleResult(
        status,
        feasible,
        time.perf_counter() - started,
        timeout,
        gap,
        objective,
        actions,
        schedule,
        model,
        message,
    )


def _transport_service(task: TransportTask, robot: RobotEntity) -> float:
    assert isinstance(robot, TransportRobot)
    return float(transport_durations(robot.location, robot, task).service)


def _assignment_travel(
    origin: tuple[float, float], task: TaskEntity, speed: float
) -> int:
    # Process positioning is atomic in the canonical simulator.
    if isinstance(task, ProcessTask):
        return 0
    return max(1, travel_duration(origin, _task_start(task), speed))


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


__all__ = [
    "GurobiMDModel",
    "GurobiOracleResult",
    "GurobiOracleStatus",
    "OracleAction",
    "OracleScheduleEntry",
    "build_gurobi_md_model",
    "replay_oracle_actions",
    "solve_gurobi_md_oracle",
]
