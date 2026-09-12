"""Residual, counterfactual MILP oracle for stationary late-rollout states."""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from baselines.gurobi_md_oracle import (
    OracleAction,
    OracleScheduleEntry,
    _assignment_travel,
    _build_replay_actions,
    _location,
    _materialize,
    build_gurobi_md_model,
)
from simulation_environment.domain_model import (
    ProcessRobot, ProcessTask, RobotEntity, SchedulingDomain, TransportRobot, TransportTask,
    TaskEntity,
)
from simulation_environment.transport_timing import travel_duration


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
        if isinstance(self.current_time, bool) or not isinstance(self.current_time, int) or self.current_time < 0:
            raise ValueError("current_time must be a non-negative integer")
        object.__setattr__(self, "robot_locations", {int(k): _location(v) for k, v in self.robot_locations.items()})
        object.__setattr__(self, "completed_task_ids", tuple(int(x) for x in self.completed_task_ids))
        object.__setattr__(self, "pending_task_ids", tuple(int(x) for x in self.pending_task_ids))
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
            origin = task.location if isinstance(task, ProcessTask) else task.delivery_location
        else:
            origin = robot.location
        speed = robot.speed if isinstance(robot, ProcessRobot) else robot.unloaded_speed
        result.append((
            travel_duration(origin, exit_location, speed),
            robot.robot_id,
        ))
    return max(result)[1] if result else None


def _compact_task(task: TaskEntity, task_ids: Mapping[int, int]) -> TaskEntity:
    if isinstance(task, ProcessTask):
        return replace(task, task_id=task_ids[task.task_id], normal_predecessors=tuple(task_ids[x] for x in task.normal_predecessors if x in task_ids), material_predecessor=None if task.material_predecessor is None or task.material_predecessor not in task_ids else task_ids[task.material_predecessor])
    return replace(task, task_id=task_ids[task.task_id], downstream_process_task_id=None if task.downstream_process_task_id is None or task.downstream_process_task_id not in task_ids else task_ids[task.downstream_process_task_id])

def residual_domain(domain: SchedulingDomain, state: ResidualMDState) -> SchedulingDomain:
    """Create a zero-based domain containing only pending tasks and snapshot robots."""
    pending = set(state.pending_task_ids)
    original_tasks = tuple(task for task in domain.tasks if task.task_id in pending)
    task_ids = residual_task_map(domain, state).original_to_compact
    tasks = tuple(_compact_task(task, task_ids) for task in original_tasks)
    normal = tuple((task_ids[a], task_ids[b]) for a, b in domain.normal_edges if a in pending and b in pending)
    satisfied_material = set(state.satisfied_material_dependencies)
    material = tuple((task_ids[a], task_ids[b]) for a, b in domain.material_edges if a in pending and b in pending and (a, b) not in satisfied_material)
    # Completed predecessors are already satisfied; edges into pending tasks are removed.
    rebuilt: list[TaskEntity] = []
    for task in tasks:
        if isinstance(task, ProcessTask):
            rebuilt.append(ProcessTask(task.task_id, task.location, task.duration, task.requirements,
                                       tuple(a for a, b in normal if b == task.task_id),
                                       next((a for a, b in material if b == task.task_id), None)))
        else:
            rebuilt.append(task)
    robots: list[RobotEntity] = []
    for robot in domain.robots:
        location = state.robot_locations.get(robot.robot_id, robot.location)
        if isinstance(robot, ProcessRobot):
            robots.append(ProcessRobot(robot.robot_id, location, robot.capabilities, robot.speed))
        else:
            robots.append(TransportRobot(robot.robot_id, location, robot.transport_capable, robot.capacity, robot.unloaded_speed, robot.loaded_speed))
    return SchedulingDomain.create(config=domain.config, tasks=rebuilt, robots=robots, normal_edges=normal, material_edges=material)


def solve_residual_forced_batch(domain: SchedulingDomain, state: ResidualMDState,
                                batch: ForcedAssignmentBatch, *, time_limit_seconds: float = 10.0,
                                threads: int = 1, forbidden_immediate_robot_ids: tuple[int, ...] = (),
                                first_action: tuple[tuple[int, int], ...] = ()) -> ResidualOracleResult:
    rd = residual_domain(domain, state)
    task_map = residual_task_map(domain, state)
    task_ids = task_map.original_to_compact
    batch = ForcedAssignmentBatch(tuple((robot_id, task_ids.get(task_id, task_id)) for robot_id, task_id in batch.assignments))
    spec = build_gurobi_md_model(rd, exit_location=state.exit_location)
    started = time.perf_counter()
    try:
        gp = importlib.import_module("gurobipy")
        model, assignment, start, completion = _materialize(spec, gp)
        first = {
            pair: model.addVar(vtype=gp.GRB.BINARY, name=f"first_{pair[0]}_{pair[1]}")
            for pair in assignment
        }
        for pair, variable in first.items():
            model.addConstr(variable <= assignment[pair], name=f"first_requires_assignment_{pair[0]}_{pair[1]}")
        for robot in rd.robots:
            robot_first = tuple(variable for (robot_id, _), variable in first.items() if robot_id == robot.robot_id)
            if robot_first:
                model.addConstr(gp.quicksum(robot_first) <= 1, name=f"at_most_one_first_{robot.robot_id}")
                for (assigned_robot_id, assigned_task_id), assigned_variable in assignment.items():
                    if assigned_robot_id == robot.robot_id:
                        model.addConstr(gp.quicksum(robot_first) >= assigned_variable, name=f"assigned_has_first_{robot.robot_id}_{assigned_task_id}")
                for (robot_id, task_id), variable in first.items():
                    if robot_id != robot.robot_id:
                        continue
                    for (other_robot_id, other_task_id), other_assignment in assignment.items():
                        if other_robot_id == robot_id and other_task_id != task_id:
                            model.addConstr(start[other_task_id] >= completion[task_id] - spec.big_m * (2 - variable - other_assignment), name=f"after_first_{robot_id}_{task_id}_{other_task_id}")
        for robot_id, task_id in batch.assignments:
            variable = assignment.get((robot_id, task_id))
            if variable is None:
                raise ValueError(f"forced assignment ({robot_id}, {task_id}) is illegal")
            model.addConstr(variable == 1, name=f"forced_assignment_{robot_id}_{task_id}")
            for (other_robot_id, other_task_id), other_variable in assignment.items():
                if other_robot_id == robot_id and other_task_id != task_id:
                    model.addConstr(start[other_task_id] >= completion[task_id] - spec.big_m * (1 - other_variable), name=f"forced_first_{robot_id}_{task_id}_{other_task_id}")
            task = next(t for t in rd.tasks if t.task_id == task_id)
            robot = next(r for r in rd.robots if r.robot_id == robot_id)
            model.addConstr(start[task_id] >= _assignment_travel(robot.location, task, getattr(robot, "speed", getattr(robot, "unloaded_speed", 1.0))), name=f"forced_earliest_{robot_id}_{task_id}")
        for robot_id in forbidden_immediate_robot_ids:
            for (candidate_robot_id, task_id), variable in assignment.items():
                if candidate_robot_id != robot_id:
                    continue
                task = next(t for t in rd.tasks if t.task_id == task_id)
                robot = next(r for r in rd.robots if r.robot_id == robot_id)
                delay = 1 + (_assignment_travel(robot.location, task, getattr(robot, "speed", getattr(robot, "unloaded_speed", 1.0))) if isinstance(task, TransportTask) else 0)
                model.addConstr(start[task_id] >= delay - spec.big_m * (1 - first[robot_id, task_id]), name=f"exact_idle_first_{robot_id}_{task_id}")
        selected_by_task: dict[int, set[int]] = {}
        for robot_id, task_id in first_action:
            compact_task_id = task_ids.get(task_id, task_id)
            selected_by_task.setdefault(compact_task_id, set()).add(robot_id)
            task = next(t for t in rd.tasks if t.task_id == compact_task_id)
            robot = next(r for r in rd.robots if r.robot_id == robot_id)
            model.addConstr(first[robot_id, compact_task_id] == 1, name=f"selected_first_{robot_id}_{compact_task_id}")
            initial = _assignment_travel(robot.location, task, getattr(robot, "speed", getattr(robot, "unloaded_speed", 1.0))) if isinstance(task, TransportTask) else 0
            model.addConstr(start[compact_task_id] == initial, name=f"exact_first_start_{robot_id}_{compact_task_id}")
        for compact_task_id, selected_robot_ids in selected_by_task.items():
            for (other_robot_id, other_task_id), variable in assignment.items():
                if other_task_id == compact_task_id and other_robot_id not in selected_robot_ids:
                    model.addConstr(variable == 0, name=f"exact_no_extra_member_{other_robot_id}_{compact_task_id}")
        model.Params.TimeLimit = float(time_limit_seconds)
        model.Params.Threads = int(threads)
        model.Params.OutputFlag = 0
        model.optimize()
        grb = gp.GRB
        status = int(model.Status)
        if int(model.SolCount) == 0:
            return ResidualOracleResult(ResidualOracleStatus.TIMEOUT if status == int(grb.TIME_LIMIT) else ResidualOracleStatus.INFEASIBLE, None, None, None, None, None, time.perf_counter()-started, message="no feasible incumbent")
        compact_schedule = tuple(sorted((OracleScheduleEntry(r, t, float(start[t].X), float(completion[t].X)) for (r, t), v in assignment.items() if float(v.X) > .5), key=lambda x:(x.start,x.task_id,x.robot_id)))
        task_horizon = float(model.getVarByName("all_tasks_complete").X)
        # The model exposes these canonical objective variables.
        tail = float(model.getVarByName("longest_exit_return").X)
        makespan = state.current_time + float(model.getVarByName("latest_exit_return").X)
        schedule = tuple(
            replace(entry, task_id=task_map.compact_to_original[entry.task_id], start=state.current_time + entry.start, completion=state.current_time + entry.completion)
            for entry in compact_schedule
        )
        last_robot = _latest_return_robot(rd, compact_schedule, state.exit_location)
        actions = tuple(replace(
            action,
            task_id=task_map.compact_to_original[action.task_id],
            planned_assignment=state.current_time + action.planned_assignment,
            planned_start=state.current_time + action.planned_start,
        ) for action in _build_replay_actions(spec, compact_schedule))
        return ResidualOracleResult(ResidualOracleStatus.OPTIMAL if status == int(grb.OPTIMAL) else ResidualOracleStatus.FEASIBLE, makespan, task_horizon, tail, last_robot, float(model.MIPGap), time.perf_counter()-started, schedule, actions, "ok")
    except ImportError:
        return ResidualOracleResult(ResidualOracleStatus.UNAVAILABLE, None, None, None, None, None, time.perf_counter()-started, message="gurobipy is unavailable")
    except Exception as error:
        return ResidualOracleResult(ResidualOracleStatus.ERROR, None, None, None, None, None, time.perf_counter()-started, message=str(error))


def enumerate_forced_batches(domain: SchedulingDomain, state: ResidualMDState) -> tuple[ForcedAssignmentBatch, ...]:
    """Enumerate non-empty combinations of legal, disjoint first actions."""
    import itertools
    rd = residual_domain(domain, state)
    tasks = {task.task_id: task for task in rd.tasks}
    robots = {robot.robot_id: robot for robot in rd.robots}
    spec = build_gurobi_md_model(rd)
    atomic: list[tuple[tuple[int, int], ...]] = []
    for task_id, task in sorted(tasks.items()):
        if isinstance(task, TransportTask):
            atomic.extend(((rid, task_id),) for rid in spec.transport_eligible_robots.get(task_id, ()))
            continue
        if task.normal_predecessors or task.material_predecessor is not None:
            continue
        eligible = tuple(spec.process_eligible_robots.get(task_id, ()))
        minimal: list[tuple[int, ...]] = []
        for size in range(1, len(eligible) + 1):
            for coalition in itertools.combinations(eligible, size):
                if not all(not required or any(robots[r].capabilities[i] for r in coalition) for i, required in enumerate(task.requirements)):
                    continue
                if not any(set(previous).issubset(coalition) for previous in minimal):
                    minimal.append(coalition)
        atomic.extend(tuple((rid, task_id) for rid in coalition) for coalition in minimal)
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
            ForcedAssignmentBatch(tuple((robot_id, compact_to_original[task_id]) for robot_id, task_id in batch.assignments))
            for batch in batches
        )
    )
