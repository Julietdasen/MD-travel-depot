"""OR-Tools CP-SAT prototype for the MD-Sadcher MILP oracle.

This is a license-free alternative to ``baselines.gurobi_md_oracle``. It
translates the same MILP formulation to Google OR-Tools' CP-SAT solver so we
can generate Path A training data (process_scarce / dependency_deep tc in
{24,42,60}) without a Gurobi WLS license.

The formulation mirrors ``gurobi_md_oracle._materialize``:

* Integer start / completion times per task
* Binary assignment[r,t] for eligible robots
* Skill coverage + inclusion-minimal coalition constraints for process tasks
* Transport singleton (exactly one robot per transport task)
* Precedence: ``start[j] >= completion[i]``
* Per-robot ordering with sequence transition times (travel time between
  consecutive tasks on the same robot) implemented via CP-SAT ``Circuit``
  (equivalent to sequenceing constraint with transition times)
* Return-travel from each robot's last task to the exit location
* Objective: ``makespan_weight * makespan + sum(assignment)``

The output types (`OracleAction`, `OracleScheduleEntry`, `GurobiOracleResult`)
are re-used so the caller side (data generation pipeline) does not need to
change.
"""

from __future__ import annotations

import math
import time
from typing import Any

from ortools.sat.python import cp_model

from baselines.md_oracle_types import (
    GurobiMDModel,
    GurobiOracleResult,
    GurobiOracleStatus,
    OracleAction,
    OracleScheduleEntry,
    _assignment_travel,
    _build_replay_actions,
    _canonical_time,
    _speed,
    _task_end,
    _transport_service,
    build_gurobi_md_model,
)
from simulation_environment.domain_model import (
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportTask,
)
from simulation_environment.transport_timing import (
    discrete_duration,
    travel_duration,
)


def solve_ortools_md_oracle(
    domain: SchedulingDomain,
    *,
    exit_location: tuple[float, float] = (0.0, 0.0),
    time_limit_seconds: float = 300.0,
    threads: int = 8,
    log_search: bool = False,
) -> GurobiOracleResult:
    """CP-SAT twin of ``solve_gurobi_md_oracle``.

    Returns the *same* result dataclass so downstream code can consume it
    interchangeably. The ``status`` uses the OPTIMAL/FEASIBLE/TIMEOUT/INFEASIBLE
    codes from GurobiOracleStatus.
    """

    if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive and finite")
    if not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    spec = build_gurobi_md_model(domain, exit_location=exit_location)
    started = time.perf_counter()

    model = cp_model.CpModel()
    tasks = {task.task_id: task for task in spec.domain.tasks}
    robots = {robot.robot_id: robot for robot in spec.domain.robots}
    eligible = {
        **spec.transport_eligible_robots,
        **spec.process_eligible_robots,
    }
    horizon = int(math.ceil(spec.big_m))

    # ------------------------------------------------------------------
    # Base variables
    # ------------------------------------------------------------------
    start: dict[int, cp_model.IntVar] = {}
    end: dict[int, cp_model.IntVar] = {}
    duration_var: dict[int, cp_model.IntVar] = {}
    for task_id in tasks:
        start[task_id] = model.NewIntVar(0, horizon, f"start_{task_id}")
        end[task_id] = model.NewIntVar(0, horizon, f"end_{task_id}")
        duration_var[task_id] = model.NewIntVar(0, horizon, f"dur_{task_id}")
        model.Add(end[task_id] == start[task_id] + duration_var[task_id])

    assignment: dict[tuple[int, int], cp_model.IntVar] = {}
    for task_id, robot_ids in eligible.items():
        for robot_id in robot_ids:
            assignment[robot_id, task_id] = model.NewBoolVar(
                f"a_{robot_id}_{task_id}"
            )

    # ------------------------------------------------------------------
    # Per-task assignment / skill / duration constraints
    # ------------------------------------------------------------------
    for task_id, task in tasks.items():
        if isinstance(task, TransportTask):
            # Exactly one transport robot performs the task.
            model.Add(
                sum(
                    assignment[robot_id, task_id]
                    for robot_id in eligible[task_id]
                )
                == 1
            )
            # Duration depends on which robot is assigned (empty travel differs
            # by robot start location, but ``_transport_service`` uses the
            # robot's own location at t=0; the sequence transitions below carry
            # the actual empty travel). Service = loading + loaded_travel +
            # unloading is what we treat as task duration.
            duration_by_robot: list[tuple[cp_model.IntVar, int]] = []
            for robot_id in eligible[task_id]:
                dur = int(
                    round(
                        _transport_service(task, robots[robot_id])
                        - travel_duration(
                            robots[robot_id].location,
                            task.pickup_location,
                            robots[robot_id].unloaded_speed,
                        )
                    )
                )
                # We want *only* the service portion (loading + loaded +
                # unloading) as the interval duration; empty-travel is a
                # transition, handled by Circuit.
                service = int(_transport_service(task, robots[robot_id]))
                duration_by_robot.append((assignment[robot_id, task_id], service))
            # duration_var = sum(assign_r * service_r) since exactly one is 1.
            model.Add(
                duration_var[task_id]
                == sum(a * s for a, s in duration_by_robot)
            )
        else:
            coverage = spec.process_skill_coverage[task_id]
            if not coverage:
                # Skill-less process task: still needs exactly one robot.
                model.Add(
                    sum(
                        assignment[robot_id, task_id]
                        for robot_id in eligible[task_id]
                    )
                    == 1
                )
            for skill, robot_ids in coverage.items():
                # Skill coverage: at least one selected robot covers this
                # required skill.
                model.Add(
                    sum(assignment[robot_id, task_id] for robot_id in robot_ids)
                    >= 1
                )

            # Coalition minimality (mirrors the Gurobi unique_skill trick).
            unique_skill: dict[tuple[int, int], cp_model.IntVar] = {}
            for skill, robot_ids in coverage.items():
                for robot_id in robot_ids:
                    var = model.NewBoolVar(
                        f"unique_{task_id}_{skill}_{robot_id}"
                    )
                    unique_skill[skill, robot_id] = var
                    # If var is 1 then robot_id is assigned.
                    model.Add(assignment[robot_id, task_id] >= var)
                    # Exclusive uniqueness: no other covering robot for the same
                    # skill can be assigned.
                    for other in robot_ids:
                        if other == robot_id:
                            continue
                        model.Add(
                            assignment[other, task_id] + var <= 1
                        )
            if unique_skill:
                for robot_id in eligible[task_id]:
                    covering = [
                        var
                        for skill in coverage
                        if (var := unique_skill.get((skill, robot_id))) is not None
                    ]
                    if covering:
                        model.Add(
                            sum(covering) >= assignment[robot_id, task_id]
                        )
                    else:
                        # Robot has no required skill for this task; forbid it.
                        model.Add(assignment[robot_id, task_id] == 0)
            # Duration is deterministic for process tasks.
            model.Add(duration_var[task_id] == discrete_duration(task.duration))

    # ------------------------------------------------------------------
    # Precedence
    # ------------------------------------------------------------------
    for source, target in (*spec.normal_precedence, *spec.material_precedence):
        model.Add(start[target] >= end[source])

    # ------------------------------------------------------------------
    # Per-robot sequence with transition times, via Circuit.
    # ------------------------------------------------------------------
    # For each robot we introduce optional intervals per eligible task, and use
    # a Circuit over (source, task1, task2, ..., sink) with arc literals that
    # imply start[next] >= end[prev] + travel_time. This matches the Gurobi
    # big-M sequence constraint but is tighter for CP-SAT.

    return_horizon = model.NewIntVar(0, horizon, "return_horizon")
    task_horizon_var = model.NewIntVar(0, horizon, "task_horizon")
    for task_id in tasks:
        model.Add(task_horizon_var >= end[task_id])

    # last_assignment[robot_id, task_id]: 1 iff task_id is the last executed
    # task on robot_id. Used for return travel horizon.
    last_assignment: dict[tuple[int, int], cp_model.IntVar] = {}
    unassigned: dict[int, cp_model.IntVar] = {}

    for robot_id, robot in robots.items():
        robot_tasks = sorted(
            task_id
            for task_id, robot_ids in eligible.items()
            if robot_id in robot_ids
        )
        # last_assignment vars.
        for task_id in robot_tasks:
            last_assignment[robot_id, task_id] = model.NewBoolVar(
                f"last_{robot_id}_{task_id}"
            )
            model.Add(
                assignment[robot_id, task_id]
                >= last_assignment[robot_id, task_id]
            )
        unassigned[robot_id] = model.NewBoolVar(f"unassigned_{robot_id}")

        # sum(last_assign) + unassigned == 1
        model.Add(
            sum(last_assignment[robot_id, t] for t in robot_tasks)
            + unassigned[robot_id]
            == 1
        )
        # unassigned = 1 iff no tasks assigned.
        # sum(assign) == 0 => unassigned = 1; sum(assign) >= 1 => unassigned = 0.
        assigns = [assignment[robot_id, t] for t in robot_tasks]
        if assigns:
            model.Add(sum(assigns) >= 1).OnlyEnforceIf(unassigned[robot_id].Not())
            for a in assigns:
                model.Add(a == 0).OnlyEnforceIf(unassigned[robot_id])

        # Initial return travel horizon when idle.
        return_target = (
            robot.home_location
            if isinstance(robot, ProcessRobot)
            else spec.exit_location
        )
        initial_return = travel_duration(
            robot.location, return_target, _speed(robot)
        )
        model.Add(return_horizon >= initial_return).OnlyEnforceIf(
            unassigned[robot_id]
        )

        # Per-task return travel horizon when it's the last task.
        for task_id in robot_tasks:
            ret = travel_duration(
                _task_end(tasks[task_id]), return_target, _speed(robot)
            )
            model.Add(return_horizon >= ret).OnlyEnforceIf(
                last_assignment[robot_id, task_id]
            )

        if not robot_tasks:
            model.Add(unassigned[robot_id] == 1)
            continue

        # Initial travel from robot start location.
        for task_id in robot_tasks:
            initial_travel = _assignment_travel(
                robot.location, tasks[task_id], _speed(robot)
            )
            model.Add(start[task_id] >= initial_travel).OnlyEnforceIf(
                assignment[robot_id, task_id]
            )

        # Circuit-based sequencing on this robot.
        # Nodes: 0 = depot (start/end), 1..N = tasks in robot_tasks (indexed by
        # position). Each robot only visits tasks assigned to it.
        node_ids = {i + 1: t for i, t in enumerate(robot_tasks)}
        arcs: list[tuple[int, int, cp_model.IntVar]] = []

        # Depot self-loop (robot idle) allowed only if all assigns are zero.
        # Circuit does not directly express "optional visits"; we emulate:
        # for each task node i, add a self-loop with literal (1 - assignment).
        for i, task_id in node_ids.items():
            skip_lit = model.NewBoolVar(f"skip_{robot_id}_{task_id}")
            model.Add(skip_lit == 1 - assignment[robot_id, task_id])
            arcs.append((i, i, skip_lit))

        # Depot -> task i: robot's first task after starting at its location.
        # We use a helper bool first_r_t which is 1 iff task_id is the first
        # task executed by robot_id.
        first_at: dict[int, cp_model.IntVar] = {}
        for i, task_id in node_ids.items():
            var = model.NewBoolVar(f"first_{robot_id}_{task_id}")
            first_at[task_id] = var
            model.Add(assignment[robot_id, task_id] >= var)
            arcs.append((0, i, var))
        # last_at: mapped to last_assignment.
        for i, task_id in node_ids.items():
            arcs.append((i, 0, last_assignment[robot_id, task_id]))

        # If unassigned, depot self-loop must be active.
        depot_self = model.NewBoolVar(f"depot_self_{robot_id}")
        arcs.append((0, 0, depot_self))
        model.Add(depot_self == unassigned[robot_id])
        # Exactly one first_at when active.
        model.Add(
            sum(first_at.values()) == 1 - unassigned[robot_id]
        )

        # Task-to-task arcs with transition time enforcement.
        for i, left in node_ids.items():
            for j, right in node_ids.items():
                if i == j:
                    continue
                arc = model.NewBoolVar(f"arc_{robot_id}_{left}_{right}")
                arcs.append((i, j, arc))
                # If arc active, both assigned.
                model.Add(assignment[robot_id, left] >= arc)
                model.Add(assignment[robot_id, right] >= arc)
                # Transition: right starts after left ends + travel.
                travel = _assignment_travel(
                    _task_end(tasks[left]), tasks[right], _speed(robot)
                )
                model.Add(
                    start[right] >= end[left] + travel
                ).OnlyEnforceIf(arc)

        model.AddCircuit(arcs)

    # ------------------------------------------------------------------
    # Objective: makespan + weighted assignment count (matches Gurobi).
    # ------------------------------------------------------------------
    makespan = model.NewIntVar(0, 2 * horizon, "latest_exit_return")
    model.Add(makespan == task_horizon_var + return_horizon)

    assignment_count = sum(assignment.values())
    weight = len(assignment) + 1
    model.Minimize(weight * makespan + assignment_count)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(threads)
    solver.parameters.log_search_progress = bool(log_search)

    status_code = solver.Solve(model)
    elapsed = time.perf_counter() - started

    if status_code == cp_model.INFEASIBLE:
        return GurobiOracleResult(
            status=GurobiOracleStatus.INFEASIBLE,
            feasible=False,
            solve_time_seconds=elapsed,
            timeout=False,
            optimality_gap=None,
            objective=None,
            action_order=(),
            schedule=(),
            model=spec,
            message="CP-SAT proved the MD model infeasible",
        )

    if status_code == cp_model.UNKNOWN or status_code == cp_model.MODEL_INVALID:
        return GurobiOracleResult(
            status=GurobiOracleStatus.ERROR
            if status_code == cp_model.MODEL_INVALID
            else GurobiOracleStatus.TIMEOUT,
            feasible=None,
            solve_time_seconds=elapsed,
            timeout=(status_code == cp_model.UNKNOWN),
            optimality_gap=None,
            objective=None,
            action_order=(),
            schedule=(),
            model=spec,
            message=f"CP-SAT returned status {solver.StatusName(status_code)}",
        )

    # FEASIBLE or OPTIMAL
    schedule_entries = []
    for (robot_id, task_id), var in assignment.items():
        if solver.Value(var) == 1:
            schedule_entries.append(
                OracleScheduleEntry(
                    robot_id=robot_id,
                    task_id=task_id,
                    start=_canonical_time(float(solver.Value(start[task_id]))),
                    completion=_canonical_time(float(solver.Value(end[task_id]))),
                )
            )
    schedule = tuple(
        sorted(schedule_entries, key=lambda x: (x.start, x.task_id, x.robot_id))
    )
    actions = _build_replay_actions(spec, schedule)

    makespan_value = _canonical_time(float(solver.Value(makespan)))
    best_bound = solver.BestObjectiveBound()
    obj_value = solver.ObjectiveValue()
    if obj_value > 0:
        gap = max(0.0, (obj_value - best_bound) / max(1e-9, abs(obj_value)))
    else:
        gap = 0.0

    return GurobiOracleResult(
        status=GurobiOracleStatus.OPTIMAL
        if status_code == cp_model.OPTIMAL
        else GurobiOracleStatus.FEASIBLE,
        feasible=True,
        solve_time_seconds=elapsed,
        timeout=(status_code == cp_model.FEASIBLE),
        optimality_gap=gap,
        objective=makespan_value,
        action_order=actions,
        schedule=schedule,
        model=spec,
        message=f"CP-SAT returned {solver.StatusName(status_code)} in {elapsed:.2f}s",
    )


__all__ = ["solve_ortools_md_oracle"]
