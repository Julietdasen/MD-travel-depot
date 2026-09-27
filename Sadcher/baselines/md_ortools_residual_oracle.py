"""OR-Tools CP-SAT twin of :mod:`baselines.gurobi_md_residual_oracle`.

The Gurobi WLS license expired on 2026-09-17.  This module re-implements the
residual, forced-batch MILP oracle on top of Google OR-Tools' CP-SAT solver
so exact-action label generation (Ticket 48 / physics-scorer data) no longer
requires a Gurobi license.

The public API mirrors the Gurobi module 1-for-1:

* :func:`solve_residual_forced_batch` — solve one ``ForcedAssignmentBatch``
  under a stationary residual snapshot and return a :class:`ResidualOracleResult`.
* :func:`enumerate_forced_batches` — re-exported from the Gurobi module (pure
  Python, no solver involved).

Pure-Python helpers (``ResidualMDState``, ``ForcedAssignmentBatch``,
``ResidualOracleResult``, ``ResidualOracleStatus``, ``residual_domain``,
``residual_task_map``) live in :mod:`baselines.gurobi_md_residual_oracle`
because they only depend on the license-free ``build_gurobi_md_model``
formulation builder.  This module reuses them as-is so downstream callers
(``data_generation.md_residual_generation``, ``baselines.exact_online_action_oracle``,
``schedulers.joint_batch_decoder``) do not need to change how they build
inputs.

Runtime callers should route through :func:`baselines.md_oracle_dispatch.
solve_residual_forced_batch` (env var ``MRTA_MILP_SOLVER``).
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any

from ortools.sat.python import cp_model

from baselines.md_oracle_types import (
    ForcedAssignmentBatch,
    OracleAction,
    OracleScheduleEntry,
    ResidualMDState,
    ResidualOracleResult,
    ResidualOracleStatus,
    _assignment_travel,
    _build_replay_actions,
    _canonical_time,
    _latest_return_robot,
    _speed,
    _task_end,
    _transport_service,
    build_gurobi_md_model,
    enumerate_forced_batches,
    residual_domain,
    residual_task_map,
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


def solve_residual_forced_batch(
    domain: SchedulingDomain,
    state: ResidualMDState,
    batch: ForcedAssignmentBatch,
    *,
    time_limit_seconds: float = 10.0,
    threads: int = 1,
    forbidden_immediate_robot_ids: tuple[int, ...] = (),
    first_action: tuple[tuple[int, int], ...] = (),
) -> ResidualOracleResult:
    """CP-SAT version of the Gurobi residual forced-batch oracle.

    Semantics match ``baselines.gurobi_md_residual_oracle.
    solve_residual_forced_batch`` exactly:

    * every ``(robot_id, task_id)`` in ``batch`` must be selected and be that
      robot's first executed task in the residual continuation;
    * every robot in ``forbidden_immediate_robot_ids`` may not start any task
      before ``t == 1`` (or ``1 + empty_travel`` for transport tasks) — this
      encodes the "cannot dispatch on this idle tick" complement used by the
      exact-action oracle;
    * every ``(robot_id, task_id)`` in ``first_action`` pins the exact-action
      selection: the pair must be the robot's first task and the task must
      have exactly those robots as its coalition.

    Objective and horizon variables mirror the Gurobi formulation
    (``all_tasks_complete``, ``longest_exit_return``, ``latest_exit_return``)
    so ``makespan``, ``remaining_task_horizon`` and ``terminal_return_tail``
    stay comparable across the two backends.
    """

    if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive and finite")
    if not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    rd = residual_domain(domain, state)
    task_map = residual_task_map(domain, state)
    task_ids = task_map.original_to_compact
    batch = ForcedAssignmentBatch(
        tuple(
            (robot_id, task_ids.get(task_id, task_id))
            for robot_id, task_id in batch.assignments
        )
    )
    spec = build_gurobi_md_model(rd, exit_location=state.exit_location)
    started = time.perf_counter()
    try:
        return _cp_solve(
            spec,
            rd,
            task_map,
            batch,
            state,
            time_limit_seconds=time_limit_seconds,
            threads=threads,
            forbidden_immediate_robot_ids=forbidden_immediate_robot_ids,
            first_action=first_action,
            started=started,
        )
    except Exception as error:  # pragma: no cover - defensive
        return ResidualOracleResult(
            ResidualOracleStatus.ERROR,
            None,
            None,
            None,
            None,
            None,
            time.perf_counter() - started,
            message=str(error),
        )


def _cp_solve(
    spec: Any,
    rd: SchedulingDomain,
    task_map: Any,
    batch: ForcedAssignmentBatch,
    state: ResidualMDState,
    *,
    time_limit_seconds: float,
    threads: int,
    forbidden_immediate_robot_ids: tuple[int, ...],
    first_action: tuple[tuple[int, int], ...],
    started: float,
) -> ResidualOracleResult:
    model = cp_model.CpModel()
    tasks = {task.task_id: task for task in spec.domain.tasks}
    robots = {robot.robot_id: robot for robot in spec.domain.robots}
    eligible = {
        **spec.transport_eligible_robots,
        **spec.process_eligible_robots,
    }
    horizon = int(math.ceil(spec.big_m))

    # Base timing variables.
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

    # Per-task assignment / skill / duration constraints (mirrors the base
    # ORTools scheduler exactly).
    for task_id, task in tasks.items():
        if isinstance(task, TransportTask):
            model.Add(
                sum(
                    assignment[robot_id, task_id]
                    for robot_id in eligible[task_id]
                )
                == 1
            )
            duration_by_robot: list[tuple[cp_model.IntVar, int]] = []
            for robot_id in eligible[task_id]:
                service = int(_transport_service(task, robots[robot_id]))
                duration_by_robot.append((assignment[robot_id, task_id], service))
            model.Add(
                duration_var[task_id]
                == sum(a * s for a, s in duration_by_robot)
            )
        else:
            coverage = spec.process_skill_coverage[task_id]
            if not coverage:
                model.Add(
                    sum(
                        assignment[robot_id, task_id]
                        for robot_id in eligible[task_id]
                    )
                    == 1
                )
            for skill, robot_ids in coverage.items():
                model.Add(
                    sum(assignment[robot_id, task_id] for robot_id in robot_ids)
                    >= 1
                )

            unique_skill: dict[tuple[int, int], cp_model.IntVar] = {}
            for skill, robot_ids in coverage.items():
                for robot_id in robot_ids:
                    var = model.NewBoolVar(
                        f"unique_{task_id}_{skill}_{robot_id}"
                    )
                    unique_skill[skill, robot_id] = var
                    model.Add(assignment[robot_id, task_id] >= var)
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
                        model.Add(assignment[robot_id, task_id] == 0)
            model.Add(duration_var[task_id] == discrete_duration(task.duration))

    # Precedence.
    for source, target in (*spec.normal_precedence, *spec.material_precedence):
        model.Add(start[target] >= end[source])

    # Per-robot sequence + first-task tracking via a Circuit.
    return_horizon = model.NewIntVar(0, horizon, "return_horizon")
    task_horizon_var = model.NewIntVar(0, horizon, "task_horizon")
    for task_id in tasks:
        model.Add(task_horizon_var >= end[task_id])

    last_assignment: dict[tuple[int, int], cp_model.IntVar] = {}
    first_at: dict[tuple[int, int], cp_model.IntVar] = {}
    unassigned: dict[int, cp_model.IntVar] = {}

    for robot_id, robot in robots.items():
        robot_tasks = sorted(
            task_id
            for task_id, robot_ids in eligible.items()
            if robot_id in robot_ids
        )
        for task_id in robot_tasks:
            last_assignment[robot_id, task_id] = model.NewBoolVar(
                f"last_{robot_id}_{task_id}"
            )
            first_at[robot_id, task_id] = model.NewBoolVar(
                f"first_{robot_id}_{task_id}"
            )
            model.Add(
                assignment[robot_id, task_id]
                >= last_assignment[robot_id, task_id]
            )
            model.Add(
                assignment[robot_id, task_id] >= first_at[robot_id, task_id]
            )
        unassigned[robot_id] = model.NewBoolVar(f"unassigned_{robot_id}")

        model.Add(
            sum(last_assignment[robot_id, t] for t in robot_tasks)
            + unassigned[robot_id]
            == 1
        )
        assigns = [assignment[robot_id, t] for t in robot_tasks]
        if assigns:
            model.Add(sum(assigns) >= 1).OnlyEnforceIf(unassigned[robot_id].Not())
            for a in assigns:
                model.Add(a == 0).OnlyEnforceIf(unassigned[robot_id])

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

        for task_id in robot_tasks:
            initial_travel = _assignment_travel(
                robot.location, tasks[task_id], _speed(robot)
            )
            model.Add(start[task_id] >= initial_travel).OnlyEnforceIf(
                assignment[robot_id, task_id]
            )

        node_ids = {i + 1: t for i, t in enumerate(robot_tasks)}
        arcs: list[tuple[int, int, cp_model.IntVar]] = []

        for i, task_id in node_ids.items():
            skip_lit = model.NewBoolVar(f"skip_{robot_id}_{task_id}")
            model.Add(skip_lit == 1 - assignment[robot_id, task_id])
            arcs.append((i, i, skip_lit))

        for i, task_id in node_ids.items():
            arcs.append((0, i, first_at[robot_id, task_id]))
        for i, task_id in node_ids.items():
            arcs.append((i, 0, last_assignment[robot_id, task_id]))

        depot_self = model.NewBoolVar(f"depot_self_{robot_id}")
        arcs.append((0, 0, depot_self))
        model.Add(depot_self == unassigned[robot_id])
        model.Add(
            sum(first_at[robot_id, t] for t in robot_tasks)
            == 1 - unassigned[robot_id]
        )

        for i, left in node_ids.items():
            for j, right in node_ids.items():
                if i == j:
                    continue
                arc = model.NewBoolVar(f"arc_{robot_id}_{left}_{right}")
                arcs.append((i, j, arc))
                model.Add(assignment[robot_id, left] >= arc)
                model.Add(assignment[robot_id, right] >= arc)
                travel = _assignment_travel(
                    _task_end(tasks[left]), tasks[right], _speed(robot)
                )
                model.Add(
                    start[right] >= end[left] + travel
                ).OnlyEnforceIf(arc)

        model.AddCircuit(arcs)

    # ------------------------------------------------------------------
    # Residual-specific constraints: forced batch, forbidden idle, exact
    # first-action pinning.
    # ------------------------------------------------------------------
    for robot_id, task_id in batch.assignments:
        if (robot_id, task_id) not in assignment:
            raise ValueError(
                f"forced assignment ({robot_id}, {task_id}) is illegal"
            )
        model.Add(assignment[robot_id, task_id] == 1)
        # Force this task to be that robot's first executed task.  The Circuit
        # constraints already enforce completion-based sequencing for
        # subsequent tasks on the same robot.
        model.Add(first_at[robot_id, task_id] == 1)
        task = tasks[task_id]
        robot = robots[robot_id]
        earliest = _assignment_travel(robot.location, task, _speed(robot))
        model.Add(start[task_id] >= earliest)

    for robot_id in forbidden_immediate_robot_ids:
        if robot_id not in robots:
            continue
        robot_tasks = [
            task_id
            for (candidate_robot_id, task_id) in assignment
            if candidate_robot_id == robot_id
        ]
        for task_id in robot_tasks:
            task = tasks[task_id]
            robot = robots[robot_id]
            speed = _speed(robot)
            travel = (
                _assignment_travel(robot.location, task, speed)
                if isinstance(task, TransportTask)
                else 0
            )
            delay = 1 + travel
            model.Add(start[task_id] >= delay).OnlyEnforceIf(
                first_at[robot_id, task_id]
            )

    selected_by_task: dict[int, set[int]] = {}
    for robot_id, task_id in first_action:
        compact_task_id = task_map.original_to_compact.get(task_id, task_id)
        selected_by_task.setdefault(compact_task_id, set()).add(robot_id)
        if (robot_id, compact_task_id) not in assignment:
            raise ValueError(
                f"exact first-action pair ({robot_id}, {compact_task_id})"
                " is illegal"
            )
        task = tasks[compact_task_id]
        robot = robots[robot_id]
        model.Add(first_at[robot_id, compact_task_id] == 1)
        initial = (
            _assignment_travel(robot.location, task, _speed(robot))
            if isinstance(task, TransportTask)
            else 0
        )
        model.Add(start[compact_task_id] == initial)
    for compact_task_id, selected_robot_ids in selected_by_task.items():
        for (other_robot_id, other_task_id), variable in assignment.items():
            if other_task_id == compact_task_id and other_robot_id not in selected_robot_ids:
                model.Add(variable == 0)

    # Objective mirrors the Gurobi formulation: minimise weighted makespan +
    # coalition size so ties break towards inclusion-minimal coalitions.
    makespan = model.NewIntVar(0, 2 * horizon, "latest_exit_return")
    model.Add(makespan == task_horizon_var + return_horizon)
    assignment_count = sum(assignment.values())
    weight = len(assignment) + 1
    model.Minimize(weight * makespan + assignment_count)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(threads)
    solver.parameters.log_search_progress = False

    status_code = solver.Solve(model)
    elapsed = time.perf_counter() - started

    if status_code == cp_model.INFEASIBLE:
        return ResidualOracleResult(
            ResidualOracleStatus.INFEASIBLE,
            None,
            None,
            None,
            None,
            None,
            elapsed,
            message="CP-SAT proved the residual model infeasible",
        )
    if status_code in (cp_model.UNKNOWN, cp_model.MODEL_INVALID):
        status = (
            ResidualOracleStatus.ERROR
            if status_code == cp_model.MODEL_INVALID
            else ResidualOracleStatus.TIMEOUT
        )
        return ResidualOracleResult(
            status,
            None,
            None,
            None,
            None,
            None,
            elapsed,
            message=f"CP-SAT returned status {solver.StatusName(status_code)}",
        )

    # FEASIBLE / OPTIMAL: extract schedule and canonical times.
    compact_entries = []
    for (robot_id, task_id), var in assignment.items():
        if solver.Value(var) == 1:
            compact_entries.append(
                OracleScheduleEntry(
                    robot_id=robot_id,
                    task_id=task_id,
                    start=_canonical_time(float(solver.Value(start[task_id]))),
                    completion=_canonical_time(float(solver.Value(end[task_id]))),
                )
            )
    compact_schedule = tuple(
        sorted(compact_entries, key=lambda x: (x.start, x.task_id, x.robot_id))
    )
    task_horizon_value = float(solver.Value(task_horizon_var))
    tail_value = float(solver.Value(return_horizon))
    makespan_value = state.current_time + float(solver.Value(makespan))
    schedule = tuple(
        replace(
            entry,
            task_id=task_map.compact_to_original[entry.task_id],
            start=state.current_time + entry.start,
            completion=state.current_time + entry.completion,
        )
        for entry in compact_schedule
    )
    last_robot = _latest_return_robot(rd, compact_schedule, state.exit_location)
    actions = tuple(
        replace(
            action,
            task_id=task_map.compact_to_original[action.task_id],
            planned_assignment=state.current_time + action.planned_assignment,
            planned_start=state.current_time + action.planned_start,
        )
        for action in _build_replay_actions(spec, compact_schedule)
    )

    best_bound = solver.BestObjectiveBound()
    obj_value = solver.ObjectiveValue()
    if obj_value > 0:
        gap = max(0.0, (obj_value - best_bound) / max(1e-9, abs(obj_value)))
    else:
        gap = 0.0

    status = (
        ResidualOracleStatus.OPTIMAL
        if status_code == cp_model.OPTIMAL
        else ResidualOracleStatus.FEASIBLE
    )
    return ResidualOracleResult(
        status,
        makespan_value,
        task_horizon_value,
        tail_value,
        last_robot,
        gap,
        elapsed,
        schedule,
        actions,
        f"CP-SAT returned {solver.StatusName(status_code)} in {elapsed:.2f}s",
    )


__all__ = [
    "ForcedAssignmentBatch",
    "ResidualMDState",
    "ResidualOracleResult",
    "ResidualOracleStatus",
    "enumerate_forced_batches",
    "residual_domain",
    "residual_task_map",
    "solve_residual_forced_batch",
]
