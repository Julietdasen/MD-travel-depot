"""Exact complete-action oracle for stationary residual MD snapshots.

This module is intentionally separate from the historical forced-batch oracle.
It defines the v1 action identity and the JSON-safe label/audit representation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from baselines.gurobi_md_residual_oracle import (
    ForcedAssignmentBatch,
    ResidualMDState,
    ResidualOracleResult,
    ResidualOracleStatus,
    enumerate_forced_batches,
    residual_domain,
    residual_task_map,
    solve_residual_forced_batch,
)
from simulation_environment.domain_model import ProcessTask, SchedulingDomain, TransportTask
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


SCHEMA = "exact-online-action-1.0"


@dataclass(frozen=True, slots=True)
class CompleteOnlineAction:
    assignments: tuple[tuple[int, int], ...]
    idle_robot_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        assignments = tuple(sorted((int(r), int(t)) for r, t in self.assignments))
        if not assignments:
            raise ValueError("complete action must be non-empty")
        if len({r for r, _ in assignments}) != len(assignments):
            raise ValueError("action may not assign a robot twice")
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "idle_robot_ids", tuple(sorted(int(x) for x in self.idle_robot_ids)))

    @property
    def key(self) -> tuple[tuple[int, int], ...]:
        return self.assignments


@dataclass(frozen=True, slots=True)
class ExactActionReplayAudit:
    first_event_assignments: tuple[tuple[int, int], ...]
    idle_robot_ids: tuple[int, ...]
    task_times: Mapping[str, Mapping[str, int | None]]
    latest_completion: int | None
    tail: int | None
    final_makespan: int | None


@dataclass(frozen=True, slots=True)
class ExactActionContinuation:
    status: ResidualOracleStatus
    final_makespan: float | None
    continuation_cost: float | None
    task_horizon: float | None
    return_tail: float | None
    schedule: tuple[Any, ...] = ()
    actions: tuple[Any, ...] = ()
    audit: ExactActionReplayAudit | None = None
    message: str = ""


def _check_stationary(state: ResidualMDState) -> None:
    # ResidualMDState is the frozen stationary representation.  Also accept
    # snapshot-like objects only when they explicitly expose robot activity.
    activities = getattr(state, "robot_states", None)
    if activities is not None:
        for value in activities.values() if isinstance(activities, Mapping) else activities:
            activity = value.get("activity") if isinstance(value, Mapping) else getattr(value, "activity", None)
            if activity is not None and str(activity).upper() not in {"AVAILABLE", "ROBOTACTIVITY.AVAILABLE"}:
                raise ValueError("exact online action oracle requires a stationary snapshot")


def enumerate_complete_online_actions(
    domain: SchedulingDomain, state: ResidualMDState
) -> tuple[CompleteOnlineAction, ...]:
    """Enumerate canonical, non-empty, v1 complete actions."""
    if not isinstance(state, ResidualMDState):
        raise TypeError("v1 exact actions require ResidualMDState")
    _check_stationary(state)
    robot_ids = {robot.robot_id for robot in domain.robots}
    actions = []
    for batch in enumerate_forced_batches(domain, state):
        assigned = tuple(batch.assignments)
        if any(robot_id not in robot_ids for robot_id, _ in assigned):
            continue
        actions.append(CompleteOnlineAction(assigned, tuple(sorted(robot_ids - {r for r, _ in assigned}))))
    return tuple(sorted({action.key: action for action in actions}.values(), key=lambda a: a.key))


def solve_exact_action_continuation(
    domain: SchedulingDomain,
    state: ResidualMDState,
    action: CompleteOnlineAction,
    *,
    time_limit_seconds: float = 10.0,
    threads: int = 1,
) -> ExactActionContinuation:
    """Solve one complete action and return absolute and continuation costs."""
    legal = {candidate.key: candidate for candidate in enumerate_complete_online_actions(domain, state)}
    if action.key not in legal:
        raise ValueError("action is not in the exact v1 action space")
    result: ResidualOracleResult = solve_residual_forced_batch(
        domain, state, ForcedAssignmentBatch(action.assignments),
        time_limit_seconds=time_limit_seconds, threads=threads,
        forbidden_immediate_robot_ids=action.idle_robot_ids, first_action=action.assignments,
    )
    cost = None if result.makespan is None else float(result.makespan - state.current_time)
    audit = replay_audit(domain, state, action, result) if result.status in (ResidualOracleStatus.OPTIMAL, ResidualOracleStatus.FEASIBLE) else None
    return ExactActionContinuation(
        result.status, result.makespan, cost, result.remaining_task_horizon,
        result.terminal_return_tail, result.schedule, result.actions, audit, result.message,
    )


def replay_audit(
    domain: SchedulingDomain, state: ResidualMDState, action: CompleteOnlineAction,
    result: ResidualOracleResult,
) -> ExactActionReplayAudit:
    """Replay a solver result and capture the fields used by schema validation."""
    rd = residual_domain(domain, state)
    mapping = residual_task_map(domain, state).original_to_compact
    sim = MDDiscreteSimulator(rd, exit_location=state.exit_location)
    first = tuple(sorted(action.assignments))
    if not result.actions:
        raise ValueError("optimal exact result has no replay actions")
    first_assignment_time = min(item.planned_assignment for item in result.actions)
    solver_first = tuple(sorted(
        (item.robot_id, item.task_id)
        for item in result.actions
        if item.planned_assignment == first_assignment_time
    ))
    if first_assignment_time != state.current_time or solver_first != first:
        raise ValueError(
            f"first-event replay mismatch: solver={solver_first}, requested={first}"
        )
    for robot_id, task_id in first:
        sim.assign(robot_id=robot_id, task_id=mapping[task_id])
    continuation = [
        (a.planned_assignment - state.current_time, a.robot_id, mapping[a.task_id])
        for a in result.actions
        if (a.robot_id, a.task_id) not in action.assignments
    ]
    while not sim.done and sim.time < 100000:
        for planned, robot_id, task_id in tuple(continuation):
            if planned <= sim.time:
                try:
                    sim.assign(robot_id=robot_id, task_id=task_id)
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"continuation replay rejected {(robot_id, task_id)} at {sim.time}"
                    ) from error
                continuation.remove((planned, robot_id, task_id))
        if sim.done:
            break
        sim.step()
    task_times: dict[str, dict[str, int | None]] = {}
    for original_id, compact_id in sorted(mapping.items()):
        runtime = sim.task_state(compact_id)
        task_times[str(original_id)] = {
            "start": None if runtime.started_at is None else state.current_time + runtime.started_at,
            "completion": None if runtime.completed_at is None else state.current_time + runtime.completed_at,
        }
    completions = [v["completion"] for v in task_times.values() if v["completion"] is not None]
    latest = max(completions, default=None)
    absolute_final = None if not sim.done else state.current_time + sim.time
    tail = None if latest is None or absolute_final is None else absolute_final - latest
    audit = ExactActionReplayAudit(first, action.idle_robot_ids, task_times, latest, tail, absolute_final)
    expected_latest = None if result.remaining_task_horizon is None else state.current_time + result.remaining_task_horizon
    checks = (
        ("latest completion", audit.latest_completion, expected_latest),
        ("tail", audit.tail, result.terminal_return_tail),
        ("final makespan", audit.final_makespan, result.makespan),
    )
    for name, observed, expected in checks:
        if observed is None or expected is None or abs(observed - expected) > 1e-6:
            raise ValueError(f"{name} replay mismatch: observed={observed}, expected={expected}")
    return audit


def label_to_json(label: ExactActionContinuation, action: CompleteOnlineAction, *, state_min_continuation_cost: float | None = None, pending_task_ids: tuple[int, ...] = ()) -> dict[str, Any]:
    cost = label.continuation_cost
    regret = None if cost is None or state_min_continuation_cost is None else cost - state_min_continuation_cost
    return {
        "schema": SCHEMA,
        "complete_first_action": {"assignments": [list(x) for x in action.assignments]},
        "idle_robot_ids": list(action.idle_robot_ids),
        "forbidden_immediate_assignments": [[r, t] for r in action.idle_robot_ids for t in pending_task_ids],
        "continuation_final_makespan": label.final_makespan,
        "continuation_cost": cost,
        "task_horizon": label.task_horizon,
        "return_tail": label.return_tail,
        "regret": regret,
        "tolerance_optimal": regret is not None and regret <= 1.0,
        "schedule": [asdict(x) for x in label.schedule],
        "continuation_actions": [asdict(x) for x in label.actions],
        "replay_audit": None if label.audit is None else asdict(label.audit),
    }


def validate_action_identity(action: CompleteOnlineAction, robot_ids: set[int]) -> None:
    if set(action.idle_robot_ids) != robot_ids - {r for r, _ in action.assignments}:
        raise ValueError("idle_robot_ids must be the assignment complement")
    if not action.assignments:
        raise ValueError("all-idle is outside exact v1 action space")


def load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("schema") != SCHEMA:
                raise ValueError("schema mismatch")
            rows.append(row)
    return tuple(rows)


def write_jsonl(path: str | Path, labels: list[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for label in labels:
            if label.get("schema") != SCHEMA:
                raise ValueError("schema mismatch")
            handle.write(json.dumps(label, sort_keys=True) + "\n")


__all__ = ["SCHEMA", "CompleteOnlineAction", "ExactActionContinuation", "ExactActionReplayAudit", "enumerate_complete_online_actions", "solve_exact_action_continuation", "label_to_json", "write_jsonl", "load_jsonl", "validate_action_identity"]
