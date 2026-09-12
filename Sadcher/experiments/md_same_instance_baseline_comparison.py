"""Independent MILP and Greedy comparison on the Ticket 46 instances."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from baselines.gurobi_md_oracle import (
    GurobiOracleStatus,
    replay_oracle_actions,
    solve_gurobi_md_oracle,
)
from experiments.md_c0_end_to_end_diagnostic_pilot import (
    FROZEN_INSTANCE_SEEDS,
    MAX_ROLLOUT_STEPS,
    _run_fixed_baseline,
    load_frozen_diagnostic_records,
)
from experiments.protocol import FailureReason
from schedulers.md_greedy_baselines import (
    MDTransportGreedy,
    TransportGreedyStrategy,
)
from schedulers.process_greedy_md import ProcessGreedyMDAdapter
from simulation_environment.domain_model import (
    ProcessRobot,
    ProcessTask,
    TransportRobot,
    TransportTask,
)
from simulation_environment.transport_timing import travel_duration
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "reports" / "md_c0_end_to_end_diagnostic_pilot_2026-09-01"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "reports" / "md_same_instance_baseline_comparison_2026-09-02"
)
EXIT_LOCATION = (0.0, 0.0)
MILP_TIME_LIMIT_SECONDS = 60.0
MILP_THREADS = 1
MODEL_SEEDS = (3101, 3102, 3103)

GREEDY_STRATEGIES = {
    "greedy_distance": TransportGreedyStrategy.DISTANCE,
    "greedy_eta": TransportGreedyStrategy.ETA,
    "greedy_unlock": TransportGreedyStrategy.UNLOCK,
    "material_solo_greedy": TransportGreedyStrategy.MATERIAL_SOLO,
}
GREEDY_METHODS = tuple(GREEDY_STRATEGIES) + ("process_md_greedy",)
EXISTING_FIXED_METHODS = ("physics_only", "eta_unlock_heuristic", "masked_greedy")
ALL_FIXED_METHODS = (*GREEDY_METHODS, *EXISTING_FIXED_METHODS)


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": statistics.median(values) if values else None,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _experiment_payload(result) -> dict[str, object]:
    return json.loads(result.to_json())


def _row_from_result(
    result,
    *,
    instance,
    method: str,
    decision_records: Sequence[Mapping[str, object]],
    solver_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = _experiment_payload(result)
    termination = payload["termination"]
    metrics = payload["metrics"]
    execution_records = payload["execution_records"]
    assert isinstance(termination, Mapping)
    assert isinstance(metrics, Mapping)
    assert isinstance(execution_records, Mapping)
    solver = {} if solver_fields is None else dict(solver_fields)
    inference_seconds = float(metrics["inference_time_seconds"])
    return {
        "schema_version": "1.0.0",
        "ticket": 47,
        "source_ticket": 46,
        "method": method,
        "model_seed": None,
        "instance_id": instance.instance_id,
        "instance_seed": instance.seed,
        "execution_status": "completed",
        "success": bool(termination["success"]),
        "failure_reason": termination["failure_reason"],
        "makespan": metrics["makespan"],
        "material_starvation": metrics["material_starvation"],
        "robot_utilization": metrics["robot_utilization"],
        "illegal_assignment_count": termination["illegal_assignment_count"],
        "repair_count": 0,
        "repair_rate": 0.0,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "fallback_reasons": [],
        "solver_calls": int(solver.get("solver_calls", 0)),
        "solver_time_seconds": float(solver.get("solver_time_seconds", 0.0)),
        "decision_count": len(decision_records),
        "assignment_count": len(execution_records.get("process", ())) + len(
            execution_records.get("transport", ())
        ),
        "latency_totals": {
            "encoder_seconds": 0.0,
            "scoring_seconds": 0.0,
            "decoder_seconds": 0.0,
            "repair_seconds": 0.0,
            "fallback_seconds": 0.0,
            "total_seconds": inference_seconds,
        },
        "wall_runtime_seconds": float(metrics["wall_time_seconds"]),
        "terminal_state": {
            "all_real_tasks_completed": termination["all_real_tasks_completed"],
            "all_robots_at_exit": termination["all_robots_at_exit"],
        },
        "checkpoint_identity": None,
        "decision_records": list(decision_records),
        "execution_records": execution_records,
        "experiment": payload,
        **solver,
    }


def _run_greedy(instance, method: str) -> dict[str, object]:
    if method not in GREEDY_STRATEGIES:
        raise ValueError(f"unknown Greedy method: {method}")
    simulator = MDDiscreteSimulator(instance.generated.domain, exit_location=EXIT_LOCATION)
    transport = MDTransportGreedy(
        GREEDY_STRATEGIES[method]
    )
    process = ProcessGreedyMDAdapter()
    decision_records: list[dict[str, object]] = []
    started = time.perf_counter()
    inference_seconds = 0.0
    steps = 0
    failure_reason: FailureReason | None = None

    while not simulator.done and steps < MAX_ROLLOUT_STEPS:
        decision_started = time.perf_counter()
        transport_assignments = transport.assign(simulator)
        process_assignments = process.assign(simulator)
        elapsed = time.perf_counter() - decision_started
        inference_seconds += elapsed
        if transport_assignments or process_assignments:
            decision_records.append(
                {
                    "decision_time": simulator.time,
                    "transport_assignments": [
                        asdict(item) for item in transport_assignments
                    ],
                    "process_assignments": [asdict(item) for item in process_assignments],
                    "total_time_seconds": elapsed,
                }
            )
        if not simulator.has_advancing_work:
            failure_reason = FailureReason.DEADLOCK
            break
        simulator.step()
        steps += 1

    if not simulator.done and failure_reason is None:
        failure_reason = FailureReason.TIMEOUT
    result = simulator.build_experiment_result(
        run_id=f"ticket47-{method}-{instance.instance_id}",
        method=method,
        instance_id=instance.instance_id,
        seed=instance.seed,
        split=instance.split,
        failure_reason=failure_reason,
        inference_time_seconds=inference_seconds,
        wall_time_seconds=time.perf_counter() - started,
        metadata={
            "strategy": method,
            "normal_path_uses_mip": False,
            "decision_records": decision_records,
        },
    )
    return _row_from_result(
        result,
        instance=instance,
        method=method,
        decision_records=decision_records,
    )


def _normalize_ticket46_row(
    row: Mapping[str, object], *, method: str
) -> dict[str, object]:
    """Carry a Ticket 46 rollout into the Ticket 47 comparison schema."""

    normalized = dict(row)
    normalized["source_ticket"] = normalized.get("ticket", 46)
    normalized["ticket"] = 47
    normalized["source_method"] = normalized.get("method")
    normalized["method"] = method
    normalized["model_seed"] = None
    normalized.setdefault("execution_status", "completed")
    if "execution_records" not in normalized:
        experiment = normalized.get("experiment")
        if isinstance(experiment, Mapping):
            execution_records = experiment.get("execution_records")
            if isinstance(execution_records, Mapping):
                normalized["execution_records"] = execution_records
    return normalized


def _run_ticket46_fixed_baseline(instance, method: str) -> dict[str, object]:
    if method != "process_md_greedy" and method not in EXISTING_FIXED_METHODS:
        raise ValueError(f"unknown Ticket 46 fixed baseline: {method}")
    return _normalize_ticket46_row(
        _run_fixed_baseline(method, instance),
        method=method,
    )


def _solver_not_replayed_row(
    instance,
    *,
    status: GurobiOracleStatus,
    solver_time_seconds: float,
    message: str,
    objective: float | None,
    optimality_gap: float | None,
    timeout: bool,
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "ticket": 47,
        "source_ticket": 46,
        "method": "milp_oracle",
        "model_seed": None,
        "instance_id": instance.instance_id,
        "instance_seed": instance.seed,
        "execution_status": f"solver_{status.value}",
        "success": None,
        "failure_reason": None,
        "makespan": None,
        "material_starvation": {},
        "robot_utilization": {},
        "illegal_assignment_count": 0,
        "repair_count": 0,
        "repair_rate": 0.0,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "fallback_reasons": [],
        "solver_calls": 1,
        "solver_time_seconds": solver_time_seconds,
        "decision_count": 0,
        "assignment_count": 0,
        "latency_totals": {
            "encoder_seconds": 0.0,
            "scoring_seconds": 0.0,
            "decoder_seconds": 0.0,
            "repair_seconds": 0.0,
            "fallback_seconds": 0.0,
            "total_seconds": solver_time_seconds,
        },
        "wall_runtime_seconds": solver_time_seconds,
        "terminal_state": {},
        "checkpoint_identity": None,
        "decision_records": [],
        "execution_records": {"process": [], "transport": []},
        "experiment": None,
        "solver_status": status.value,
        "solver_feasible": None,
        "solver_timeout": timeout,
        "solver_optimality_gap": optimality_gap,
        "solver_objective": objective,
        "solver_message": message,
    }


def _run_milp(
    instance,
    *,
    time_limit_seconds: float = MILP_TIME_LIMIT_SECONDS,
    threads: int = MILP_THREADS,
) -> dict[str, object]:
    started = time.perf_counter()
    oracle = solve_gurobi_md_oracle(
        instance.generated.domain,
        exit_location=EXIT_LOCATION,
        time_limit_seconds=time_limit_seconds,
        threads=threads,
    )
    common_solver = {
        "solver_status": oracle.status.value,
        "solver_feasible": oracle.feasible,
        "solver_timeout": oracle.timeout,
        "solver_optimality_gap": oracle.optimality_gap,
        "solver_objective": oracle.objective,
        "solver_message": oracle.message,
        "solver_time_seconds": oracle.solve_time_seconds,
        "solver_calls": 1,
    }
    if oracle.feasible is not True:
        row = _solver_not_replayed_row(
            instance,
            status=oracle.status,
            solver_time_seconds=oracle.solve_time_seconds,
            message=oracle.message,
            objective=oracle.objective,
            optimality_gap=oracle.optimality_gap,
            timeout=oracle.timeout,
        )
        row.update(common_solver)
        row["wall_runtime_seconds"] = time.perf_counter() - started
        return row

    replay = replay_oracle_actions(
        instance.generated.domain,
        oracle.action_order,
        exit_location=EXIT_LOCATION,
        run_id=f"ticket47-milp-{instance.instance_id}",
        instance_id=instance.instance_id,
        seed=instance.seed,
        split=instance.split,
        max_steps=MAX_ROLLOUT_STEPS,
    )
    decisions = [asdict(action) for action in oracle.action_order]
    row = _row_from_result(
        replay,
        instance=instance,
        method="milp_oracle",
        decision_records=decisions,
        solver_fields=common_solver,
    )
    row["wall_runtime_seconds"] = time.perf_counter() - started
    row["latency_totals"]["total_seconds"] = row["wall_runtime_seconds"]
    row["oracle_schedule"] = [asdict(entry) for entry in oracle.schedule]
    return row


def _success_value(row: Mapping[str, object]) -> bool | None:
    value = row.get("success")
    return value if isinstance(value, bool) else None


def _row_starvation(row: Mapping[str, object]) -> float | None:
    values = row.get("material_starvation")
    if not isinstance(values, Mapping) or not values:
        return None
    return statistics.fmean(float(value) for value in values.values())


def _row_utilization(row: Mapping[str, object]) -> float | None:
    values = row.get("robot_utilization")
    if not isinstance(values, Mapping) or not values:
        return None
    return statistics.fmean(float(value) for value in values.values())


def aggregate_baseline_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize rows while excluding solver-unavailable runs from success rate."""

    evaluated = [row for row in rows if _success_value(row) is not None]
    successes = [row for row in evaluated if _success_value(row) is True]
    makespans = [
        float(row["makespan"])
        for row in successes
        if row.get("makespan") is not None
    ]
    failure_counts = Counter(
        str(row["failure_reason"])
        for row in evaluated
        if _success_value(row) is False and row.get("failure_reason") is not None
    )
    status_counts = Counter(
        str(row["solver_status"])
        for row in rows
        if row.get("solver_status") is not None
    )
    return {
        "run_count": len(rows),
        "evaluated_count": len(evaluated),
        "unavailable_count": len(rows) - len(evaluated),
        "success_count": len(successes),
        "success_rate": len(successes) / len(evaluated) if evaluated else None,
        "failure_counts": dict(sorted(failure_counts.items())),
        "solver_status_counts": dict(sorted(status_counts.items())),
        "makespan_on_successes": _summary(makespans),
        "mean_material_starvation": _mean(
            [value for row in evaluated if (value := _row_starvation(row)) is not None]
        ),
        "mean_robot_utilization": _mean(
            [value for row in evaluated if (value := _row_utilization(row)) is not None]
        ),
        "mean_wall_runtime_seconds": _mean(
            [float(row.get("wall_runtime_seconds", 0.0)) for row in rows]
        ),
        "mean_solver_time_seconds": _mean(
            [float(row.get("solver_time_seconds", 0.0)) for row in rows]
        ),
        "solver_calls": sum(int(row.get("solver_calls", 0)) for row in rows),
        "illegal_assignment_count": sum(
            int(row.get("illegal_assignment_count", 0)) for row in evaluated
        ),
        "repair_count": sum(int(row.get("repair_count", 0)) for row in rows),
        "fallback_count": sum(int(row.get("fallback_count", 0)) for row in rows),
        "mean_decision_count": _mean(
            [float(row.get("decision_count", 0)) for row in evaluated]
        ),
    }


def _index_rows(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, int], Mapping[str, object]]:
    indexed = {
        (str(row["instance_id"]), int(row.get("instance_seed", 0))): row for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("paired rows must have unique instance_id/seed keys")
    return indexed


def paired_baseline_rows(
    reference: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare paired rows and expose unavailable and survivor sets explicitly."""

    reference_by_key = _index_rows(reference)
    candidate_by_key = _index_rows(candidate)
    if reference_by_key.keys() != candidate_by_key.keys():
        raise ValueError("paired rows must contain the same instance keys")
    common_success_deltas: list[float] = []
    starvation_deltas: list[float] = []
    reference_successes = 0
    candidate_successes = 0
    reference_only = 0
    candidate_only = 0
    unavailable_pairs = 0
    for key in sorted(reference_by_key):
        base = reference_by_key[key]
        contender = candidate_by_key[key]
        base_success = _success_value(base)
        candidate_success = _success_value(contender)
        reference_successes += base_success is True
        candidate_successes += candidate_success is True
        if base_success is True and candidate_success is True:
            base_makespan = base.get("makespan")
            candidate_makespan = contender.get("makespan")
            if base_makespan is not None and candidate_makespan is not None:
                common_success_deltas.append(
                    float(candidate_makespan) - float(base_makespan)
                )
        elif base_success is True:
            reference_only += 1
        elif candidate_success is True:
            candidate_only += 1
        if base_success is None or candidate_success is None:
            unavailable_pairs += 1
        base_starvation = _row_starvation(base)
        candidate_starvation = _row_starvation(contender)
        if base_starvation is not None and candidate_starvation is not None:
            starvation_deltas.append(candidate_starvation - base_starvation)

    count = len(reference)
    reference_evaluated = sum(
        _success_value(row) is not None for row in reference
    )
    candidate_evaluated = sum(
        _success_value(row) is not None for row in candidate
    )
    reference_rate = (
        reference_successes / reference_evaluated
        if reference_evaluated
        else None
    )
    candidate_rate = (
        candidate_successes / candidate_evaluated
        if candidate_evaluated
        else None
    )
    mean_delta = _mean(common_success_deltas)
    return {
        "reference_method": reference[0].get("method") if reference else None,
        "candidate_method": candidate[0].get("method") if candidate else None,
        "matched_pairs": count,
        "reference_evaluated_count": reference_evaluated,
        "candidate_evaluated_count": candidate_evaluated,
        "common_successful_pairs": len(common_success_deltas),
        "reference_success_rate": reference_rate,
        "candidate_success_rate": candidate_rate,
        "success_rate_difference": (
            None
            if reference_rate is None or candidate_rate is None
            else candidate_rate - reference_rate
        ),
        "candidate_minus_reference": {
            "count": len(common_success_deltas),
            "mean": mean_delta,
            "median": statistics.median(common_success_deltas)
            if common_success_deltas
            else None,
            "improvement_candidate_minus_reference": (
                None if mean_delta is None else -mean_delta
            ),
        },
        "starvation_candidate_minus_reference": _summary(starvation_deltas),
        "survivor_bias": {
            "reference_only_successes": reference_only,
            "candidate_only_successes": candidate_only,
            "both_failed_or_unavailable": sum(
                _success_value(reference_by_key[key]) is not True
                and _success_value(candidate_by_key[key]) is not True
                for key in reference_by_key
            ),
        },
        "unavailable_pairs": unavailable_pairs,
    }


def _execution_records(row: Mapping[str, object]) -> Mapping[str, object]:
    records = row.get("execution_records")
    if isinstance(records, Mapping):
        return records
    experiment = row.get("experiment")
    if isinstance(experiment, Mapping):
        nested = experiment.get("execution_records")
        if isinstance(nested, Mapping):
            return nested
    return {"process": [], "transport": []}


def _task_order(row: Mapping[str, object], kind: str, time_key: str) -> list[int]:
    records = _execution_records(row).get(kind, ())
    if not isinstance(records, Sequence):
        return []
    ordered = sorted(
        (record for record in records if isinstance(record, Mapping)),
        key=lambda record: (
            float(record.get(time_key) or 0),
            int(record.get("task_id", 0)),
        ),
    )
    return [int(record["task_id"]) for record in ordered]


def _latest_task_completion(row: Mapping[str, object]) -> float | None:
    values: list[float] = []
    records = _execution_records(row)
    for kind in ("process", "transport"):
        entries = records.get(kind, ())
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("completed_at") is not None:
                values.append(float(entry["completed_at"]))
    return max(values) if values else None


def _event_entries(
    row: Mapping[str, object], kind: str
) -> tuple[Mapping[str, object], ...]:
    entries = _execution_records(row).get(kind, ())
    if not isinstance(entries, Sequence):
        return ()
    return tuple(entry for entry in entries if isinstance(entry, Mapping))


def _task_completion_map(row: Mapping[str, object]) -> dict[int, float]:
    return {
        int(entry["task_id"]): float(entry["completed_at"])
        for kind in ("process", "transport")
        for entry in _event_entries(row, kind)
        if entry.get("completed_at") is not None
    }


def _terminal_robot_details(instance, row: Mapping[str, object]) -> list[dict[str, object]]:
    tasks = {task.task_id: task for task in instance.generated.domain.tasks}
    last_by_robot: dict[int, tuple[float, int, tuple[float, float]]] = {}
    for kind in ("process", "transport"):
        for entry in _event_entries(row, kind):
            if entry.get("completed_at") is None:
                continue
            task_id = int(entry["task_id"])
            task = tasks[task_id]
            robot_ids = (
                (int(entry["robot_id"]),)
                if kind == "transport"
                else tuple(int(robot_id) for robot_id in entry["robot_ids"])
            )
            if isinstance(task, TransportTask):
                location = task.delivery_location
            else:
                assert isinstance(task, ProcessTask)
                location = task.location
            completed = float(entry["completed_at"])
            for robot_id in robot_ids:
                previous = last_by_robot.get(robot_id)
                if previous is None or (completed, task_id) >= (
                    previous[0],
                    previous[1],
                ):
                    last_by_robot[robot_id] = (completed, task_id, location)

    details: list[dict[str, object]] = []
    for robot in sorted(instance.generated.domain.robots, key=lambda item: item.robot_id):
        previous = last_by_robot.get(robot.robot_id)
        if previous is None:
            location = robot.location
            last_task_id = None
            last_task_completed = None
        else:
            last_task_completed, last_task_id, location = previous
        if isinstance(robot, ProcessRobot):
            speed = robot.speed
        else:
            assert isinstance(robot, TransportRobot)
            speed = robot.unloaded_speed
        details.append(
            {
                "robot_id": robot.robot_id,
                "last_task_id": last_task_id,
                "last_task_completed": last_task_completed,
                "location": location,
                "return_duration": travel_duration(location, EXIT_LOCATION, speed),
            }
        )
    return details


def _format_order(order: Sequence[int]) -> str:
    return " -> ".join(str(task_id) for task_id in order) if order else "none"


def _format_event_trace(row: Mapping[str, object], kind: str) -> str:
    time_key = "assigned_at" if kind == "transport" else "started_at"
    parts: list[str] = []
    for entry in sorted(
        _event_entries(row, kind),
        key=lambda item: (
            float(item.get(time_key) or 0),
            int(item.get("task_id", 0)),
        ),
    ):
        task_id = int(entry["task_id"])
        started = _format(entry[time_key])
        completed = _format(entry["completed_at"])
        if kind == "transport":
            parts.append(f"{task_id}@{started}->{completed}")
        else:
            robots = ",".join(str(robot_id) for robot_id in entry["robot_ids"])
            parts.append(f"{task_id}[{robots}]@{started}->{completed}")
    return ", ".join(parts) if parts else "none"


def _format_location(location: Sequence[float]) -> str:
    return "(" + ",".join(f"{float(value):g}" for value in location) + ")"


def _representative_trace_lines(
    groups: Mapping[str, Sequence[Mapping[str, object]]],
    records: Sequence[Any],
    analysis_map: Mapping[str, object],
) -> list[str]:
    if not groups or not records:
        return []
    instances = {instance.instance_id: instance for instance in records}
    cases = (
        ("C0_seed3101", "md-c0-diagnostic-46002"),
        ("C0_seed3102", "md-c0-diagnostic-46027"),
    )
    lines = [
        "",
        "### Representative Event Traces",
        "",
        "Times are start/complete; return tail is the time from the latest real-task completion to the terminal exit.",
    ]
    for candidate_name, instance_id in cases:
        instance = instances.get(instance_id)
        reference_row = next(
            (
                row
                for row in groups.get("milp_oracle", ())
                if row.get("instance_id") == instance_id
            ),
            None,
        )
        candidate_row = next(
            (
                row
                for row in groups.get(candidate_name, ())
                if row.get("instance_id") == instance_id
            ),
            None,
        )
        analysis = analysis_map.get(f"{candidate_name}_vs_milp_oracle")
        if (
            instance is None
            or reference_row is None
            or candidate_row is None
            or not isinstance(analysis, Mapping)
        ):
            continue
        instance_analysis = next(
            (
                row
                for row in analysis.get("instances", ())
                if isinstance(row, Mapping)
                and row.get("instance_id") == instance_id
            ),
            None,
        )
        if not isinstance(instance_analysis, Mapping):
            continue
        reference_details = _terminal_robot_details(instance, reference_row)
        candidate_details = _terminal_robot_details(instance, candidate_row)
        reference_tail_owner = max(
            reference_details, key=lambda row: int(row["return_duration"])
        )
        candidate_tail_owner = max(
            candidate_details, key=lambda row: int(row["return_duration"])
        )
        reference_completion = _task_completion_map(reference_row)
        candidate_completion = _task_completion_map(candidate_row)
        delayed = sorted(
            (
                (
                    task_id,
                    reference_completion[task_id],
                    candidate_completion[task_id],
                    candidate_completion[task_id] - reference_completion[task_id],
                )
                for task_id in reference_completion.keys() & candidate_completion.keys()
                if candidate_completion[task_id] - reference_completion[task_id] > 0.5
            ),
            key=lambda item: item[3],
            reverse=True,
        )[:3]
        material_delays = []
        for source, target in instance.generated.domain.material_edges:
            if (
                source in reference_completion
                and source in candidate_completion
                and target in reference_completion
                and target in candidate_completion
                and candidate_completion[source] > reference_completion[source]
                and candidate_completion[target] > reference_completion[target]
            ):
                material_delays.append(
                    f"{source}->{target}: source "
                    f"{_format(reference_completion[source])}->{_format(candidate_completion[source])}, "
                    f"task {target} "
                    f"{_format(reference_completion[target])}->{_format(candidate_completion[target])}"
                )
        lines.append(
            f"- {instance_id}: MILP makespan "
            f"{_format(instance_analysis['reference_makespan'])} "
            f"(latest task {_format(instance_analysis['reference_latest_task_completion'])}, "
            f"tail {_format(instance_analysis['reference_return_tail'])}) versus "
            f"{candidate_name} "
            f"{_format(instance_analysis['candidate_makespan'])} "
            f"(latest task {_format(instance_analysis['candidate_latest_task_completion'])}, "
            f"tail {_format(instance_analysis['candidate_return_tail'])}); "
            f"delta {_format(instance_analysis['makespan_delta_candidate_minus_reference'])}."
        )
        lines.append(
            f"  - Transport order: "
            f"{_format_order(instance_analysis['reference_transport_order'])} versus "
            f"{_format_order(instance_analysis['candidate_transport_order'])}; "
            f"process order: "
            f"{_format_order(instance_analysis['reference_process_order'])} versus "
            f"{_format_order(instance_analysis['candidate_process_order'])}."
        )
        lines.append(
            f"  - Transport events (task@assigned->complete): MILP "
            f"{_format_event_trace(reference_row, 'transport')}; "
            f"{candidate_name} "
            f"{_format_event_trace(candidate_row, 'transport')}."
        )
        if delayed:
            delayed_text = "; ".join(
                f"task {task_id} {_format(reference_time)}->{_format(candidate_time)} "
                f"(+{_format(delta)})"
                for task_id, reference_time, candidate_time, delta in delayed
            )
            lines.append(f"  - Latest completion changes: {delayed_text}.")
        if material_delays:
            lines.append(
                "  - Material-linked delay: " + "; ".join(material_delays) + "."
            )
        lines.append(
            f"  - Longest terminal return: MILP robot "
            f"{reference_tail_owner['robot_id']} from task "
            f"{reference_tail_owner['last_task_id']} at "
            f"{_format_location(reference_tail_owner['location'])} "
            f"takes {reference_tail_owner['return_duration']}; "
            f"{candidate_name} robot {candidate_tail_owner['robot_id']} from task "
            f"{candidate_tail_owner['last_task_id']} at "
            f"{_format_location(candidate_tail_owner['location'])} "
            f"takes {candidate_tail_owner['return_duration']}."
        )
    return lines


def _first_difference(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_task, right_task) in enumerate(zip(left, right)):
        if left_task != right_task:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _diagnostic_signal(
    makespan_delta: float | None,
    starvation_delta: float | None,
    return_tail_delta: float | None,
    transport_order_same: bool,
) -> str:
    if makespan_delta is None:
        return "not_comparable"
    if abs(makespan_delta) < 1e-9:
        return "same_makespan"
    if starvation_delta is not None and starvation_delta > 0.5:
        return "more_material_starvation"
    if return_tail_delta is not None and return_tail_delta > 0.5:
        return "longer_terminal_return_tail"
    if not transport_order_same:
        return "different_transport_order"
    return "process_order_or_coalition_timing"


def analyze_instance_differences(
    reference: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return per-instance signals explaining a paired makespan difference."""

    reference_by_key = _index_rows(reference)
    candidate_by_key = _index_rows(candidate)
    if reference_by_key.keys() != candidate_by_key.keys():
        raise ValueError("difference analysis requires identical paired keys")
    rows: list[dict[str, object]] = []
    for key in sorted(reference_by_key):
        base = reference_by_key[key]
        contender = candidate_by_key[key]
        base_success = _success_value(base)
        candidate_success = _success_value(contender)
        base_makespan = base.get("makespan")
        candidate_makespan = contender.get("makespan")
        makespan_delta = (
            None
            if base_makespan is None or candidate_makespan is None
            else float(candidate_makespan) - float(base_makespan)
        )
        base_starvation = _row_starvation(base)
        candidate_starvation = _row_starvation(contender)
        starvation_delta = (
            None
            if base_starvation is None or candidate_starvation is None
            else candidate_starvation - base_starvation
        )
        base_completion = _latest_task_completion(base)
        candidate_completion = _latest_task_completion(contender)
        base_tail = (
            None
            if base_makespan is None or base_completion is None
            else float(base_makespan) - base_completion
        )
        candidate_tail = (
            None
            if candidate_makespan is None or candidate_completion is None
            else float(candidate_makespan) - candidate_completion
        )
        tail_delta = (
            None
            if base_tail is None or candidate_tail is None
            else candidate_tail - base_tail
        )
        base_transport = _task_order(base, "transport", "assigned_at")
        candidate_transport = _task_order(contender, "transport", "assigned_at")
        base_process = _task_order(base, "process", "started_at")
        candidate_process = _task_order(contender, "process", "started_at")
        rows.append(
            {
                "instance_id": key[0],
                "instance_seed": key[1],
                "reference_success": base_success,
                "candidate_success": candidate_success,
                "reference_makespan": base_makespan,
                "candidate_makespan": candidate_makespan,
                "makespan_delta_candidate_minus_reference": makespan_delta,
                "makespan_improvement_candidate_vs_reference": (
                    None if makespan_delta is None else -makespan_delta
                ),
                "reference_material_starvation_mean": base_starvation,
                "candidate_material_starvation_mean": candidate_starvation,
                "material_starvation_delta_candidate_minus_reference": starvation_delta,
                "reference_latest_task_completion": base_completion,
                "candidate_latest_task_completion": candidate_completion,
                "reference_return_tail": base_tail,
                "candidate_return_tail": candidate_tail,
                "return_tail_delta_candidate_minus_reference": tail_delta,
                "reference_transport_order": base_transport,
                "candidate_transport_order": candidate_transport,
                "transport_order_same": base_transport == candidate_transport,
                "first_transport_order_difference": _first_difference(
                    base_transport, candidate_transport
                ),
                "reference_process_order": base_process,
                "candidate_process_order": candidate_process,
                "process_order_same": base_process == candidate_process,
                "diagnostic_signal": _diagnostic_signal(
                    makespan_delta,
                    starvation_delta,
                    tail_delta,
                    base_transport == candidate_transport,
                ),
            }
        )
    return rows


def _analysis_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    signals = Counter(str(row["diagnostic_signal"]) for row in rows)
    comparable = [
        row
        for row in rows
        if row.get("makespan_delta_candidate_minus_reference") is not None
    ]
    worst = sorted(
        comparable,
        key=lambda row: float(row["makespan_delta_candidate_minus_reference"]),
        reverse=True,
    )[:5]
    best = sorted(
        comparable,
        key=lambda row: float(row["makespan_delta_candidate_minus_reference"]),
    )[:5]
    def deltas(field: str) -> list[float]:
        return [
            float(row[field])
            for row in comparable
            if row.get(field) is not None
        ]

    tail_deltas = deltas("return_tail_delta_candidate_minus_reference")
    starvation_deltas = deltas(
        "material_starvation_delta_candidate_minus_reference"
    )
    return {
        "instance_count": len(rows),
        "comparable_count": len(comparable),
        "signal_counts": dict(sorted(signals.items())),
        "mean_makespan_delta": _mean(
            [float(row["makespan_delta_candidate_minus_reference"]) for row in comparable]
        ),
        "mean_latest_task_completion_delta": _mean(
            [
                float(row["candidate_latest_task_completion"])
                - float(row["reference_latest_task_completion"])
                for row in comparable
                if row.get("candidate_latest_task_completion") is not None
                and row.get("reference_latest_task_completion") is not None
            ]
        ),
        "mean_return_tail_delta": _mean(tail_deltas),
        "mean_material_starvation_delta": _mean(starvation_deltas),
        "transport_order_same_count": sum(
            bool(row["transport_order_same"]) for row in comparable
        ),
        "process_order_same_count": sum(
            bool(row["process_order_same"]) for row in comparable
        ),
        "worst_instances": worst,
        "best_instances": best,
    }


def _load_c0_rows(package_root: Path, records) -> dict[str, list[dict[str, object]]]:
    source = package_root / "rollouts_json_safe_recheck"
    expected_keys = {(record.instance_id, record.seed) for record in records}
    grouped: dict[str, list[dict[str, object]]] = {}
    for seed in MODEL_SEEDS:
        path = source / f"C0_seed{seed}.jsonl"
        rows = _read_jsonl(path)
        keys = {(str(row["instance_id"]), int(row["instance_seed"])) for row in rows}
        if keys != expected_keys or len(rows) != len(records):
            raise ValueError(f"C0 raw rollout does not match frozen package: {path}")
        normalized = []
        for row in rows:
            item = dict(row)
            item["source_ticket"] = item.get("ticket", 46)
            item["ticket"] = 47
            item["source_method"] = item.get("method")
            item["method"] = f"C0_seed{seed}"
            item["model_seed"] = seed
            item.setdefault("execution_status", "completed")
            item["source_rollout"] = str(path.resolve())
            normalized.append(item)
        grouped[f"C0_seed{seed}"] = normalized
    return grouped


def _protocol_payload(
    package_root: Path,
    records,
    *,
    milp_time_limit_seconds: float,
    milp_threads: int,
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "ticket": 47,
        "source_ticket": 46,
        "type": "independent_diagnostic_comparison",
        "package_root": str(package_root.resolve()),
        "instance_count": len(records),
        "instance_seeds": [record.seed for record in records],
        "methods": [
            "milp_oracle",
            *ALL_FIXED_METHODS,
            "C0_seed3101",
            "C0_seed3102",
            "C0_seed3103",
        ],
        "milp": {
            "implementation": "baselines.gurobi_md_oracle",
            "time_limit_seconds": milp_time_limit_seconds,
            "threads": milp_threads,
            "objective": "minimize_latest_robot_exit_return",
        },
        "terminal_protocol": {
            "exit_location": list(EXIT_LOCATION),
            "max_rollout_steps": MAX_ROLLOUT_STEPS,
            "canonical_simulator_replay": True,
        },
        "comparison_rules": {
            "success_rate_denominator": "evaluated_runs_only",
            "makespan": "jointly_successful_pairs_only",
            "unavailable_solver_is_not_rollout_failure": True,
            "failed_makespan": None,
        },
        "diagnostic_only": True,
        "production_claim": False,
    }


def _comparison_pairs(
    groups: Mapping[str, Sequence[Mapping[str, object]]]
) -> dict[str, dict[str, object]]:
    pairs: dict[str, dict[str, object]] = {}
    for candidate_name in groups:
        if candidate_name.startswith("C0_"):
            references = ("milp_oracle", *ALL_FIXED_METHODS)
        elif candidate_name in ALL_FIXED_METHODS:
            references = ("milp_oracle",)
        else:
            continue
        for reference_name in references:
            key = f"{candidate_name}_vs_{reference_name}"
            comparison = paired_baseline_rows(
                groups[reference_name], groups[candidate_name]
            )
            comparison["reference_group"] = reference_name
            comparison["candidate_group"] = candidate_name
            pairs[key] = comparison
    return pairs


def _difference_payload(
    groups: Mapping[str, Sequence[Mapping[str, object]]]
) -> dict[str, object]:
    analyses: dict[str, object] = {}
    for candidate_name, candidate_rows in groups.items():
        if candidate_name.startswith("C0_"):
            references = ("milp_oracle", *ALL_FIXED_METHODS)
        elif candidate_name in ALL_FIXED_METHODS:
            references = ("milp_oracle",)
        else:
            continue
        for reference_name in references:
            key = f"{candidate_name}_vs_{reference_name}"
            rows = analyze_instance_differences(groups[reference_name], candidate_rows)
            analyses[key] = {
                "reference": reference_name,
                "candidate": candidate_name,
                "summary": _analysis_summary(rows),
                "instances": rows,
            }
    return {"schema_version": "1.0.0", "ticket": 47, "comparisons": analyses}


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_report(
    aggregates: Mapping[str, Mapping[str, object]],
    comparisons: Mapping[str, Mapping[str, object]],
    differences: Mapping[str, object],
    milp_rows: Sequence[Mapping[str, object]],
    *,
    groups: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    records: Sequence[Any] = (),
) -> str:
    lines = [
        "# Ticket 47 Same-Instance MILP and Greedy Comparison",
        "",
        "Diagnostic comparison only; no production or final-model claim.",
        "",
        "## Summary",
        "",
        "| Method | Evaluated | Success rate | Mean makespan | Mean starvation | Solver status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    order = [
        "milp_oracle",
        *GREEDY_METHODS,
        *EXISTING_FIXED_METHODS,
        "C0_seed3101",
        "C0_seed3102",
        "C0_seed3103",
    ]
    for name in order:
        summary = aggregates[name]
        makespan = summary["makespan_on_successes"]
        assert isinstance(makespan, Mapping)
        statuses = summary["solver_status_counts"]
        status_text = ", ".join(
            f"{key}:{value}" for key, value in dict(statuses).items()
        ) or "none"
        lines.append(
            f"| {name} | {summary['evaluated_count']}/{summary['run_count']} | "
            f"{_format(summary['success_rate'])} | {_format(makespan['mean'])} | "
            f"{_format(summary['mean_material_starvation'])} | {status_text} |"
        )

    milp_statuses = Counter(str(row.get("solver_status")) for row in milp_rows)
    lines.extend(
        [
            "",
            "## C0 Compared With Baselines",
            "",
            "Negative makespan delta means the candidate is faster; positive starvation delta means more waiting.",
            "",
            "| Candidate | Reference | Common success | Makespan delta | Improvement | Starvation delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for candidate_name in (
        "C0_seed3101",
        "C0_seed3102",
        "C0_seed3103",
    ):
        for reference_name in ("milp_oracle", *ALL_FIXED_METHODS):
            key = f"{candidate_name}_vs_{reference_name}"
            comparison = comparisons.get(key)
            if not isinstance(comparison, Mapping):
                continue
            delta = comparison["candidate_minus_reference"]
            starvation = comparison["starvation_candidate_minus_reference"]
            assert isinstance(delta, Mapping)
            assert isinstance(starvation, Mapping)
            lines.append(
                f"| {comparison['candidate_method']} | {comparison['reference_method']} | "
                f"{comparison['common_successful_pairs']} | {_format(delta['mean'])} | "
                f"{_format(delta['improvement_candidate_minus_reference'])} | "
                f"{_format(starvation['mean'])} |"
            )

    lines.extend(
        [
            "",
            "## Where Differences Come From",
            "",
            "The machine-readable `difference_analysis.json` records transport/process order, material starvation, latest task completion, and terminal return tail for every pair.",
            "The MILP chooses transport order, process coalition, robot reuse, and terminal positions jointly; C0 commits to online local assignments.",
            "The common terminal protocol requires every robot to return to (0,0), so a schedule with earlier real-task completion can still have a longer makespan.",
            "The following decomposition is descriptive evidence of where the paired gap appears, not a causal intervention.",
        ]
    )
    analysis_map = differences["comparisons"]
    assert isinstance(analysis_map, Mapping)
    lines.extend(
        [
            "",
            "### Component Decomposition Against MILP",
            "",
            "The makespan delta is decomposed into the latest real-task completion delta and the terminal return-tail delta.",
            "",
            "| Candidate | Makespan delta | Latest task completion delta | Return-tail delta | Starvation delta | Same transport order | Same process order |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in MODEL_SEEDS:
        key = f"C0_seed{seed}_vs_milp_oracle"
        analysis = analysis_map.get(key)
        if not isinstance(analysis, Mapping):
            continue
        summary = analysis["summary"]
        assert isinstance(summary, Mapping)
        comparable_count = int(summary["comparable_count"])
        lines.append(
            f"| C0_seed{seed} | {_format(summary['mean_makespan_delta'])} | "
            f"{_format(summary['mean_latest_task_completion_delta'])} | "
            f"{_format(summary['mean_return_tail_delta'])} | "
            f"{_format(summary['mean_material_starvation_delta'])} | "
            f"{summary['transport_order_same_count']}/{comparable_count} | "
            f"{summary['process_order_same_count']}/{comparable_count} |"
        )
    c0_milp_rows: list[Mapping[str, object]] = []
    for seed in MODEL_SEEDS:
        analysis = analysis_map.get(f"C0_seed{seed}_vs_milp_oracle")
        if not isinstance(analysis, Mapping):
            continue
        instances = analysis.get("instances", ())
        if not isinstance(instances, Sequence):
            continue
        c0_milp_rows.extend(
            row
            for row in instances
            if isinstance(row, Mapping)
            and row.get("makespan_delta_candidate_minus_reference") is not None
        )
    if c0_milp_rows:
        mean_gap = _mean(
            [
                float(row["makespan_delta_candidate_minus_reference"])
                for row in c0_milp_rows
            ]
        )
        mean_task_delta = _mean(
            [
                float(row["candidate_latest_task_completion"])
                - float(row["reference_latest_task_completion"])
                for row in c0_milp_rows
                if row.get("candidate_latest_task_completion") is not None
                and row.get("reference_latest_task_completion") is not None
            ]
        )
        mean_tail_delta = _mean(
            [
                float(row["return_tail_delta_candidate_minus_reference"])
                for row in c0_milp_rows
                if row.get("return_tail_delta_candidate_minus_reference") is not None
            ]
        )
        tail_share = (
            None
            if mean_gap is None or mean_gap <= 0 or mean_tail_delta is None
            else 100.0 * mean_tail_delta / mean_gap
        )
        lines.append(
            f"Across {len(c0_milp_rows)} paired C0/MILP instances, the mean "
            f"makespan gap is {_format(mean_gap)}: latest real-task completion "
            f"contributes {_format(mean_task_delta)}, terminal return tail "
            f"contributes {_format(mean_tail_delta)} "
            f"({_format(tail_share)}% of the mean gap)."
        )
    for key in ("C0_seed3101_vs_milp_oracle", "C0_seed3102_vs_milp_oracle", "C0_seed3103_vs_milp_oracle"):
        analysis = analysis_map.get(key)
        if not isinstance(analysis, Mapping):
            continue
        summary = analysis["summary"]
        assert isinstance(summary, Mapping)
        lines.append(
            f"- `{key}`: mean makespan delta {_format(summary['mean_makespan_delta'])}; "
            f"signals {dict(summary['signal_counts'])}."
        )
        worst = summary["worst_instances"]
        assert isinstance(worst, Sequence)
        for row in list(worst)[:2]:
            assert isinstance(row, Mapping)
            lines.append(
                f"  - {row['instance_id']}: delta {_format(row['makespan_delta_candidate_minus_reference'])}, "
                f"starvation delta {_format(row['material_starvation_delta_candidate_minus_reference'])}, "
                f"signal `{row['diagnostic_signal']}`."
            )
    lines.extend(_representative_trace_lines(groups or {}, records, analysis_map))
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is a diagnostic comparison and makes no production or final-model claim.",
            f"- MILP solver statuses across {len(milp_rows)} instances: {dict(sorted(milp_statuses.items()))}; rows without an incumbent remain null and are excluded from the evaluated-run denominator.",
            "- Ticket 20/44/45 were not executed by this comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def run_same_instance_comparison(
    package_root: str | Path = SOURCE_ROOT,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    milp_time_limit_seconds: float = MILP_TIME_LIMIT_SECONDS,
    milp_threads: int = MILP_THREADS,
) -> dict[str, Path]:
    """Run all fixed baselines on the exact Ticket 46 frozen instances."""

    package_path = Path(package_root)
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite comparison output: {destination}")
    records = load_frozen_diagnostic_records(package_path)
    if tuple(record.seed for record in records) != FROZEN_INSTANCE_SEEDS:
        raise ValueError("comparison package does not use the Ticket 46 frozen seeds")
    if milp_time_limit_seconds <= 0 or not math.isfinite(milp_time_limit_seconds):
        raise ValueError("milp_time_limit_seconds must be positive and finite")
    if isinstance(milp_threads, bool) or milp_threads <= 0:
        raise ValueError("milp_threads must be a positive integer")

    started = time.perf_counter()
    destination.mkdir(parents=True)
    _write_json(
        destination / "comparison_protocol.json",
        _protocol_payload(
            package_path,
            records,
            milp_time_limit_seconds=milp_time_limit_seconds,
            milp_threads=milp_threads,
        ),
    )
    groups = _load_c0_rows(package_path, records)
    raw_root = destination / "rollouts"
    for name, rows in groups.items():
        _write_jsonl(raw_root / f"{name}.jsonl", rows)

    milp_rows = [
        _run_milp(
            record,
            time_limit_seconds=milp_time_limit_seconds,
            threads=milp_threads,
        )
        for record in records
    ]
    groups["milp_oracle"] = milp_rows
    _write_jsonl(raw_root / "milp_oracle.jsonl", milp_rows)
    for method in GREEDY_METHODS:
        if method == "process_md_greedy":
            rows = [_run_ticket46_fixed_baseline(record, method) for record in records]
        else:
            rows = [_run_greedy(record, method) for record in records]
        groups[method] = rows
        _write_jsonl(raw_root / f"{method}.jsonl", rows)
    for method in EXISTING_FIXED_METHODS:
        rows = [_run_ticket46_fixed_baseline(record, method) for record in records]
        groups[method] = rows
        _write_jsonl(raw_root / f"{method}.jsonl", rows)

    aggregates = {
        name: aggregate_baseline_rows(rows) for name, rows in groups.items()
    }
    comparisons = _comparison_pairs(groups)
    differences = _difference_payload(groups)
    failures = [
        row
        for rows in groups.values()
        for row in rows
        if _success_value(row) is False or _success_value(row) is None
    ]
    runtime = {
        "ticket": 47,
        "package_root": str(package_path.resolve()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "milp_time_limit_seconds": milp_time_limit_seconds,
        "milp_threads": milp_threads,
        "methods_run_sequentially": True,
        "elapsed_seconds": time.perf_counter() - started,
        "milp_status_counts": dict(
            sorted(Counter(str(row["solver_status"]) for row in milp_rows).items())
        ),
    }
    _write_json(
        destination / "aggregate_metrics.json",
        {"ticket": 47, "baseline_package": str(package_path.resolve()), "methods": aggregates},
    )
    _write_json(
        destination / "paired_comparisons.json",
        {"ticket": 47, "comparisons": comparisons},
    )
    _write_json(destination / "difference_analysis.json", differences)
    _write_json(destination / "runtime_and_hardware.json", runtime)
    _write_json(destination / "failures.json", {"ticket": 47, "failures": failures})
    (destination / "final_report.md").write_text(
        _render_report(
            aggregates, comparisons, differences, milp_rows,
            groups=groups,
            records=records,
        ),
        encoding="utf-8",
    )
    return {
        "aggregate_metrics": destination / "aggregate_metrics.json",
        "paired_comparisons": destination / "paired_comparisons.json",
        "difference_analysis": destination / "difference_analysis.json",
        "final_report": destination / "final_report.md",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--milp-time-limit", type=float, default=MILP_TIME_LIMIT_SECONDS)
    parser.add_argument("--milp-threads", type=int, default=MILP_THREADS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = run_same_instance_comparison(
        args.package_root,
        output_root=args.output_root,
        milp_time_limit_seconds=args.milp_time_limit,
        milp_threads=args.milp_threads,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


__all__ = [
    "GREEDY_METHODS",
    "MILP_THREADS",
    "MILP_TIME_LIMIT_SECONDS",
    "aggregate_baseline_rows",
    "analyze_instance_differences",
    "main",
    "paired_baseline_rows",
    "run_same_instance_comparison",
]


if __name__ == "__main__":
    raise SystemExit(main())
