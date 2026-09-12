"""Reproducible evidence for the process-only legacy scheduling paths."""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.protocol import ExperimentResult, task_level_split
from helper_functions.schedules import Full_Horizon_Schedule


LEGACY_INSTANCE_FIELDS = frozenset(
    {"Q", "R", "T_e", "T_t", "task_locations", "precedence_constraints"}
)
_ARRAY_FIELDS = ("Q", "R", "T_e", "T_t", "task_locations")
LEGACY_FLOAT_TOLERANCE = 1e-5  # Official schedules store times to five decimals.

class LegacyTimingSemantics(str, Enum):
    """Time accounting used by a legacy schedule producer."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


def load_legacy_problem(path: str | Path) -> dict[str, Any]:
    """Load an unconverted SADCHER instance and reject schema drift."""

    source = Path(path)
    with source.open(encoding="utf-8") as file:
        problem = json.load(file)

    if not isinstance(problem, dict):
        raise ValueError("legacy problem must be a JSON object")
    actual_fields = set(problem)
    if actual_fields != LEGACY_INSTANCE_FIELDS:
        missing = sorted(LEGACY_INSTANCE_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - LEGACY_INSTANCE_FIELDS)
        raise ValueError(
            f"legacy problem schema mismatch; missing={missing}, unexpected={unexpected}"
        )

    loaded = dict(problem)
    for field_name in _ARRAY_FIELDS:
        loaded[field_name] = np.asarray(loaded[field_name])
    precedence = loaded["precedence_constraints"]
    if precedence is not None and not isinstance(precedence, list):
        raise ValueError("precedence_constraints must be a list or null")
    return loaded


def validate_legacy_schedule(
    problem_instance: Mapping[str, Any],
    schedule: Full_Horizon_Schedule,
    *,
    timing_semantics: LegacyTimingSemantics,
) -> dict[str, list[int]]:
    """Return task assignments after checking legacy skill and precedence feasibility."""

    _require_legacy_fields(problem_instance)
    if not isinstance(timing_semantics, LegacyTimingSemantics):
        raise ValueError("timing_semantics must be a LegacyTimingSemantics")
    capabilities = np.asarray(problem_instance["Q"], dtype=bool)
    requirements = np.asarray(problem_instance["R"], dtype=bool)
    durations = np.asarray(problem_instance["T_e"], dtype=float)
    travel_times = np.asarray(problem_instance["T_t"], dtype=float)
    n_robots = capabilities.shape[0]
    n_tasks = requirements.shape[0] - 2
    expected_nodes = n_tasks + 2
    if (
        durations.shape != (expected_nodes,)
        or travel_times.shape != (expected_nodes, expected_nodes)
    ):
        raise ValueError("legacy timing arrays do not match the task count")

    if schedule.n_robots != n_robots:
        raise ValueError(
            f"schedule has {schedule.n_robots} robots; expected {n_robots}"
        )
    if schedule.n_tasks != n_tasks:
        raise ValueError(f"schedule has {schedule.n_tasks} tasks; expected {n_tasks}")
    if not _is_finite_number(schedule.makespan) or schedule.makespan < 0:
        raise ValueError("schedule makespan must be a non-negative finite number")

    assignments: dict[int, list[int]] = {task: [] for task in range(1, n_tasks + 1)}
    task_intervals: dict[int, list[tuple[float, float]]] = {
        task: [] for task in range(1, n_tasks + 1)
    }
    return_arrivals: list[float] = []
    for robot_id, robot_schedule in schedule.robot_schedules.items():
        if robot_id not in range(n_robots):
            raise ValueError(f"schedule references unknown robot {robot_id}")
        seen_tasks: set[int] = set()
        previous_task = 0
        previous_end = 0.0
        for task_id, start_time, end_time in robot_schedule:
            if task_id not in assignments:
                raise ValueError(f"schedule references non-real task {task_id}")
            if task_id in seen_tasks:
                raise ValueError(f"robot {robot_id} visits task {task_id} more than once")
            if not _is_finite_number(start_time) or not _is_finite_number(end_time):
                raise ValueError("schedule times must be finite numbers")
            if start_time < 0 or end_time < start_time:
                raise ValueError(f"task {task_id} has an invalid execution interval")
            if start_time + LEGACY_FLOAT_TOLERANCE < previous_end:
                raise ValueError(f"robot {robot_id} has overlapping task intervals")
            travel_time = float(travel_times[previous_task, task_id])
            actual_duration = float(end_time) - float(start_time)
            required_duration = float(durations[task_id])
            if timing_semantics is LegacyTimingSemantics.CONTINUOUS:
                earliest_arrival = previous_end + travel_time
                expected_duration = required_duration
            else:
                initial_departure_offset = 1 if previous_task == 0 else 0
                earliest_arrival = (
                    previous_end + math.ceil(travel_time) - initial_departure_offset
                )
                expected_duration = max(0.0, required_duration - 1.0)
            if start_time + LEGACY_FLOAT_TOLERANCE < earliest_arrival:
                raise ValueError(f"robot {robot_id} cannot reach task {task_id} in time")
            if abs(actual_duration - expected_duration) > LEGACY_FLOAT_TOLERANCE:
                raise ValueError(f"task {task_id} has an invalid execution duration")
            if end_time > schedule.makespan + LEGACY_FLOAT_TOLERANCE:
                raise ValueError(f"task {task_id} ends after the schedule makespan")
            seen_tasks.add(task_id)
            assignments[task_id].append(robot_id)
            task_intervals[task_id].append((float(start_time), float(end_time)))
            previous_task = task_id
            previous_end = float(end_time)
        travel_to_exit = float(travel_times[previous_task, n_tasks + 1])
        if timing_semantics is LegacyTimingSemantics.DISCRETE:
            travel_to_exit = math.ceil(travel_to_exit)
        return_arrivals.append(previous_end + travel_to_exit)

    expected_makespan = max(return_arrivals)
    if abs(float(schedule.makespan) - expected_makespan) > LEGACY_FLOAT_TOLERANCE:
        raise ValueError("schedule makespan does not show every robot returned to exit")

    for task_id, robot_ids in assignments.items():
        if not robot_ids:
            raise ValueError(f"real task {task_id} has no assigned robot")
        provided_skills = np.logical_or.reduce(capabilities[robot_ids], axis=0)
        if not np.all(provided_skills[requirements[task_id]]):
            raise ValueError(f"assignment does not cover task {task_id} requirements")
        starts, ends = zip(*task_intervals[task_id])
        if (
            max(starts) - min(starts) > LEGACY_FLOAT_TOLERANCE
            or max(ends) - min(ends) > LEGACY_FLOAT_TOLERANCE
        ):
            raise ValueError(f"task {task_id} coalition does not execute together")

    precedence_constraints = (
        problem_instance["precedence_constraints"]
        if problem_instance["precedence_constraints"] is not None
        else []
    )
    for edge in precedence_constraints:
        if not isinstance(edge, (list, tuple, np.ndarray)) or len(edge) != 2:
            raise ValueError("each precedence constraint must contain two task IDs")
        predecessor, successor = edge
        if (
            not isinstance(predecessor, (int, np.integer))
            or isinstance(predecessor, (bool, np.bool_))
            or not isinstance(successor, (int, np.integer))
            or isinstance(successor, (bool, np.bool_))
            or predecessor not in task_intervals
            or successor not in task_intervals
        ):
            raise ValueError(f"invalid precedence task IDs: {edge!r}")
        predecessor_end = max(end for _, end in task_intervals[int(predecessor)])
        successor_start = min(start for start, _ in task_intervals[int(successor)])
        if successor_start + LEGACY_FLOAT_TOLERANCE < predecessor_end:
            raise ValueError(
                f"precedence {predecessor}->{successor} is violated by the schedule"
            )

    return {str(task): sorted(robot_ids) for task, robot_ids in assignments.items()}


def build_legacy_regression_evidence(
    *,
    problem_instance: Mapping[str, Any],
    schedule: Full_Horizon_Schedule,
    method: str,
    instance_id: str,
    seed: int,
    timing_semantics: LegacyTimingSemantics,
    checkpoint_path: str | Path | None = None,
    checkpoint_device: str | None = None,
) -> dict[str, Any]:
    """Build a versioned result plus a stable hash for one successful legacy run."""

    feasible_assignments = validate_legacy_schedule(
        problem_instance, schedule, timing_semantics=timing_semantics
    )
    regression_metadata: dict[str, Any] = {
        "material_delivery_enabled": False,
        "timing_semantics": timing_semantics.value,
        "legacy_instance_fields": sorted(LEGACY_INSTANCE_FIELDS),
        "schedule": _canonical_schedule(schedule),
        "feasible_assignments": feasible_assignments,
    }
    if checkpoint_path is not None:
        if checkpoint_device != "cpu":
            raise ValueError("legacy checkpoint evidence must record a CPU rollout")
        regression_metadata.update(
            {
                "checkpoint_device": checkpoint_device,
                "checkpoint_sha256": _sha256_file(Path(checkpoint_path)),
            }
        )
    elif checkpoint_device is not None:
        raise ValueError("checkpoint_device requires checkpoint_path")

    result = ExperimentResult.succeeded(
        run_id=f"{method}-{instance_id}-{seed}",
        method=method,
        instance_id=instance_id,
        seed=seed,
        split=task_level_split(instance_id),
        makespan=float(schedule.makespan),
        all_real_tasks_completed=True,
        all_robots_at_exit=True,
        metadata={"legacy_regression": regression_metadata},
    ).to_dict()
    return {"result_hash": hash_legacy_result(result), "result": result}


def hash_legacy_result(result: Mapping[str, Any]) -> str:
    """Hash the canonical protocol result without including its envelope hash."""

    canonical = json.dumps(
        result, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def save_legacy_regression_evidence(
    path: str | Path, evidence: Mapping[str, Any]
) -> None:
    """Persist evidence only after verifying that its result hash is intact."""

    result = evidence.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("legacy evidence requires a result mapping")
    expected_hash = hash_legacy_result(result)
    if evidence.get("result_hash") != expected_hash:
        raise ValueError("legacy evidence result hash does not match its result")
    Path(path).write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_legacy_fields(problem_instance: Mapping[str, Any]) -> None:
    if set(problem_instance) != LEGACY_INSTANCE_FIELDS:
        raise ValueError("regression evidence requires the exact legacy instance schema")


def _canonical_schedule(schedule: Full_Horizon_Schedule) -> dict[str, Any]:
    return {
        "makespan": float(schedule.makespan),
        "n_tasks": schedule.n_tasks,
        "n_robots": schedule.n_robots,
        "robot_schedules": {
            str(robot_id): [
                {
                    "task": int(task_id),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                }
                for task_id, start_time, end_time in robot_schedule
            ]
            for robot_id, robot_schedule in sorted(schedule.robot_schedules.items())
        },
    }


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
