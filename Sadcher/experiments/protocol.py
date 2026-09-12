"""Versioned result, termination, split, seed, and statistics protocol."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "1.0.0"
PAIRED_EVALUATION_SEEDS = tuple(range(30))
SPLIT_HASH_SEED = 2025
NORMAL_95_PERCENT_Z = 1.96


class DatasetSplit(str, Enum):
    """Task-group split names used by every training and evaluation pipeline."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class FailureReason(str, Enum):
    """Stable terminal failure codes from the MD-SADCHER++ domain contract."""

    STATIC_INFEASIBLE = "static_infeasible"
    ROBOT_TYPE_MISMATCH = "robot_type_mismatch"
    NO_CAPABLE_TRANSPORT_ROBOT = "no_capable_transport_robot"
    INVALID_GRAPH = "invalid_graph"
    DEADLOCK = "deadlock"
    SCHEDULER_FAILURE = "scheduler_failure"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class DescriptiveStatistics:
    """Mean, sample standard deviation, and normal-approximation 95% CI."""

    count: int
    mean: float | None
    sample_standard_deviation: float | None
    ci95_low: float | None
    ci95_high: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "mean": self.mean,
            "sample_standard_deviation": self.sample_standard_deviation,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
        }


@dataclass(frozen=True, slots=True)
class ResultSummary:
    total_runs: int
    successful_runs: int
    success_rate: float | None
    failure_counts: Mapping[str, int]
    makespan: DescriptiveStatistics

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "success_rate": self.success_rate,
            "failure_counts": dict(self.failure_counts),
            "makespan": self.makespan.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PairedMakespanSummary:
    matched_pairs: int
    common_successful_pairs: int
    candidate_minus_baseline: DescriptiveStatistics

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_pairs": self.matched_pairs,
            "common_successful_pairs": self.common_successful_pairs,
            "candidate_minus_baseline": self.candidate_minus_baseline.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """One auditable run in the versioned experiment result schema."""

    run_id: str
    method: str
    instance_id: str
    seed: int
    split: DatasetSplit
    success: bool
    makespan: float | None
    failure_reason: FailureReason | None
    all_real_tasks_completed: bool
    all_robots_at_exit: bool
    illegal_assignment_count: int = 0
    material_starvation: Mapping[str, float] = field(default_factory=dict)
    robot_utilization: Mapping[str, float] = field(default_factory=dict)
    inference_time_seconds: float = 0.0
    wall_time_seconds: float = 0.0
    process_execution_records: Sequence[Mapping[str, Any]] = field(
        default_factory=tuple
    )
    transport_execution_records: Sequence[Mapping[str, Any]] = field(
        default_factory=tuple
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        for field_name in ("run_id", "method", "instance_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must be {PROTOCOL_VERSION!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.split, DatasetSplit):
            raise ValueError("split must be a DatasetSplit")
        if not isinstance(self.success, bool):
            raise ValueError("success must be boolean")
        if not isinstance(self.all_real_tasks_completed, bool):
            raise ValueError("all_real_tasks_completed must be boolean")
        if not isinstance(self.all_robots_at_exit, bool):
            raise ValueError("all_robots_at_exit must be boolean")
        if (
            not isinstance(self.illegal_assignment_count, int)
            or isinstance(self.illegal_assignment_count, bool)
            or self.illegal_assignment_count < 0
        ):
            raise ValueError("illegal_assignment_count must be a non-negative integer")

        if self.success:
            if not self.all_real_tasks_completed or not self.all_robots_at_exit:
                raise ValueError(
                    "successful run requires all real tasks completed and all robots at exit"
                )
            if self.illegal_assignment_count:
                raise ValueError("successful run cannot contain an illegal assignment")
            _validate_non_negative_finite("makespan", self.makespan)
            if self.failure_reason is not None:
                raise ValueError("successful run cannot have a failure reason")
        else:
            if self.makespan is not None:
                raise ValueError("failed run requires a null makespan")
            if not isinstance(self.failure_reason, FailureReason):
                raise ValueError("failed run requires a stable FailureReason")

        _validate_metric_map("material starvation", self.material_starvation)
        _validate_metric_map(
            "robot utilization", self.robot_utilization, maximum=1.0
        )
        _validate_non_negative_finite(
            "inference_time_seconds", self.inference_time_seconds
        )
        _validate_non_negative_finite("wall_time_seconds", self.wall_time_seconds)

        object.__setattr__(
            self,
            "material_starvation",
            MappingProxyType(dict(self.material_starvation)),
        )
        object.__setattr__(
            self,
            "robot_utilization",
            MappingProxyType(dict(self.robot_utilization)),
        )
        object.__setattr__(
            self,
            "process_execution_records",
            _freeze_execution_records(
                "process_execution_records", self.process_execution_records
            ),
        )
        object.__setattr__(
            self,
            "transport_execution_records",
            _freeze_execution_records(
                "transport_execution_records", self.transport_execution_records
            ),
        )
        object.__setattr__(
            self, "metadata", _freeze_json_mapping("metadata", self.metadata)
        )

    @classmethod
    def succeeded(
        cls,
        *,
        run_id: str,
        method: str,
        instance_id: str,
        seed: int,
        split: DatasetSplit,
        makespan: float,
        all_real_tasks_completed: bool,
        all_robots_at_exit: bool,
        illegal_assignment_count: int = 0,
        material_starvation: Mapping[str, float] | None = None,
        robot_utilization: Mapping[str, float] | None = None,
        inference_time_seconds: float = 0.0,
        wall_time_seconds: float = 0.0,
        process_execution_records: Sequence[Mapping[str, Any]] = (),
        transport_execution_records: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentResult:
        return cls(
            run_id=run_id,
            method=method,
            instance_id=instance_id,
            seed=seed,
            split=split,
            success=True,
            makespan=makespan,
            failure_reason=None,
            all_real_tasks_completed=all_real_tasks_completed,
            all_robots_at_exit=all_robots_at_exit,
            illegal_assignment_count=illegal_assignment_count,
            material_starvation={} if material_starvation is None else material_starvation,
            robot_utilization={} if robot_utilization is None else robot_utilization,
            inference_time_seconds=inference_time_seconds,
            wall_time_seconds=wall_time_seconds,
            process_execution_records=process_execution_records,
            transport_execution_records=transport_execution_records,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def failed(
        cls,
        *,
        run_id: str,
        method: str,
        instance_id: str,
        seed: int,
        split: DatasetSplit,
        reason: FailureReason,
        all_real_tasks_completed: bool,
        all_robots_at_exit: bool,
        illegal_assignment_count: int = 0,
        material_starvation: Mapping[str, float] | None = None,
        robot_utilization: Mapping[str, float] | None = None,
        inference_time_seconds: float = 0.0,
        wall_time_seconds: float = 0.0,
        process_execution_records: Sequence[Mapping[str, Any]] = (),
        transport_execution_records: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentResult:
        return cls(
            run_id=run_id,
            method=method,
            instance_id=instance_id,
            seed=seed,
            split=split,
            success=False,
            makespan=None,
            failure_reason=reason,
            all_real_tasks_completed=all_real_tasks_completed,
            all_robots_at_exit=all_robots_at_exit,
            illegal_assignment_count=illegal_assignment_count,
            material_starvation={} if material_starvation is None else material_starvation,
            robot_utilization={} if robot_utilization is None else robot_utilization,
            inference_time_seconds=inference_time_seconds,
            wall_time_seconds=wall_time_seconds,
            process_execution_records=process_execution_records,
            transport_execution_records=transport_execution_records,
            metadata={} if metadata is None else metadata,
        )

    @property
    def pairing_key(self) -> tuple[str, int]:
        return self.instance_id, self.seed

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "run_id": self.run_id,
            "method": self.method,
            "dataset": {
                "instance_id": self.instance_id,
                "seed": self.seed,
                "split": self.split.value,
            },
            "termination": {
                "success": self.success,
                "failure_reason": (
                    self.failure_reason.value if self.failure_reason else None
                ),
                "all_real_tasks_completed": self.all_real_tasks_completed,
                "all_robots_at_exit": self.all_robots_at_exit,
                "illegal_assignment_count": self.illegal_assignment_count,
            },
            "metrics": {
                "makespan": self.makespan,
                "material_starvation": dict(self.material_starvation),
                "robot_utilization": dict(self.robot_utilization),
                "inference_time_seconds": self.inference_time_seconds,
                "wall_time_seconds": self.wall_time_seconds,
            },
            "execution_records": {
                "process": [
                    _to_plain_json(record)
                    for record in self.process_execution_records
                ],
                "transport": [
                    _to_plain_json(record)
                    for record in self.transport_execution_records
                ],
            },
            "metadata": _to_plain_json(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")
        )


def task_level_split(task_group_id: str | int) -> DatasetSplit:
    """Assign an entire task/instance group to a stable 80/10/10 split."""

    if isinstance(task_group_id, bool) or not isinstance(task_group_id, (str, int)):
        raise ValueError("task_group_id must be a non-empty string or integer")
    normalized_id = str(task_group_id).strip()
    if not normalized_id:
        raise ValueError("task_group_id must be a non-empty string or integer")

    digest = hashlib.sha256(
        f"{SPLIT_HASH_SEED}:{normalized_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < 0.8:
        return DatasetSplit.TRAIN
    if bucket < 0.9:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


def describe(values: Iterable[float]) -> DescriptiveStatistics:
    """Summarize finite observations without silently dropping any value."""

    observations = list(values)
    for value in observations:
        _validate_signed_finite("observation", value)
    count = len(observations)
    if count == 0:
        return DescriptiveStatistics(0, None, None, None, None)

    mean = statistics.fmean(observations)
    if count == 1:
        return DescriptiveStatistics(1, mean, None, None, None)

    sample_standard_deviation = statistics.stdev(observations)
    margin = NORMAL_95_PERCENT_Z * sample_standard_deviation / math.sqrt(count)
    return DescriptiveStatistics(
        count,
        mean,
        sample_standard_deviation,
        mean - margin,
        mean + margin,
    )


def summarize_results(results: Iterable[ExperimentResult]) -> ResultSummary:
    """Report failures separately and summarize makespan over successes only."""

    runs = list(results)
    successful = [run for run in runs if run.success]
    failure_counts = Counter(
        run.failure_reason.value for run in runs if run.failure_reason is not None
    )
    return ResultSummary(
        total_runs=len(runs),
        successful_runs=len(successful),
        success_rate=(len(successful) / len(runs) if runs else None),
        failure_counts=dict(sorted(failure_counts.items())),
        makespan=describe(run.makespan for run in successful if run.makespan is not None),
    )


def paired_makespan_difference(
    baseline: Sequence[ExperimentResult], candidate: Sequence[ExperimentResult]
) -> PairedMakespanSummary:
    """Summarize candidate-minus-baseline over matching, jointly successful runs."""

    baseline_by_key = _index_unique_results(baseline, "baseline")
    candidate_by_key = _index_unique_results(candidate, "candidate")
    if baseline_by_key.keys() != candidate_by_key.keys():
        raise ValueError(
            "paired comparisons require identical instance/seed keys for both methods"
        )
    matched_keys = sorted(baseline_by_key)
    differences: list[float] = []
    for key in matched_keys:
        baseline_run = baseline_by_key[key]
        candidate_run = candidate_by_key[key]
        if baseline_run.split is not candidate_run.split:
            raise ValueError(f"paired run {key!r} has inconsistent dataset splits")
        if baseline_run.success and candidate_run.success:
            assert baseline_run.makespan is not None
            assert candidate_run.makespan is not None
            differences.append(candidate_run.makespan - baseline_run.makespan)

    return PairedMakespanSummary(
        matched_pairs=len(matched_keys),
        common_successful_pairs=len(differences),
        candidate_minus_baseline=describe(differences),
    )


def _index_unique_results(
    results: Sequence[ExperimentResult], label: str
) -> dict[tuple[str, int], ExperimentResult]:
    indexed: dict[tuple[str, int], ExperimentResult] = {}
    for result in results:
        if result.pairing_key in indexed:
            raise ValueError(f"duplicate {label} pairing key: {result.pairing_key!r}")
        indexed[result.pairing_key] = result
    return indexed


def _freeze_execution_records(
    name: str, records: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError(f"{name} must be a sequence of mappings")

    frozen_records = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{name}[{index}] must be a mapping")
        frozen_records.append(_freeze_json_mapping(f"{name}[{index}]", record))
    return tuple(frozen_records)


def _freeze_json_mapping(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    frozen = _freeze_json_value(name, value)
    assert isinstance(frozen, Mapping)
    return frozen


def _freeze_json_value(name: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen_items = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings")
            frozen_items[key] = _freeze_json_value(f"{name}.{key}", item)
        return MappingProxyType(frozen_items)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(f"{name}[{index}]", item)
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{name} must contain only finite JSON values")


def _to_plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_json(item) for item in value]
    return value


def _validate_metric_map(
    name: str, values: Mapping[str, float], *, maximum: float | None = None
) -> None:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    for identifier, value in values.items():
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{name} identifiers must be non-empty strings")
        _validate_non_negative_finite(name, value)
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} values must be at most {maximum}")


def _validate_non_negative_finite(name: str, value: float | None) -> None:
    _validate_signed_finite(name, value)
    assert value is not None
    if value < 0:
        raise ValueError(f"{name} must be non-negative and finite")


def _validate_signed_finite(name: str, value: float | None) -> None:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be numeric and finite")
