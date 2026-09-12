"""Paired end-to-end latency and quality evaluation for MD decoders."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from experiments.protocol import ExperimentResult, describe, summarize_results
from schedulers.online_md_scheduler import OnlineSchedulerResult


LATENCY_COMPONENTS = (
    "encoder",
    "scoring",
    "decoder",
    "repair",
    "fallback",
    "total",
)


@dataclass(frozen=True, slots=True)
class LatencyPercentiles:
    count: int
    p50_seconds: float | None
    p95_seconds: float | None
    p99_seconds: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "p50_seconds": self.p50_seconds,
            "p95_seconds": self.p95_seconds,
            "p99_seconds": self.p99_seconds,
        }


@dataclass(frozen=True, slots=True)
class LatencyEvaluationMethod:
    name: str
    decoder_kind: str
    runner: Callable[[str, int], OnlineSchedulerResult]
    exact_reference: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("latency evaluation method name must not be empty")
        if self.decoder_kind not in {
            "learned_decoder",
            "masked_greedy",
            "exact_mip",
            "baseline",
        }:
            raise ValueError("unsupported decoder_kind")
        if self.exact_reference != (self.decoder_kind == "exact_mip"):
            raise ValueError("exact_reference must identify the exact_mip method")


@dataclass(frozen=True, slots=True)
class MDLatencyEvaluationReport:
    paired_keys: tuple[tuple[str, int], ...]
    methods: tuple[LatencyEvaluationMethod, ...]
    results: dict[str, tuple[OnlineSchedulerResult, ...]]
    exact_reference_name: str

    def to_dict(self) -> dict:
        exact = self.results[self.exact_reference_name]
        method_payload = {}
        for method in self.methods:
            runs = self.results[method.name]
            method_payload[method.name] = {
                "decoder_kind": method.decoder_kind,
                "quality": _quality_summary(runs, exact),
                "latency": _latency_summary(runs),
                "raw_results": [run.experiment.to_dict() for run in runs],
                "raw_decisions": [
                    {
                        "instance_id": run.experiment.instance_id,
                        "seed": run.experiment.seed,
                        "decision_time": decision.decision_time,
                        "encoder_seconds": decision.encoder_time_seconds,
                        "scoring_seconds": decision.scoring_time_seconds,
                        "decoder_seconds": decision.decoder_time_seconds,
                        "repair_seconds": decision.repair_time_seconds,
                        "fallback_seconds": decision.fallback_time_seconds,
                        "total_seconds": decision.total_time_seconds,
                        "fallback_used": decision.fallback_used,
                    }
                    for run in runs
                    for decision in run.decision_records
                ],
            }
        return {
            "paired_keys": [list(key) for key in self.paired_keys],
            "exact_reference": self.exact_reference_name,
            "methods": method_payload,
            "quality_latency_tradeoff": {
                name: {
                    "oracle_regret_mean": payload["quality"]["oracle_regret"][
                        "mean"
                    ],
                    "total_latency_p95_seconds": payload["latency"]["total"][
                        "p95_seconds"
                    ],
                    "success_rate": payload["quality"]["success_rate"],
                }
                for name, payload in method_payload.items()
            },
        }


def run_paired_latency_evaluation(
    cases: Sequence[tuple[str, int]],
    methods: Sequence[LatencyEvaluationMethod],
) -> MDLatencyEvaluationReport:
    paired_keys = tuple(cases)
    if not paired_keys or len(set(paired_keys)) != len(paired_keys):
        raise ValueError("latency cases must be non-empty unique paired keys")
    method_list = tuple(methods)
    if not method_list or len({item.name for item in method_list}) != len(method_list):
        raise ValueError("latency methods must have unique names")
    exact = tuple(item for item in method_list if item.exact_reference)
    if len(exact) != 1:
        raise ValueError("latency evaluation requires exactly one exact MIP reference")

    results: dict[str, tuple[OnlineSchedulerResult, ...]] = {}
    expected_contract = None
    for method in method_list:
        runs = tuple(method.runner(*key) for key in paired_keys)
        if tuple(run.experiment.pairing_key for run in runs) != paired_keys:
            raise ValueError(f"method {method.name} did not preserve paired case order")
        for run in runs:
            experiment = run.experiment
            if experiment.illegal_assignment_count != 0:
                raise ValueError("latency evaluation rejects illegal assignments")
            contract = (
                experiment.split,
                experiment.metadata.get("hard_mask_source"),
                experiment.metadata.get("terminal_protocol"),
                experiment.protocol_version,
            )
            if contract[1:] != (
                "hard_feasibility",
                "canonical_md_simulator",
                experiment.protocol_version,
            ):
                raise ValueError("run does not use the canonical hard mask and terminal protocol")
            if expected_contract is None:
                expected_contract = contract
            elif contract != expected_contract:
                raise ValueError("methods must use the same split, mask, and terminal protocol")
            _validate_decision_timings(run)
        results[method.name] = runs
    return MDLatencyEvaluationReport(
        paired_keys, method_list, results, exact[0].name
    )


def latency_percentiles(values: Iterable[float]) -> LatencyPercentiles:
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("latency values must be non-negative and finite")
    if not ordered:
        return LatencyPercentiles(0, None, None, None)
    return LatencyPercentiles(
        len(ordered),
        _percentile(ordered, 0.50),
        _percentile(ordered, 0.95),
        _percentile(ordered, 0.99),
    )


def _latency_summary(runs: tuple[OnlineSchedulerResult, ...]) -> dict:
    records = tuple(record for run in runs for record in run.decision_records)
    values = {
        "encoder": (record.encoder_time_seconds for record in records),
        "scoring": (record.scoring_time_seconds for record in records),
        "decoder": (record.decoder_time_seconds for record in records),
        "repair": (record.repair_time_seconds for record in records),
        "fallback": (record.fallback_time_seconds for record in records),
        "total": (record.total_time_seconds for record in records),
    }
    return {
        component: latency_percentiles(values[component]).to_dict()
        for component in LATENCY_COMPONENTS
    }


def _quality_summary(
    runs: tuple[OnlineSchedulerResult, ...],
    exact_runs: tuple[OnlineSchedulerResult, ...],
) -> dict:
    experiments = tuple(run.experiment for run in runs)
    summary = summarize_results(experiments).to_dict()
    successful = tuple(result for result in experiments if result.success)
    decisions = tuple(record for run in runs for record in run.decision_records)
    fallback_count = sum(record.fallback_used for record in decisions)
    summary.update(
        {
            "material_starvation": describe(
                value for result in successful for value in result.material_starvation.values()
            ).to_dict(),
            "robot_utilization": describe(
                value for result in successful for value in result.robot_utilization.values()
            ).to_dict(),
            "failure_reasons": dict(
                Counter(
                    result.failure_reason.value
                    for result in experiments
                    if result.failure_reason is not None
                )
            ),
            "illegal_assignment_count": sum(
                result.illegal_assignment_count for result in experiments
            ),
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / len(decisions) if decisions else 0.0,
            "oracle_regret": describe(
                _paired_oracle_regrets(experiments, exact_runs)
            ).to_dict(),
        }
    )
    return summary


def _paired_oracle_regrets(
    candidates: tuple[ExperimentResult, ...],
    exact_runs: tuple[OnlineSchedulerResult, ...],
) -> tuple[float, ...]:
    oracle = {run.experiment.pairing_key: run.experiment for run in exact_runs}
    regrets = []
    for candidate in candidates:
        reference = oracle[candidate.pairing_key]
        if not candidate.success or not reference.success:
            continue
        assert candidate.makespan is not None and reference.makespan is not None
        if reference.makespan <= 0:
            raise ValueError("exact MIP makespan must be positive")
        regrets.append((candidate.makespan - reference.makespan) / reference.makespan)
    return tuple(regrets)


def _validate_decision_timings(run: OnlineSchedulerResult) -> None:
    for record in run.decision_records:
        components = (
            record.encoder_time_seconds,
            record.scoring_time_seconds,
            record.decoder_time_seconds,
            record.repair_time_seconds,
            record.fallback_time_seconds,
            record.total_time_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in components):
            raise ValueError("decision timings must be non-negative and finite")


def _percentile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


__all__ = [
    "LATENCY_COMPONENTS",
    "LatencyEvaluationMethod",
    "LatencyPercentiles",
    "MDLatencyEvaluationReport",
    "latency_percentiles",
    "run_paired_latency_evaluation",
]
