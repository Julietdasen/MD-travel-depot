"""Paired main-comparison reporting for MD schedulers and baselines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from experiments.protocol import (
    DescriptiveStatistics,
    ExperimentResult,
    PairedMakespanSummary,
    describe,
    paired_makespan_difference,
    summarize_results,
)


class MethodSemantics(str, Enum):
    ORIGINAL_SADCHER = "process_only_original_sadcher"
    CARRY_AS_SKILL = "carry_without_material_delivery_semantics"
    MATERIAL_SOLO = "explicit_material_solo_transport"
    FULL_MD_POLICY = "full_md_policy"
    MD_GREEDY = "md_greedy"
    MD_METAHEURISTIC = "md_metaheuristic"
    MD_ORACLE = "md_oracle"


@dataclass(frozen=True, slots=True)
class ComparisonMethod:
    name: str
    semantics: MethodSemantics
    runner: Callable[[str, int], ExperimentResult]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("comparison method name must not be empty")
        if not isinstance(self.semantics, MethodSemantics):
            raise TypeError("semantics must be a MethodSemantics")


@dataclass(frozen=True, slots=True)
class MDComparisonReport:
    paired_keys: tuple[tuple[str, int], ...]
    methods: tuple[ComparisonMethod, ...]
    results: dict[str, tuple[ExperimentResult, ...]]

    def compare(self, baseline: str, candidate: str) -> PairedMakespanSummary:
        return paired_makespan_difference(
            self.results[baseline], self.results[candidate]
        )

    def to_dict(self) -> dict:
        return {
            "paired_keys": [list(key) for key in self.paired_keys],
            "methods": {
                method.name: _method_summary(
                    method.semantics, self.results[method.name]
                )
                for method in self.methods
            },
            "raw_results": {
                name: [result.to_dict() for result in results]
                for name, results in self.results.items()
            },
        }


def run_paired_md_comparison(
    cases: Sequence[tuple[str, int]],
    methods: Sequence[ComparisonMethod],
) -> MDComparisonReport:
    paired_keys = tuple(cases)
    if not paired_keys or len(set(paired_keys)) != len(paired_keys):
        raise ValueError("comparison cases must be non-empty unique paired keys")
    method_list = tuple(methods)
    if not method_list or len({method.name for method in method_list}) != len(method_list):
        raise ValueError("comparison methods must have unique names")
    results = {}
    for method in method_list:
        runs = tuple(method.runner(instance_id, seed) for instance_id, seed in paired_keys)
        if tuple(run.pairing_key for run in runs) != paired_keys:
            raise ValueError(f"method {method.name} did not preserve paired case order")
        results[method.name] = runs
    return MDComparisonReport(paired_keys, method_list, results)


def _method_summary(
    semantics: MethodSemantics, results: tuple[ExperimentResult, ...]
) -> dict:
    summary = summarize_results(results).to_dict()
    successful = tuple(result for result in results if result.success)
    summary.update(
        {
            "semantics": semantics.value,
            "material_starvation": _flatten_metric(
                successful, "material_starvation"
            ).to_dict(),
            "robot_utilization": _flatten_metric(
                successful, "robot_utilization"
            ).to_dict(),
            "inference_time_seconds": describe(
                result.inference_time_seconds for result in results
            ).to_dict(),
            "wall_time_seconds": describe(
                result.wall_time_seconds for result in results
            ).to_dict(),
            "gurobi_gap": describe(
                float(result.metadata["gurobi_gap"])
                for result in results
                if result.metadata.get("gurobi_gap") is not None
            ).to_dict(),
        }
    )
    return summary


def _flatten_metric(
    results: tuple[ExperimentResult, ...], attribute: str
) -> DescriptiveStatistics:
    values = []
    for result in results:
        metric = getattr(result, attribute)
        values.extend(metric.values())
    return describe(values)


__all__ = [
    "ComparisonMethod",
    "MDComparisonReport",
    "MethodSemantics",
    "run_paired_md_comparison",
]
