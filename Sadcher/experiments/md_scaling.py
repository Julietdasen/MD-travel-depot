"""Paired scaling/generalization aggregation with strict split isolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from experiments.protocol import (
    DatasetSplit,
    ExperimentResult,
    describe,
    summarize_results,
)


@dataclass(frozen=True, slots=True)
class ScalingCase:
    factor: str
    instance_id: str
    seed: int
    generation_config: Mapping[str, int | float]

    def __post_init__(self) -> None:
        if not self.factor.strip() or not self.instance_id.strip():
            raise ValueError("scaling factor and instance_id must not be empty")


@dataclass(frozen=True, slots=True)
class ScalingReport:
    cases: tuple[ScalingCase, ...]
    methods: tuple[str, ...]
    results: Mapping[str, tuple[ExperimentResult, ...]]

    def to_dict(self) -> dict:
        groups = {}
        for factor in sorted({case.factor for case in self.cases}):
            indices = [
                index for index, case in enumerate(self.cases) if case.factor == factor
            ]
            groups[factor] = {}
            for method in self.methods:
                runs = tuple(self.results[method][index] for index in indices)
                summary = summarize_results(runs).to_dict()
                successful = tuple(run for run in runs if run.success)
                summary["material_starvation"] = describe(
                    value
                    for run in successful
                    for value in run.material_starvation.values()
                ).to_dict()
                summary["robot_utilization"] = describe(
                    value
                    for run in successful
                    for value in run.robot_utilization.values()
                ).to_dict()
                summary["inference_time_seconds"] = describe(
                    run.inference_time_seconds for run in runs
                ).to_dict()
                groups[factor][method] = summary
        return {
            "paired_case_count": len(self.cases),
            "cases": [
                {
                    "factor": case.factor,
                    "instance_id": case.instance_id,
                    "seed": case.seed,
                    "generation_config": dict(case.generation_config),
                }
                for case in self.cases
            ],
            "groups": groups,
        }


def run_scaling_generalization(
    cases: Sequence[ScalingCase],
    methods: Sequence[str],
    runner: Callable[[str, ScalingCase], ExperimentResult],
    *,
    train_instance_ids: Sequence[str],
) -> ScalingReport:
    case_list = tuple(cases)
    method_list = tuple(methods)
    if not case_list or len({(case.instance_id, case.seed) for case in case_list}) != len(case_list):
        raise ValueError("scaling cases must have unique paired keys")
    if not method_list or len(set(method_list)) != len(method_list):
        raise ValueError("scaling methods must be unique")
    train_ids = set(train_instance_ids)
    if any(case.instance_id in train_ids for case in case_list):
        raise ValueError("scaling test instances overlap training instances")
    results = {}
    expected_keys = tuple((case.instance_id, case.seed) for case in case_list)
    for method in method_list:
        runs = tuple(runner(method, case) for case in case_list)
        if tuple(run.pairing_key for run in runs) != expected_keys:
            raise ValueError(f"method {method} changed paired case identity")
        if any(run.split is not DatasetSplit.TEST for run in runs):
            raise ValueError("scaling/generalization runs must use test split")
        results[method] = runs
    return ScalingReport(case_list, method_list, results)


__all__ = ["ScalingCase", "ScalingReport", "run_scaling_generalization"]
