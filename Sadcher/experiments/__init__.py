"""Shared experiment contracts and utilities."""

from experiments.protocol import (
    PAIRED_EVALUATION_SEEDS,
    PROTOCOL_VERSION,
    SPLIT_HASH_SEED,
    DatasetSplit,
    ExperimentResult,
    FailureReason,
    describe,
    paired_makespan_difference,
    summarize_results,
    task_level_split,
)

__all__ = [
    "PAIRED_EVALUATION_SEEDS",
    "PROTOCOL_VERSION",
    "SPLIT_HASH_SEED",
    "DatasetSplit",
    "ExperimentResult",
    "FailureReason",
    "describe",
    "paired_makespan_difference",
    "summarize_results",
    "task_level_split",
]
