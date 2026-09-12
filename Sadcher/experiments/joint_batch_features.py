"""Joint batch feature helpers for residual IL."""

from __future__ import annotations

from typing import Sequence

import torch


def batch_incidence(
    assignments: Sequence[tuple[int, int]],
    robot_count: int,
    task_count: int,
) -> torch.Tensor:
    """Build a deterministic robot-task incidence matrix."""
    if not assignments:
        raise ValueError("joint batch must not be empty")
    result = torch.zeros((robot_count, task_count), dtype=torch.float32)
    for robot, task in assignments:
        if not (0 <= robot < robot_count and 0 <= task < task_count):
            raise ValueError("assignment is outside the feature matrix")
        result[robot, task] = 1.0
    return result


def pooled_batch_features(
    pair_representation: torch.Tensor,
    assignments: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Return mean pair, mean global context and batch metadata."""
    if pair_representation.ndim != 4 or pair_representation.shape[0] != 1:
        raise ValueError("pair representation must have shape [1, R, T, D]")
    if not assignments:
        raise ValueError("joint batch must not be empty")
    selected = torch.stack([pair_representation[0, r, t] for r, t in assignments])
    pooled = selected.mean(dim=0)
    context = pair_representation[0].mean(dim=(0, 1))
    metadata = pair_representation.new_tensor(
        [len(assignments), len({task for _, task in assignments}), len({robot for robot, _ in assignments})]
    )
    return torch.cat((pooled, context, metadata))
