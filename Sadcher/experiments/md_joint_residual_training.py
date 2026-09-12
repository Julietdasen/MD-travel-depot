"""Optional joint-batch correction head for the residual IL policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from experiments.joint_batch_features import pooled_batch_features
from experiments.md_residual_tail_training import ResidualCostPredictions, ResidualTailPolicy


@dataclass(frozen=True, slots=True)
class JointBatchPrediction:
    total: torch.Tensor
    task: torch.Tensor
    tail: torch.Tensor


class JointResidualTailPolicy(ResidualTailPolicy):
    """Edge residual policy extended with non-additive whole-batch correction."""

    def __init__(self, base_policy: nn.Module, *, pair_feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(base_policy, pair_feature_dim=pair_feature_dim, hidden_dim=hidden_dim, cost_to_score_weight=1.0)
        self.joint_head = nn.Sequential(
            nn.Linear(pair_feature_dim * 2 + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3)
        )
        nn.init.zeros_(self.joint_head[-1].weight)
        nn.init.zeros_(self.joint_head[-1].bias)

    def predict_batch(
        self,
        pair_representation: torch.Tensor,
        costs: ResidualCostPredictions,
        assignments: Sequence[tuple[int, int]],
    ) -> JointBatchPrediction:
        if not assignments:
            raise ValueError("joint batch must not be empty")
        edge = tuple(
            sum((head[0, robot, task] for robot, task in assignments), head.new_zeros(()))
            for head in (costs.total, costs.task, costs.tail)
        )
        correction = self.joint_head(pooled_batch_features(pair_representation, assignments))
        return JointBatchPrediction(*(base + delta for base, delta in zip(edge, correction, strict=True)))


def joint_batch_ranking_loss(
    predictions: Sequence[torch.Tensor],
    regrets: Sequence[float],
    *,
    margin: float = 0.1,
    tie_timestep: float = 1.0,
) -> torch.Tensor:
    """Prefer lower-regret legal batches while ignoring near ties."""
    if len(predictions) != len(regrets) or len(predictions) == 0:
        raise ValueError("predictions and regrets must be non-empty and aligned")
    terms = []
    for index, left in enumerate(regrets):
        for other, right in enumerate(regrets[index + 1:], index + 1):
            delta = left - right
            if abs(delta) <= tie_timestep:
                continue
            winner, loser = (index, other) if delta < 0 else (other, index)
            terms.append(torch.relu(predictions[winner] - predictions[loser] + margin))
    return torch.stack(terms).mean() if terms else predictions[0] * 0.0
