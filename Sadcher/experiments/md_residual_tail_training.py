"""Training utilities for augmenting C0 with residual edge cost heads."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data_generation.md_residual_dataset import ResidualBatchLabel


@dataclass(frozen=True, slots=True)
class ResidualCostPredictions:
    total: torch.Tensor
    task: torch.Tensor
    tail: torch.Tensor


class ResidualTailPolicy(nn.Module):
    """Wrap any existing C0 policy while preserving its ordinary forward API."""

    def __init__(self, base_policy: nn.Module, robot_feature_dim: int | None = None, task_feature_dim: int | None = None, hidden_dim: int = 64, *, pair_feature_dim: int | None = None, cost_to_score_weight: float = 0.0) -> None:
        super().__init__()
        self.base_policy = base_policy
        if not isinstance(cost_to_score_weight, (int, float)) or cost_to_score_weight < 0:
            raise ValueError("cost_to_score_weight must be non-negative")
        self.cost_to_score_weight = float(cost_to_score_weight)
        width = pair_feature_dim if pair_feature_dim is not None else int(robot_feature_dim or 0) + int(task_feature_dim or 0)
        if width <= 0:
            raise ValueError("a positive pair feature dimension is required")
        self.cost_heads = nn.ModuleList(nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)) for _ in range(3))

    def _scores_and_pair(self, robot_features, task_features, task_adjacencies=None, *, md_inputs=None):
        if hasattr(self.base_policy, "forward_with_diagnostics"):
            diagnostics = self.base_policy.forward_with_diagnostics(
                robot_features,
                task_features,
                task_adjacencies,
                md_inputs=md_inputs,
            )
            pair = getattr(diagnostics, "pair_representation", None)
            if pair is not None:
                return diagnostics.scores, pair
        scores = self.base_policy(
            robot_features,
            task_features,
            task_adjacencies,
            md_inputs=md_inputs,
        )
        if hasattr(self.base_policy, "pair_representation"):
            pair = self.base_policy.pair_representation(
                robot_features,
                task_features,
                task_adjacencies,
                md_inputs=md_inputs,
            )
        else:
            robot = robot_features.unsqueeze(2).expand(-1, -1, task_features.shape[1], -1)
            task = task_features.unsqueeze(1).expand(-1, robot_features.shape[1], -1, -1)
            pair = torch.cat((robot, task), dim=-1)
        return scores, pair

    def forward(self, robot_features, task_features, task_adjacencies=None, *, md_inputs=None):
        scores, pair = self._scores_and_pair(
            robot_features,
            task_features,
            task_adjacencies,
            md_inputs=md_inputs,
        )
        if self.cost_to_score_weight == 0.0:
            return scores
        total = self.cost_heads[0](pair).squeeze(-1)
        adjusted = scores[..., :total.shape[-1]] - self.cost_to_score_weight * total
        return torch.cat((adjusted, scores[..., total.shape[-1]:]), dim=-1)

    def forward_with_costs(self, robot_features, task_features, task_adjacencies=None, *, md_inputs=None) -> tuple[torch.Tensor, ResidualCostPredictions]:
        scores, pair = self._scores_and_pair(
            robot_features,
            task_features,
            task_adjacencies,
            md_inputs=md_inputs,
        )
        costs = ResidualCostPredictions(*(head(pair).squeeze(-1) for head in self.cost_heads))
        if self.cost_to_score_weight:
            total = costs.total
            adjusted = scores[..., :total.shape[-1]] - self.cost_to_score_weight * total
            scores = torch.cat((adjusted, scores[..., total.shape[-1]:]), dim=-1)
        return scores, costs


def residual_tail_loss(predicted: ResidualCostPredictions, target: ResidualCostPredictions, *, rank_loss: torch.Tensor | None = None, c0_loss: torch.Tensor | None = None, regression_masks: ResidualCostPredictions | None = None) -> torch.Tensor:
    """Smooth-L1 regression plus optional ranking and legacy terms."""
    values = []
    for actual, expected, mask in zip((predicted.total, predicted.task, predicted.tail), (target.total, target.task, target.tail), (None if regression_masks is None else (regression_masks.total, regression_masks.task, regression_masks.tail)), strict=True):
        if mask is None:
            values.append(nn.functional.smooth_l1_loss(actual, expected))
        elif bool(mask.any()):
            values.append(nn.functional.smooth_l1_loss(actual[mask], expected[mask]))
    regression = sum(values, torch.zeros((), device=predicted.total.device))
    result = 0.25 * regression
    if rank_loss is not None:
        result = result + rank_loss
    if c0_loss is not None:
        result = result + c0_loss
    return result


def save_residual_checkpoint(path: str | Path, policy: ResidualTailPolicy, optimizer: torch.optim.Optimizer, *, epoch: int, metrics: dict[str, Any] | None = None) -> None:
    torch.save({"model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "metrics": {} if metrics is None else metrics}, path)


def load_residual_checkpoint(path: str | Path, policy: ResidualTailPolicy, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def batch_margin_ranking_loss(edge_scores: torch.Tensor, labels: tuple[ResidualBatchLabel, ...], *, margin: float = 0.1, tie_timestep: float = 1.0) -> torch.Tensor:
    """Rank lower-regret batches above worse batches; ties are omitted."""
    terms = []
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            delta = float(left.regret) - float(right.regret)
            if abs(delta) <= tie_timestep:
                continue
            left_score = sum(edge_scores[r, t] for r, t in left.batch.assignments)
            right_score = sum(edge_scores[r, t] for r, t in right.batch.assignments)
            if delta < 0:
                terms.append(torch.relu(torch.as_tensor(margin, device=edge_scores.device) - left_score + right_score))
            else:
                terms.append(torch.relu(torch.as_tensor(margin, device=edge_scores.device) - right_score + left_score))
    return torch.stack(terms).mean() if terms else edge_scores.sum() * 0.0


def select_development_checkpoint(rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("development rows must not be empty")
    return min(rows, key=lambda row: (float(row["forced_action_regret"]), float(row["terminal_return_tail"]), int(row["epoch"])))
