"""Optional decoder-aware structured loss and V(s) auxiliary supervision."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_generation.md_expert_dataset import MDDecisionSample
from models.md_policy import MDPolicyInputs, MDSchedulerNetwork
from simulation_environment.domain_model import TaskType


@dataclass(frozen=True, slots=True)
class MDTrainingEnhancementConfig:
    use_structured_loss: bool = False
    use_value_head: bool = False
    structured_weight: float = 1.0
    value_weight: float = 0.1
    margin_weight: float = 1.0
    max_feasible_assignments: int = 100_000

    def __post_init__(self) -> None:
        for name in ("use_structured_loss", "use_value_head"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("structured_weight", "value_weight", "margin_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative and finite")
        if (
            isinstance(self.max_feasible_assignments, bool)
            or not isinstance(self.max_feasible_assignments, int)
            or self.max_feasible_assignments <= 0
        ):
            raise ValueError("max_feasible_assignments must be positive")


@dataclass(frozen=True, slots=True)
class OptionalTrainingLosses:
    total: torch.Tensor
    structured: torch.Tensor
    value: torch.Tensor
    predicted_value: torch.Tensor | None


class MDStateValueHead(nn.Module):
    """Type-conditioned attention pooling for remaining-makespan V(s)."""

    def __init__(self, *, hidden_dim: int):
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.robot_attention = nn.Linear(hidden_dim, 1)
        self.process_attention = nn.Linear(hidden_dim, 1)
        self.transport_attention = nn.Linear(hidden_dim, 1)
        self.value_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        robot_context: torch.Tensor,
        task_context: torch.Tensor,
        task_is_transport: torch.Tensor,
    ) -> torch.Tensor:
        if robot_context.ndim != 3 or task_context.ndim != 3:
            raise ValueError("value contexts must be batched token tensors")
        if tuple(task_is_transport.shape) != tuple(task_context.shape[:2]):
            raise ValueError("task_is_transport shape must match task tokens")
        robot_pool = _attention_pool(
            robot_context,
            self.robot_attention(robot_context).squeeze(-1),
            torch.ones(
                robot_context.shape[:2],
                dtype=torch.bool,
                device=robot_context.device,
            ),
        )
        transport_mask = task_is_transport.to(
            device=task_context.device, dtype=torch.bool
        )
        process_pool = _attention_pool(
            task_context,
            self.process_attention(task_context).squeeze(-1),
            ~transport_mask,
        )
        transport_pool = _attention_pool(
            task_context,
            self.transport_attention(task_context).squeeze(-1),
            transport_mask,
        )
        return self.value_mlp(
            torch.cat((robot_pool, process_pool, transport_pool), dim=-1)
        ).squeeze(-1)


def coalition_feasible_assignments(
    sample: MDDecisionSample,
    *,
    max_assignments: int = 100_000,
) -> tuple[torch.Tensor, ...]:
    """Enumerate the exact small-state feasible set without a solver."""

    if not isinstance(sample, MDDecisionSample):
        raise TypeError("sample must be an MDDecisionSample")
    hard_mask = torch.tensor(sample.hard_feasibility_mask, dtype=torch.bool)
    choices = [
        (-1, *torch.where(hard_mask[robot_index])[0].tolist())
        for robot_index in range(len(sample.robot_ids))
    ]
    combination_count = math.prod(len(options) for options in choices)
    if combination_count > max_assignments:
        raise ValueError(
            f"feasible enumeration would inspect {combination_count} assignments"
        )
    tasks = {task.task_id: task for task in sample.tasks}
    robots = {robot.robot_id: robot for robot in sample.robots}
    candidates = []
    for selected_columns in itertools.product(*choices):
        assignment = torch.zeros_like(hard_mask)
        for robot_index, task_column in enumerate(selected_columns):
            if task_column >= 0:
                assignment[robot_index, task_column] = True
        if _joint_assignment_feasible(
            assignment, sample, tasks=tasks, robots=robots
        ):
            candidates.append(assignment)
    if not candidates:
        raise RuntimeError("coalition feasible set is unexpectedly empty")
    return tuple(candidates)


def decoder_aware_structured_loss(
    scores: torch.Tensor,
    samples: Sequence[MDDecisionSample],
    *,
    margin_weight: float = 1.0,
    max_assignments: int = 100_000,
) -> torch.Tensor:
    """Structured hinge whose loss-augmented argmax is a detached constant."""

    batch = tuple(samples)
    if scores.ndim != 3 or scores.shape[0] != len(batch):
        raise ValueError("scores must have one robot-task matrix per sample")
    losses = []
    for index, sample in enumerate(batch):
        expected_shape = (len(sample.robot_ids), len(sample.task_ids))
        if tuple(scores[index].shape) != expected_shape:
            raise ValueError("score shape does not match decision sample")
        expert = torch.tensor(
            sample.expert_assignment,
            dtype=scores.dtype,
            device=scores.device,
        )
        candidates = coalition_feasible_assignments(
            sample, max_assignments=max_assignments
        )
        detached_scores = scores[index].detach()
        best_candidate = None
        best_objective = float("-inf")
        best_delta = 0.0
        for candidate_cpu in candidates:
            candidate = candidate_cpu.to(
                device=scores.device, dtype=scores.dtype
            )
            delta = margin_weight * float(torch.abs(candidate - expert).sum())
            objective = float((detached_scores * candidate).sum()) + delta
            if objective > best_objective:
                best_objective = objective
                best_candidate = candidate.detach()
                best_delta = delta
        assert best_candidate is not None
        hinge = (
            (scores[index] * (best_candidate - expert)).sum() + best_delta
        )
        losses.append(torch.relu(hinge))
    return torch.stack(losses).mean()


def compute_optional_training_losses(
    scores: torch.Tensor,
    samples: Sequence[MDDecisionSample],
    *,
    policy: MDSchedulerNetwork,
    md_inputs: MDPolicyInputs,
    config: MDTrainingEnhancementConfig,
    value_head: MDStateValueHead | None = None,
) -> OptionalTrainingLosses:
    zero = scores.sum() * 0.0
    structured = zero
    value = zero
    prediction = None
    if config.use_structured_loss:
        structured = decoder_aware_structured_loss(
            scores,
            samples,
            margin_weight=config.margin_weight,
            max_assignments=config.max_feasible_assignments,
        )
    if config.use_value_head:
        if value_head is None:
            raise ValueError("value_head is required when use_value_head=true")
        robot_metadata = md_inputs.robot_metadata.to(
            device=scores.device, dtype=scores.dtype
        )
        task_metadata = md_inputs.task_metadata.to(
            device=scores.device, dtype=scores.dtype
        )
        robot_context = policy.robot_md_adapter(robot_metadata)
        task_context = policy.task_md_adapter(task_metadata)
        prediction = value_head(
            robot_context,
            task_context,
            md_inputs.task_is_transport.to(scores.device),
        )
        labels = torch.tensor(
            [sample.remaining_makespan for sample in samples],
            dtype=scores.dtype,
            device=scores.device,
        )
        labels = labels / labels.max().clamp_min(1.0)
        value = F.huber_loss(prediction, labels)
    total = config.structured_weight * structured + config.value_weight * value
    return OptionalTrainingLosses(total, structured, value, prediction)


def _joint_assignment_feasible(
    assignment: torch.Tensor,
    sample: MDDecisionSample,
    *,
    tasks,
    robots,
) -> bool:
    for task_column, task_id in enumerate(sample.task_ids):
        selected_rows = torch.where(assignment[:, task_column])[0].tolist()
        if not selected_rows:
            continue
        task = tasks[task_id]
        if task.task_type == TaskType.TRANSPORT.value:
            if len(selected_rows) > 1:
                return False
            continue
        coverage = [False] * len(task.requirements)
        selected_robot_ids = [sample.robot_ids[row] for row in selected_rows]
        for robot_id in selected_robot_ids:
            capabilities = robots[robot_id].capabilities
            coverage = [
                current or capability
                for current, capability in zip(
                    coverage, capabilities, strict=True
                )
            ]
        if not all(
            not required or coverage[index]
            for index, required in enumerate(task.requirements)
        ):
            return False
        if any(task.requirements):
            for removed_id in selected_robot_ids:
                reduced = [False] * len(task.requirements)
                for robot_id in selected_robot_ids:
                    if robot_id == removed_id:
                        continue
                    reduced = [
                        current or capability
                        for current, capability in zip(
                            reduced, robots[robot_id].capabilities, strict=True
                        )
                    ]
                if all(
                    not required or reduced[index]
                    for index, required in enumerate(task.requirements)
                ):
                    return False
        elif len(selected_rows) != 1:
            return False
    return True


def _attention_pool(
    tokens: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    has_tokens = mask.any(dim=1, keepdim=True)
    safe_logits = torch.where(has_tokens, masked_logits, torch.zeros_like(logits))
    weights = torch.softmax(safe_logits, dim=1) * mask
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


__all__ = [
    "MDStateValueHead",
    "MDTrainingEnhancementConfig",
    "OptionalTrainingLosses",
    "coalition_feasible_assignments",
    "compute_optional_training_losses",
    "decoder_aware_structured_loss",
]
