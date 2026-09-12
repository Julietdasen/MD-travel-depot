"""Deterministic, solver-free CPU smoke checks for the MD policy."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch
import torch.nn.functional as F

from data_generation.md_expert_dataset import MDDecisionSample
from models.md_legacy_features import legacy_policy_inputs_from_samples
from models.md_policy import MDPolicyConfig, MDSchedulerNetwork
from models.md_policy_features import md_policy_inputs_from_samples


@dataclass(frozen=True, slots=True)
class MDPolicySmokeResult:
    sample_count: int
    train_loss: float
    validation_loss: float
    forward_finite: bool
    backward_finite: bool
    validation_finite: bool
    illegal_assignment_count: int
    selected_actions: tuple[tuple[int, int, int], ...]
    hard_mask_digest: str
    split_digest: str


def independent_ablation_configs(
    base: MDPolicyConfig,
) -> dict[str, MDPolicyConfig]:
    if not isinstance(base, MDPolicyConfig):
        raise TypeError("base must be an MDPolicyConfig")
    return {
        "typed_edges": replace(base, use_typed_edges=False),
        "downstream": replace(base, use_downstream_encoding=False),
        "eta": replace(base, use_transport_eta=False),
        "capacity": replace(base, use_capacity_features=False),
    }


def run_md_policy_smoke(
    samples: Sequence[MDDecisionSample],
    *,
    md_config: MDPolicyConfig,
    seed: int,
) -> MDPolicySmokeResult:
    """Run one deterministic IL step and a masked one-step rollout on CPU."""

    batch = tuple(samples)
    if not batch:
        raise ValueError("samples must not be empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not md_config.enabled:
        raise ValueError("MD smoke requires an enabled MDPolicyConfig")
    torch.manual_seed(seed)
    legacy = legacy_policy_inputs_from_samples(batch)
    md_inputs = md_policy_inputs_from_samples(batch)
    model = MDSchedulerNetwork(
        robot_input_dimensions=legacy.robot_features.shape[-1],
        task_input_dimension=legacy.task_features.shape[-1],
        embed_dim=16,
        ff_dim=32,
        n_transformer_heads=4,
        n_transformer_layers=1,
        n_gatn_heads=4,
        n_gatn_layers=1,
        dropout=0.0,
        use_idle=False,
        md_config=md_config,
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    targets = torch.tensor(
        [sample.expert_assignment for sample in batch], dtype=torch.float32
    )
    feasible = md_inputs.hard_feasibility_mask

    model.train()
    scores = model(
        legacy.robot_features,
        legacy.task_features,
        legacy.task_adjacency,
        md_inputs=md_inputs,
    )
    if not torch.any(feasible):
        raise ValueError("smoke batch must contain at least one feasible pair")
    loss = F.binary_cross_entropy_with_logits(scores[feasible], targets[feasible])
    forward_finite = bool(torch.isfinite(scores).all() and torch.isfinite(loss))
    optimizer.zero_grad()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    backward_finite = bool(
        gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    )
    optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_scores = model(
            legacy.robot_features,
            legacy.task_features,
            legacy.task_adjacency,
            md_inputs=md_inputs,
        )
        validation_loss = F.binary_cross_entropy_with_logits(
            validation_scores[feasible], targets[feasible]
        )
    validation_finite = bool(
        torch.isfinite(validation_scores).all()
        and torch.isfinite(validation_loss)
    )

    selected = []
    illegal_count = 0
    for batch_index, sample in enumerate(batch):
        for robot_index, robot_id in enumerate(sample.robot_ids):
            legal = feasible[batch_index, robot_index]
            if not torch.any(legal):
                continue
            legal_scores = validation_scores[batch_index, robot_index].masked_fill(
                ~legal, model.masked_score
            )
            task_index = int(torch.argmax(legal_scores).item())
            if not bool(legal[task_index]):
                illegal_count += 1
                continue
            selected.append((batch_index, robot_id, sample.task_ids[task_index]))

    train_loss = float(loss.detach())
    validation_loss_value = float(validation_loss)
    if not math.isfinite(train_loss) or not math.isfinite(validation_loss_value):
        raise RuntimeError("MD policy smoke produced a non-finite loss")
    return MDPolicySmokeResult(
        sample_count=len(batch),
        train_loss=train_loss,
        validation_loss=validation_loss_value,
        forward_finite=forward_finite,
        backward_finite=backward_finite,
        validation_finite=validation_finite,
        illegal_assignment_count=illegal_count,
        selected_actions=tuple(selected),
        hard_mask_digest=_tensor_digest(feasible),
        split_digest=_split_digest(batch),
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()
    return f"sha256:{digest}"


def _split_digest(samples: tuple[MDDecisionSample, ...]) -> str:
    value = "|".join(
        f"{sample.instance_id}:{sample.split.value}" for sample in samples
    )
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


__all__ = [
    "MDPolicySmokeResult",
    "independent_ablation_configs",
    "run_md_policy_smoke",
]
