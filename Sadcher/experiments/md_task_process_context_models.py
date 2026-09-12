"""Research-only equal-budget context ablation models for Ticket 43."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.md_enhanced_policy import (
    MDEnhancedPolicyConfig,
    MDEnhancedSchedulerNetwork,
    MDPolicyDiagnostics,
)
from models.md_policy import MD_OPPORTUNITY_FEATURE_COUNT, MDPolicyInputs
from models.scheduler_network import SchedulerNetwork


CONTEXT_ABLATION_VARIANTS: Final = (
    "current_pair_aware",
    "transport_context_only",
    "process_context_only",
    "transport_process_late_fusion",
    "transport_calibration_control",
)
PARAMETER_BUDGET: Final = 12459
_CONTEXT_INPUT_DIM: Final = 39


class _SingleContextHead(nn.Module):
    """Two-axis attention over one visible context with no inactive branch."""

    def __init__(self) -> None:
        super().__init__()
        hidden = 20
        refinement = 33
        self.input_projection = nn.Linear(_CONTEXT_INPUT_DIM, hidden)
        self.row_attention = nn.MultiheadAttention(
            hidden, 4, batch_first=True
        )
        self.column_attention = nn.MultiheadAttention(
            hidden, 4, batch_first=True
        )
        self.row_gate = nn.Parameter(torch.tensor(0.01))
        self.column_gate = nn.Parameter(torch.tensor(0.01))
        self.refinement = nn.Linear(hidden, refinement)
        self.output = nn.Linear(refinement, 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        tokens = F.relu(self.input_projection(context))
        batch_size, robot_count, task_count, hidden = tokens.shape
        row = tokens.reshape(batch_size * robot_count, task_count, hidden)
        row_update = _exclude_focal_attention(self.row_attention, row)
        row = row + torch.tanh(self.row_gate) * row_update
        row_context = row.reshape(
            batch_size, robot_count, task_count, hidden
        )
        column = row_context.permute(0, 2, 1, 3).reshape(
            batch_size * task_count, robot_count, hidden
        )
        column_update = _exclude_focal_attention(
            self.column_attention, column
        )
        column = column + torch.tanh(self.column_gate) * column_update
        fused = column.reshape(
            batch_size, task_count, robot_count, hidden
        ).permute(0, 2, 1, 3)
        return self.output(F.relu(self.refinement(fused))).squeeze(-1)


class _LateFusionContextHead(nn.Module):
    """Encode disjoint transport/process branches before parameter-free concat."""

    def __init__(self) -> None:
        super().__init__()
        branch_hidden = 12
        fusion_hidden = 103
        self.transport_projection = nn.Linear(
            _CONTEXT_INPUT_DIM, branch_hidden
        )
        self.process_projection = nn.Linear(
            _CONTEXT_INPUT_DIM, branch_hidden
        )
        self.transport_attention = nn.MultiheadAttention(
            branch_hidden, 4, batch_first=True
        )
        self.process_attention = nn.MultiheadAttention(
            branch_hidden, 4, batch_first=True
        )
        self.transport_gate = nn.Parameter(torch.tensor(0.01))
        self.process_gate = nn.Parameter(torch.tensor(0.01))
        self.fusion = nn.Linear(2 * branch_hidden, fusion_hidden)
        self.output = nn.Linear(fusion_hidden, 1)

    def forward(
        self,
        transport_context: torch.Tensor,
        process_context: torch.Tensor,
    ) -> torch.Tensor:
        transport = F.relu(self.transport_projection(transport_context))
        process = F.relu(self.process_projection(process_context))
        batch_size, robot_count, task_count, hidden = transport.shape

        transport_column = transport.permute(0, 2, 1, 3).reshape(
            batch_size * task_count, robot_count, hidden
        )
        transport_update = _exclude_focal_attention(
            self.transport_attention, transport_column
        )
        transport = (
            transport_column
            + torch.tanh(self.transport_gate) * transport_update
        ).reshape(batch_size, task_count, robot_count, hidden).permute(
            0, 2, 1, 3
        )

        process_row = process.reshape(
            batch_size * robot_count, task_count, hidden
        )
        process_update = _exclude_focal_attention(
            self.process_attention, process_row
        )
        process = (
            process_row + torch.tanh(self.process_gate) * process_update
        ).reshape(batch_size, robot_count, task_count, hidden)

        fused = torch.cat((transport, process), dim=-1)
        return self.output(F.relu(self.fusion(fused))).squeeze(-1)


class _ContextOnlySchedulerNetwork(SchedulerNetwork):
    masked_score = -1.0e9

    def __init__(self, variant: str) -> None:
        super().__init__(
            robot_input_dimensions=7,
            task_input_dimension=9,
            embed_dim=16,
            ff_dim=32,
            n_transformer_heads=4,
            n_transformer_layers=1,
            n_gatn_heads=4,
            n_gatn_layers=1,
            dropout=0.0,
            use_idle=False,
        )
        self.variant = variant
        self.context_head: nn.Module
        if variant == "transport_process_late_fusion":
            self.context_head = _LateFusionContextHead()
        else:
            self.context_head = _SingleContextHead()
        self._validation_config = MDEnhancedPolicyConfig(hidden_dim=16)

    def forward(
        self,
        robot_features: torch.Tensor,
        task_features: torch.Tensor,
        task_adjacencies: torch.Tensor | None = None,
        *,
        md_inputs: MDPolicyInputs | None = None,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(
            robot_features,
            task_features,
            task_adjacencies,
            md_inputs=md_inputs,
        ).scores

    def forward_with_diagnostics(
        self,
        robot_features: torch.Tensor,
        task_features: torch.Tensor,
        task_adjacencies: torch.Tensor | None = None,
        *,
        md_inputs: MDPolicyInputs | None = None,
    ) -> MDPolicyDiagnostics:
        if md_inputs is None:
            raise ValueError("md_inputs are required for context ablations")
        legacy_encoding = SchedulerNetwork.encode(
            self, robot_features, task_features, task_adjacencies
        )
        legacy_scores = SchedulerNetwork.score_encoding(
            self, legacy_encoding, robot_features, task_features
        )
        batch_size, robot_count, _ = robot_features.shape
        _, task_count, _ = task_features.shape
        md_inputs.validate(
            batch_size=batch_size,
            robot_count=robot_count,
            task_count=task_count,
            config=self._validation_config,
        )
        device = robot_features.device
        dtype = robot_features.dtype
        transport_context = _transport_context(
            legacy_encoding.robot_tokens,
            task_features,
            md_inputs,
            device=device,
            dtype=dtype,
        )
        process_context = _process_context(
            legacy_encoding.task_tokens,
            md_inputs,
            robot_count=robot_count,
            device=device,
            dtype=dtype,
        )
        if self.variant == "transport_process_late_fusion":
            head = self.context_head
            assert isinstance(head, _LateFusionContextHead)
            raw_residual = head(transport_context, process_context)
        else:
            head = self.context_head
            assert isinstance(head, _SingleContextHead)
            context = (
                transport_context
                if self.variant == "transport_context_only"
                else process_context
            )
            raw_residual = head(context)
        bounded_residual = 0.75 * torch.tanh(raw_residual)

        pair_metadata = md_inputs.pair_metadata.to(device=device, dtype=dtype)
        eta = pair_metadata[..., 0].clamp_min(0)
        physics = -0.5 * eta / (1.0 + eta)
        transport_mask = md_inputs.task_is_transport.to(device=device).unsqueeze(1)
        physics = torch.where(transport_mask, physics, torch.zeros_like(physics))
        raw_residual = torch.where(
            transport_mask, raw_residual, torch.zeros_like(raw_residual)
        )
        bounded_residual = torch.where(
            transport_mask, bounded_residual, torch.zeros_like(bounded_residual)
        )
        legacy_raw = legacy_scores[..., :task_count]
        legacy_transport = 0.15 * torch.tanh(legacy_raw / 0.15)
        legacy_component = torch.where(
            transport_mask, legacy_transport, legacy_raw
        )
        hard_mask = md_inputs.hard_feasibility_mask.to(device=device)
        combined_score = (
            legacy_component + physics + bounded_residual
        ).masked_fill(~hard_mask, self.masked_score)
        baseline_score = (legacy_component + physics).masked_fill(
            ~hard_mask, self.masked_score
        )
        action_flip = (
            torch.argmax(combined_score, dim=-1)
            != torch.argmax(baseline_score, dim=-1)
        )
        eligible = hard_mask & transport_mask
        eligible_float = eligible.to(dtype=dtype)
        eligible_count = eligible_float.sum(dim=(1, 2)).clamp_min(1.0)
        component_scales = torch.stack(
            tuple(
                (component.abs() * eligible_float).sum(dim=(1, 2))
                / eligible_count
                for component in (
                    legacy_component,
                    physics,
                    bounded_residual,
                )
            ),
            dim=-1,
        )
        saturation = bounded_residual.abs() >= 0.95 * 0.75
        residual_saturation_rate = (
            (saturation & eligible).to(dtype=dtype).sum(dim=(1, 2))
            / eligible_count
        )
        return MDPolicyDiagnostics(
            scores=combined_score,
            legacy_raw=legacy_raw,
            legacy_component=legacy_component,
            physics_utility=physics,
            raw_residual=raw_residual,
            bounded_residual=bounded_residual,
            combined_score=combined_score,
            action_flip=action_flip,
            component_scales=component_scales,
            residual_saturation_rate=residual_saturation_rate,
        )


class _TransportCalibrationControlNetwork(MDEnhancedSchedulerNetwork):
    """Current pair-aware context with a tied positive physics calibration."""

    def _physics_utility(self, eta: torch.Tensor) -> torch.Tensor:
        positive_scale = 2.0 * torch.sigmoid(
            self.opportunity_projection.weight.mean()
        )
        return positive_scale * super()._physics_utility(eta)


def build_context_ablation_model(
    variant: str,
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> nn.Module:
    if variant not in CONTEXT_ABLATION_VARIANTS:
        raise ValueError(f"unknown context ablation variant: {variant}")
    torch.manual_seed(seed)
    if variant in (
        "current_pair_aware",
        "transport_calibration_control",
    ):
        config = MDEnhancedPolicyConfig(
            hidden_dim=16,
            residual_bound=0.75,
            legacy_transport_bound=0.15,
            physics_scale=0.5,
            eta_scale=1.0,
            use_cross_attention=False,
            use_pair_aware_attention=True,
            use_signed_opportunity_prior=False,
        )
        model_class = (
            MDEnhancedSchedulerNetwork
            if variant == "current_pair_aware"
            else _TransportCalibrationControlNetwork
        )
        model: nn.Module = model_class(
            robot_input_dimensions=7,
            task_input_dimension=9,
            embed_dim=16,
            ff_dim=32,
            n_transformer_heads=4,
            n_transformer_layers=1,
            n_gatn_heads=4,
            n_gatn_layers=1,
            dropout=0.0,
            use_idle=False,
            md_config=config,
        )
    else:
        model = _ContextOnlySchedulerNetwork(variant)
    model = model.to(torch.device(device))
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if parameter_count != PARAMETER_BUDGET:
        raise RuntimeError(
            f"{variant} parameter budget is {parameter_count}, "
            f"expected {PARAMETER_BUDGET}"
        )
    return model


def _transport_context(
    robot_tokens: torch.Tensor,
    task_features: torch.Tensor,
    md_inputs: MDPolicyInputs,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, robot_count, hidden = robot_tokens.shape
    task_count = task_features.shape[1]
    expanded_robot = robot_tokens.unsqueeze(2).expand(
        -1, -1, task_count, -1
    )
    local_task = F.pad(
        task_features.to(device=device, dtype=dtype),
        (0, hidden - task_features.shape[-1]),
    )
    expanded_task = local_task.unsqueeze(1).expand(
        -1, robot_count, -1, -1
    )
    robot_metadata = md_inputs.robot_metadata.to(device=device, dtype=dtype)
    robot_transport = robot_metadata[..., 2:4].unsqueeze(2).expand(
        -1, -1, task_count, -1
    )
    pair_metadata = md_inputs.pair_metadata.to(device=device, dtype=dtype)
    context = torch.cat(
        (expanded_robot, expanded_task, robot_transport, pair_metadata),
        dim=-1,
    )
    transport_mask = md_inputs.task_is_transport.to(device=device)
    context = context * transport_mask.unsqueeze(1).unsqueeze(-1)
    assert context.shape == (batch_size, robot_count, task_count, 39)
    return context


def _process_context(
    task_tokens: torch.Tensor,
    md_inputs: MDPolicyInputs,
    *,
    robot_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, task_count, hidden = task_tokens.shape
    task_metadata = md_inputs.task_metadata.to(device=device, dtype=dtype)
    downstream_index = md_inputs.downstream_task_index.to(device=device)
    valid_downstream = downstream_index >= 0
    gather_token = downstream_index.clamp_min(0).unsqueeze(-1).expand(
        -1, -1, hidden
    )
    downstream_tokens = torch.gather(task_tokens, 1, gather_token)
    downstream_tokens = downstream_tokens * valid_downstream.unsqueeze(-1)
    gather_metadata = downstream_index.clamp_min(0).unsqueeze(-1).expand(
        -1, -1, task_metadata.shape[-1]
    )
    downstream_metadata = torch.gather(task_metadata, 1, gather_metadata)
    downstream_metadata = (
        downstream_metadata * valid_downstream.unsqueeze(-1)
    )
    opportunity = (
        torch.zeros(
            batch_size,
            robot_count,
            task_count,
            MD_OPPORTUNITY_FEATURE_COUNT,
            device=device,
            dtype=dtype,
        )
        if md_inputs.opportunity_context is None
        else md_inputs.opportunity_context.to(device=device, dtype=dtype)
    )
    focal_metadata = task_metadata.unsqueeze(1).expand(
        -1, robot_count, -1, -1
    )
    expanded_downstream_tokens = downstream_tokens.unsqueeze(1).expand(
        -1, robot_count, -1, -1
    )
    expanded_downstream_metadata = downstream_metadata.unsqueeze(1).expand(
        -1, robot_count, -1, -1
    )
    material_edges = md_inputs.typed_adjacency.to(
        device=device, dtype=dtype
    )[:, 1].sum(dim=-1)
    material_edges = material_edges.unsqueeze(1).unsqueeze(-1).expand(
        -1, robot_count, -1, -1
    )
    valid = valid_downstream.to(dtype=dtype).unsqueeze(1).unsqueeze(-1).expand(
        -1, robot_count, -1, -1
    )
    context = torch.cat(
        (
            focal_metadata,
            expanded_downstream_tokens,
            expanded_downstream_metadata,
            opportunity,
            material_edges,
            valid,
        ),
        dim=-1,
    )
    assert context.shape == (batch_size, robot_count, task_count, 39)
    return context


def _exclude_focal_attention(
    attention: nn.MultiheadAttention,
    tokens: torch.Tensor,
) -> torch.Tensor:
    axis_length = tokens.shape[1]
    if axis_length <= 1:
        return torch.zeros_like(tokens)
    exclude_self = torch.eye(
        axis_length,
        dtype=torch.bool,
        device=tokens.device,
    )
    update, _ = attention(
        tokens,
        tokens,
        tokens,
        attn_mask=exclude_self,
        need_weights=False,
    )
    return update


__all__ = [
    "CONTEXT_ABLATION_VARIANTS",
    "PARAMETER_BUDGET",
    "build_context_ablation_model",
]
