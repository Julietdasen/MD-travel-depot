"""Cross-attended MD policy with fixed physics and bounded neural residual."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.md_policy import (
    MD_OPPORTUNITY_DIRECTIONS,
    MD_OPPORTUNITY_FEATURE_COUNT,
    MDOpportunityFeature,
    MDPolicyConfig,
    MDPolicyInputs,
    MDSchedulerNetwork,
    _incoming_mean,
)
from models.scheduler_network import SchedulerNetwork


@dataclass(frozen=True, slots=True)
class MDEnhancedPolicyConfig(MDPolicyConfig):
    use_cross_attention: bool = True
    use_pair_aware_attention: bool = False
    cross_attention_heads: int = 4
    residual_bound: float = 0.25
    legacy_transport_bound: float = 0.25
    physics_scale: float = 1.0
    eta_scale: float = 1.0
    use_critical_path_proxy: bool = True
    use_last_material_blocker: bool = True
    use_coalition_context: bool = True
    use_capacity_scarcity: bool = True
    use_signed_opportunity_prior: bool = True

    def __post_init__(self) -> None:
        MDPolicyConfig.__post_init__(self)
        for name in ("use_cross_attention", "use_pair_aware_attention"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.use_cross_attention and self.use_pair_aware_attention:
            raise ValueError(
                "token cross-attention and pair-aware attention are mutually exclusive"
            )
        for name in (
            "use_critical_path_proxy",
            "use_last_material_blocker",
            "use_coalition_context",
            "use_capacity_scarcity",
            "use_signed_opportunity_prior",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if (
            isinstance(self.cross_attention_heads, bool)
            or not isinstance(self.cross_attention_heads, int)
            or self.cross_attention_heads <= 0
        ):
            raise ValueError("cross_attention_heads must be a positive integer")
        if self.hidden_dim % self.cross_attention_heads:
            raise ValueError("hidden_dim must be divisible by cross_attention_heads")
        if (
            isinstance(self.residual_bound, bool)
            or not isinstance(self.residual_bound, (int, float))
            or not math.isfinite(self.residual_bound)
            or self.residual_bound < 0
        ):
            raise ValueError("residual_bound must be non-negative and finite")
        for name in ("legacy_transport_bound", "physics_scale", "eta_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class MDPolicyDiagnostics:
    scores: torch.Tensor
    legacy_raw: torch.Tensor
    legacy_component: torch.Tensor
    physics_utility: torch.Tensor
    raw_residual: torch.Tensor
    bounded_residual: torch.Tensor
    combined_score: torch.Tensor
    action_flip: torch.Tensor
    component_scales: torch.Tensor
    residual_saturation_rate: torch.Tensor
    pair_representation: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "MDPolicyDiagnostics":
        """Return a copy with every diagnostic tensor on one device."""
        return MDPolicyDiagnostics(
            scores=self.scores.to(device),
            legacy_raw=self.legacy_raw.to(device),
            legacy_component=self.legacy_component.to(device),
            physics_utility=self.physics_utility.to(device),
            raw_residual=self.raw_residual.to(device),
            bounded_residual=self.bounded_residual.to(device),
            combined_score=self.combined_score.to(device),
            action_flip=self.action_flip.to(device),
            component_scales=self.component_scales.to(device),
            residual_saturation_rate=self.residual_saturation_rate.to(device),
            pair_representation=(
                None if self.pair_representation is None
                else self.pair_representation.to(device)
            ),
        )

    @property
    def downstream_residual(self) -> torch.Tensor:
        """Backward-compatible name for the bounded downstream residual."""

        return self.bounded_residual


class MDEnhancedSchedulerNetwork(MDSchedulerNetwork):
    """Use context attention without ever treating action masks as context masks."""

    def __init__(
        self,
        robot_input_dimensions,
        task_input_dimension,
        embed_dim,
        ff_dim,
        n_transformer_heads,
        n_transformer_layers,
        n_gatn_heads,
        n_gatn_layers,
        dropout=0.0,
        use_idle=True,
        *,
        md_config: MDEnhancedPolicyConfig | None = None,
    ):
        config = MDEnhancedPolicyConfig() if md_config is None else md_config
        if not isinstance(config, MDEnhancedPolicyConfig):
            raise TypeError("md_config must be an MDEnhancedPolicyConfig")
        super().__init__(
            robot_input_dimensions,
            task_input_dimension,
            embed_dim,
            ff_dim,
            n_transformer_heads,
            n_transformer_layers,
            n_gatn_heads,
            n_gatn_layers,
            dropout,
            use_idle,
            md_config=config,
        )
        hidden = config.hidden_dim
        heads = config.cross_attention_heads
        self.robot_to_task_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.task_to_robot_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.robot_to_task_gate = nn.Parameter(torch.zeros(()))
        self.task_to_robot_gate = nn.Parameter(torch.zeros(()))
        self.robot_backbone_adapter = nn.Linear(embed_dim, hidden)
        self.task_backbone_adapter = nn.Linear(embed_dim, hidden)
        nn.init.zeros_(self.transport_score_mlp[-1].weight)
        nn.init.zeros_(self.transport_score_mlp[-1].bias)
        self.transport_context_gate = nn.Parameter(torch.tensor(0.01))
        initial_magnitude = math.log(math.expm1(0.1))
        self.opportunity_raw_magnitudes = nn.Parameter(
            torch.full((MD_OPPORTUNITY_FEATURE_COUNT,), initial_magnitude)
        )
        if not config.use_signed_opportunity_prior:
            self.opportunity_raw_magnitudes.requires_grad_(False)
        self.opportunity_projection = nn.Linear(
            MD_OPPORTUNITY_FEATURE_COUNT, 1, bias=False
        )
        nn.init.zeros_(self.opportunity_projection.weight)

    @property
    def enhanced_config(self) -> MDEnhancedPolicyConfig:
        config = self.md_config
        assert isinstance(config, MDEnhancedPolicyConfig)
        return config

    def _exchange_context(
        self, robot_context: torch.Tensor, task_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enhanced_config.use_cross_attention:
            return robot_context, task_context
        robot_update, _ = self.robot_to_task_attention(
            robot_context, task_context, task_context, need_weights=False
        )
        task_update, _ = self.task_to_robot_attention(
            task_context, robot_context, robot_context, need_weights=False
        )
        robot_context = (
            robot_context
            + torch.tanh(self.robot_to_task_gate) * robot_update
        )
        task_context = (
            task_context
            + torch.tanh(self.task_to_robot_gate) * task_update
        )
        return robot_context, task_context

    @staticmethod
    def _without_focal_pair_attention(
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        """Attend to competitors while excluding the focal pair itself."""

        axis_length = query.shape[1]
        if axis_length <= 1:
            return torch.zeros_like(query)
        exclude_self = torch.eye(
            axis_length,
            dtype=torch.bool,
            device=query.device,
        )
        update, _ = attention(
            query,
            key_value,
            key_value,
            attn_mask=exclude_self,
            need_weights=False,
        )
        return update

    def _exchange_pair_context(
        self,
        local_pair_input: torch.Tensor,
        relational_pair_input: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate pair relations along task and robot competition axes."""

        pair_projection = self.transport_score_mlp[0]
        pair_activation = self.transport_score_mlp[1]
        local_tokens = pair_activation(pair_projection(local_pair_input))
        if not self.enhanced_config.use_pair_aware_attention:
            return local_tokens
        relational_tokens = pair_activation(
            pair_projection(relational_pair_input)
        )
        batch_size, robot_count, task_count, hidden = local_tokens.shape

        row_query = local_tokens.reshape(
            batch_size * robot_count, task_count, hidden
        )
        row_source = relational_tokens.reshape(
            batch_size * robot_count, task_count, hidden
        )
        row_update = self._without_focal_pair_attention(
            self.robot_to_task_attention,
            row_query,
            row_source,
        ).reshape(batch_size, robot_count, task_count, hidden)
        row_context = (
            local_tokens
            + torch.tanh(self.robot_to_task_gate) * row_update
        )

        column_query = row_context.permute(0, 2, 1, 3).reshape(
            batch_size * task_count, robot_count, hidden
        )
        column_source = (relational_tokens + row_update).permute(
            0, 2, 1, 3
        ).reshape(batch_size * task_count, robot_count, hidden)
        column_update = self._without_focal_pair_attention(
            self.task_to_robot_attention,
            column_query,
            column_source,
        ).reshape(batch_size, task_count, robot_count, hidden)
        return row_context + torch.tanh(
            self.task_to_robot_gate
        ) * column_update.permute(0, 2, 1, 3)

    def _physics_utility(self, eta: torch.Tensor) -> torch.Tensor:
        return (
            -self.enhanced_config.physics_scale
            * eta
            / (self.enhanced_config.eta_scale + eta)
        )

    def forward(
        self,
        robot_features,
        task_features,
        task_adjacencies=None,
        *,
        md_inputs: MDPolicyInputs | None = None,
    ):
        return self.forward_with_diagnostics(
            robot_features,
            task_features,
            task_adjacencies,
            md_inputs=md_inputs,
        ).scores

    def forward_with_diagnostics(
        self,
        robot_features,
        task_features,
        task_adjacencies=None,
        *,
        md_inputs: MDPolicyInputs | None = None,
    ) -> MDPolicyDiagnostics:
        legacy_encoding = SchedulerNetwork.encode(
            self, robot_features, task_features, task_adjacencies
        )
        legacy_scores = SchedulerNetwork.score_encoding(
            self, legacy_encoding, robot_features, task_features
        )
        batch_size, robot_count, _ = robot_features.shape
        _, task_count, _ = task_features.shape
        if not self.enhanced_config.enabled:
            legacy_task_scores = legacy_scores[..., :task_count]
            zeros = torch.zeros_like(legacy_task_scores)
            return MDPolicyDiagnostics(
                scores=legacy_scores,
                legacy_raw=legacy_task_scores,
                legacy_component=legacy_task_scores,
                physics_utility=zeros,
                raw_residual=zeros,
                bounded_residual=zeros,
                combined_score=legacy_task_scores,
                action_flip=torch.zeros(
                    batch_size,
                    robot_count,
                    dtype=torch.bool,
                    device=legacy_scores.device,
                ),
                component_scales=legacy_scores.new_zeros((batch_size, 3)),
                residual_saturation_rate=legacy_scores.new_zeros(batch_size),
            )
        if md_inputs is None:
            raise ValueError("md_inputs are required when MD policy is enabled")
        md_inputs.validate(
            batch_size=batch_size,
            robot_count=robot_count,
            task_count=task_count,
            config=self.enhanced_config,
        )
        device = robot_features.device
        dtype = robot_features.dtype
        robot_metadata = md_inputs.robot_metadata.to(device=device, dtype=dtype)
        task_metadata = md_inputs.task_metadata.to(device=device, dtype=dtype)
        pair_metadata = md_inputs.pair_metadata.to(device=device, dtype=dtype)
        opportunity_context = (
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
        opportunity_context = opportunity_context.clone()
        if not self.enhanced_config.use_critical_path_proxy:
            opportunity_context[..., MDOpportunityFeature.CRITICAL_PATH_PROXY] = 0
        if not self.enhanced_config.use_last_material_blocker:
            opportunity_context[..., MDOpportunityFeature.LAST_MATERIAL_BLOCKER] = 0
        if not self.enhanced_config.use_coalition_context:
            opportunity_context[
                ..., MDOpportunityFeature.COALITION_AVAILABILITY
            ] = 0
            opportunity_context[..., MDOpportunityFeature.COALITION_SCARCITY] = 0
        if not self.enhanced_config.use_capacity_scarcity:
            opportunity_context[..., MDOpportunityFeature.CAPACITY_SCARCITY] = 0

        if not self.enhanced_config.use_capacity_features:
            robot_metadata = robot_metadata.clone()
            task_metadata = task_metadata.clone()
            pair_metadata = pair_metadata.clone()
            robot_metadata[..., 2] = 0
            task_metadata[..., 1] = 0
            pair_metadata[..., 1:3] = 0
        if not self.enhanced_config.use_transport_eta:
            pair_metadata = pair_metadata.clone()
            pair_metadata[..., 0] = 0
            pair_metadata[..., 4] = 0

        robot_context = self.robot_md_adapter(robot_metadata) + self.robot_backbone_adapter(legacy_encoding.robot_tokens)
        task_context = self.task_md_adapter(task_metadata) + self.task_backbone_adapter(legacy_encoding.task_tokens)
        if self.enhanced_config.use_typed_edges:
            typed_adjacency = md_inputs.typed_adjacency.to(
                device=device, dtype=dtype
            )
            task_context = (
                task_context
                + _incoming_mean(
                    typed_adjacency[:, 0], self.normal_relation(task_context)
                )
                + _incoming_mean(
                    typed_adjacency[:, 1], self.material_relation(task_context)
                )
            )

        robot_context, task_context = self._exchange_context(
            robot_context, task_context
        )

        downstream_context = torch.zeros_like(task_context)
        if self.enhanced_config.use_downstream_encoding:
            downstream_index = md_inputs.downstream_task_index.to(device=device)
            valid_downstream = downstream_index >= 0
            gather_index = downstream_index.clamp_min(0).unsqueeze(-1).expand(
                -1, -1, task_context.shape[-1]
            )
            downstream_context = torch.gather(task_context, 1, gather_index)
            downstream_context = downstream_context * valid_downstream.unsqueeze(-1)

        expanded_robot = robot_context.unsqueeze(2).expand(
            -1, -1, task_count, -1
        )
        expanded_task = task_context.unsqueeze(1).expand(
            -1, robot_count, -1, -1
        )
        expanded_downstream = downstream_context.unsqueeze(1).expand(
            -1, robot_count, -1, -1
        )
        # ETA is reserved for the monotonic physics term; the neural residual
        # cannot learn an opposing shortcut through raw distance or duration.
        residual_pair_metadata = pair_metadata.clone()
        residual_pair_metadata[..., 0] = 0
        residual_pair_metadata[..., 4] = 0
        local_pair_input = torch.cat(
            (
                expanded_robot,
                expanded_task,
                expanded_downstream,
                residual_pair_metadata,
            ),
            dim=-1,
        )
        if self.enhanced_config.use_pair_aware_attention:
            relational_pair_input = torch.cat(
                (
                    expanded_robot,
                    expanded_task,
                    expanded_downstream,
                    pair_metadata,
                ),
                dim=-1,
            )
            pair_context = self._exchange_pair_context(
                local_pair_input,
                relational_pair_input,
            )
            neural_raw = self.transport_score_mlp[2](pair_context).squeeze(-1)
        else:
            neural_raw = self.transport_score_mlp(local_pair_input).squeeze(-1)
        directions = opportunity_context.new_tensor(MD_OPPORTUNITY_DIRECTIONS)
        residual_pair_representation = torch.cat(
            (pair_context if self.enhanced_config.use_pair_aware_attention else local_pair_input,
             pair_metadata,
             opportunity_context),
            dim=-1,
        )
        opportunity_weights = F.softplus(self.opportunity_raw_magnitudes)
        signed_opportunity_prior = (
            opportunity_context * directions * opportunity_weights
        ).sum(dim=-1)
        if not self.enhanced_config.use_signed_opportunity_prior:
            signed_opportunity_prior = torch.zeros_like(signed_opportunity_prior)
        learned_opportunity = self.opportunity_projection(
            opportunity_context
        ).squeeze(-1)
        raw_residual = (
            neural_raw
            + self.transport_context_gate
            * (expanded_task + expanded_downstream).mean(dim=-1)
            + signed_opportunity_prior
            + learned_opportunity
        )
        bounded_residual = self.enhanced_config.residual_bound * torch.tanh(
            raw_residual
        )

        eta = pair_metadata[..., 0].clamp_min(0)
        physics = self._physics_utility(eta)
        transport_mask = md_inputs.task_is_transport.to(device=device).unsqueeze(1)
        physics = torch.where(transport_mask, physics, torch.zeros_like(physics))
        raw_residual = torch.where(
            transport_mask, raw_residual, torch.zeros_like(raw_residual)
        )
        bounded_residual = torch.where(
            transport_mask, bounded_residual, torch.zeros_like(bounded_residual)
        )

        legacy_raw = legacy_scores[..., :task_count]
        legacy_transport = (
            self.enhanced_config.legacy_transport_bound
            * torch.tanh(
                legacy_raw / self.enhanced_config.legacy_transport_bound
            )
        )
        legacy_component = torch.where(
            transport_mask, legacy_transport, legacy_raw
        )
        combined_unmasked = legacy_component + physics + bounded_residual
        hard_mask = md_inputs.hard_feasibility_mask.to(device=device)
        combined_score = combined_unmasked.masked_fill(
            ~hard_mask, self.masked_score
        )
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
        saturation = (
            bounded_residual.abs()
            >= 0.95 * self.enhanced_config.residual_bound
        )
        residual_saturation_rate = (
            (saturation & eligible).to(dtype=dtype).sum(dim=(1, 2))
            / eligible_count
        )

        scores = combined_score
        if self.use_idle:
            scores = torch.cat(
                (scores, legacy_scores[..., task_count:]), dim=-1
            )
        return MDPolicyDiagnostics(
            scores=scores,
            legacy_raw=legacy_raw,
            legacy_component=legacy_component,
            physics_utility=physics,
            raw_residual=raw_residual,
            bounded_residual=bounded_residual,
            combined_score=combined_score,
            action_flip=action_flip,
            pair_representation=residual_pair_representation,
            component_scales=component_scales,
            residual_saturation_rate=residual_saturation_rate,
        )


__all__ = [
    "MDEnhancedPolicyConfig",
    "MDEnhancedSchedulerNetwork",
    "MDPolicyDiagnostics",
]
