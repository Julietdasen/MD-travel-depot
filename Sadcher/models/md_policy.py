"""Legacy-preserving material-delivery policy adapters and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

import torch
import torch.nn as nn

from models.scheduler_network import SchedulerNetwork


class MDOpportunityFeature(IntEnum):
    CRITICAL_PATH_PROXY = 0
    LAST_MATERIAL_BLOCKER = 1
    COALITION_AVAILABILITY = 2
    COALITION_SCARCITY = 3
    CAPACITY_SCARCITY = 4


MD_OPPORTUNITY_FEATURE_COUNT = len(MDOpportunityFeature)
MD_OPPORTUNITY_DIRECTIONS = (1.0, 1.0, 1.0, -1.0, 1.0)
MDOpportunityValues = tuple[float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class MDOpportunityContext:
    critical_path_proxy: float = 0.0
    last_material_blocker: float = 0.0
    coalition_availability: float = 0.0
    coalition_scarcity: float = 0.0
    capacity_scarcity: float = 0.0

    @property
    def values(self) -> MDOpportunityValues:
        return (
            self.critical_path_proxy,
            self.last_material_blocker,
            self.coalition_availability,
            self.coalition_scarcity,
            self.capacity_scarcity,
        )


@dataclass(frozen=True, slots=True)
class MDPolicyConfig:
    enabled: bool = True
    robot_metadata_dim: int = 4
    task_metadata_dim: int = 8
    pair_metadata_dim: int = 5
    hidden_dim: int = 64
    use_typed_edges: bool = True
    use_downstream_encoding: bool = True
    use_transport_eta: bool = True
    use_capacity_features: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "use_typed_edges",
            "use_downstream_encoding",
            "use_transport_eta",
            "use_capacity_features",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in (
            "robot_metadata_dim",
            "task_metadata_dim",
            "pair_metadata_dim",
            "hidden_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.robot_metadata_dim < 4:
            raise ValueError("robot_metadata_dim must include the four MD features")
        if self.task_metadata_dim < 8:
            raise ValueError("task_metadata_dim must include the eight MD features")
        if self.pair_metadata_dim < 5:
            raise ValueError("pair_metadata_dim must include the five MD features")


@dataclass(frozen=True, slots=True)
class MDPolicyInputs:
    """Separate metadata tensors; legacy 7/9 feature tensors stay unchanged."""

    robot_metadata: torch.Tensor
    task_metadata: torch.Tensor
    pair_metadata: torch.Tensor
    task_is_transport: torch.Tensor
    typed_adjacency: torch.Tensor
    downstream_task_index: torch.Tensor
    hard_feasibility_mask: torch.Tensor
    opportunity_context: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "MDPolicyInputs":
        """Return a copy with every metadata tensor on one device."""
        return MDPolicyInputs(
            robot_metadata=self.robot_metadata.to(device),
            task_metadata=self.task_metadata.to(device),
            pair_metadata=self.pair_metadata.to(device),
            task_is_transport=self.task_is_transport.to(device),
            typed_adjacency=self.typed_adjacency.to(device),
            downstream_task_index=self.downstream_task_index.to(device),
            hard_feasibility_mask=self.hard_feasibility_mask.to(device),
            opportunity_context=(
                None
                if self.opportunity_context is None
                else self.opportunity_context.to(device)
            ),
        )

    def validate(
        self,
        *,
        batch_size: int,
        robot_count: int,
        task_count: int,
        config: MDPolicyConfig,
    ) -> None:
        expected = {
            "robot_metadata": (
                self.robot_metadata,
                (batch_size, robot_count, config.robot_metadata_dim),
            ),
            "task_metadata": (
                self.task_metadata,
                (batch_size, task_count, config.task_metadata_dim),
            ),
            "pair_metadata": (
                self.pair_metadata,
                (
                    batch_size,
                    robot_count,
                    task_count,
                    config.pair_metadata_dim,
                ),
            ),
            "task_is_transport": (
                self.task_is_transport,
                (batch_size, task_count),
            ),
            "typed_adjacency": (
                self.typed_adjacency,
                (batch_size, 2, task_count, task_count),
            ),
            "downstream_task_index": (
                self.downstream_task_index,
                (batch_size, task_count),
            ),
            "hard_feasibility_mask": (
                self.hard_feasibility_mask,
                (batch_size, robot_count, task_count),
            ),
        }
        for name, (tensor, shape) in expected.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.opportunity_context is not None:
            if not isinstance(self.opportunity_context, torch.Tensor):
                raise TypeError("opportunity_context must be a torch.Tensor")
            opportunity_shape = (
                batch_size,
                robot_count,
                task_count,
                MD_OPPORTUNITY_FEATURE_COUNT,
            )
            if tuple(self.opportunity_context.shape) != opportunity_shape:
                raise ValueError(
                    f"opportunity_context must have shape {opportunity_shape}"
                )
            if not torch.isfinite(self.opportunity_context).all():
                raise ValueError("opportunity_context must contain finite values")
        if self.task_is_transport.dtype is not torch.bool:
            raise ValueError("task_is_transport must be boolean")
        if self.hard_feasibility_mask.dtype is not torch.bool:
            raise ValueError("hard_feasibility_mask must be boolean")
        if self.downstream_task_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("downstream_task_index must be an integer tensor")
        if torch.any(self.downstream_task_index >= task_count) or torch.any(
            self.downstream_task_index < -1
        ):
            raise ValueError("downstream_task_index values must be -1 or task indices")


class MDSchedulerNetwork(SchedulerNetwork):
    """Add transport-only scoring while keeping every process score on legacy path."""

    masked_score = -1.0e9

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
        md_config: MDPolicyConfig | None = None,
    ):
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
        )
        self.md_config = MDPolicyConfig() if md_config is None else md_config
        hidden = self.md_config.hidden_dim
        self.robot_md_adapter = _adapter(
            self.md_config.robot_metadata_dim, hidden
        )
        self.task_md_adapter = _adapter(self.md_config.task_metadata_dim, hidden)
        self.normal_relation = nn.Linear(hidden, hidden, bias=False)
        self.material_relation = nn.Linear(hidden, hidden, bias=False)
        self.transport_score_mlp = nn.Sequential(
            nn.Linear(3 * hidden + self.md_config.pair_metadata_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def load_legacy_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> nn.modules.module._IncompatibleKeys:
        """Load all shape-compatible legacy keys without changing their names."""

        normalized = {
            (key[len("scheduler_net.") :] if key.startswith("scheduler_net.") else key): value
            for key, value in state_dict.items()
        }
        current = self.state_dict()
        compatible = {
            key: value
            for key, value in normalized.items()
            if key in current and current[key].shape == value.shape
        }
        return self.load_state_dict(compatible, strict=False)

    def forward(
        self,
        robot_features,
        task_features,
        task_adjacencies=None,
        *,
        md_inputs: MDPolicyInputs | None = None,
    ):
        legacy_scores = super().forward(
            robot_features, task_features, task_adjacencies
        )
        if not self.md_config.enabled:
            return legacy_scores
        if md_inputs is None:
            raise ValueError("md_inputs are required when MD policy is enabled")

        batch_size, robot_count, _ = robot_features.shape
        _, task_count, _ = task_features.shape
        md_inputs.validate(
            batch_size=batch_size,
            robot_count=robot_count,
            task_count=task_count,
            config=self.md_config,
        )
        device = robot_features.device
        dtype = robot_features.dtype
        robot_metadata = md_inputs.robot_metadata.to(device=device, dtype=dtype)
        task_metadata = md_inputs.task_metadata.to(device=device, dtype=dtype)
        pair_metadata = md_inputs.pair_metadata.to(device=device, dtype=dtype)

        if not self.md_config.use_capacity_features:
            robot_metadata = robot_metadata.clone()
            task_metadata = task_metadata.clone()
            pair_metadata = pair_metadata.clone()
            robot_metadata[..., 2] = 0
            task_metadata[..., 1] = 0
            pair_metadata[..., 1:3] = 0
        if not self.md_config.use_transport_eta:
            pair_metadata = pair_metadata.clone()
            pair_metadata[..., 0] = 0
            pair_metadata[..., 4] = 0

        robot_context = self.robot_md_adapter(robot_metadata)
        task_context = self.task_md_adapter(task_metadata)
        if self.md_config.use_typed_edges:
            typed_adjacency = md_inputs.typed_adjacency.to(
                device=device, dtype=dtype
            )
            normal_message = _incoming_mean(
                typed_adjacency[:, 0], self.normal_relation(task_context)
            )
            material_message = _incoming_mean(
                typed_adjacency[:, 1], self.material_relation(task_context)
            )
            task_context = task_context + normal_message + material_message

        downstream_context = torch.zeros_like(task_context)
        if self.md_config.use_downstream_encoding:
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
        transport_residual = self.transport_score_mlp(
            torch.cat(
                (
                    expanded_robot,
                    expanded_task,
                    expanded_downstream,
                    pair_metadata,
                ),
                dim=-1,
            )
        ).squeeze(-1)

        task_scores = legacy_scores[..., :task_count]
        task_is_transport = md_inputs.task_is_transport.to(device=device)
        combined = torch.where(
            task_is_transport.unsqueeze(1),
            task_scores + transport_residual,
            task_scores,
        )
        hard_mask = md_inputs.hard_feasibility_mask.to(device=device)
        combined = combined.masked_fill(~hard_mask, self.masked_score)
        if self.use_idle:
            return torch.cat((combined, legacy_scores[..., task_count:]), dim=-1)
        return combined


def _adapter(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _incoming_mean(adjacency: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    incoming = adjacency.transpose(1, 2)
    degree = incoming.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.bmm(incoming, values) / degree


__all__ = [
    "MD_OPPORTUNITY_DIRECTIONS",
    "MD_OPPORTUNITY_FEATURE_COUNT",
    "MDOpportunityContext",
    "MDOpportunityFeature",
    "MDOpportunityValues",
    "MDPolicyConfig",
    "MDPolicyInputs",
    "MDSchedulerNetwork",
]
