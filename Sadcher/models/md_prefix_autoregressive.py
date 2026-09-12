"""Supervised autoregressive prefix decoder for fixed-shape MD instances.

This module is intentionally independent of RLlib and production scheduling.
It keeps the existing ``[robot] -> task`` action representation while making
each robot's logits conditional on the preceding robot choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from reinforcement_learning.md_joint_action import oracle_aligned_action_masks


ROBOT_FEATURE_DIM = 7
TASK_FEATURE_DIM = 9
ROBOT_METADATA_DIM = 4
TASK_METADATA_DIM = 8
PAIR_METADATA_DIM = 5
OPPORTUNITY_DIM = 5
SKILL_DIM = 3
PREFIX_DYNAMIC_DIM = 10


@dataclass(frozen=True, slots=True)
class MDSequentialObservation:
    """Tensor contract consumed by :class:`MDPrefixAutoregressivePolicy`."""

    robot_features: torch.Tensor
    task_features: torch.Tensor
    robot_metadata: torch.Tensor
    task_metadata: torch.Tensor
    pair_metadata: torch.Tensor
    task_is_transport: torch.Tensor
    typed_adjacency: torch.Tensor
    opportunity_context: torch.Tensor
    process_covered_skills: torch.Tensor
    action_mask: torch.Tensor

    @classmethod
    def from_mapping(cls, observation: Mapping[str, torch.Tensor]) -> "MDSequentialObservation":
        required = (
            "robot_features",
            "task_features",
            "robot_metadata",
            "task_metadata",
            "pair_metadata",
            "task_is_transport",
            "typed_adjacency",
            "opportunity_context",
            "process_covered_skills",
            "action_mask",
        )
        missing = [key for key in required if key not in observation]
        if missing:
            raise ValueError(f"observation is missing fields: {missing}")
        result = cls(*(observation[key] for key in required))
        result.validate()
        return result

    def validate(self) -> None:
        tensors = (
            self.robot_features,
            self.task_features,
            self.robot_metadata,
            self.task_metadata,
            self.pair_metadata,
            self.task_is_transport,
            self.typed_adjacency,
            self.opportunity_context,
            self.process_covered_skills,
            self.action_mask,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("all observation fields must be tensors")
        if self.robot_features.ndim != 3 or self.task_features.ndim != 3:
            raise ValueError("robot_features and task_features must be [B,N,F]")
        batch, robots, robot_width = self.robot_features.shape
        batch_tasks, tasks, task_width = self.task_features.shape
        if batch != batch_tasks or robot_width != ROBOT_FEATURE_DIM or task_width != TASK_FEATURE_DIM:
            raise ValueError("unexpected robot/task feature shapes")
        expected = {
            "robot_metadata": (batch, robots, ROBOT_METADATA_DIM),
            "task_metadata": (batch, tasks, TASK_METADATA_DIM),
            "pair_metadata": (batch, robots, tasks, PAIR_METADATA_DIM),
            "task_is_transport": (batch, tasks),
            "typed_adjacency": (batch, 2, tasks, tasks),
            "opportunity_context": (batch, robots, tasks, OPPORTUNITY_DIM),
            "process_covered_skills": (batch, tasks, SKILL_DIM),
            "action_mask": (batch, robots, tasks + 1),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.task_is_transport.dtype is not torch.bool:
            raise ValueError("task_is_transport must be boolean")
        if self.action_mask.dtype not in (torch.bool, torch.float32, torch.float64):
            raise ValueError("action_mask must be bool or floating point")
        if not all(bool(torch.isfinite(value).all()) for value in tensors if value.is_floating_point()):
            raise ValueError("observation contains non-finite values")

    def as_mapping(self) -> dict[str, torch.Tensor]:
        return {
            "robot_features": self.robot_features,
            "task_features": self.task_features,
            "robot_metadata": self.robot_metadata,
            "task_metadata": self.task_metadata,
            "pair_metadata": self.pair_metadata,
            "task_is_transport": self.task_is_transport,
            "typed_adjacency": self.typed_adjacency,
            "opportunity_context": self.opportunity_context,
            "process_covered_skills": self.process_covered_skills,
            "action_mask": self.action_mask,
        }


@dataclass(frozen=True, slots=True)
class MDSequentialEncoding:
    robot_tokens: torch.Tensor
    task_tokens: torch.Tensor
    pair_features: torch.Tensor
    global_context: torch.Tensor
    observation: MDSequentialObservation


@dataclass(frozen=True, slots=True)
class MDPrefixState:
    hidden: torch.Tensor
    covered_skills: torch.Tensor
    selected_count: torch.Tensor
    selected_robots: torch.Tensor
    used_transport: torch.Tensor
    active_process: torch.Tensor
    completed_process: torch.Tensor
    selected_any: torch.Tensor


@dataclass(frozen=True, slots=True)
class MDDecodedAction:
    actions: torch.Tensor
    log_probability: torch.Tensor
    beam_width: int


@dataclass(frozen=True, slots=True)
class MDCandidateActions:
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    beam_width: int


class MDPrefixAutoregressivePolicy(nn.Module):
    """Cross-attended state encoder with a causal fixed-robot decoder."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        attention_heads: int = 4,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        use_prefix: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim = hidden_dim
        self.use_prefix = bool(use_prefix)
        self.robot_input = nn.Linear(ROBOT_FEATURE_DIM + ROBOT_METADATA_DIM, hidden_dim)
        self.task_input = nn.Linear(TASK_FEATURE_DIM + TASK_METADATA_DIM + 1, hidden_dim)
        layer = lambda: nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.robot_encoder = nn.TransformerEncoder(layer(), num_layers=1)
        self.task_encoder = nn.TransformerEncoder(layer(), num_layers=1)
        self.normal_relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.material_relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.robot_to_task = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.task_to_robot = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.robot_norm = nn.LayerNorm(hidden_dim)
        self.task_norm = nn.LayerNorm(hidden_dim)
        self.pair_input = nn.Linear(PAIR_METADATA_DIM + OPPORTUNITY_DIM, hidden_dim)
        self.dynamic_input = nn.Linear(PREFIX_DYNAMIC_DIM, hidden_dim)
        self.prefix_init = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.query = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.Tanh())
        self.task_score = nn.Sequential(
            nn.Linear(2 * hidden_dim + PAIR_METADATA_DIM + OPPORTUNITY_DIM + PREFIX_DYNAMIC_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.idle_score = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.action_gru = nn.GRUCell(
            2 * hidden_dim + PAIR_METADATA_DIM + OPPORTUNITY_DIM + PREFIX_DYNAMIC_DIM + 1,
            hidden_dim,
        )

    def encode(self, observation: Mapping[str, torch.Tensor] | MDSequentialObservation) -> MDSequentialEncoding:
        obs = observation if isinstance(observation, MDSequentialObservation) else MDSequentialObservation.from_mapping(observation)
        robot = self.robot_encoder(self.robot_input(torch.cat((obs.robot_features, obs.robot_metadata), dim=-1)))
        task_input = torch.cat(
            (obs.task_features, obs.task_metadata, obs.task_is_transport.float().unsqueeze(-1)), dim=-1
        )
        task = self.task_encoder(self.task_input(task_input))
        typed = obs.typed_adjacency.to(dtype=task.dtype)
        normal = torch.bmm(typed[:, 0], self.normal_relation(task))
        material = torch.bmm(typed[:, 1], self.material_relation(task))
        degree = (typed.sum(dim=1).sum(dim=-1, keepdim=True)).clamp_min(1.0)
        task = task + (normal + material) / degree
        robot_update, _ = self.robot_to_task(robot, task, task, need_weights=False)
        task_update, _ = self.task_to_robot(task, robot, robot, need_weights=False)
        robot = self.robot_norm(robot + robot_update)
        task = self.task_norm(task + task_update)
        pair = torch.cat((obs.pair_metadata, obs.opportunity_context), dim=-1)
        global_context = torch.cat((robot.mean(dim=1), task.mean(dim=1)), dim=-1)
        return MDSequentialEncoding(robot, task, pair, global_context, obs)

    def initial_prefix_state(self, encoding: MDSequentialEncoding) -> MDPrefixState:
        obs = encoding.observation
        requirements = obs.task_features[..., 3:6].float().clamp_min(0.0)
        covered = obs.process_covered_skills.float().clamp_min(0.0)
        assigned = obs.task_features[..., 7] > 0.5
        complete = (~obs.task_is_transport) & assigned & ((covered >= requirements).all(dim=-1))
        active = (~obs.task_is_transport) & assigned & ~complete
        return MDPrefixState(
            hidden=self.prefix_init(encoding.global_context),
            covered_skills=covered,
            selected_count=assigned.float(),
            selected_robots=torch.zeros(
                (encoding.robot_tokens.shape[0], encoding.robot_tokens.shape[1]),
                dtype=torch.bool,
                device=encoding.robot_tokens.device,
            ),
            used_transport=obs.task_is_transport & assigned,
            active_process=active,
            completed_process=complete,
            selected_any=assigned.any(dim=-1),
        )

    def _dynamic_features(self, encoding: MDSequentialEncoding, state: MDPrefixState) -> torch.Tensor:
        requirements = encoding.observation.task_features[..., 3:6].float().clamp_min(0.0)
        remaining = (requirements - state.covered_skills).clamp_min(0.0)
        return torch.cat(
            (
                state.covered_skills,
                remaining,
                (state.selected_count > 0).float().unsqueeze(-1),
                state.completed_process.float().unsqueeze(-1),
                state.selected_count.unsqueeze(-1),
                state.used_transport.float().unsqueeze(-1),
            ),
            dim=-1,
        )

    def step(
        self,
        encoding: MDSequentialEncoding,
        prefix_state: MDPrefixState,
        robot_index: int,
        *,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if robot_index < 0 or robot_index >= encoding.robot_tokens.shape[1]:
            raise IndexError("robot_index is outside the encoded robot set")
        dynamic = self._dynamic_features(encoding, prefix_state)
        if not self.use_prefix:
            dynamic = torch.zeros_like(dynamic)
        dynamic_tokens = self.dynamic_input(dynamic)
        task_tokens = encoding.task_tokens + dynamic_tokens
        robot = encoding.robot_tokens[:, robot_index]
        query = self.query(torch.cat((robot, prefix_state.hidden, encoding.global_context), dim=-1))
        query_tasks = query.unsqueeze(1).expand(-1, task_tokens.shape[1], -1)
        pair = encoding.pair_features[:, robot_index]
        score_input = torch.cat((query_tasks, task_tokens, pair, dynamic), dim=-1)
        task_logits = self.task_score(score_input).squeeze(-1)
        idle_logits = self.idle_score(torch.cat((query, prefix_state.hidden, encoding.global_context), dim=-1))
        logits = torch.cat((idle_logits, task_logits), dim=-1)
        if action_mask is not None:
            if tuple(action_mask.shape) != tuple(logits.shape):
                raise ValueError("action_mask shape does not match decoder logits")
            logits = logits.masked_fill(~action_mask.bool(), -1.0e9)
        return logits

    def advance(
        self,
        encoding: MDSequentialEncoding,
        prefix_state: MDPrefixState,
        robot_index: int,
        choices: torch.Tensor,
    ) -> MDPrefixState:
        choices = choices.long().reshape(-1)
        batch, tasks = encoding.task_tokens.shape[:2]
        if choices.shape[0] != batch:
            raise ValueError("choices batch size does not match encoding")
        dynamic = self._dynamic_features(encoding, prefix_state)
        task_index = (choices - 1).clamp_min(0).clamp_max(tasks - 1)
        gather_index = task_index.view(batch, 1, 1).expand(-1, 1, self.hidden_dim)
        selected_task = encoding.task_tokens.gather(1, gather_index).squeeze(1)
        pair = encoding.pair_features[:, robot_index].gather(
            1, task_index.view(batch, 1, 1).expand(-1, 1, encoding.pair_features.shape[-1])
        ).squeeze(1)
        selected_dynamic = dynamic.gather(
            1, task_index.view(batch, 1, 1).expand(-1, 1, dynamic.shape[-1])
        ).squeeze(1)
        is_idle = (choices == 0).float().unsqueeze(-1)
        action_input = torch.cat(
            (encoding.robot_tokens[:, robot_index], selected_task, pair, selected_dynamic, is_idle), dim=-1
        )
        hidden = self.action_gru(action_input, prefix_state.hidden) if self.use_prefix else prefix_state.hidden

        selected = choices > 0
        task_one_hot = F.one_hot(task_index, num_classes=tasks).float() * selected.float().unsqueeze(-1)
        selected_count = prefix_state.selected_count + task_one_hot
        selected_robots = prefix_state.selected_robots.clone()
        selected_robots[:, robot_index] = selected
        transport = encoding.observation.task_is_transport
        used_transport = prefix_state.used_transport | (task_one_hot.bool() & transport)
        capabilities = (encoding.observation.robot_features[:, robot_index, 3:6] > 0.5).float()
        process_selection = task_one_hot * (~transport).float()
        added_skills = process_selection.unsqueeze(-1) * capabilities.unsqueeze(1)
        covered = torch.maximum(prefix_state.covered_skills, added_skills)
        requirements = encoding.observation.task_features[..., 3:6].float().clamp_min(0.0)
        complete = (~transport) & (selected_count > 0) & ((covered >= requirements).all(dim=-1))
        active = (~transport) & (selected_count > 0) & ~complete
        return MDPrefixState(
            hidden=hidden,
            covered_skills=covered,
            selected_count=selected_count,
            selected_robots=selected_robots,
            used_transport=used_transport,
            active_process=active,
            completed_process=complete,
            selected_any=prefix_state.selected_any | selected,
        )

    def log_prob(
        self,
        observation: Mapping[str, torch.Tensor] | MDSequentialObservation,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        encoding = self.encode(observation)
        actions = actions.long()
        batch, robots = encoding.robot_tokens.shape[:2]
        if tuple(actions.shape) != (batch, robots):
            raise ValueError("actions must have shape [B, robot_count]")
        masks = oracle_aligned_action_masks(encoding.observation.as_mapping(), actions)
        state = self.initial_prefix_state(encoding)
        total = actions.new_zeros((batch,), dtype=encoding.robot_tokens.dtype)
        for index in range(robots):
            logits = self.step(encoding, state, index, action_mask=masks[:, index])
            selected = masks[:, index].gather(1, actions[:, index].unsqueeze(-1)).squeeze(-1)
            if not bool(selected.all()):
                raise ValueError("actions contain an action outside oracle-aligned support")
            total = total + F.log_softmax(logits, dim=-1).gather(1, actions[:, index].unsqueeze(-1)).squeeze(-1)
            state = self.advance(encoding, state, index, actions[:, index])
        return total

    @torch.no_grad()
    def beam_candidates(
        self,
        observation: Mapping[str, torch.Tensor] | MDSequentialObservation,
        *,
        beam_width: int = 16,
    ) -> MDCandidateActions:
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        encoding = self.encode(observation)
        if encoding.robot_tokens.shape[0] != 1:
            raise ValueError("beam_decode currently expects batch size one")
        robots = encoding.robot_tokens.shape[1]
        initial = self.initial_prefix_state(encoding)
        beams: list[tuple[list[int], float, MDPrefixState]] = [([], 0.0, initial)]
        completion_cache: dict[int, dict[tuple[object, ...], bool]] = {}
        for index in range(robots):
            expanded: list[tuple[list[int], float, MDPrefixState]] = []
            for prefix, score, state in beams:
                partial = torch.full((1, robots), -1, dtype=torch.long, device=encoding.robot_tokens.device)
                if prefix:
                    partial[0, : len(prefix)] = torch.tensor(prefix, device=partial.device)
                masks = oracle_aligned_action_masks(
                    encoding.observation.as_mapping(),
                    partial,
                    _completion_cache=completion_cache,
                )
                logits = self.step(encoding, state, index, action_mask=masks[:, index])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                valid_choices = torch.nonzero(masks[0, index], as_tuple=False).squeeze(-1)
                valid_scores = log_probs.index_select(0, valid_choices)
                values, offsets = torch.topk(valid_scores, min(beam_width, valid_scores.numel()))
                choices = valid_choices.index_select(0, offsets)
                for value, choice in zip(values.tolist(), choices.tolist(), strict=True):
                    next_state = self.advance(encoding, state, index, torch.tensor([choice], device=partial.device))
                    expanded.append((prefix + [int(choice)], score + float(value), next_state))
            expanded.sort(key=lambda item: (-item[1], item[0]))
            beams = expanded[:beam_width]
        return MDCandidateActions(
            actions=torch.tensor(
                [actions for actions, _score, _state in beams],
                dtype=torch.long,
                device=encoding.robot_tokens.device,
            ),
            log_probabilities=torch.tensor(
                [score for _actions, score, _state in beams],
                dtype=encoding.robot_tokens.dtype,
                device=encoding.robot_tokens.device,
            ),
            beam_width=beam_width,
        )

    @torch.no_grad()
    def beam_decode(
        self,
        observation: Mapping[str, torch.Tensor] | MDSequentialObservation,
        *,
        beam_width: int = 16,
    ) -> MDDecodedAction:
        candidates = self.beam_candidates(observation, beam_width=beam_width)
        return MDDecodedAction(
            actions=candidates.actions[:1],
            log_probability=candidates.log_probabilities[:1],
            beam_width=beam_width,
        )


__all__ = [
    "DecoderResult",
    "MDCandidateActions",
    "MDDecodedAction",
    "MDPrefixAutoregressivePolicy",
    "MDPrefixState",
    "MDSequentialEncoding",
    "MDSequentialObservation",
]


# Public name used by the prototype plan; keep the descriptive dataclass name
# above for callers that want to inspect the returned action metadata.
DecoderResult = MDDecodedAction
