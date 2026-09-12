"""Small, MD-specific building blocks for conservative IL -> PPO fine-tuning.

This module deliberately contains no solver calls and no policy architecture.  It
is usable from a Gym/SKRL runner once an MD environment adapter is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True, slots=True)
class MDFineTuneConfig:
    """Guardrails for a policy initialized from an offline IL checkpoint."""

    bc_weight: float = 0.10
    kl_weight: float = 0.05
    illegal_penalty: float = 1.0
    completion_bonus: float = 0.05
    starvation_penalty: float = 0.02
    terminal_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "bc_weight",
            "kl_weight",
            "illegal_penalty",
            "completion_bonus",
            "starvation_penalty",
            "terminal_scale",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")


class MDRewardTracker:
    """Compute potential-based dense rewards from canonical MD metrics."""

    def __init__(self, *, greedy_makespan: float, config: MDFineTuneConfig | None = None):
        if greedy_makespan <= 0:
            raise ValueError("greedy_makespan must be positive")
        self.greedy_makespan = float(greedy_makespan)
        self.config = MDFineTuneConfig() if config is None else config
        self._previous_time = 0.0
        self._previous_completed = 0
        self._previous_starvation = 0.0

    def reset(self) -> None:
        self._previous_time = 0.0
        self._previous_completed = 0
        self._previous_starvation = 0.0

    def step(
        self,
        *,
        time: float,
        completed_tasks: int,
        starvation: Mapping[str, float] | None = None,
        terminal: bool = False,
        makespan: float | None = None,
        illegal_assignments: int = 0,
    ) -> float:
        """Return a bounded reward; positive progress reduces remaining time."""
        if time < self._previous_time or completed_tasks < self._previous_completed:
            raise ValueError("time and completed_tasks must be monotonic")
        starvation_total = sum(float(value) for value in (starvation or {}).values())
        reward = (self._previous_time - float(time)) / self.greedy_makespan
        reward += self.config.completion_bonus * (
            completed_tasks - self._previous_completed
        )
        reward -= self.config.starvation_penalty * max(
            0.0, starvation_total - self._previous_starvation
        )
        reward -= self.config.illegal_penalty * max(0, illegal_assignments)
        if terminal and makespan is not None:
            reward += self.config.terminal_scale * (
                self.greedy_makespan - float(makespan)
            ) / self.greedy_makespan
        self._previous_time = float(time)
        self._previous_completed = int(completed_tasks)
        self._previous_starvation = starvation_total
        return float(reward)


def il_regularized_ppo_loss(
    ppo_loss: torch.Tensor,
    *,
    policy_expert_log_prob: torch.Tensor,
    il_expert_log_prob: torch.Tensor,
    old_expert_log_prob: torch.Tensor | None = None,
    config: MDFineTuneConfig | None = None,
) -> torch.Tensor:
    """Add behavior-cloning and optional KL-to-IL penalties to a PPO objective.

    ``expert_log_prob`` must be computed with the frozen IL policy on the same
    legal action mask.  Using log-probabilities keeps this helper compatible
    with multi-categorical robot actions.
    """
    cfg = MDFineTuneConfig() if config is None else config
    if policy_expert_log_prob.shape != il_expert_log_prob.shape:
        raise ValueError("policy and IL expert log-probs must have equal shape")
    bc_loss = -policy_expert_log_prob.mean()
    total = ppo_loss + cfg.bc_weight * bc_loss
    if old_expert_log_prob is not None:
        if old_expert_log_prob.shape != policy_expert_log_prob.shape:
            raise ValueError("old_expert_log_prob must match policy log-prob shape")
        # A first-order KL proxy, useful when the PPO implementation already
        # supplies the old-policy ratio and only the IL policy is extra.
        kl_proxy = (policy_expert_log_prob - il_expert_log_prob).mean()
        total = total + cfg.kl_weight * kl_proxy
    return total


@dataclass(frozen=True, slots=True)
class GoNoGoResult:
    go: bool
    reasons: tuple[str, ...]


def evaluate_go_no_go(
    *,
    il_makespan: float,
    rl_makespan: float,
    il_success_rate: float,
    rl_success_rate: float,
    rl_illegal_assignments: int,
    il_material_starvation: float,
    rl_material_starvation: float,
    rl_p95_latency: float,
    il_p95_latency: float,
    min_relative_makespan_gain: float = 0.01,
) -> GoNoGoResult:
    """Apply the pre-registered safety gates for an IL -> RL comparison."""
    reasons: list[str] = []
    if il_makespan <= 0 or rl_makespan <= 0:
        raise ValueError("makespans must be positive")
    gain = (il_makespan - rl_makespan) / il_makespan
    if gain < min_relative_makespan_gain:
        reasons.append("makespan_gain_below_threshold")
    if rl_success_rate < il_success_rate:
        reasons.append("success_rate_regressed")
    if rl_illegal_assignments != 0:
        reasons.append("illegal_assignments_nonzero")
    if rl_material_starvation > il_material_starvation:
        reasons.append("material_starvation_regressed")
    if rl_p95_latency > il_p95_latency * 1.10:
        reasons.append("p95_latency_regressed")
    return GoNoGoResult(go=not reasons, reasons=tuple(reasons))
