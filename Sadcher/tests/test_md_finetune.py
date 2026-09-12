import pytest
import torch

from reinforcement_learning.md_finetune import (
    MDRewardTracker,
    evaluate_go_no_go,
    il_regularized_ppo_loss,
)


def test_reward_tracker_is_dense_and_terminal_aware():
    tracker = MDRewardTracker(greedy_makespan=10.0)
    first = tracker.step(time=1, completed_tasks=0)
    second = tracker.step(time=2, completed_tasks=1)
    terminal = tracker.step(time=5, completed_tasks=2, terminal=True, makespan=5)
    assert first < 0
    assert second > first
    assert terminal > 0


def test_regularized_loss_requires_matching_log_prob_shapes():
    base = torch.tensor(2.0, requires_grad=True)
    result = il_regularized_ppo_loss(
        base,
        policy_expert_log_prob=torch.tensor([-0.2, -0.3], requires_grad=True),
        il_expert_log_prob=torch.tensor([-0.1, -0.4]),
        old_expert_log_prob=torch.tensor([-0.2, -0.2]),
    )
    result.backward()
    assert result.item() > 2.0
    with pytest.raises(ValueError):
        il_regularized_ppo_loss(
            base,
            policy_expert_log_prob=torch.zeros(2),
            il_expert_log_prob=torch.zeros(1),
        )


def test_go_no_go_rejects_any_safety_regression():
    result = evaluate_go_no_go(
        il_makespan=100,
        rl_makespan=95,
        il_success_rate=1.0,
        rl_success_rate=1.0,
        rl_illegal_assignments=0,
        il_material_starvation=2.0,
        rl_material_starvation=2.0,
        il_p95_latency=10.0,
        rl_p95_latency=10.0,
    )
    assert result.go
    blocked = evaluate_go_no_go(
        il_makespan=100,
        rl_makespan=99.5,
        il_success_rate=1.0,
        rl_success_rate=0.9,
        rl_illegal_assignments=1,
        il_material_starvation=2.0,
        rl_material_starvation=3.0,
        il_p95_latency=10.0,
        rl_p95_latency=12.0,
    )
    assert not blocked.go
    assert len(blocked.reasons) == 5
