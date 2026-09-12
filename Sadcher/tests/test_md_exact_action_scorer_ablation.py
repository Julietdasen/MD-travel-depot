import torch

from experiments.md_exact_action_scorer_ablation import (
    ActionExample,
    DeepSetsActionScorer,
    StateExample,
    TrainingConfig,
    evaluate,
    state_loss,
)


def test_deepsets_score_is_assignment_order_invariant():
    torch.manual_seed(1)
    model = DeepSetsActionScorer(hidden_dim=8)
    edges = torch.randn(3, 7)

    assert torch.allclose(model(edges), model(edges[[2, 0, 1]]))


def test_tie_aware_loss_and_evaluation_are_finite():
    model = DeepSetsActionScorer(hidden_dim=8)
    state = StateExample(
        "snapshot",
        2,
        (
            ActionExample(((0, 1),), 0, torch.zeros(1, 7), 1.0, True),
            ActionExample(((1, 2),), 1, torch.ones(1, 7), 4.0, False),
        ),
    )

    loss = state_loss(model, state, TrainingConfig(epochs=1))
    metrics = evaluate(model, (state,))

    assert bool(torch.isfinite(loss))
    assert metrics["state_count"] == 1
    assert 0.0 <= metrics["tolerance_optimal_top1"] <= 1.0

