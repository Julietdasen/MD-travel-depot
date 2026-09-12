import torch

from experiments.md_minimal_physics_scorer import PhysicsSetScorer, evaluate


def test_physics_set_scorer_is_assignment_order_invariant():
    torch.manual_seed(7)
    model = PhysicsSetScorer(hidden=8)
    assignments = torch.randn(3, 17)

    assert torch.allclose(model(assignments), model(assignments[[2, 0, 1]]))


def test_baseline_evaluation_uses_candidate_rank():
    states = [
        {
            "pending_count": 2,
            "actions": [
                {"rank": 1, "regret": 0.0, "optimal": True},
                {"rank": 0, "regret": 3.0, "optimal": False},
            ],
        }
    ]

    metrics = evaluate(None, states)

    assert metrics["tolerance_optimal_top1"] == 0.0
    assert metrics["mean_regret"] == 3.0
