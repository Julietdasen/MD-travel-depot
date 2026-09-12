import unittest

import torch

from models.md_policy import MDPolicyConfig, MDPolicyInputs, MDSchedulerNetwork
from models.scheduler_network import SchedulerNetwork


def _network(*, enabled: bool = True, **flags) -> MDSchedulerNetwork:
    torch.manual_seed(19)
    return MDSchedulerNetwork(
        robot_input_dimensions=7,
        task_input_dimension=9,
        embed_dim=16,
        ff_dim=32,
        n_transformer_heads=4,
        n_transformer_layers=1,
        n_gatn_heads=4,
        n_gatn_layers=1,
        use_idle=False,
        md_config=MDPolicyConfig(enabled=enabled, hidden_dim=16, **flags),
    )


def _inputs(*, hard_mask=None, eta: float = 0.25) -> MDPolicyInputs:
    robot_metadata = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 2.0, 0.8]]]
    )
    task_metadata = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0],
                [1.0, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.4, 0.0],
            ]
        ]
    )
    pair_metadata = torch.zeros(1, 2, 3, 5)
    pair_metadata[:, :, 1, :] = torch.tensor([eta, 1.0, 1.0, 0.8, 0.1])
    typed_adjacency = torch.zeros(1, 2, 3, 3)
    typed_adjacency[0, 0, 2, 0] = 1.0
    typed_adjacency[0, 1, 1, 0] = 1.0
    return MDPolicyInputs(
        robot_metadata=robot_metadata,
        task_metadata=task_metadata,
        pair_metadata=pair_metadata,
        task_is_transport=torch.tensor([[False, True, False]]),
        typed_adjacency=typed_adjacency,
        downstream_task_index=torch.tensor([[-1, 0, -1]]),
        hard_feasibility_mask=(
            torch.ones(1, 2, 3, dtype=torch.bool)
            if hard_mask is None
            else hard_mask
        ),
    )


class MDPolicyScoringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.robot_features = torch.randn(1, 2, 7)
        self.task_features = torch.randn(1, 3, 9)
        self.task_adjacency = torch.ones(1, 3, 3)

    def test_disabled_md_path_is_bitwise_legacy_and_loads_legacy_state(self):
        legacy = SchedulerNetwork(
            7, 9, 16, 32, 4, 1, 4, 1, use_idle=False
        )
        policy = _network(enabled=False)
        load_result = policy.load_legacy_state_dict(legacy.state_dict())

        expected = legacy(
            self.robot_features, self.task_features, self.task_adjacency
        )
        actual = policy(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=_inputs(),
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(load_result.missing_keys)
        self.assertFalse(load_result.unexpected_keys)
        self.assertEqual(policy.robot_embedding.in_features, 7)
        self.assertEqual(policy.task_embedding.in_features, 9)

    def test_md_transport_head_preserves_process_scores_and_applies_hard_mask(self):
        policy = _network()
        mask = torch.tensor(
            [[[True, False, True], [False, True, False]]], dtype=torch.bool
        )
        md_inputs = _inputs(hard_mask=mask)
        legacy_scores = super(MDSchedulerNetwork, policy).forward(
            self.robot_features, self.task_features, self.task_adjacency
        )

        scores = policy(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=md_inputs,
        )

        self.assertEqual(tuple(scores.shape), (1, 2, 3))
        self.assertEqual(scores[0, 0, 0], legacy_scores[0, 0, 0])
        self.assertEqual(scores[0, 0, 2], legacy_scores[0, 0, 2])
        self.assertEqual(scores[0, 1, 1].isfinite(), torch.tensor(True))
        self.assertTrue(torch.all(scores[~mask] == policy.masked_score))
        self.assertNotEqual(scores[0, 1, 1], legacy_scores[0, 1, 1])

    def test_transport_score_consumes_eta_capacity_and_downstream_context(self):
        policy = _network()
        baseline = policy(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=_inputs(eta=0.1),
        )
        changed = _inputs(eta=0.9)
        changed.pair_metadata[:, :, 1, 2] = -0.5
        changed.task_metadata[:, 0, 6] = 0.9
        scored = policy(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=changed,
        )

        self.assertFalse(torch.allclose(baseline[:, :, 1], scored[:, :, 1]))
        self.assertTrue(torch.equal(baseline[:, :, 0], scored[:, :, 0]))
        self.assertTrue(torch.equal(baseline[:, :, 2], scored[:, :, 2]))

    def test_blocked_tasks_remain_context_visible_but_action_ineligible(self):
        policy = _network()
        mask = torch.ones(1, 2, 3, dtype=torch.bool)
        mask[:, :, 0] = False
        first = _inputs(hard_mask=mask)
        second = _inputs(hard_mask=mask)
        second.task_metadata[:, 0, 6] = 1.0

        score_one = policy(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=first,
        )
        score_two = policy(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=second,
        )

        self.assertTrue(torch.all(score_one[:, :, 0] == policy.masked_score))
        self.assertTrue(torch.all(score_two[:, :, 0] == policy.masked_score))
        self.assertFalse(torch.allclose(score_one[:, :, 1], score_two[:, :, 1]))


if __name__ == "__main__":
    unittest.main()
