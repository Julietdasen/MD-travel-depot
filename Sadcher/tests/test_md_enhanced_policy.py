import unittest
from dataclasses import replace

import torch

from models.md_enhanced_policy import (
    MDEnhancedPolicyConfig,
    MDEnhancedSchedulerNetwork,
)
from models.md_policy import MD_OPPORTUNITY_FEATURE_COUNT, MDPolicyInputs
from models.scheduler_network import SchedulerNetwork


def _inputs(
    eta: float = 0.25,
    *,
    blocked_value: float = 0.2,
    opportunity: tuple[float, float, float, float, float] = (0, 0, 0, 0, 0),
):
    task_metadata = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, blocked_value, 1.0],
                [1.0, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.4, 1.0],
            ]
        ]
    )
    pair_metadata = torch.zeros(1, 2, 3, 5)
    pair_metadata[:, :, 1, :] = torch.tensor([eta, 1.0, 1.0, 0.8, 0.1])
    hard_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    hard_mask[:, :, 0] = False
    typed_adjacency = torch.zeros(1, 2, 3, 3)
    typed_adjacency[0, 1, 1, 0] = 1.0
    opportunity_context = torch.zeros(
        1, 2, 3, MD_OPPORTUNITY_FEATURE_COUNT
    )
    opportunity_context[:, :, 1, :] = torch.tensor(opportunity)
    return MDPolicyInputs(
        robot_metadata=torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 2.0, 0.8]]]
        ),
        task_metadata=task_metadata,
        pair_metadata=pair_metadata,
        task_is_transport=torch.tensor([[False, True, False]]),
        typed_adjacency=typed_adjacency,
        downstream_task_index=torch.tensor([[-1, 0, -1]]),
        hard_feasibility_mask=hard_mask,
        opportunity_context=opportunity_context,
    )


def _network(
    *,
    enabled=True,
    cross_attention=True,
    pair_aware_attention=False,
    **config_overrides,
):
    torch.manual_seed(53)
    return MDEnhancedSchedulerNetwork(
        robot_input_dimensions=7,
        task_input_dimension=9,
        embed_dim=16,
        ff_dim=32,
        n_transformer_heads=4,
        n_transformer_layers=1,
        n_gatn_heads=4,
        n_gatn_layers=1,
        use_idle=False,
        md_config=MDEnhancedPolicyConfig(
            enabled=enabled,
            hidden_dim=16,
            use_cross_attention=cross_attention,
            use_pair_aware_attention=pair_aware_attention,
            cross_attention_heads=4,
            residual_bound=0.2,
            legacy_transport_bound=0.25,
            **config_overrides,
        ),
    )


class MDEnhancedPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(59)
        self.robot_features = torch.randn(1, 2, 7)
        self.task_features = torch.randn(1, 3, 9)
        self.task_adjacency = torch.ones(1, 3, 3)

    def test_pair_representation_contains_context_metadata_and_opportunity(self):
        diagnostics = _network(
            cross_attention=False,
            pair_aware_attention=True,
        ).forward_with_diagnostics(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=_inputs(),
        )
        self.assertIsNotNone(diagnostics.pair_representation)
        self.assertEqual(
            diagnostics.pair_representation.shape[:3],
            (1, 2, 3),
        )
        self.assertGreater(
            diagnostics.pair_representation.shape[-1],
            16,
        )


    def test_disabled_enhancement_is_bitwise_legacy_compatible(self):
        legacy = SchedulerNetwork(7, 9, 16, 32, 4, 1, 4, 1, use_idle=False)
        policy = _network(enabled=False)
        policy.load_legacy_state_dict(legacy.state_dict())

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
        self.assertEqual(policy.robot_embedding.in_features, 7)
        self.assertEqual(policy.task_embedding.in_features, 9)

    def test_transport_decomposition_is_monotonic_and_residual_is_bounded(self):
        policy = _network()
        low = policy.forward_with_diagnostics(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=_inputs(eta=0.1),
        )
        high = policy.forward_with_diagnostics(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=_inputs(eta=0.9),
        )
        legacy = SchedulerNetwork.forward(
            policy, self.robot_features, self.task_features, self.task_adjacency
        )

        self.assertTrue(torch.all(high.physics_utility[:, :, 1] < low.physics_utility[:, :, 1]))
        self.assertTrue(torch.all(high.scores[:, :, 1] < low.scores[:, :, 1]))
        self.assertLessEqual(float(low.downstream_residual.abs().max()), 0.2)
        self.assertLessEqual(float(high.downstream_residual.abs().max()), 0.2)
        self.assertTrue(torch.equal(low.scores[:, :, 2], legacy[:, :, 2]))
        expected_transport = (
            low.legacy_component[:, :, 1]
            + low.physics_utility[:, :, 1]
            + low.downstream_residual[:, :, 1]
        )
        self.assertTrue(torch.allclose(low.scores[:, :, 1], expected_transport))
        self.assertLessEqual(float(low.legacy_component[:, :, 1].abs().max()), 0.25)
        self.assertEqual(tuple(low.component_scales.shape), (1, 3))
        self.assertEqual(tuple(low.action_flip.shape), (1, 2))
        self.assertTrue(torch.equal(low.raw_residual[:, :, 2], torch.zeros(1, 2)))

    def test_pair_aware_attention_uses_competitor_eta_not_focal_eta(self):
        policy = _network(
            cross_attention=False,
            pair_aware_attention=True,
        )
        policy.robot_to_task_gate.data.fill_(1.0)
        policy.task_to_robot_gate.data.fill_(1.0)
        policy.transport_score_mlp[-1].weight.data.fill_(0.1)
        base_inputs = _inputs(eta=0.3)

        competitor_metadata = base_inputs.pair_metadata.clone()
        competitor_metadata[0, 1, 1, 0] = 0.9
        competitor_inputs = replace(
            base_inputs,
            pair_metadata=competitor_metadata,
        )
        base = policy.forward_with_diagnostics(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=base_inputs,
        )
        competitor = policy.forward_with_diagnostics(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=competitor_inputs,
        )

        self.assertFalse(
            torch.allclose(
                base.raw_residual[0, 0, 1],
                competitor.raw_residual[0, 0, 1],
            )
        )
        self.assertTrue(
            torch.equal(
                base.physics_utility[0, 0, 1],
                competitor.physics_utility[0, 0, 1],
            )
        )

        focal_metadata = base_inputs.pair_metadata.clone()
        focal_metadata[0, 0, 1, 0] = 0.9
        focal_inputs = replace(base_inputs, pair_metadata=focal_metadata)
        focal = policy.forward_with_diagnostics(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=focal_inputs,
        )
        self.assertTrue(
            torch.allclose(
                base.raw_residual[0, 0, 1],
                focal.raw_residual[0, 0, 1],
            )
        )
        self.assertLess(
            float(focal.physics_utility[0, 0, 1]),
            float(base.physics_utility[0, 0, 1]),
        )

    def test_pair_aware_and_token_cross_attention_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            MDEnhancedPolicyConfig(
                use_cross_attention=True,
                use_pair_aware_attention=True,
            )

    def test_blocked_task_is_cross_attention_context_but_never_an_action(self):
        policy = _network()
        policy.robot_to_task_gate.data.fill_(1.0)
        policy.task_to_robot_gate.data.fill_(1.0)
        first = policy(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=_inputs(blocked_value=0.1),
        )
        second = policy(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=_inputs(blocked_value=1.0),
        )

        self.assertTrue(torch.all(first[:, :, 0] == policy.masked_score))
        self.assertTrue(torch.all(second[:, :, 0] == policy.masked_score))
        self.assertFalse(torch.allclose(first[:, :, 1], second[:, :, 1]))

    def test_zero_distance_forward_and_backward_are_finite(self):
        policy = _network()
        robot_features = torch.zeros(1, 2, 7)
        task_features = torch.zeros(1, 3, 9)
        scores = policy(
            robot_features,
            task_features,
            torch.zeros(1, 3, 3),
            md_inputs=_inputs(eta=0.0),
        )

        self.assertTrue(torch.isfinite(scores).all())
        scores[scores != policy.masked_score].sum().backward()
        gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_no_prior_probe_starts_neutral_and_learns_opportunity_weights(self):
        policy = _network(
            cross_attention=False,
            use_signed_opportunity_prior=False,
        )
        self.assertFalse(policy.opportunity_raw_magnitudes.requires_grad)
        self.assertTrue(
            torch.equal(
                policy.opportunity_projection.weight,
                torch.zeros_like(policy.opportunity_projection.weight),
            )
        )

        diagnostics = policy.forward_with_diagnostics(
            self.robot_features,
            self.task_features,
            self.task_adjacency,
            md_inputs=_inputs(opportunity=(1, 0, 0, 0, 0)),
        )
        diagnostics.bounded_residual[:, :, 1].sum().backward()

        gradient = policy.opportunity_projection.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_opportunity_interventions_have_calibrated_independent_directions(self):
        base_inputs = _inputs()
        interventions = {
            "critical_path": ((1, 0, 0, 0, 0), "use_critical_path_proxy", 1),
            "last_blocker": ((0, 1, 0, 0, 0), "use_last_material_blocker", 1),
            "coalition_availability": (
                (0, 0, 1, 0, 0),
                "use_coalition_context",
                1,
            ),
            "coalition_scarcity": (
                (0, 0, 0, 1, 0),
                "use_coalition_context",
                -1,
            ),
            "capacity_scarcity": ((0, 0, 0, 0, 1), "use_capacity_scarcity", 1),
        }
        for name, (opportunity, switch, expected_direction) in interventions.items():
            policy = _network(cross_attention=False)
            baseline = policy.forward_with_diagnostics(
                self.robot_features,
                self.task_features,
                self.task_adjacency,
                md_inputs=base_inputs,
            )
            changed = policy.forward_with_diagnostics(
                self.robot_features,
                self.task_features,
                self.task_adjacency,
                md_inputs=_inputs(opportunity=opportunity),
            )
            delta = (
                changed.downstream_residual[:, :, 1]
                - baseline.downstream_residual[:, :, 1]
            )
            self.assertTrue(torch.all(expected_direction * delta > 0), name)

            disabled = _network(cross_attention=False, **{switch: False})
            disabled_base = disabled.forward_with_diagnostics(
                self.robot_features,
                self.task_features,
                self.task_adjacency,
                md_inputs=base_inputs,
            )
            disabled_changed = disabled.forward_with_diagnostics(
                self.robot_features,
                self.task_features,
                self.task_adjacency,
                md_inputs=_inputs(opportunity=opportunity),
            )
            self.assertTrue(
                torch.equal(
                    disabled_base.downstream_residual,
                    disabled_changed.downstream_residual,
                ),
                name,
            )



if __name__ == "__main__":
    unittest.main()
