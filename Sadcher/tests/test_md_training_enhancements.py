import unittest
from unittest.mock import patch

import torch

from models.md_enhanced_policy import (
    MDEnhancedPolicyConfig,
    MDEnhancedSchedulerNetwork,
)
from models.md_policy_features import md_policy_inputs_from_samples
from models.md_training_enhancements import (
    MDStateValueHead,
    MDTrainingEnhancementConfig,
    coalition_feasible_assignments,
    compute_optional_training_losses,
    decoder_aware_structured_loss,
)
from tests.test_md_policy_smoke import _samples


class MDTrainingEnhancementTests(unittest.TestCase):
    def test_feasible_set_preserves_coalitions_singletons_and_hard_mask(self):
        sample = _samples()[0]
        candidates = coalition_feasible_assignments(sample)

        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertTrue(
                torch.all(
                    ~candidate
                    | torch.tensor(sample.hard_feasibility_mask, dtype=torch.bool)
                )
            )
            transport_column = sample.task_ids.index(3)
            self.assertLessEqual(int(candidate[:, transport_column].sum()), 1)

    def test_structured_loss_uses_constant_inference_and_finite_score_gradient(self):
        samples = _samples()
        scores = torch.randn(2, 2, 3, requires_grad=True)

        with patch("pulp.LpProblem.solve") as solve:
            loss = decoder_aware_structured_loss(scores, samples)
            loss.backward()

        solve.assert_not_called()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(scores.grad)
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_value_head_is_state_value_not_assignment_value(self):
        samples = _samples()
        md_inputs = md_policy_inputs_from_samples(samples)
        torch.manual_seed(67)
        policy = MDEnhancedSchedulerNetwork(
            7,
            9,
            16,
            32,
            4,
            1,
            4,
            1,
            use_idle=False,
            md_config=MDEnhancedPolicyConfig(hidden_dim=16),
        )
        value_head = MDStateValueHead(hidden_dim=16)
        robot_context = policy.robot_md_adapter(md_inputs.robot_metadata)
        task_context = policy.task_md_adapter(md_inputs.task_metadata)

        first = value_head(
            robot_context, task_context, md_inputs.task_is_transport
        )
        second = value_head(
            robot_context, task_context, md_inputs.task_is_transport
        )

        self.assertEqual(tuple(first.shape), (2,))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first, second))

    def test_optional_losses_are_disabled_by_default_and_independently_enabled(self):
        samples = _samples()
        md_inputs = md_policy_inputs_from_samples(samples)
        policy = MDEnhancedSchedulerNetwork(
            7,
            9,
            16,
            32,
            4,
            1,
            4,
            1,
            use_idle=False,
            md_config=MDEnhancedPolicyConfig(hidden_dim=16),
        )
        scores = torch.randn(2, 2, 3, requires_grad=True)

        disabled = compute_optional_training_losses(
            scores,
            samples,
            policy=policy,
            md_inputs=md_inputs,
            config=MDTrainingEnhancementConfig(),
        )
        self.assertEqual(float(disabled.total), 0.0)
        self.assertIsNone(disabled.predicted_value)

        value_head = MDStateValueHead(hidden_dim=16)
        enabled = compute_optional_training_losses(
            scores,
            samples,
            policy=policy,
            md_inputs=md_inputs,
            value_head=value_head,
            config=MDTrainingEnhancementConfig(
                use_structured_loss=True,
                use_value_head=True,
            ),
        )
        self.assertTrue(torch.isfinite(enabled.total))
        self.assertGreaterEqual(float(enabled.structured), 0.0)
        self.assertGreaterEqual(float(enabled.value), 0.0)
        self.assertIsNotNone(enabled.predicted_value)


if __name__ == "__main__":
    unittest.main()
