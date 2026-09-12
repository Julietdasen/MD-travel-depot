import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.md_policy_signal_gate import (
    SEMANTIC_FAMILIES,
    build_semantic_twins,
    run_md_policy_signal_gate,
)


class MDPolicySignalGateTests(unittest.TestCase):
    def test_twins_cover_each_semantic_factor_with_same_legality_and_flipped_oracle(self):
        twins = build_semantic_twins(pairs_per_family=2, seed=71)

        self.assertEqual(len(twins), 2 * len(SEMANTIC_FAMILIES))
        self.assertEqual(
            Counter(twin.family for twin in twins),
            {family: 2 for family in SEMANTIC_FAMILIES},
        )
        for twin in twins:
            self.assertEqual(twin.before.hard_mask, twin.after.hard_mask)
            self.assertNotEqual(twin.before.oracle_action, twin.after.oracle_action)
            self.assertEqual(twin.before.intervention_family, twin.family)
            self.assertEqual(twin.after.intervention_family, twin.family)
        templates = {
            (twin.before.eta, twin.before.opportunity, twin.after.opportunity)
            for twin in twins
        }
        self.assertEqual(len(templates), len(twins))

    def test_small_gate_is_deterministic_finite_and_writes_auditable_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = run_md_policy_signal_gate(
                Path(temporary) / "first",
                pairs_per_family=2,
                epochs=2,
                seed=73,
            )
            second = run_md_policy_signal_gate(
                Path(temporary) / "second",
                pairs_per_family=2,
                epochs=2,
                seed=73,
            )

            first_summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
            second_summary = json.loads(second.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_summary["ticket"], 19)
            self.assertEqual(first_summary["status"], "semantic_signal_gate")
            self.assertEqual(first_summary["pairs_per_family"], 2)
            self.assertEqual(first_summary["oracle"]["kind"], "exact_proxy_mip")
            self.assertFalse(
                first_summary["controls"]["signed_opportunity_prior_enabled"]
            )
            self.assertEqual(
                first_summary["controls"]["unique_pair_template_count"],
                2 * len(SEMANTIC_FAMILIES),
            )
            self.assertEqual(
                first_summary["controls"]["train_evaluation_template_overlap"], 0
            )
            self.assertEqual(
                first_summary["controls"]["confidence_interval_unit"],
                "semantic_pair",
            )
            self.assertEqual(
                first_summary["pre_registered_kill_criterion"][
                    "minimum_agreement_gain_over_initial"
                ],
                0.05,
            )
            self.assertEqual(
                set(first_summary["methods"]),
                {
                    "physics_only",
                    "residual_only",
                    "no_downstream_no_coalition",
                    "matched_parameter_mlp",
                    "cross_attention_full",
                    "eta_unlock_heuristic",
                },
            )
            self.assertEqual(
                first_summary["methods"]["matched_parameter_mlp"]["parameter_count"],
                first_summary["methods"]["cross_attention_full"]["parameter_count"],
            )
            self.assertNotIn("ticket_20", json.dumps(first_summary))
            for metrics in first_summary["methods"].values():
                for key in (
                    "first_action_agreement",
                    "semantic_action_flip_accuracy",
                    "mean_one_step_oracle_regret",
                    "mean_residual_magnitude",
                    "initial_evaluation_agreement",
                ):
                    self.assertTrue(math.isfinite(metrics[key]))
                self.assertEqual(len(metrics["agreement_ci95"]), 2)
                self.assertEqual(len(metrics["flip_ci95"]), 2)
            self.assertTrue(first.report_path.is_file())
            self.assertIn(
                "not a Ticket 20 benchmark",
                first.report_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
