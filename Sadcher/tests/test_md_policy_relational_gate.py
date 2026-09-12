import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.md_policy_relational_gate import (
    CANDIDATE_TASK_COUNT,
    RELATIONAL_FAMILIES,
    ROBOT_COUNT,
    build_relational_twins,
    select_margin_balanced_twins,
    run_md_policy_relational_gate,
)


class MDPolicyRelationalGateTests(unittest.TestCase):
    def test_twins_are_unique_nonlocal_multi_robot_conflicts(self):
        twins = build_relational_twins(pairs_per_family=2, seed=91)

        self.assertEqual(len(twins), 2 * len(RELATIONAL_FAMILIES))
        self.assertEqual(
            Counter(twin.family for twin in twins),
            {family: 2 for family in RELATIONAL_FAMILIES},
        )
        signatures = set()
        for twin in twins:
            self.assertEqual(
                len(twin.before.robot_positions),
                ROBOT_COUNT,
            )
            self.assertEqual(
                len(twin.before.task_pickups),
                CANDIDATE_TASK_COUNT,
            )
            self.assertEqual(twin.before.hard_mask, twin.after.hard_mask)
            self.assertNotEqual(
                twin.before.oracle_action,
                twin.after.oracle_action,
            )
            if twin.family.startswith("competitor_robot"):
                self.assertNotEqual(
                    twin.before.oracle_action // CANDIDATE_TASK_COUNT,
                    twin.changed_entity_index,
                )
                self.assertNotEqual(
                    twin.after.oracle_action // CANDIDATE_TASK_COUNT,
                    twin.changed_entity_index,
                )
            else:
                self.assertNotEqual(
                    twin.before.oracle_action % CANDIDATE_TASK_COUNT,
                    twin.changed_entity_index,
                )
                self.assertNotEqual(
                    twin.after.oracle_action % CANDIDATE_TASK_COUNT,
                    twin.changed_entity_index,
                )
            signatures.add(
                (
                    twin.before.robot_positions,
                    twin.before.task_pickups,
                    twin.after.robot_positions,
                    twin.after.task_pickups,
                    twin.before.downstream_priorities,
                    twin.after.downstream_priorities,
                )
            )
        self.assertEqual(len(signatures), len(twins))

    def test_margin_balanced_selection_is_deterministic_and_disjoint(self):
        twins = build_relational_twins(pairs_per_family=250, seed=2028)
        selected = select_margin_balanced_twins(
            twins, pairs_per_family=2, min_margin=0.02
        )
        self.assertEqual(len(selected), 2 * len(RELATIONAL_FAMILIES))
        self.assertEqual(
            Counter(twin.family for twin in selected),
            {family: 2 for family in RELATIONAL_FAMILIES},
        )
        for twin in selected:
            self.assertGreaterEqual(
                min(
                    max(twin.before.action_values) - sorted(twin.before.action_values)[-2],
                    max(twin.after.action_values) - sorted(twin.after.action_values)[-2],
                ),
                0.02,
            )
        self.assertEqual(
            selected,
            select_margin_balanced_twins(
                twins, pairs_per_family=2, min_margin=0.02
            ),
        )


    def test_margin_balanced_arguments_must_be_paired(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "requires min_margin"):
                run_md_policy_relational_gate(
                    Path(temporary) / "invalid",
                    pairs_per_family=2,
                    selection_pool_per_family=20,
                    epochs=1,
                    seed=2028,
                    model_seeds=(101, 102, 103),
                )
            with self.assertRaisesRegex(ValueError, "requires selection_pool_per_family"):
                run_md_policy_relational_gate(
                    Path(temporary) / "invalid-margin",
                    pairs_per_family=2,
                    min_margin=0.02,
                    epochs=1,
                    seed=2028,
                    model_seeds=(101, 102, 103),
                )

    def test_margin_balanced_gate_is_explicitly_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_md_policy_relational_gate(
                Path(temporary) / "diagnostic",
                pairs_per_family=2,
                selection_pool_per_family=20,
                min_margin=0.02,
                epochs=2,
                seed=2028,
                model_seeds=(101, 102, 103),
            )
            summary = json.loads(result.summary_path.read_text())
            self.assertEqual(summary["report_phase"], "margin_balanced_diagnostic")
            self.assertEqual(summary["sampling"]["selection_pool_per_family"], 20)
            self.assertEqual(summary["sampling"]["minimum_state_margin"], 0.02)
            self.assertEqual(
                summary["decision"]["pair_aware_relational_claim"],
                "diagnostic_only",
            )
            self.assertFalse(summary["decision"]["ticket_17_formal_data_authorized"])
            self.assertFalse(summary["decision"]["ticket_20_authorized"])

    def test_small_gate_is_deterministic_and_auditable(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = run_md_policy_relational_gate(
                Path(temporary) / "first",
                pairs_per_family=2,
                epochs=2,
                seed=93,
                model_seeds=(101, 102, 103),
            )
            second = run_md_policy_relational_gate(
                Path(temporary) / "second",
                pairs_per_family=2,
                epochs=2,
                seed=93,
                model_seeds=(101, 102, 103),
            )
            first_summary = json.loads(first.summary_path.read_text())
            second_summary = json.loads(second.summary_path.read_text())

            first_duration = first_summary.pop("wall_time_seconds")
            second_duration = second_summary.pop("wall_time_seconds")
            self.assertGreater(first_duration, 0)
            self.assertGreater(second_duration, 0)
            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_summary["transport_robot_count"], 3)
            self.assertEqual(
                first_summary["controls"]["train_evaluation_template_overlap"],
                0,
            )
            self.assertFalse(
                first_summary["controls"]["production_dataset_written"]
            )
            self.assertIn(
                first_summary["decision"]["pair_aware_relational_claim"],
                {"go", "no_go", "inconclusive"},
            )
            self.assertEqual(first_summary["model_seeds"], [101, 102, 103])
            self.assertIn("pair_aware_attention", first_summary["methods"])
            self.assertEqual(
                first_summary["pre_registered_comparison_validity_check"][
                    "minimum_train_agreement"
                ],
                0.8,
            )
            self.assertIn(
                "minimum_train_agreement_for_architecture_comparison",
                first_summary["pre_registered_go_criterion"],
            )
            self.assertEqual(
                first_summary["methods"]["matched_parameter_mlp"][
                    "parameter_count"
                ],
                first_summary["methods"]["cross_attention_full"][
                    "parameter_count"
                ],
            )
            self.assertEqual(
                first_summary["methods"]["matched_parameter_mlp"][
                    "parameter_count"
                ],
                first_summary["methods"]["pair_aware_attention"][
                    "parameter_count"
                ],
            )
            for method in (
                "matched_parameter_mlp",
                "cross_attention_full",
                "pair_aware_attention",
            ):
                self.assertEqual(
                    len(first_summary["methods"][method]["per_seed"]),
                    3,
                )
            self.assertIsNotNone(
                first_summary["methods"]["cross_attention_full"][
                    "train_first_action_agreement"
                ]
            )
            self.assertEqual(
                set(
                    first_summary["methods"]["cross_attention_full"][
                        "context_gate_magnitudes"
                    ]
                ),
                {"robot_to_task", "task_to_robot"},
            )
            for metrics in first_summary["methods"].values():
                for key in (
                    "first_action_agreement",
                    "relational_flip_accuracy",
                    "mean_completion_proxy_regret",
                ):
                    self.assertTrue(math.isfinite(metrics[key]))
            self.assertTrue(
                first_summary["controls"]["parameter_budget_equal"]
            )
            self.assertTrue(first.report_path.is_file())
            pair_aware_metrics = first_summary["methods"]["pair_aware_attention"]
            self.assertIn("margin_strata", pair_aware_metrics)
            for stratum in ("near_tie_lt_0.01", "less_ambiguous_ge_0.01", "clear_ge_0.05"):
                self.assertIn(stratum, pair_aware_metrics["margin_strata"])
                self.assertGreaterEqual(pair_aware_metrics["margin_strata"][stratum]["state_count"], 0)
            self.assertEqual(
                sum(row["state_count"] for row in pair_aware_metrics["margin_strata"].values()),
                first_summary["evaluation_pair_count"] * 2,
            )
            report_text = first.report_path.read_text()
            self.assertIn("Report phase:", report_text)
            self.assertIn("Utility@0.05", report_text)
            self.assertIn("Margin-stratified diagnostics", report_text)
            self.assertIn(
                "not Ticket 20",
                first.report_path.read_text(),
            )


if __name__ == "__main__":
    unittest.main()
