import json
import tempfile
import unittest
from pathlib import Path

from experiments.md_relational_data_audit import (
    AUDIT_MARGIN_STRATA,
    BASELINE_NAMES,
    PERTURBATION_MAGNITUDES,
    margin_stratum_name,
    run_relational_data_audit,
)
from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES


class MDRelationalDataAuditTests(unittest.TestCase):
    def test_audit_is_reproducible_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = run_relational_data_audit(
                root / "first",
                pairs_per_family=3,
                seed=290,
                perturbation_seed=291,
            )
            second = run_relational_data_audit(
                root / "second",
                pairs_per_family=3,
                seed=290,
                perturbation_seed=291,
            )

            self.assertEqual(
                json.loads(first.summary_path.read_text()),
                json.loads(second.summary_path.read_text()),
            )
            self.assertEqual(
                first.report_path.read_text(), second.report_path.read_text()
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_relational_data_audit(
                    root / "first",
                    pairs_per_family=3,
                    seed=290,
                    perturbation_seed=291,
                )

    def test_train_evaluation_templates_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_relational_data_audit(
                Path(temporary) / "audit",
                pairs_per_family=3,
                seed=292,
                perturbation_seed=293,
            )
            summary = json.loads(result.summary_path.read_text())

        self.assertEqual(
            summary["controls"]["train_evaluation_template_overlap"], 0
        )
        self.assertFalse(
            summary["controls"]["production_expert_dataset_written"]
        )
        self.assertEqual(
            {row["split"] for row in summary["pair_records"]},
            {"train", "evaluation"},
        )
        self.assertEqual(
            set(summary["family_statistics"]), set(RELATIONAL_FAMILIES)
        )
        for row in summary["family_statistics"].values():
            self.assertEqual(row["pair_count"], 3)
            self.assertEqual(row["oracle_action_flip_rate"], 1.0)
            self.assertIn("before_margin_distribution", row)
            self.assertIn("after_margin_distribution", row)

    def test_four_margin_strata_are_disjoint_and_cover_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_relational_data_audit(
                Path(temporary) / "audit",
                pairs_per_family=4,
                seed=294,
                perturbation_seed=295,
            )
            summary = json.loads(result.summary_path.read_text())

        strata = summary["evaluation_margin_strata"]
        self.assertEqual(
            set(strata), {name for name, _lower, _upper in AUDIT_MARGIN_STRATA}
        )
        state_ids = [
            state_id for row in strata.values() for state_id in row["state_ids"]
        ]
        self.assertEqual(len(state_ids), len(set(state_ids)))
        self.assertEqual(
            len(state_ids), summary["evaluation_pair_count"] * 2
        )
        self.assertTrue(
            summary["controls"]["margin_strata_are_mutually_exclusive"]
        )
        self.assertTrue(
            summary["controls"]["margin_strata_cover_all_evaluation_states"]
        )
        self.assertEqual(margin_stratum_name(0.0), "near_tie_lt_0.01")
        self.assertEqual(margin_stratum_name(0.01), "small_ge_0.01_lt_0.05")
        self.assertEqual(
            margin_stratum_name(0.05), "moderate_ge_0.05_lt_0.10"
        )
        self.assertEqual(margin_stratum_name(0.10), "clear_ge_0.10")

    def test_perturbation_statistics_and_rule_baselines_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_result = run_relational_data_audit(
                root / "first",
                pairs_per_family=3,
                seed=296,
                perturbation_seed=297,
            )
            second_result = run_relational_data_audit(
                root / "second",
                pairs_per_family=3,
                seed=296,
                perturbation_seed=297,
            )
            summary = json.loads(first_result.summary_path.read_text())
            second_summary = json.loads(second_result.summary_path.read_text())
        perturbations = summary["perturbations"]
        self.assertEqual(perturbations, second_summary["perturbations"])
        self.assertEqual(
            perturbations["comparison_count"],
            summary["pair_count"] * 2 * len(PERTURBATION_MAGNITUDES),
        )
        self.assertEqual(
            sum(
                row["label_flip_count"]
                for row in perturbations["by_feature"].values()
            ),
            perturbations["label_flip_count"],
        )
        self.assertEqual(set(summary["baselines"]), set(BASELINE_NAMES))
        for metrics in summary["baselines"].values():
            self.assertGreaterEqual(metrics["first_action_agreement"], 0.0)
            self.assertLessEqual(metrics["first_action_agreement"], 1.0)
            self.assertEqual(set(metrics["by_family"]), set(RELATIONAL_FAMILIES))


if __name__ == "__main__":
    unittest.main()
