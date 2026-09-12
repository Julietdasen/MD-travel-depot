import json
import tempfile
import unittest
from pathlib import Path

from experiments.md_pair_structured_train_diagnostic import (
    run_pair_structured_train_diagnostic,
)
from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES


class MDPairStructuredTrainDiagnosticTests(unittest.TestCase):
    def test_train_comparison_is_deterministic_balanced_and_stops_before_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = run_pair_structured_train_diagnostic(
                root / "first",
                candidate_pool_per_family=8,
                train_pairs_per_family=4,
                pairs_per_family_per_batch=2,
                epochs=1,
                data_seed=3030,
                model_seeds=(101, 102, 103),
            )
            second = run_pair_structured_train_diagnostic(
                root / "second",
                candidate_pool_per_family=8,
                train_pairs_per_family=4,
                pairs_per_family_per_batch=2,
                epochs=1,
                data_seed=3030,
                model_seeds=(101, 102, 103),
            )
            first_summary = json.loads(first.summary_path.read_text())
            second_summary = json.loads(second.summary_path.read_text())

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first.report_path.read_text(), second.report_path.read_text())
            self.assertEqual(first_summary["status"], "pair_structured_train_diagnostic")
            self.assertEqual(first_summary["train_pair_count"], 16)
            self.assertEqual(first_summary["evaluation_pair_count"], 0)
            self.assertEqual(first_summary["training"]["batches_per_epoch"], 2)
            self.assertEqual(
                first_summary["training"]["family_pairs_per_batch"],
                {family: 2 for family in RELATIONAL_FAMILIES},
            )
            self.assertEqual(
                set(first_summary["variants"]),
                {"family_balanced_state_only", "family_balanced_pair_structured"},
            )
            for variant in first_summary["variants"].values():
                self.assertEqual(
                    set(variant["methods"]),
                    {"matched_parameter_mlp", "pair_aware_attention"},
                )
                for method in variant["methods"].values():
                    self.assertEqual(len(method["per_seed"]), 3)
                    for row in method["per_seed"]:
                        self.assertEqual(
                            set(row["oracle_flip_pair_exact_by_family"]),
                            set(RELATIONAL_FAMILIES),
                        )
            self.assertEqual(first_summary["decision"], "diagnostic_only")
            self.assertFalse(first_summary["controls"]["held_out_data_read"])
            self.assertEqual(
                first_summary["controls"]["held_out_stage_authorized"],
                False,
            )
            self.assertFalse(
                first_summary["controls"]["production_expert_dataset_written"]
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_pair_structured_train_diagnostic(
                    root / "first",
                    candidate_pool_per_family=8,
                    train_pairs_per_family=4,
                    pairs_per_family_per_batch=2,
                    epochs=1,
                    data_seed=3030,
                    model_seeds=(101, 102, 103),
                )

    def test_non_formal_passing_conditions_cannot_authorize_held_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_pair_structured_train_diagnostic(
                Path(temporary) / "non-formal",
                candidate_pool_per_family=20,
                train_pairs_per_family=20,
                pairs_per_family_per_batch=5,
                epochs=1,
                data_seed=3030,
                model_seeds=(101, 102, 103),
                minimum_overall_train_agreement=0.0,
                minimum_family_flip_pair_exact=0.0,
                maximum_residual_saturation_rate=1.0,
            )
            summary = json.loads(result.summary_path.read_text())

        self.assertFalse(summary["formal_protocol_run"])
        self.assertTrue(summary["candidate_training_conditions_met"])
        self.assertEqual(summary["decision"], "diagnostic_only")
        self.assertFalse(summary["controls"]["held_out_stage_authorized"])


if __name__ == "__main__":
    unittest.main()
