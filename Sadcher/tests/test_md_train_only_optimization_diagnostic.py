import json
import re
import tempfile
import unittest
from pathlib import Path

from experiments.md_train_only_optimization_diagnostic import (
    FORMAL_PROTOCOL,
    run_train_only_optimization_diagnostic,
)
from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES


class MDTrainOnlyOptimizationDiagnosticTests(unittest.TestCase):
    def test_small_diagnostic_is_deterministic_and_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = run_train_only_optimization_diagnostic(
                root / "first",
                candidate_pool_per_family=8,
                train_pairs_per_family=4,
                epochs=1,
                data_seed=3030,
                model_seeds=(101, 102, 103),
            )
            second = run_train_only_optimization_diagnostic(
                root / "second",
                candidate_pool_per_family=8,
                train_pairs_per_family=4,
                epochs=1,
                data_seed=3030,
                model_seeds=(101, 102, 103),
            )
            first_summary = json.loads(first.summary_path.read_text())
            second_summary = json.loads(second.summary_path.read_text())

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first.report_path.read_text(), second.report_path.read_text())
            self.assertEqual(first_summary["report_phase"], "train_only_optimization_diagnostic")
            self.assertEqual(first_summary["protocol_class"], "exploratory")
            self.assertEqual(first_summary["train_pair_count"], 16)
            self.assertEqual(first_summary["train_state_count"], 32)
            self.assertEqual(first_summary["evaluation_pair_count"], 0)
            self.assertEqual(first_summary["evaluation_state_count"], 0)
            self.assertEqual(
                set(first_summary["methods"]),
                {"matched_parameter_mlp", "pair_aware_attention"},
            )
            self.assertEqual(first_summary["model_seeds"], [101, 102, 103])
            self.assertEqual(
                first_summary["selection_controls"],
                {
                    "label_used_for_selection": False,
                    "margin_used_for_selection": False,
                    "model_output_used_for_selection": False,
                    "oracle_flip_used_for_selection": False,
                    "selection_order": "first_generated_pairs_per_family",
                },
            )
            self.assertFalse(first_summary["controls"]["held_out_data_read"])
            self.assertFalse(first_summary["controls"]["model_evaluation_run"])
            self.assertFalse(first_summary["controls"]["ticket_17_unfrozen"])
            self.assertFalse(first_summary["controls"]["ticket_20_unfrozen"])
            self.assertFalse(
                first_summary["controls"]["production_expert_dataset_written"]
            )
            self.assertIn(
                first_summary["decision"],
                {"optimization_feasible", "optimization_not_feasible"},
            )
            serialized = json.dumps(first_summary) + first.report_path.read_text()
            self.assertIsNone(re.search(r"\b(?:go|no_go|inconclusive)\b", serialized))
            for method in first_summary["methods"].values():
                self.assertEqual(len(method["per_seed"]), 3)
                for row in method["per_seed"]:
                    self.assertEqual(
                        set(row["oracle_flip_pair_exact_by_family"]),
                        set(RELATIONAL_FAMILIES),
                    )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_train_only_optimization_diagnostic(
                    root / "first",
                    candidate_pool_per_family=8,
                    train_pairs_per_family=4,
                    epochs=1,
                    data_seed=3030,
                    model_seeds=(101, 102, 103),
                )

    def test_formal_protocol_matches_the_approved_exploratory_run(self):
        self.assertEqual(FORMAL_PROTOCOL["candidate_pool_per_family"], 5000)
        self.assertEqual(FORMAL_PROTOCOL["train_pairs_per_family"], 500)
        self.assertEqual(FORMAL_PROTOCOL["data_seed"], 3030)
        self.assertEqual(FORMAL_PROTOCOL["model_seeds"], (3101, 3102, 3103))
        self.assertEqual(FORMAL_PROTOCOL["epochs"], 200)
        self.assertEqual(FORMAL_PROTOCOL["training_objective"], "bounded_margin")
        self.assertEqual(FORMAL_PROTOCOL["margin"], 0.1)
        self.assertEqual(FORMAL_PROTOCOL["minimum_overall_train_agreement"], 0.7)
        self.assertEqual(FORMAL_PROTOCOL["minimum_family_flip_pair_exact"], 0.6)
        self.assertEqual(FORMAL_PROTOCOL["maximum_residual_saturation_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
