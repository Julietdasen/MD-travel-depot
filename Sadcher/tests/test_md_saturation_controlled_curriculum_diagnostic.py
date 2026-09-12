import json
import tempfile
import unittest
from pathlib import Path

from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES
from experiments.md_saturation_controlled_curriculum_diagnostic import (
    run_saturation_controlled_curriculum_diagnostic,
)


class MDSaturationControlledCurriculumDiagnosticTests(unittest.TestCase):
    def test_reduced_run_is_deterministic_regularized_and_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "candidate_pool_per_family": 8,
                "train_pairs_per_family": 4,
                "pairs_per_family_per_batch": 2,
                "pretrain_epochs": 1,
                "fine_tune_epochs": 2,
                "checkpoint_interval": 1,
                "data_seed": 3030,
                "model_seeds": (101, 102, 103),
            }
            first = run_saturation_controlled_curriculum_diagnostic(
                root / "first", **kwargs
            )
            second = run_saturation_controlled_curriculum_diagnostic(
                root / "second", **kwargs
            )
            first_summary = json.loads(first.summary_path.read_text())
            second_summary = json.loads(second.summary_path.read_text())

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first.report_path.read_text(), second.report_path.read_text())
            self.assertEqual(
                first_summary["status"],
                "saturation_controlled_curriculum_diagnostic",
            )
            self.assertFalse(first_summary["formal_protocol_run"])
            self.assertEqual(first_summary["decision"], "diagnostic_only")
            self.assertEqual(first_summary["evaluation_pair_count"], 0)
            self.assertEqual(
                first_summary["training"]["raw_residual_saturation_penalty"],
                {
                    "boundary": "atanh(0.95)",
                    "eligible_actions": "hard_feasible_transport",
                    "objective": "mean_excess_absolute_raw_residual",
                    "weight": 0.1,
                },
            )
            for method in first_summary["methods"].values():
                self.assertEqual(len(method["per_seed"]), 3)
                for row in method["per_seed"]:
                    self.assertEqual(
                        set(row["oracle_flip_pair_exact_by_family"]),
                        set(RELATIONAL_FAMILIES),
                    )
            self.assertFalse(first_summary["controls"]["held_out_data_read"])
            self.assertFalse(first_summary["controls"]["held_out_stage_authorized"])
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_saturation_controlled_curriculum_diagnostic(
                    root / "first", **kwargs
                )

    def test_non_formal_passing_candidate_cannot_authorize_held_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_saturation_controlled_curriculum_diagnostic(
                Path(temporary) / "non-formal",
                candidate_pool_per_family=20,
                train_pairs_per_family=20,
                pairs_per_family_per_batch=5,
                pretrain_epochs=1,
                fine_tune_epochs=1,
                checkpoint_interval=1,
                data_seed=3030,
                model_seeds=(101, 102, 103),
                minimum_overall_train_agreement=0.0,
                minimum_family_flip_pair_exact=0.0,
                maximum_residual_saturation_rate=1.0,
            )
            summary = json.loads(result.summary_path.read_text())

        self.assertTrue(summary["candidate_training_conditions_met"])
        self.assertEqual(summary["decision"], "diagnostic_only")
        self.assertFalse(summary["all_training_conditions_met"])
        self.assertFalse(summary["controls"]["held_out_stage_authorized"])


if __name__ == "__main__":
    unittest.main()
