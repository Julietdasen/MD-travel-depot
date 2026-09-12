import json
import tempfile
import unittest
from pathlib import Path

from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES
from experiments.md_train_error_diagnosis import run_train_error_diagnosis


class MDTrainErrorDiagnosisTests(unittest.TestCase):
    def test_diagnosis_is_deterministic_complete_and_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = run_train_error_diagnosis(
                root / "first",
                candidate_pool_per_family=8,
                train_pairs_per_family=4,
                epochs=1,
                data_seed=3030,
                model_seeds=(101, 102, 103),
            )
            second = run_train_error_diagnosis(
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
            self.assertEqual(first_summary["status"], "train_error_diagnosis")
            self.assertFalse(first_summary["formal_protocol_run"])
            self.assertEqual(first_summary["train_pair_count"], 16)
            self.assertEqual(first_summary["train_state_count"], 32)
            self.assertEqual(first_summary["evaluation_pair_count"], 0)
            self.assertFalse(first_summary["controls"]["held_out_data_read"])
            self.assertFalse(first_summary["controls"]["held_out_stage_authorized"])
            self.assertFalse(
                first_summary["controls"]["production_expert_dataset_written"]
            )
            self.assertIn("Reduced non-formal run", first.report_path.read_text())
            self.assertEqual(
                sum(first_summary["data_profile"]["oracle_action_histogram"].values()),
                32,
            )
            for method in first_summary["methods"].values():
                self.assertEqual(len(method["per_seed"]), 3)
                for row in method["per_seed"]:
                    self.assertEqual(sum(row["pair_outcomes"].values()), 16)
                    self.assertEqual(
                        sum(
                            split["pair_count"]
                            for split in row["flip_status_metrics"].values()
                        ),
                        16,
                    )
                    self.assertEqual(set(row["family_metrics"]), set(RELATIONAL_FAMILIES))
                    self.assertEqual(
                        sum(
                            family["pair_count"]
                            for family in row["family_metrics"].values()
                        ),
                        16,
                    )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_train_error_diagnosis(
                    root / "first",
                    candidate_pool_per_family=8,
                    train_pairs_per_family=4,
                    epochs=1,
                    data_seed=3030,
                    model_seeds=(101, 102, 103),
                )


if __name__ == "__main__":
    unittest.main()
