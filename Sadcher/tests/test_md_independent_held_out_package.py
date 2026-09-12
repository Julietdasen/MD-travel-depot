import json
import tempfile
import unittest
from pathlib import Path

from experiments.md_independent_held_out_package import (
    HELD_OUT_MARGIN_STRATA,
    run_independent_held_out_package,
)
from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES


class MDIndependentHeldOutPackageTests(unittest.TestCase):
    def test_package_is_deterministic_stratified_disjoint_and_model_blind(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "candidate_pool_per_family": 500,
                "train_pairs_per_family": 20,
                "quota_per_family_stratum": 1,
                "train_seed": 3030,
                "evaluation_seed": 3939,
                "perturbation_seed": 3940,
            }
            first = run_independent_held_out_package(root / "first", **kwargs)
            second = run_independent_held_out_package(root / "second", **kwargs)
            first_summary = json.loads(first.summary_path.read_text())
            second_summary = json.loads(second.summary_path.read_text())
            first_package = json.loads(first.package_path.read_text())
            second_package = json.loads(second.package_path.read_text())
            different_seed_kwargs = dict(kwargs)
            different_seed_kwargs["perturbation_seed"] = 3941
            different = run_independent_held_out_package(
                root / "different", **different_seed_kwargs
            )
            different_package = json.loads(different.package_path.read_text())

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_package, second_package)
            self.assertNotEqual(
                first_package["evaluation_records"],
                different_package["evaluation_records"],
            )
            self.assertEqual(first.report_path.read_text(), second.report_path.read_text())
            self.assertEqual(first_summary["protocol_status"], "diagnostic_only")
            self.assertFalse(first_summary["formal_protocol_run"])
            self.assertTrue(first_summary["evaluation_package_written"])
            self.assertFalse(first_summary["ticket_40_authorized"])
            self.assertEqual(first_summary["train_evaluation_template_overlap"], 0)
            self.assertEqual(first_package["evaluation_pair_count"], 16)
            self.assertEqual(first_package["evaluation_state_count"], 32)
            self.assertEqual(
                set(first_package["pair_margin_strata"]),
                {name for name, _lower, _upper in HELD_OUT_MARGIN_STRATA},
            )
            cell_counts = {
                (row["family"], row["pair_margin_stratum"]): 0
                for row in first_summary["cells"]
            }
            for record in first_package["evaluation_records"]:
                cell_counts[(record["family"], record["pair_margin_stratum"])] += 1
            self.assertEqual(
                cell_counts,
                {
                    (family, stratum): 1
                    for family in RELATIONAL_FAMILIES
                    for stratum, _lower, _upper in HELD_OUT_MARGIN_STRATA
                },
            )
            self.assertEqual(
                sum(
                    record["pair_margin_stratum"] == "high_ge_0.05"
                    for record in first_package["evaluation_records"]
                ),
                len(RELATIONAL_FAMILIES),
            )
            controls = first_package["selection_controls"]
            self.assertFalse(controls["oracle_action_used_for_selection"])
            self.assertFalse(controls["oracle_flip_used_for_selection"])
            self.assertFalse(controls["model_output_used_for_selection"])
            self.assertFalse(first_summary["model_training_or_evaluation_run"])
            self.assertFalse(
                first_summary["controls"]["production_expert_dataset_written"]
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_independent_held_out_package(root / "first", **kwargs)


if __name__ == "__main__":
    unittest.main()
