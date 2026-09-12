import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.md_context_ablation_development_package import (
    CORRECTED_HELD_OUT_PACKAGE,
    FORMAL_PROTOCOL,
    HELD_OUT_MARGIN_STRATA,
    render_context_ablation_development_report,
    run_context_ablation_development_package,
)
from experiments.md_context_ablation_relational_data import RELATIONAL_FAMILIES


FROZEN_REPORT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "md_context_ablation_development_package_2026-08-30"
)


class MDContextAblationDevelopmentPackageTests(unittest.TestCase):
    def test_import_does_not_load_torch_policy_or_training_modules(self):
        repository_root = Path(__file__).resolve().parents[1]
        command = (
            "import sys; "
            "import experiments.md_context_ablation_development_package; "
            "forbidden = [name for name in sys.modules "
            "if name == 'torch' "
            "or 'md_policy_relational_gate' in name "
            "or 'md_train_only_optimization_diagnostic' in name]; "
            "assert not forbidden, forbidden"
        )

        subprocess.run(
            [sys.executable, "-c", command],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_generator_does_not_import_training_or_model_modules(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "md_context_ablation_development_package.py"
        )
        tree = ast.parse(source_path.read_text())
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }

        self.assertFalse(
            {
                module
                for module in imported_modules
                if "train" in module
                or "policy" in module
                or module.startswith("models")
            }
        )

    def test_reduced_package_is_deterministic_stratified_and_model_blind(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "development_candidates_per_family": 500,
                "quota_per_family_stratum": 1,
            }

            first = run_context_ablation_development_package(
                root / "first", **kwargs
            )
            second = run_context_ablation_development_package(
                root / "second", **kwargs
            )
            first_summary = json.loads(first.summary_path.read_text())
            second_summary = json.loads(second.summary_path.read_text())
            first_package = json.loads(first.package_path.read_text())
            second_package = json.loads(second.package_path.read_text())

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_package, second_package)
            self.assertEqual(
                first.markdown_path.read_text(), second.markdown_path.read_text()
            )
            self.assertFalse(first_summary["formal_protocol_run"])
            self.assertFalse(first_summary["ticket_43_authorized"])
            self.assertEqual(first_summary["package_replay_mismatch_count"], 0)
            self.assertEqual(
                first_summary["training_development_template_overlap"], 0
            )
            self.assertEqual(
                first_summary["prior_held_out_development_template_overlap"], 0
            )
            self.assertEqual(len(first_summary["cells"]), 16)
            self.assertTrue(
                all(row["selected_pair_count"] == 1 for row in first_summary["cells"])
            )
            self.assertEqual(first_package["development_pair_count"], 16)
            self.assertEqual(first_package["development_state_count"], 32)
            self.assertEqual(
                len({row["pair_id"] for row in first_package["development_records"]}),
                16,
            )
            self.assertTrue(
                all(
                    [state["state_role"] for state in row["states"]]
                    == ["before", "after"]
                    for row in first_package["development_records"]
                )
            )
            cell_counts = {
                (row["family"], row["pair_margin_stratum"]): 0
                for row in first_summary["cells"]
            }
            for record in first_package["development_records"]:
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
                first_package["selection_controls"]["selection_fields"],
                ["family", "pair_margin"],
            )
            for forbidden in (
                "oracle_action_used_for_selection",
                "oracle_flip_used_for_selection",
                "model_output_used_for_selection",
                "ticket_41_classification_used_for_selection",
            ):
                self.assertFalse(first_package["selection_controls"][forbidden])
            self.assertEqual(
                first.markdown_path.read_text(),
                render_context_ablation_development_report(first_summary),
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_context_ablation_development_package(root / "first", **kwargs)

    def test_both_generation_seeds_change_selected_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "training_candidate_pool_per_family": 100,
                "training_pairs_per_family": 20,
                "development_candidates_per_family": 500,
                "quota_per_family_stratum": 1,
            }
            baseline = run_context_ablation_development_package(
                root / "baseline", **kwargs
            )
            different_base = run_context_ablation_development_package(
                root / "different-base",
                development_base_state_seed=4244,
                **kwargs,
            )
            different_perturbation = run_context_ablation_development_package(
                root / "different-perturbation",
                development_perturbation_seed=4245,
                **kwargs,
            )

            baseline_records = json.loads(
                baseline.package_path.read_text()
            )["development_records"]
            base_records = json.loads(
                different_base.package_path.read_text()
            )["development_records"]
            perturbation_records = json.loads(
                different_perturbation.package_path.read_text()
            )["development_records"]
            self.assertNotEqual(baseline_records, base_records)
            self.assertNotEqual(baseline_records, perturbation_records)

    def test_insufficient_supply_stops_without_writing_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_context_ablation_development_package(
                Path(temporary) / "infeasible",
                training_candidate_pool_per_family=20,
                training_pairs_per_family=10,
                development_candidates_per_family=20,
                quota_per_family_stratum=20,
            )
            summary = json.loads(result.summary_path.read_text())

            self.assertEqual(summary["status"], "development_package_infeasible")
            self.assertEqual(summary["protocol_status"], "infeasible")
            self.assertTrue(summary["insufficient_cells"])
            self.assertFalse(summary["development_package_written"])
            self.assertFalse(summary["ticket_43_authorized"])
            self.assertIsNone(result.package_path)
            self.assertFalse(
                (
                    result.summary_path.parent
                    / "context_ablation_development_package.json"
                ).exists()
            )

    def test_formal_protocol_is_the_frozen_ticket_42_protocol(self):
        self.assertEqual(FORMAL_PROTOCOL["training_seed"], 3030)
        self.assertEqual(FORMAL_PROTOCOL["training_pairs_per_family"], 500)
        self.assertEqual(
            FORMAL_PROTOCOL["development_candidates_per_family"], 10000
        )
        self.assertEqual(FORMAL_PROTOCOL["development_base_state_seed"], 4242)
        self.assertEqual(FORMAL_PROTOCOL["development_perturbation_seed"], 4243)
        self.assertEqual(FORMAL_PROTOCOL["quota_per_family_stratum"], 50)

    def test_frozen_package_has_complete_formal_counts_and_checks(self):
        summary = json.loads(
            (
                FROZEN_REPORT_DIRECTORY
                / "context_ablation_development_summary.json"
            ).read_text()
        )
        package = json.loads(
            (
                FROZEN_REPORT_DIRECTORY
                / "context_ablation_development_package.json"
            ).read_text()
        )
        markdown = (
            FROZEN_REPORT_DIRECTORY
            / "context_ablation_development_package.md"
        ).read_text()

        self.assertEqual(summary["status"], "development_package_frozen")
        self.assertTrue(summary["formal_protocol_run"])
        self.assertTrue(summary["ticket_43_authorized"])
        self.assertEqual(summary["development_pair_count"], 800)
        self.assertEqual(summary["unique_development_pair_count"], 800)
        self.assertEqual(summary["development_state_count"], 1600)
        self.assertEqual(summary["pairs_with_exactly_before_after_states"], 800)
        self.assertEqual(summary["package_replay_mismatch_count"], 0)
        self.assertEqual(summary["corrected_held_out_replay_mismatch_count"], 0)
        self.assertEqual(summary["training_development_template_overlap"], 0)
        self.assertEqual(
            summary["prior_held_out_development_template_overlap"], 0
        )
        self.assertEqual(len(summary["cells"]), 16)
        self.assertTrue(
            all(
                row["available_candidate_count"] >= 50
                and row["selected_pair_count"] == 50
                for row in summary["cells"]
            )
        )
        self.assertEqual(len(package["development_records"]), 800)
        self.assertEqual(
            len({row["pair_id"] for row in package["development_records"]}),
            800,
        )
        self.assertEqual(
            markdown, render_context_ablation_development_report(summary)
        )

    def test_formal_protocol_replay_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            replay = run_context_ablation_development_package(
                Path(temporary) / "formal-replay"
            )

            self.assertEqual(
                replay.summary_path.read_bytes(),
                (
                    FROZEN_REPORT_DIRECTORY
                    / "context_ablation_development_summary.json"
                ).read_bytes(),
            )
            self.assertEqual(
                replay.package_path.read_bytes(),
                (
                    FROZEN_REPORT_DIRECTORY
                    / "context_ablation_development_package.json"
                ).read_bytes(),
            )
            self.assertEqual(
                replay.markdown_path.read_bytes(),
                (
                    FROZEN_REPORT_DIRECTORY
                    / "context_ablation_development_package.md"
                ).read_bytes(),
            )

    def test_replay_mismatch_and_template_overlap_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            held_out = json.loads(CORRECTED_HELD_OUT_PACKAGE.read_text())
            held_out["evaluation_records"][0]["pair_id"] = "tampered"
            tampered_path = root / "tampered-held-out.json"
            tampered_path.write_text(json.dumps(held_out))
            reduced = {
                "training_candidate_pool_per_family": 20,
                "training_pairs_per_family": 10,
                "development_candidates_per_family": 500,
                "quota_per_family_stratum": 1,
            }

            mismatch = run_context_ablation_development_package(
                root / "mismatch",
                corrected_held_out_package_path=tampered_path,
                **reduced,
            )
            mismatch_summary = json.loads(mismatch.summary_path.read_text())
            overlap = run_context_ablation_development_package(
                root / "overlap",
                training_candidate_pool_per_family=20,
                training_pairs_per_family=10,
                development_candidates_per_family=5000,
                development_base_state_seed=3939,
                development_perturbation_seed=3940,
                quota_per_family_stratum=1,
            )
            overlap_summary = json.loads(overlap.summary_path.read_text())

            self.assertGreater(
                mismatch_summary["corrected_held_out_replay_mismatch_count"], 0
            )
            self.assertEqual(
                mismatch_summary["status"], "development_package_infeasible"
            )
            self.assertIsNone(mismatch.package_path)
            self.assertGreater(
                overlap_summary[
                    "prior_held_out_development_template_overlap"
                ],
                0,
            )
            self.assertEqual(
                overlap_summary["status"], "development_package_infeasible"
            )
            self.assertIsNone(overlap.package_path)


if __name__ == "__main__":
    unittest.main()
