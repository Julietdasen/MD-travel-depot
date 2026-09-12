import json
import tempfile
import unittest
from pathlib import Path

from experiments.md_independent_held_out_evaluation import (
    _render_report,
    run_independent_held_out_evaluation,
)
from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES


PACKAGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "md_independent_held_out_package_protocol_corrected_2026-08-28"
    / "independent_held_out_package.json"
)


class MDIndependentHeldOutEvaluationTests(unittest.TestCase):
    def test_reduced_training_replays_frozen_package_deterministically(self):
        package_before = PACKAGE_PATH.read_text()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "package_path": PACKAGE_PATH,
                "pretrain_epochs": 1,
                "fine_tune_epochs": 1,
                "checkpoint_interval": 1,
            }
            first = run_independent_held_out_evaluation(root / "first", **kwargs)
            second = run_independent_held_out_evaluation(root / "second", **kwargs)
            first_summary = json.loads(first.summary_path.read_text())
            second_summary = json.loads(second.summary_path.read_text())

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first.report_path.read_text(), second.report_path.read_text())
            self.assertEqual(
                first.report_path.read_text(),
                _render_report(first_summary),
            )
            self.assertEqual(PACKAGE_PATH.read_text(), package_before)
            self.assertFalse(first_summary["formal_protocol_run"])
            self.assertEqual(first_summary["decision"], "diagnostic_only")
            self.assertEqual(first_summary["evaluation_pair_count"], 400)
            self.assertEqual(first_summary["evaluation_state_count"], 800)
            self.assertEqual(first_summary["package_replay_mismatch_count"], 0)
            self.assertEqual(first_summary["train_evaluation_template_overlap"], 0)
            self.assertEqual(
                set(first_summary["models"]),
                {"matched_parameter_mlp", "pair_aware_attention"},
            )
            self.assertEqual(
                set(first_summary["baselines"]),
                {"local_eta", "eta_plus_priority", "competitor_aware_relational"},
            )
            for baseline in first_summary["baselines"].values():
                for grouping in (
                    "by_family",
                    "by_pair_margin_stratum",
                    "by_oracle_flip_status",
                ):
                    for metrics in baseline[grouping].values():
                        self.assertEqual(
                            metrics["residual_saturation_rate"], 0.0
                        )
            for model in first_summary["models"].values():
                self.assertEqual(len(model["per_seed"]), 3)
                self.assertEqual(
                    set(model["aggregate"]["by_family"]),
                    set(RELATIONAL_FAMILIES),
                )
                self.assertEqual(
                    set(model["aggregate"]["by_pair_margin_stratum"]),
                    {
                        "near_tie_lt_0.01",
                        "small_ge_0.01_lt_0.03",
                        "medium_ge_0.03_lt_0.05",
                        "high_ge_0.05",
                    },
                )
                grouped_results = [
                    model["aggregate"],
                    *(row["evaluation"] for row in model["per_seed"]),
                ]
                for result in grouped_results:
                    for grouping in (
                        "by_family",
                        "by_pair_margin_stratum",
                        "by_oracle_flip_status",
                    ):
                        for metrics in result[grouping].values():
                            self.assertIn(
                                "residual_saturation_rate", metrics
                            )
            self.assertFalse(
                first_summary["controls"]["production_expert_dataset_written"]
            )
            self.assertFalse(first_summary["controls"]["ticket_20_unfrozen"])
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_independent_held_out_evaluation(root / "first", **kwargs)


    def test_rejects_modified_formal_package_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = json.loads(PACKAGE_PATH.read_text())
            payload["evaluation_provenance"][
                "quota_per_family_stratum"
            ] = 24
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                ValueError, "does not match frozen Ticket 39 protocol"
            ):
                run_independent_held_out_evaluation(
                    root / "metadata-output",
                    package_path=metadata_path,
                    pretrain_epochs=1,
                    fine_tune_epochs=1,
                    checkpoint_interval=1,
                )

            payload = json.loads(PACKAGE_PATH.read_text())
            records = payload["evaluation_records"]
            records[0], records[1] = records[1], records[0]
            records_path = root / "records.json"
            records_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                ValueError, "replay did not match frozen records"
            ):
                run_independent_held_out_evaluation(
                    root / "records-output",
                    package_path=records_path,
                    pretrain_epochs=1,
                    fine_tune_epochs=1,
                    checkpoint_interval=1,
                )


if __name__ == "__main__":
    unittest.main()
