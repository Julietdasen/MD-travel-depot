import json
import tempfile
import unittest
from pathlib import Path

from experiments.md_unconditioned_relational_audit import (
    PAIR_MARGIN_STRATA,
    run_unconditioned_relational_candidate_audit,
)
from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES


class MDUnconditionedRelationalAuditTests(unittest.TestCase):
    def test_candidate_audit_is_deterministic_and_not_label_conditioned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = run_unconditioned_relational_candidate_audit(
                root / "first",
                candidates_per_family=20,
                seed=3030,
            )
            second = run_unconditioned_relational_candidate_audit(
                root / "second",
                candidates_per_family=20,
                seed=3030,
            )
            first_summary = json.loads(first.summary_path.read_text())
            second_summary = json.loads(second.summary_path.read_text())

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first.report_path.read_text(), second.report_path.read_text())
            self.assertEqual(first_summary["candidate_pair_count"], 80)
            self.assertEqual(first_summary["oracle_flip_pair_count"], 26)
            self.assertEqual(first_summary["oracle_no_flip_pair_count"], 54)
            self.assertEqual(
                {
                    family: row["oracle_flip_pair_count"]
                    for family, row in first_summary["family_statistics"].items()
                },
                {
                    "competitor_robot_position": 8,
                    "competitor_robot_speed": 8,
                    "alternative_task_pickup": 6,
                    "alternative_downstream_priority": 4,
                },
            )
            self.assertEqual(
                first_summary["selection_controls"]["label_used_for_acceptance"],
                False,
            )
            self.assertEqual(
                first_summary["selection_controls"]["margin_used_for_acceptance"],
                False,
            )
            self.assertFalse(
                first_summary["controls"]["production_expert_dataset_written"]
            )
            self.assertTrue(first.report_path.is_file())
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_unconditioned_relational_candidate_audit(
                    root / "first",
                    candidates_per_family=20,
                    seed=3030,
                )

    def test_family_margin_cells_cover_every_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_unconditioned_relational_candidate_audit(
                Path(temporary) / "audit",
                candidates_per_family=24,
                seed=3031,
            )
            summary = json.loads(result.summary_path.read_text())

        self.assertEqual(set(summary["family_statistics"]), set(RELATIONAL_FAMILIES))
        expected_strata = {name for name, _lower, _upper in PAIR_MARGIN_STRATA}
        for family in RELATIONAL_FAMILIES:
            family_row = summary["family_statistics"][family]
            self.assertEqual(family_row["candidate_pair_count"], 24)
            self.assertEqual(set(family_row["margin_strata"]), expected_strata)
            self.assertEqual(
                sum(
                    row["pair_count"]
                    for row in family_row["margin_strata"].values()
                ),
                24,
            )
            self.assertEqual(
                sum(
                    row["flip_pair_count"] + row["no_flip_pair_count"]
                    for row in family_row["margin_strata"].values()
                ),
                24,
            )
        self.assertEqual(
            summary["train_pair_count"] + summary["evaluation_pair_count"],
            summary["candidate_pair_count"],
        )
        self.assertEqual(
            summary["controls"]["train_evaluation_template_overlap"], 0
        )


if __name__ == "__main__":
    unittest.main()
