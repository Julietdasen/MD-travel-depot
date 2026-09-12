import json
import tempfile
import unittest
from pathlib import Path

from experiments.md_frozen_relational_evaluation import (
    run_frozen_margin_stratified_relational_evaluation,
)
from experiments.md_unconditioned_relational_audit import (
    run_unconditioned_relational_candidate_audit,
)
from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES


class MDFrozenRelationalEvaluationTests(unittest.TestCase):
    def test_insufficient_frozen_cells_write_infeasible_report_and_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = run_unconditioned_relational_candidate_audit(
                root / "candidate-audit",
                candidates_per_family=24,
                seed=3130,
            )
            result = run_frozen_margin_stratified_relational_evaluation(
                root / "frozen-evaluation",
                candidate_audit_path=audit.summary_path,
                quota_per_family_stratum=1,
                train_seed=3131,
                evaluation_seed=3132,
                perturbation_seed=3133,
            )
            summary = json.loads(result.summary_path.read_text())

            self.assertEqual(summary["protocol_status"], "infeasible")
            self.assertFalse(summary["evaluation_package_written"])
            self.assertIsNone(result.package_path)
            self.assertGreater(len(summary["insufficient_cells"]), 0)
            self.assertEqual(
                {row["family"] for row in summary["insufficient_cells"]},
                set(RELATIONAL_FAMILIES),
            )
            self.assertFalse(summary["selection_controls"]["oracle_flip_used"])
            self.assertEqual(
                summary["frozen_protocol"]["selection_order"],
                "RELATIONAL_FAMILIES order, then PAIR_MARGIN_STRATA order, "
                "then ascending deterministic candidate index",
            )
            self.assertFalse(
                summary["controls"]["production_expert_dataset_written"]
            )
            self.assertTrue(result.report_path.is_file())
            self.assertIn("Protocol infeasible", result.report_path.read_text())
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_frozen_margin_stratified_relational_evaluation(
                    root / "frozen-evaluation",
                    candidate_audit_path=audit.summary_path,
                    quota_per_family_stratum=1,
                    train_seed=3131,
                    evaluation_seed=3132,
                    perturbation_seed=3133,
                )

    def test_infeasible_decision_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = run_unconditioned_relational_candidate_audit(
                root / "candidate-audit",
                candidates_per_family=20,
                seed=3134,
            )
            first = run_frozen_margin_stratified_relational_evaluation(
                root / "first",
                candidate_audit_path=audit.summary_path,
                quota_per_family_stratum=2,
                train_seed=3135,
                evaluation_seed=3136,
                perturbation_seed=3137,
            )
            second = run_frozen_margin_stratified_relational_evaluation(
                root / "second",
                candidate_audit_path=audit.summary_path,
                quota_per_family_stratum=2,
                train_seed=3135,
                evaluation_seed=3136,
                perturbation_seed=3137,
            )

            self.assertEqual(
                json.loads(first.summary_path.read_text()),
                json.loads(second.summary_path.read_text()),
            )
            self.assertEqual(
                first.report_path.read_text(), second.report_path.read_text()
            )


if __name__ == "__main__":
    unittest.main()
