import json
import tempfile
import unittest
from pathlib import Path

from experiments.md_c0_end_to_end_diagnostic_pilot import (
    FROZEN_INSTANCE_SEEDS,
    _finite_or_none,
    freeze_diagnostic_package,
    paired_makespan_summary,
    spearman_rank_correlation,
)


class C0EndToEndDiagnosticPilotTests(unittest.TestCase):
    def test_frozen_package_is_model_blind_and_write_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = freeze_diagnostic_package(root)

            self.assertEqual(summary["instance_seeds"], list(FROZEN_INSTANCE_SEEDS))
            self.assertEqual(summary["instance_count"], 30)
            self.assertFalse(summary["model_loaded_during_generation"])
            self.assertTrue((root / "frozen_protocol.json").exists())
            self.assertTrue((root / "frozen_instances.json").exists())
            self.assertTrue((root / "package_validation.json").exists())
            with self.assertRaises(FileExistsError):
                freeze_diagnostic_package(root)

    def test_paired_makespan_summary_keeps_failures_out_of_survivor_comparison(self):
        baseline = (
            {"instance_id": "a", "success": True, "makespan": 10.0},
            {"instance_id": "b", "success": True, "makespan": 20.0},
            {"instance_id": "c", "success": False, "makespan": None},
        )
        candidate = (
            {"instance_id": "a", "success": True, "makespan": 8.0},
            {"instance_id": "b", "success": False, "makespan": None},
            {"instance_id": "c", "success": True, "makespan": 30.0},
        )

        summary = paired_makespan_summary(baseline, candidate)

        self.assertEqual(summary["matched_pairs"], 3)
        self.assertEqual(summary["common_successful_pairs"], 1)
        self.assertEqual(summary["candidate_minus_baseline"]["mean"], -2.0)
        self.assertEqual(summary["survivor_bias"]["baseline_only_successes"], 1)
        self.assertEqual(summary["survivor_bias"]["candidate_only_successes"], 1)

    def test_spearman_rank_correlation_is_deterministic(self):
        self.assertAlmostEqual(
            spearman_rank_correlation((1.0, 2.0, 3.0), (3.0, 1.0, 2.0)),
            -0.5,
        )
        self.assertIsNone(spearman_rank_correlation((1.0,), (2.0,)))

    def test_unbounded_confidence_has_standard_json_representation(self):
        payload = {
            "confidence": _finite_or_none(float("inf")),
            "confidence_is_unbounded": True,
        }

        self.assertIsNone(payload["confidence"])
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)


if __name__ == "__main__":
    unittest.main()
