import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.md_independent_held_out_evaluation import _replay_package, _load_package
from experiments.md_pickup_high_margin_error_attribution import (
    MODEL_METHODS,
    _render_report,
    run_pickup_high_margin_error_attribution,
)
from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES

PACKAGE = Path(__file__).resolve().parents[1] / "reports" / "md_independent_held_out_package_protocol_corrected_2026-08-28" / "independent_held_out_package.json"
STRATA = {"near_tie_lt_0.01", "small_ge_0.01_lt_0.03", "medium_ge_0.03_lt_0.05", "high_ge_0.05"}


class MDPickupHighMarginAttributionTests(unittest.TestCase):
    def test_reduced_run_is_deterministic_and_post_hoc_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {"package_path": PACKAGE, "pretrain_epochs": 1, "fine_tune_epochs": 1, "checkpoint_interval": 1}
            first = run_pickup_high_margin_error_attribution(root / "first", **kwargs)
            second = run_pickup_high_margin_error_attribution(root / "second", **kwargs)
            a = json.loads(first.summary_path.read_text())
            b = json.loads(second.summary_path.read_text())
            self.assertEqual(a, b)
            self.assertEqual(first.report_path.read_text(), second.report_path.read_text())
            self.assertEqual(first.report_path.read_text(), _render_report(a))
            self.assertEqual(a["diagnostic_scope"], "post_hoc_diagnostic_only")
            self.assertEqual(a["evaluation_pair_count"], 400)
            self.assertEqual(a["evaluation_state_count"], 800)
            self.assertEqual(a["package_replay_mismatch_count"], 0)
            self.assertEqual(a["train_evaluation_template_overlap"], 0)
            self.assertFalse(a["controls"]["architecture_gate_reopened"])
            self.assertEqual(sum(a["comparison"]["win_tie_loss"].values()), 400)
            self.assertEqual(sum(a["comparison"]["high_margin_transitions"]["state"].values()), 600)
            self.assertEqual(sum(a["comparison"]["high_margin_transitions"]["exact_pair"].values()), 300)
            for grouping in ("win_tie_loss_by_family", "win_tie_loss_by_margin_stratum"):
                for counts in a["comparison"][grouping].values():
                    self.assertEqual(sum(counts.values()), 100)
            self.assertEqual(sum(sum(counts.values()) for counts in a["comparison"]["win_tie_loss_by_oracle_flip_status"].values()), 400)

    def test_records_groups_and_score_decomposition_are_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_pickup_high_margin_error_attribution(Path(temporary) / "run", pretrain_epochs=1, fine_tune_epochs=1, checkpoint_interval=1)
            summary = json.loads(result.summary_path.read_text())
        for method in MODEL_METHODS:
            model = summary["models"][method]
            self.assertEqual(model["parameter_count"], 12459)
            self.assertEqual(len(model["per_seed"]), 3)
            aggregate = model["aggregate"]
            self.assertEqual(aggregate["overall"]["pair_count"], 400)
            self.assertEqual(set(aggregate["by_family"]), set(RELATIONAL_FAMILIES))
            self.assertEqual(set(aggregate["by_margin_stratum"]), STRATA)
            self.assertEqual(set(aggregate["by_oracle_flip_status"]), {"oracle_flip", "oracle_no_flip"})
            for seed in model["per_seed"]:
                records = seed["records"]
                self.assertEqual(len(records), 400)
                self.assertEqual(len({r["pair_id"] for r in records}), 400)
                self.assertEqual({r["pair_margin_stratum"] for r in records}, STRATA)
                self.assertEqual({r["oracle_flip_status"] for r in records}, {"oracle_flip", "oracle_no_flip"})
                for row in records:
                    self.assertEqual({row["before"]["state_id"].rsplit("-", 1)[-1], row["after"]["state_id"].rsplit("-", 1)[-1]}, {"before", "after"})
                    decomposition = row["before"]["score_decomposition"]
                    for r, mask_row in enumerate(decomposition["hard_mask"]):
                        for t, feasible in enumerate(mask_row):
                            if feasible:
                                expected = decomposition["legacy"][r][t] + decomposition["physics"][r][t] + decomposition["bounded_residual"][r][t]
                                self.assertTrue(math.isclose(decomposition["combined"][r][t], expected, abs_tol=1e-6))
                            else:
                                self.assertIsNone(decomposition["combined"][r][t])
                    if row["family"] == "alternative_task_pickup":
                        features = row["pickup_features"]
                        self.assertEqual(len(features["states"]), 2)
                        for state in features["states"]:
                            self.assertAlmostEqual(state["alternative_robot_to_pickup_eta"], state["alternative_pickup_distance"] / 2.0)

    def test_frozen_package_replay_contract_is_unchanged(self):
        payload = _load_package(PACKAGE)
        provenance = payload["train_provenance"]
        evaluation = payload["evaluation_provenance"]
        _, _, mismatch, overlap = _replay_package(payload, candidate_pool_per_family=provenance["candidate_pool_per_family"], train_pairs_per_family=provenance["pairs_per_family"], train_seed=provenance["seed"], evaluation_seed=evaluation["seed"], perturbation_seed=payload["perturbation_seed"], quota_per_family_stratum=evaluation["quota_per_family_stratum"])
        self.assertEqual(mismatch, 0)
        self.assertEqual(overlap, 0)


if __name__ == "__main__":
    unittest.main()
