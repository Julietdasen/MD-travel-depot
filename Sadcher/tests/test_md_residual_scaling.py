import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from baselines.md_oracle_types import (
    ForcedAssignmentBatch,
    ResidualMDState,
    ResidualOracleResult,
    ResidualOracleStatus,
    residual_task_map,
)
from data_generation.md_residual_dataset import (
    FORMAL_RESIDUAL_SPLIT_PLAN,
    ResidualBatchLabel,
    ResidualDecisionSample,
    ResidualSplitPlan,
    joint_projection_error,
    project_edge_labels,
)
from data_generation.md_residual_generation import (
    SnapshotCandidate,
    label_snapshot,
    snapshot_key,
)
from data_generation.md_residual_pipeline import (
    ResidualGenerationConfig,
    encode_training_features,
    generate_sharded_dataset,
)
from experiments.md_residual_evaluation import _attach_state_regret, _paired
from imitation_learning.md_residual_train import shape_grouped_mixed_batches
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from tests.fixtures.md_diagnostic_scenarios import LATE_MATERIAL

def _candidate(seed=11, method="masked_greedy"):
    simulator = MDDiscreteSimulator(
        LATE_MATERIAL.domain,
        exit_location=LATE_MATERIAL.exit_location,
    )
    return SnapshotCandidate(
        "instance-11",
        seed,
        method,
        LATE_MATERIAL.domain,
        simulator,
        encode_training_features(simulator),
    )

class ResidualScalingTests(unittest.TestCase):
    def test_formal_split_boundaries_and_disjoint_validation(self):
        self.assertEqual(FORMAL_RESIDUAL_SPLIT_PLAN.split_for_seed(75000), "train")
        self.assertEqual(FORMAL_RESIDUAL_SPLIT_PLAN.split_for_seed(75600), "development")
        self.assertEqual(FORMAL_RESIDUAL_SPLIT_PLAN.split_for_seed(75800), "test")
        self.assertEqual(
            ResidualSplitPlan.from_dict(FORMAL_RESIDUAL_SPLIT_PLAN.to_dict()),
            FORMAL_RESIDUAL_SPLIT_PLAN,
        )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            ResidualSplitPlan((1,), (1,), (2,))

    def test_snapshot_key_includes_rollout_method_and_satisfied_dependencies(self):
        left = _candidate(method="masked_greedy")
        right = _candidate(method="legacy_c0_seed3101")
        self.assertNotEqual(snapshot_key(left), snapshot_key(right))

    def test_task_map_uses_domain_order_not_pending_tuple_order(self):
        domain = LATE_MATERIAL.domain
        ordered = tuple(task.task_id for task in domain.tasks)
        state = ResidualMDState(
            0,
            {robot.robot_id: robot.location for robot in domain.robots},
            (),
            tuple(reversed(ordered)),
        )
        mapping = residual_task_map(domain, state)
        self.assertEqual(
            tuple(mapping.original_to_compact),
            ordered,
        )
        self.assertEqual(
            tuple(mapping.compact_to_original[index + 1] for index in range(len(ordered))),
            ordered,
        )

    def test_joint_projection_error_reports_non_additive_batch_cost(self):
        labels = (
            ResidualBatchLabel(ForcedAssignmentBatch(((0, 1),)), 10, 8, 2, 0),
            ResidualBatchLabel(ForcedAssignmentBatch(((1, 2),)), 10, 8, 2, 1),
            ResidualBatchLabel(ForcedAssignmentBatch(((0, 1), (1, 2))), 12, 9, 3, 1),
        )
        edges = project_edge_labels(labels)
        self.assertGreater(joint_projection_error(labels, edges), 0.0)

    def test_nonoptimal_batch_cache_resumes_without_second_solver_call(self):
        calls = []
        def solver(*args, **kwargs):
            calls.append(1)
            return ResidualOracleResult(
                ResidualOracleStatus.TIMEOUT,
                None,
                None,
                None,
                None,
                None,
                0.1,
                message="timeout",
            )

        plan = ResidualSplitPlan((11,), (12,), (13,))
        with tempfile.TemporaryDirectory() as directory:
            first = label_snapshot(
                _candidate(),
                solver=solver,
                split_plan=plan,
                batch_cache_dir=Path(directory),
            )
            second = label_snapshot(
                _candidate(),
                solver=solver,
                split_plan=plan,
                batch_cache_dir=Path(directory),
            )
        self.assertEqual(first.reason, "timeout")
        self.assertEqual(second.reason, "timeout")
        self.assertEqual(len(calls), 1)

    def test_shape_grouped_minibatches_are_strictly_half_and_half(self):
        candidate = _candidate()
        residual = ResidualDecisionSample(
            "instance-11",
            11,
            "train",
            ResidualMDState(0, {}, (), (1,)),
            (),
            (),
            {
                **candidate.training_features,
                "rollout_method": "masked_greedy",
            },
            (),
        )
        c0 = SimpleNamespace(
            robot_ids=tuple(candidate.training_features["robot_ids"]),
            task_ids=tuple(candidate.training_features["task_ids"]),
        )
        batches = tuple(shape_grouped_mixed_batches(
            (residual, residual, residual),
            (c0,),
            batch_size=4,
            seed=1,
        ))
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(left) == len(right) == 2 for left, right in batches))

    def test_paired_metrics_require_corresponding_model_seed(self):
        rows = [
            {
                "method": method,
                "model_seed": 3101,
                "instance_seed": seed,
                "success": True,
                "final_makespan": makespan,
                "latest_task_completion": completion,
                "illegal_assignment_count": 0,
            }
            for seed, method, makespan, completion in (
                (75800, "legacy_c0", 20, 15),
                (75800, "residual", 18, 15),
                (75801, "legacy_c0", 30, 20),
                (75801, "residual", 29, 21),
            )
        ]
        result = _paired(rows, 3101)
        self.assertEqual(result["mean_final_makespan_delta"], -1.5)
        self.assertTrue(result["task_completion_constraint_met"])

    def test_state_regret_uses_best_observed_same_progress_state(self):
        rows = [
            {
                "instance_id": "a",
                "regret_records": [{"pending_task_ids": [2, 3], "best_total_cost": 12.0}],
            },
            {
                "instance_id": "a",
                "regret_records": [{"pending_task_ids": [2, 3], "best_total_cost": 10.0}],
            },
        ]
        _attach_state_regret(rows)
        self.assertEqual(rows[0]["state_regret"], 2.0)
        self.assertEqual(rows[1]["state_regret"], 0.0)
        self.assertEqual(
            rows[0]["regret_records"][0]["state_regret"],
            2.0,
        )

    def test_resume_rejects_a_different_model_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps({
                    "split_plan": FORMAL_RESIDUAL_SPLIT_PLAN.to_dict(),
                    "model_seed": 3101,
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "model seed"):
                generate_sharded_dataset(
                    (),
                    directory,
                    config=ResidualGenerationConfig(model_seed=3102),
                    resume=True,
                )

if __name__ == "__main__":
    unittest.main()

