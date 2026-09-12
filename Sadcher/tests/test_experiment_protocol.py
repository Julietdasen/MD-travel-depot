import json
import math
import unittest

from experiments.protocol import (
    PAIRED_EVALUATION_SEEDS,
    DatasetSplit,
    ExperimentResult,
    FailureReason,
    describe,
    paired_makespan_difference,
    summarize_results,
    task_level_split,
)


class ExperimentResultTests(unittest.TestCase):
    def test_success_requires_tasks_complete_and_robots_at_exit(self):
        result = ExperimentResult.succeeded(
            run_id="greedy-instance-7-seed-2",
            method="greedy",
            instance_id="instance-7",
            seed=2,
            split=DatasetSplit.TEST,
            makespan=42,
            all_real_tasks_completed=True,
            all_robots_at_exit=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.makespan, 42)
        self.assertIsNone(result.failure_reason)

        with self.assertRaisesRegex(ValueError, "all real tasks.*exit"):
            ExperimentResult.succeeded(
                run_id="invalid",
                method="greedy",
                instance_id="instance-7",
                seed=2,
                split=DatasetSplit.TEST,
                makespan=42,
                all_real_tasks_completed=True,
                all_robots_at_exit=False,
            )

    def test_failure_has_null_makespan_and_stable_reason_code(self):
        result = ExperimentResult.failed(
            run_id="sadcher-instance-9-seed-4",
            method="sadcher",
            instance_id="instance-9",
            seed=4,
            split=DatasetSplit.VALIDATION,
            reason=FailureReason.DEADLOCK,
            all_real_tasks_completed=False,
            all_robots_at_exit=False,
        )

        payload = result.to_dict()
        self.assertFalse(payload["termination"]["success"])
        self.assertIsNone(payload["metrics"]["makespan"])
        self.assertEqual(payload["termination"]["failure_reason"], "deadlock")
        self.assertIsNone(json.loads(result.to_json())["metrics"]["makespan"])

    def test_failure_cannot_carry_a_makespan(self):
        with self.assertRaisesRegex(ValueError, "failed run.*null makespan"):
            ExperimentResult(
                run_id="invalid",
                method="sadcher",
                instance_id="instance-9",
                seed=4,
                split=DatasetSplit.TEST,
                success=False,
                makespan=100,
                failure_reason=FailureReason.TIMEOUT,
                all_real_tasks_completed=False,
                all_robots_at_exit=False,
            )

    def test_metrics_schema_is_json_serializable_and_validated(self):
        result = ExperimentResult.succeeded(
            run_id="md-instance-2-seed-1",
            method="md_sadcher",
            instance_id="instance-2",
            seed=1,
            split=DatasetSplit.TEST,
            makespan=12.5,
            all_real_tasks_completed=True,
            all_robots_at_exit=True,
            material_starvation={"process-3": 2.0},
            robot_utilization={"robot-0": 0.75},
            inference_time_seconds=0.02,
            wall_time_seconds=0.5,
        )

        payload = json.loads(result.to_json())
        self.assertEqual(payload["protocol_version"], "1.0.0")
        self.assertEqual(payload["dataset"]["split"], "test")
        self.assertEqual(payload["metrics"]["material_starvation"], {"process-3": 2.0})
        self.assertEqual(payload["metrics"]["robot_utilization"], {"robot-0": 0.75})

        with self.assertRaisesRegex(ValueError, "robot utilization"):
            ExperimentResult.succeeded(
                run_id="invalid-utilization",
                method="md_sadcher",
                instance_id="instance-2",
                seed=1,
                split=DatasetSplit.TEST,
                makespan=12.5,
                all_real_tasks_completed=True,
                all_robots_at_exit=True,
                robot_utilization={"robot-0": 1.01},
            )

    def test_success_rejects_any_illegal_assignment(self):
        with self.assertRaisesRegex(ValueError, "illegal assignment"):
            ExperimentResult.succeeded(
                run_id="invalid-assignment",
                method="md_sadcher",
                instance_id="instance-2",
                seed=1,
                split=DatasetSplit.TEST,
                makespan=12.5,
                all_real_tasks_completed=True,
                all_robots_at_exit=True,
                illegal_assignment_count=1,
            )

    def test_validated_mappings_cannot_be_mutated_after_construction(self):
        utilization = {"robot-0": 0.75}
        result = ExperimentResult.succeeded(
            run_id="immutable-result",
            method="md_sadcher",
            instance_id="instance-2",
            seed=1,
            split=DatasetSplit.TEST,
            makespan=12.5,
            all_real_tasks_completed=True,
            all_robots_at_exit=True,
            robot_utilization=utilization,
        )

        utilization["robot-0"] = 2.0
        self.assertEqual(result.robot_utilization["robot-0"], 0.75)
        with self.assertRaises(TypeError):
            result.robot_utilization["robot-0"] = 2.0

    def test_metadata_must_be_a_mapping(self):
        with self.assertRaisesRegex(ValueError, "metadata must be a mapping"):
            ExperimentResult.succeeded(
                run_id="invalid-metadata",
                method="md_sadcher",
                instance_id="instance-2",
                seed=1,
                split=DatasetSplit.TEST,
                makespan=12.5,
                all_real_tasks_completed=True,
                all_robots_at_exit=True,
                metadata=["not", "a", "mapping"],
            )

    def test_result_schema_reserves_structured_execution_records(self):
        process_record = {"task_id": "process-3", "started_at": 4}
        result = ExperimentResult.succeeded(
            run_id="records-result",
            method="md_sadcher",
            instance_id="instance-2",
            seed=1,
            split=DatasetSplit.TEST,
            makespan=12.5,
            all_real_tasks_completed=True,
            all_robots_at_exit=True,
            process_execution_records=[process_record],
        )

        process_record["started_at"] = 99
        payload = result.to_dict()
        self.assertEqual(
            payload["execution_records"]["process"],
            [{"task_id": "process-3", "started_at": 4}],
        )
        self.assertEqual(payload["execution_records"]["transport"], [])


class ReproducibilityTests(unittest.TestCase):
    def test_paired_evaluation_seed_list_is_frozen(self):
        self.assertEqual(PAIRED_EVALUATION_SEEDS, tuple(range(30)))

    def test_task_level_split_is_stable_for_every_sample_in_a_task_group(self):
        first = task_level_split("instance-17")
        self.assertEqual(first, task_level_split("instance-17"))
        self.assertEqual(first, DatasetSplit.TRAIN)
        self.assertEqual(task_level_split("instance-0"), DatasetSplit.VALIDATION)
        self.assertEqual(task_level_split("instance-5"), DatasetSplit.TEST)


class StatisticsTests(unittest.TestCase):
    def test_describe_uses_sample_standard_deviation_and_normal_95_percent_ci(self):
        summary = describe([1.0, 2.0, 3.0])

        self.assertEqual(summary.count, 3)
        self.assertEqual(summary.mean, 2.0)
        self.assertEqual(summary.sample_standard_deviation, 1.0)
        margin = 1.96 / math.sqrt(3)
        self.assertAlmostEqual(summary.ci95_low, 2.0 - margin)
        self.assertAlmostEqual(summary.ci95_high, 2.0 + margin)

    def test_result_summary_does_not_impute_failed_makespans(self):
        successful = _success("instance-1", 0, 10)
        failed = _failure("instance-2", 0, FailureReason.TIMEOUT)

        summary = summarize_results([successful, failed])

        self.assertEqual(summary.total_runs, 2)
        self.assertEqual(summary.successful_runs, 1)
        self.assertEqual(summary.success_rate, 0.5)
        self.assertEqual(summary.failure_counts, {"timeout": 1})
        self.assertEqual(summary.makespan.count, 1)
        self.assertEqual(summary.makespan.mean, 10)

    def test_paired_difference_matches_instance_and_seed_and_uses_common_successes(self):
        baseline = [
            _success("instance-1", 0, 10),
            _success("instance-2", 0, 20),
            _failure("instance-3", 0, FailureReason.DEADLOCK),
        ]
        candidate = [
            _success("instance-1", 0, 8),
            _failure("instance-2", 0, FailureReason.TIMEOUT),
            _success("instance-3", 0, 30),
        ]

        summary = paired_makespan_difference(baseline, candidate)

        self.assertEqual(summary.matched_pairs, 3)
        self.assertEqual(summary.common_successful_pairs, 1)
        self.assertEqual(summary.candidate_minus_baseline.mean, -2)

    def test_paired_difference_rejects_a_missing_instance_seed_pair(self):
        baseline = [
            _success("instance-1", 0, 10),
            _success("instance-2", 0, 20),
        ]
        candidate = [_success("instance-1", 0, 8)]

        with self.assertRaisesRegex(ValueError, "identical instance/seed keys"):
            paired_makespan_difference(baseline, candidate)


def _success(instance_id: str, seed: int, makespan: float) -> ExperimentResult:
    return ExperimentResult.succeeded(
        run_id=f"method-{instance_id}-{seed}",
        method="method",
        instance_id=instance_id,
        seed=seed,
        split=task_level_split(instance_id),
        makespan=makespan,
        all_real_tasks_completed=True,
        all_robots_at_exit=True,
    )


def _failure(
    instance_id: str, seed: int, reason: FailureReason
) -> ExperimentResult:
    return ExperimentResult.failed(
        run_id=f"method-{instance_id}-{seed}",
        method="method",
        instance_id=instance_id,
        seed=seed,
        split=task_level_split(instance_id),
        reason=reason,
        all_real_tasks_completed=False,
        all_robots_at_exit=False,
    )


if __name__ == "__main__":
    unittest.main()
