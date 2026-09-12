import unittest

from experiments.md_scaling import ScalingCase, run_scaling_generalization
from experiments.protocol import DatasetSplit, ExperimentResult, FailureReason


class MDScalingTests(unittest.TestCase):
    def test_scaling_report_is_paired_split_safe_and_failure_aware(self):
        cases = (
            ScalingCase("small", "test-small", 0, {"task_count": 8}),
            ScalingCase("large", "test-large", 1, {"task_count": 16}),
        )

        def runner(method, case):
            if method == "full" and case.factor == "large":
                return ExperimentResult.failed(
                    run_id=f"{method}-{case.instance_id}",
                    method=method,
                    instance_id=case.instance_id,
                    seed=case.seed,
                    split=DatasetSplit.TEST,
                    reason=FailureReason.TIMEOUT,
                    all_real_tasks_completed=False,
                    all_robots_at_exit=False,
                )
            return ExperimentResult.succeeded(
                run_id=f"{method}-{case.instance_id}",
                method=method,
                instance_id=case.instance_id,
                seed=case.seed,
                split=DatasetSplit.TEST,
                makespan=10 + case.seed,
                all_real_tasks_completed=True,
                all_robots_at_exit=True,
                material_starvation={"1": 2.0},
            )

        report = run_scaling_generalization(
            cases,
            ("base", "full"),
            runner,
            train_instance_ids=("train-a", "train-b"),
        ).to_dict()

        self.assertEqual(report["paired_case_count"], 2)
        self.assertEqual(report["groups"]["large"]["full"]["success_rate"], 0.0)
        self.assertEqual(
            report["groups"]["large"]["full"]["failure_counts"],
            {"timeout": 1},
        )
        self.assertIsNone(report["groups"]["large"]["full"]["makespan"]["mean"])
        self.assertEqual(
            report["groups"]["small"]["base"]["material_starvation"]["mean"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
