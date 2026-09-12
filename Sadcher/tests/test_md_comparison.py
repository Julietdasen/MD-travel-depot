import unittest

from experiments.md_comparison import (
    ComparisonMethod,
    MethodSemantics,
    run_paired_md_comparison,
)
from experiments.protocol import DatasetSplit, ExperimentResult


def _result(method, instance, seed, makespan, starvation, utilization, gap=None):
    return ExperimentResult.succeeded(
        run_id=f"{method}-{instance}-{seed}",
        method=method,
        instance_id=instance,
        seed=seed,
        split=DatasetSplit.TEST,
        makespan=makespan,
        all_real_tasks_completed=True,
        all_robots_at_exit=True,
        material_starvation={"1": starvation},
        robot_utilization={"0": utilization},
        inference_time_seconds=0.01 * makespan,
        wall_time_seconds=0.02 * makespan,
        metadata={} if gap is None else {"gurobi_gap": gap},
    )


class MDComparisonTests(unittest.TestCase):
    def test_paired_report_keeps_semantics_failures_quality_and_runtime(self):
        cases = (("a", 0), ("b", 1))
        methods = (
            ComparisonMethod(
                "carry_as_skill",
                MethodSemantics.CARRY_AS_SKILL,
                lambda instance, seed: _result(
                    "carry_as_skill", instance, seed, 12 + seed, 4, 0.4
                ),
            ),
            ComparisonMethod(
                "md_policy",
                MethodSemantics.FULL_MD_POLICY,
                lambda instance, seed: _result(
                    "md_policy", instance, seed, 10 + seed, 2, 0.6
                ),
            ),
            ComparisonMethod(
                "gurobi",
                MethodSemantics.MD_ORACLE,
                lambda instance, seed: _result(
                    "gurobi", instance, seed, 9 + seed, 1, 0.7, gap=0.01
                ),
            ),
        )

        report = run_paired_md_comparison(cases, methods)
        payload = report.to_dict()

        self.assertEqual(payload["paired_keys"], [["a", 0], ["b", 1]])
        self.assertEqual(payload["methods"]["md_policy"]["success_rate"], 1.0)
        self.assertEqual(
            payload["methods"]["md_policy"]["material_starvation"]["mean"],
            2.0,
        )
        self.assertEqual(
            payload["methods"]["md_policy"]["robot_utilization"]["mean"],
            0.6,
        )
        self.assertIn("inference_time_seconds", payload["methods"]["md_policy"])
        self.assertEqual(payload["methods"]["gurobi"]["gurobi_gap"]["mean"], 0.01)
        comparison = report.compare("carry_as_skill", "md_policy")
        self.assertEqual(comparison.matched_pairs, 2)
        self.assertEqual(comparison.candidate_minus_baseline.mean, -2.0)


if __name__ == "__main__":
    unittest.main()
