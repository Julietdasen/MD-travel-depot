import unittest
from experiments.md_latency_evaluation import (
    LatencyEvaluationMethod,
    latency_percentiles,
    run_paired_latency_evaluation,
)
from experiments.protocol import DatasetSplit, ExperimentResult
from schedulers.online_md_scheduler import (
    OnlineDecisionRecord,
    OnlineSchedulerResult,
)


def _run(method: str, instance_id: str, seed: int, makespan: float, scale: float):
    experiment = ExperimentResult.succeeded(
        run_id=f"{method}-{instance_id}-{seed}",
        method=method,
        instance_id=instance_id,
        seed=seed,
        split=DatasetSplit.TEST,
        makespan=makespan,
        all_real_tasks_completed=True,
        all_robots_at_exit=True,
        material_starvation={"1": makespan / 10},
        robot_utilization={"0": 0.75},
        metadata={
            "hard_mask_source": "hard_feasibility",
            "terminal_protocol": "canonical_md_simulator",
        },
    )
    decision = OnlineDecisionRecord(
        decision_time=0,
        assignments=((0, 1),),
        confidence=1.0,
        encoder_time_seconds=scale,
        scoring_time_seconds=2 * scale,
        decoder_time_seconds=3 * scale,
        repair_time_seconds=scale / 2,
        fallback_time_seconds=4 * scale if method == "exact" else 0.0,
        total_time_seconds=10 * scale,
        fallback_used=method == "exact",
    )
    return OnlineSchedulerResult(experiment, (decision,), (), None)


class MDLatencyEvaluationTests(unittest.TestCase):
    def test_reports_components_quality_regret_and_tradeoff_on_paired_cases(self):
        cases = (("a", 1), ("b", 2))
        makespans = {
            "learned": {"a": 11.0, "b": 22.0},
            "greedy": {"a": 12.0, "b": 24.0},
            "exact": {"a": 10.0, "b": 20.0},
        }

        def method(name, kind, scale, exact=False):
            return LatencyEvaluationMethod(
                name,
                kind,
                lambda instance, seed: _run(
                    name, instance, seed, makespans[name][instance], scale
                ),
                exact_reference=exact,
            )

        report = run_paired_latency_evaluation(
            cases,
            (
                method("learned", "learned_decoder", 0.001),
                method("greedy", "masked_greedy", 0.0005),
                method("exact", "exact_mip", 0.01, exact=True),
            ),
        ).to_dict()

        learned = report["methods"]["learned"]
        self.assertEqual(set(learned["latency"]), {
            "encoder", "scoring", "decoder", "repair", "fallback", "total"
        })
        self.assertAlmostEqual(learned["quality"]["oracle_regret"]["mean"], 0.1)
        self.assertEqual(learned["quality"]["illegal_assignment_count"], 0)
        self.assertEqual(learned["quality"]["fallback_rate"], 0.0)
        self.assertEqual(report["methods"]["exact"]["quality"]["fallback_rate"], 1.0)
        self.assertGreater(
            report["quality_latency_tradeoff"]["exact"]["total_latency_p95_seconds"],
            report["quality_latency_tradeoff"]["learned"]["total_latency_p95_seconds"],
        )

    def test_rejects_noncanonical_or_unpaired_runs(self):
        exact = LatencyEvaluationMethod(
            "exact",
            "exact_mip",
            lambda instance, seed: _run("exact", instance, seed, 10, 0.01),
            exact_reference=True,
        )
        mismatched = LatencyEvaluationMethod(
            "learned",
            "learned_decoder",
            lambda instance, seed: _run("learned", "wrong", seed, 11, 0.001),
        )
        with self.assertRaisesRegex(ValueError, "paired case order"):
            run_paired_latency_evaluation((("a", 1),), (mismatched, exact))

    def test_percentiles_use_linear_interpolation(self):
        summary = latency_percentiles((1.0, 2.0, 3.0, 4.0))
        self.assertEqual(summary.p50_seconds, 2.5)
        self.assertAlmostEqual(summary.p95_seconds, 3.85)
        self.assertAlmostEqual(summary.p99_seconds, 3.97)


if __name__ == "__main__":
    unittest.main()
