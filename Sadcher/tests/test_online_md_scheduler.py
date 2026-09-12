import unittest
from unittest.mock import patch

import torch

from data_generation.md_instance_generator import (
    MDGeneratorConfig,
    generate_md_instance,
)
from experiments.protocol import DatasetSplit
from schedulers.md_constrained_decoder import MaskedGreedyDecoder, simulator_hard_mask
from schedulers.online_md_scheduler import (
    ExplicitMIPFallback,
    FallbackOutcome,
    OnlineMDScheduler,
    ScoreOutput,
)
from tests.test_md_constrained_decoder import _domain, _scores


class OnlineMDSchedulerTests(unittest.TestCase):
    def test_normal_online_path_completes_without_any_solver_call(self):
        scheduler = OnlineMDScheduler(
            scorer=lambda simulator: _scores(),
            confidence_threshold=None,
            max_steps=50,
        )

        with patch("pulp.LpProblem.solve") as solve:
            result = scheduler.run(
                _domain(),
                run_id="normal-online",
                instance_id="online-domain",
                seed=3,
                split=DatasetSplit.TEST,
            )

        solve.assert_not_called()
        self.assertTrue(result.experiment.success)
        self.assertEqual(result.experiment.illegal_assignment_count, 0)
        self.assertEqual(result.fallback_call_count, 0)
        self.assertTrue(result.decision_records)
        self.assertTrue(all(not item.fallback_used for item in result.decision_records))

    def test_structured_score_output_preserves_encoder_and_scoring_timings(self):
        scheduler = OnlineMDScheduler(
            scorer=lambda simulator: ScoreOutput(_scores(), 0.001, 0.002),
            max_steps=50,
        )

        result = scheduler.run(
            _domain(),
            run_id="timed-online",
            instance_id="online-domain",
            seed=3,
            split=DatasetSplit.TEST,
        )

        self.assertTrue(result.experiment.success)
        self.assertTrue(result.decision_records)
        self.assertTrue(
            all(item.encoder_time_seconds == 0.001 for item in result.decision_records)
        )
        self.assertTrue(
            all(item.scoring_time_seconds == 0.002 for item in result.decision_records)
        )
        self.assertTrue(
            all(item.repair_time_seconds >= 0 for item in result.decision_records)
        )

    def test_low_confidence_explicitly_calls_mip_and_records_it(self):
        fallback = ExplicitMIPFallback(time_limit_seconds=2, threads=1)
        scheduler = OnlineMDScheduler(
            scorer=lambda simulator: torch.zeros_like(_scores()),
            decoder=MaskedGreedyDecoder(),
            fallback=fallback,
            confidence_threshold=1.0,
            max_steps=50,
        )

        result = scheduler.run(
            _domain(),
            run_id="fallback-online",
            instance_id="online-domain",
            seed=3,
            split=DatasetSplit.TEST,
        )

        self.assertTrue(result.experiment.success)
        self.assertGreater(result.fallback_call_count, 0)
        self.assertTrue(any(item.fallback_used for item in result.decision_records))
        for record in result.fallback_records:
            self.assertEqual(record.trigger_reason, "low_confidence")
            self.assertGreaterEqual(record.solver_time_seconds, 0.0)
            self.assertTrue(record.assignments)
            self.assertIsNone(record.failure_reason)

    def test_failed_fallback_returns_stable_failure_without_fake_makespan(self):
        def failed_fallback(scores, simulator):
            return FallbackOutcome.failed("mip_no_incumbent", 0.01)

        scheduler = OnlineMDScheduler(
            scorer=lambda simulator: torch.zeros_like(_scores()),
            fallback=failed_fallback,
            confidence_threshold=1.0,
            max_steps=10,
        )

        result = scheduler.run(
            _domain(),
            run_id="failed-fallback",
            instance_id="online-domain",
            seed=3,
            split=DatasetSplit.TEST,
        )

        self.assertFalse(result.experiment.success)
        self.assertIsNone(result.experiment.makespan)
        self.assertEqual(result.failure_reason, "mip_no_incumbent")
        self.assertEqual(result.fallback_call_count, 1)
        self.assertEqual(
            result.fallback_records[0].failure_reason, "mip_no_incumbent"
        )

    def test_waits_for_advancing_work_when_pairwise_mask_has_no_coalition(self):
        domain = generate_md_instance(
            MDGeneratorConfig(
                seed=45999,
                task_count=12,
                transport_ratio=0.25,
                precedence_density=0.25,
                critical_path_length=3,
                capacity_slack=0.2,
                speed_ratio=0.8,
                process_robot_count=3,
                transport_robot_count=2,
                skill_count=3,
            )
        ).domain
        scheduler = OnlineMDScheduler(
            scorer=lambda simulator: torch.zeros_like(
                simulator_hard_mask(simulator), dtype=torch.float32
            ),
            confidence_threshold=None,
            max_steps=10_000,
        )

        result = scheduler.run(
            domain,
            run_id="wait-online",
            instance_id="wait-domain",
            seed=45999,
            split=DatasetSplit.TEST,
        )

        self.assertTrue(result.experiment.success)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.fallback_call_count, 0)
        self.assertTrue(any(not row.assignments for row in result.decision_records))

    def test_skips_non_dispatchable_waits_without_changing_assignments(self):
        domain = generate_md_instance(
            MDGeneratorConfig(
                seed=45999,
                task_count=12,
                transport_ratio=0.25,
                precedence_density=0.25,
                critical_path_length=3,
                capacity_slack=0.2,
                speed_ratio=0.8,
                process_robot_count=3,
                transport_robot_count=2,
                skill_count=3,
            )
        ).domain
        polling_calls = 0
        event_gated_calls = 0

        def polling_scorer(simulator):
            nonlocal polling_calls
            polling_calls += 1
            return torch.zeros_like(
                simulator_hard_mask(simulator), dtype=torch.float32
            )

        def event_gated_scorer(simulator):
            nonlocal event_gated_calls
            event_gated_calls += 1
            return torch.zeros_like(
                simulator_hard_mask(simulator), dtype=torch.float32
            )

        polling = OnlineMDScheduler(
            scorer=polling_scorer,
            confidence_threshold=None,
            max_steps=10_000,
        ).run(
            domain,
            run_id="polling-online",
            instance_id="wait-domain",
            seed=45999,
            split=DatasetSplit.TEST,
        )
        event_gated = OnlineMDScheduler(
            scorer=event_gated_scorer,
            confidence_threshold=None,
            max_steps=10_000,
            skip_non_dispatchable_states=True,
        ).run(
            domain,
            run_id="event-gated-online",
            instance_id="wait-domain",
            seed=45999,
            split=DatasetSplit.TEST,
        )

        self.assertTrue(polling.experiment.success)
        self.assertTrue(event_gated.experiment.success)
        self.assertEqual(polling.experiment.makespan, event_gated.experiment.makespan)
        self.assertEqual(
            polling.experiment.material_starvation,
            event_gated.experiment.material_starvation,
        )
        self.assertEqual(
            [
                (record.decision_time, record.assignments)
                for record in polling.decision_records
                if record.assignments
            ],
            [
                (record.decision_time, record.assignments)
                for record in event_gated.decision_records
                if record.assignments
            ],
        )
        self.assertLess(event_gated_calls, polling_calls)
        self.assertFalse(
            any(not record.assignments for record in event_gated.decision_records)
        )


if __name__ == "__main__":
    unittest.main()
