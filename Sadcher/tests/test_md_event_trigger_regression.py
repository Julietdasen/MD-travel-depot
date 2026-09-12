import unittest

from experiments.md_event_trigger_regression import (
    EVENT_GATED,
    POLLING,
    build_regression_summary,
)


def _row(condition, *, seed, instance_id, scorer_calls, latency, wall_time):
    return {
        "condition": condition,
        "model_seed": seed,
        "instance_id": instance_id,
        "success": True,
        "failure_reason": None,
        "makespan": 100.0,
        "material_starvation": {"1": 4.0, "2": 6.0},
        "robot_utilization": {"0": 0.5},
        "illegal_assignment_count": 0,
        "terminal_state": {
            "all_real_tasks_completed": True,
            "all_robots_at_exit": True,
        },
        "action_trace": [{"decision_time": 0, "assignments": [[0, 1]]}],
        "fallback_trace": [],
        "scored_decision_count": scorer_calls,
        "empty_scored_decision_count": scorer_calls - 1,
        "fallback_count": 0,
        "solver_time_seconds": 0.0,
        "latency_totals": {
            "encoder_seconds": latency / 5,
            "scoring_seconds": latency / 5,
            "decoder_seconds": latency / 5,
            "repair_seconds": latency / 5,
            "fallback_seconds": latency / 5,
            "total_seconds": latency,
        },
        "wall_runtime_seconds": wall_time,
    }


class EventTriggerRegressionTests(unittest.TestCase):
    def test_reports_semantic_equivalence_and_paired_efficiency_reductions(self):
        polling = [
            _row(POLLING, seed=3101, instance_id="a", scorer_calls=10, latency=2, wall_time=3),
            _row(POLLING, seed=3101, instance_id="b", scorer_calls=8, latency=1, wall_time=2),
        ]
        event_gated = [
            _row(EVENT_GATED, seed=3101, instance_id="a", scorer_calls=4, latency=1, wall_time=2),
            _row(EVENT_GATED, seed=3101, instance_id="b", scorer_calls=5, latency=0.5, wall_time=1),
        ]

        summary = build_regression_summary(polling, event_gated)

        self.assertTrue(summary["semantic_regression"]["all_outcomes_identical"])
        self.assertEqual(summary["semantic_regression"]["identical_outcome_pair_count"], 2)
        self.assertEqual(summary["conditions"][POLLING]["total_scorer_calls"], 18)
        self.assertEqual(summary["conditions"][EVENT_GATED]["total_scorer_calls"], 9)
        self.assertEqual(summary["total_reductions_percent"]["scorer_calls"], 50.0)
        self.assertLess(
            summary["paired_deltas"]["total_decision_latency_seconds"][
                "event_gated_minus_polling_mean"
            ],
            0.0,
        )

    def test_surfaces_action_trace_mismatch(self):
        polling = [_row(POLLING, seed=3101, instance_id="a", scorer_calls=2, latency=1, wall_time=1)]
        event_gated = [_row(EVENT_GATED, seed=3101, instance_id="a", scorer_calls=1, latency=1, wall_time=1)]
        event_gated[0]["action_trace"] = [
            {"decision_time": 1, "assignments": [[0, 1]]}
        ]

        summary = build_regression_summary(polling, event_gated)

        self.assertFalse(summary["semantic_regression"]["all_outcomes_identical"])
        self.assertEqual(
            summary["semantic_regression"]["mismatches"][0]["different_fields"],
            ["action_trace"],
        )


if __name__ == "__main__":
    unittest.main()
