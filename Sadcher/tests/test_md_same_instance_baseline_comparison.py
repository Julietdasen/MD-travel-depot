import unittest

from experiments.md_same_instance_baseline_comparison import (
    DEFAULT_OUTPUT_ROOT,
    _normalize_ticket46_row,
    analyze_instance_differences,
    aggregate_baseline_rows,
    paired_baseline_rows,
)


class SameInstanceBaselineComparisonTests(unittest.TestCase):
    def test_default_output_root_is_preregistered_directory(self):
        self.assertEqual(
            DEFAULT_OUTPUT_ROOT.name, "md_same_instance_baseline_comparison_2026-09-02"
        )

    def test_unavailable_solver_is_not_counted_as_rollout_failure(self):
        rows = (
            {
                "instance_id": "a",
                "success": True,
                "makespan": 10.0,
                "execution_status": "completed",
                "material_starvation": {"1": 2.0},
                "robot_utilization": {"0": 0.5},
            },
            {
                "instance_id": "b",
                "success": None,
                "makespan": None,
                "execution_status": "solver_unavailable",
                "solver_status": "unavailable",
                "material_starvation": {},
                "robot_utilization": {},
            },
        )

        summary = aggregate_baseline_rows(rows)

        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(summary["evaluated_count"], 1)
        self.assertEqual(summary["unavailable_count"], 1)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["failure_counts"], {})

    def test_paired_comparison_keeps_survivor_bias_explicit(self):
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

        comparison = paired_baseline_rows(baseline, candidate)

        self.assertEqual(comparison["matched_pairs"], 3)
        self.assertEqual(comparison["common_successful_pairs"], 1)
        self.assertEqual(
            comparison["candidate_minus_reference"]["mean"], -2.0
        )
        self.assertEqual(comparison["survivor_bias"]["reference_only_successes"], 1)
        self.assertEqual(comparison["survivor_bias"]["candidate_only_successes"], 1)

    def test_paired_success_rates_use_each_method_evaluated_denominator(self):
        reference = (
            {"instance_id": "a", "success": True},
            {"instance_id": "b", "success": None},
        )
        candidate = (
            {"instance_id": "a", "success": True},
            {"instance_id": "b", "success": False},
        )

        comparison = paired_baseline_rows(reference, candidate)

        self.assertEqual(comparison["reference_evaluated_count"], 1)
        self.assertEqual(comparison["candidate_evaluated_count"], 2)
        self.assertEqual(comparison["reference_success_rate"], 1.0)
        self.assertEqual(comparison["candidate_success_rate"], 0.5)
        self.assertEqual(comparison["unavailable_pairs"], 1)

    def test_ticket46_row_is_labeled_and_keeps_execution_records(self):
        row = _normalize_ticket46_row(
            {
                "ticket": 46,
                "method": "C0",
                "experiment": {
                    "execution_records": {"process": [], "transport": []}
                },
            },
            method="physics_only",
        )

        self.assertEqual(row["ticket"], 47)
        self.assertEqual(row["source_ticket"], 46)
        self.assertEqual(row["source_method"], "C0")
        self.assertEqual(row["method"], "physics_only")
        self.assertEqual(row["execution_records"], {"process": [], "transport": []})

    def test_difference_analysis_decomposes_makespan_delta(self):
        reference = (
            {
                "instance_id": "a",
                "instance_seed": 1,
                "success": True,
                "makespan": 20.0,
                "material_starvation": {"1": 2.0},
                "execution_records": {
                    "process": [
                        {"task_id": 1, "started_at": 0, "completed_at": 10}
                    ],
                    "transport": [],
                },
            },
        )
        candidate = (
            {
                "instance_id": "a",
                "instance_seed": 1,
                "success": True,
                "makespan": 30.0,
                "material_starvation": {"1": 3.0},
                "execution_records": {
                    "process": [
                        {"task_id": 1, "started_at": 0, "completed_at": 15}
                    ],
                    "transport": [],
                },
            },
        )

        row = analyze_instance_differences(reference, candidate)[0]

        self.assertEqual(row["makespan_delta_candidate_minus_reference"], 10.0)
        self.assertEqual(row["candidate_latest_task_completion"], 15.0)
        self.assertEqual(row["reference_latest_task_completion"], 10.0)
        self.assertEqual(row["return_tail_delta_candidate_minus_reference"], 5.0)
        self.assertEqual(
            row["makespan_delta_candidate_minus_reference"],
            row["candidate_latest_task_completion"]
            - row["reference_latest_task_completion"]
            + row["return_tail_delta_candidate_minus_reference"],
        )


if __name__ == "__main__":
    unittest.main()
