import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

IMITATION_LEARNING_ROOT = Path(__file__).resolve().parents[1] / "imitation_learning"
sys.path.insert(0, str(IMITATION_LEARNING_ROOT))

from baselines.aswale_23.MILP_solver import milp_scheduling
from baselines.aswale_23.greedy_solver import greedy_scheduling
from benchmarking.benchmark_helpers import run_one_simulation
from experiments.legacy_regression import (
    LEGACY_INSTANCE_FIELDS,
    LegacyTimingSemantics,
    build_legacy_regression_evidence,
    hash_legacy_result,
    load_legacy_problem,
    save_legacy_regression_evidence,
)
from helper_functions.schedules import Full_Horizon_Schedule
from imitation_learning.dataset import LazyLoadedSchedulingDataset


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "legacy_8t3r3s"
PROBLEM_PATH = FIXTURE_ROOT / "problem_instances" / "problem_instance_1p_000000.json"
SOLUTION_PATH = FIXTURE_ROOT / "solutions" / "optimal_schedule_1p_000000.json"
CHECKPOINT_PATH = (
    ROOT
    / "imitation_learning"
    / "checkpoints"
    / "hyperparam_2_8t3r3s"
    / "best_checkpoint.pt"
)


EXPECTED_ASSIGNMENTS = {
    "1": [0],
    "2": [0, 2],
    "3": [1],
    "4": [0],
    "5": [2],
    "6": [0],
    "7": [1],
    "8": [1, 2],
}

EXPECTED_RESULT_HASHES = {
    "legacy_greedy": "sha256:a0b0166e767d4d62feec841bad41d6d600dc94c7e8340620b6959e83c60992cb",
    "legacy_milp": "sha256:51fd80203c306f9cb8be971801c4d35fb200a4ba353c9dce2977927bd1104c7d",
    "legacy_sadcher": "sha256:c514325c3fc79dc97c5b8b93544eafd7c1e7def6ffe9570144794a12ca00e567",
}

def assignments(evidence):
    return evidence["result"]["metadata"]["legacy_regression"][
        "feasible_assignments"
    ]


class OfficialLegacyFixtureTests(unittest.TestCase):
    def test_official_fixture_keeps_the_original_process_only_schema(self):
        problem = load_legacy_problem(PROBLEM_PATH)

        self.assertEqual(set(problem), LEGACY_INSTANCE_FIELDS)
        self.assertNotIn("material_delivery", problem)
        self.assertEqual(problem["Q"].shape, (3, 3))
        self.assertEqual(problem["R"].shape, (10, 3))

    def test_existing_dataset_loader_reads_the_unconverted_fixture(self):
        dataset = LazyLoadedSchedulingDataset(
            FIXTURE_ROOT / "problem_instances", FIXTURE_ROOT / "solutions"
        )

        robot_features, task_features, reward, mask, adjacency = dataset[0]

        self.assertGreater(len(dataset), 0)
        self.assertEqual(tuple(robot_features.shape), (3, 7))
        self.assertEqual(tuple(task_features.shape), (8, 9))
        self.assertEqual(tuple(reward.shape), (3, 9))
        self.assertEqual(tuple(mask.shape), (3, 9))
        self.assertEqual(tuple(adjacency.shape), (8, 8))


class LegacySolverRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = load_legacy_problem(PROBLEM_PATH)

    def assert_evidence_contract(self, evidence, expected_makespan):
        result = evidence["result"]
        self.assertEqual(result["dataset"]["seed"], 0)
        self.assertTrue(result["termination"]["success"])
        self.assertAlmostEqual(result["metrics"]["makespan"], expected_makespan)
        self.assertFalse(
            result["metadata"]["legacy_regression"][
                "material_delivery_enabled"
            ]
        )
        self.assertEqual(evidence["result_hash"], hash_legacy_result(result))
        self.assertEqual(evidence["result_hash"], EXPECTED_RESULT_HASHES[result["method"]])

    def test_legacy_greedy_schedule_and_assignments_are_frozen(self):
        schedule = greedy_scheduling(self.problem, print_flag=False)
        evidence = build_legacy_regression_evidence(
            problem_instance=self.problem,
            schedule=schedule,
            method="legacy_greedy",
            instance_id="official-8t3r3s-1p-000000",
            seed=0,
            timing_semantics=LegacyTimingSemantics.CONTINUOUS,
        )

        self.assert_evidence_contract(evidence, 594.0197997477819)
        self.assertEqual(
            assignments(evidence),
            EXPECTED_ASSIGNMENTS,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "greedy.json"
            save_legacy_regression_evidence(output, evidence)
            self.assertEqual(json.loads(output.read_text()), evidence)

    def test_small_milp_matches_the_official_optimal_schedule(self):
        expected_schedule = Full_Horizon_Schedule.from_dict(
            json.loads(SOLUTION_PATH.read_text())
        )
        schedule = milp_scheduling(
            self.problem, n_threads=1, cutoff_time_seconds=30
        )

        self.assertIsNotNone(schedule)
        evidence = build_legacy_regression_evidence(
            problem_instance=self.problem,
            schedule=schedule,
            method="legacy_milp",
            instance_id="official-8t3r3s-1p-000000",
            seed=0,
            timing_semantics=LegacyTimingSemantics.CONTINUOUS,
        )

        self.assert_evidence_contract(evidence, expected_schedule.makespan)
        self.assertEqual(
            assignments(evidence),
            EXPECTED_ASSIGNMENTS,
        )

    def test_success_evidence_requires_return_to_exit(self):
        schedule = greedy_scheduling(self.problem, print_flag=False)
        schedule.makespan += 10

        with self.assertRaisesRegex(ValueError, "returned to exit"):
            build_legacy_regression_evidence(
                problem_instance=self.problem,
                schedule=schedule,
                method="invalid_legacy_greedy",
                instance_id="official-8t3r3s-1p-000000",
                seed=0,
                timing_semantics=LegacyTimingSemantics.CONTINUOUS,
            )

    def test_invalid_precedence_reference_has_a_clear_error(self):
        schedule = greedy_scheduling(self.problem, print_flag=False)
        invalid_problem = dict(self.problem)
        invalid_problem["precedence_constraints"] = [[99, 3]]

        with self.assertRaisesRegex(ValueError, "invalid precedence task IDs"):
            build_legacy_regression_evidence(
                problem_instance=invalid_problem,
                schedule=schedule,
                method="invalid_legacy_greedy",
                instance_id="official-8t3r3s-1p-000000",
                seed=0,
                timing_semantics=LegacyTimingSemantics.CONTINUOUS,
            )

    def test_original_sadcher_checkpoint_rolls_out_on_cpu(self):
        np.random.seed(0)
        torch.manual_seed(0)
        worst_case_makespan = float(
            np.sum(self.problem["T_e"])
            + sum(
                np.max(self.problem["T_t"][task])
                for task in range(len(self.problem["T_e"]))
            )
        )

        with mock.patch("torch.cuda.is_available", return_value=False):
            makespan, feasible, _, schedule = run_one_simulation(
                self.problem,
                "sadcher",
                str(CHECKPOINT_PATH),
                worst_case_makespan=worst_case_makespan,
            )

        self.assertTrue(feasible)
        self.assertEqual(makespan, 567)
        self.assertEqual(schedule.n_tasks, len(self.problem["T_e"]))
        schedule = Full_Horizon_Schedule(
            makespan, schedule.robot_schedules, len(self.problem["R"]) - 2
        )
        evidence = build_legacy_regression_evidence(
            problem_instance=self.problem,
            schedule=schedule,
            method="legacy_sadcher",
            instance_id="official-8t3r3s-1p-000000",
            seed=0,
            timing_semantics=LegacyTimingSemantics.DISCRETE,
            checkpoint_path=CHECKPOINT_PATH,
            checkpoint_device="cpu",
        )

        self.assert_evidence_contract(evidence, 567)
        self.assertEqual(
            assignments(evidence),
            EXPECTED_ASSIGNMENTS,
        )
        regression = evidence["result"]["metadata"]["legacy_regression"]
        self.assertEqual(regression["checkpoint_device"], "cpu")
        self.assertEqual(
            regression["checkpoint_sha256"],
            "8c391003af62d9b99399563e85790e4af6b65162120c79917077dc02882bb10e",
        )


if __name__ == "__main__":
    unittest.main()
