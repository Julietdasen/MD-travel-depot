import unittest

import baselines.md_metaheuristics as metaheuristics
from baselines.md_metaheuristics import (
    CandidateQuality,
    GeneticAlgorithmConfig,
    SimulatedAnnealingConfig,
    run_genetic_algorithm,
    run_simulated_annealing,
)
from experiments.protocol import DatasetSplit
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)
from tests.fixtures.md_diagnostic_scenarios import LATE_MATERIAL
def _assignment_choice_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=True),
        tasks=(
            TransportTask(1, (0, 0), (10, 0), 1, 0, 0),
            ProcessTask(2, (10, 0), 1, (True,)),
        ),
        robots=(
            ProcessRobot(0, (10, 0), (True,)),
            TransportRobot(1, (0, 0), True, 1, 1, loaded_speed=1),
            TransportRobot(2, (0, 0), True, 1, 1, loaded_speed=10),
        ),
        material_edges=((1, 2),),
    )




class MDMetaheuristicTests(unittest.TestCase):
    def test_genetic_algorithm_is_seeded_and_replays_through_the_simulator(self):
        config = GeneticAlgorithmConfig(population_size=6, generations=3, seed=17)
        kwargs = dict(
            domain=LATE_MATERIAL.domain,
            exit_location=LATE_MATERIAL.exit_location,
            run_id="ga",
            instance_id="late-material",
            split=DatasetSplit.TEST,
            max_steps=50,
            config=config,
        )

        first = run_genetic_algorithm(**kwargs)
        second = run_genetic_algorithm(**kwargs)

        self.assertTrue(first.success)
        self.assertEqual(first.method, "genetic_algorithm")
        self.assertEqual(first.metadata["best_order"], second.metadata["best_order"])
        self.assertEqual(
            first.metadata["best_robot_order"],
            second.metadata["best_robot_order"],
        )
        self.assertEqual(first.makespan, second.makespan)
        self.assertEqual(first.metadata["evaluations"], second.metadata["evaluations"])
        self.assertGreater(len(first.transport_execution_records), 0)
        self.assertGreater(len(first.process_execution_records), 0)

    def test_genetic_algorithm_searches_robot_assignment_preference(self):
        result = run_genetic_algorithm(
            _assignment_choice_domain(),
            run_id="ga-assignment",
            instance_id="assignment-choice",
            split=DatasetSplit.TEST,
            max_steps=50,
            config=GeneticAlgorithmConfig(
                population_size=12,
                generations=3,
                seed=4,
            ),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.transport_execution_records[0]["robot_id"], 2)
        robot_order = result.metadata["best_robot_order"]
        self.assertLess(robot_order.index(2), robot_order.index(1))


    def test_genetic_algorithm_checks_time_budget_during_initial_population(self):
        result = run_genetic_algorithm(
            LATE_MATERIAL.domain,
            exit_location=LATE_MATERIAL.exit_location,
            run_id="ga-budget",
            instance_id="late-material",
            split=DatasetSplit.TEST,
            max_steps=50,
            config=GeneticAlgorithmConfig(
                population_size=20,
                generations=20,
                seed=5,
                time_budget_seconds=1e-9,
            ),
        )

        self.assertTrue(result.metadata["stopped_by_time_budget"])
        self.assertEqual(result.metadata["evaluations"], 1)
        self.assertEqual(result.metadata["generations_completed"], 0)

    def test_annealing_delta_preserves_lexicographic_failure_dominance(self):
        successful_but_slow = CandidateQuality(0, 10_000_000.0, 8)
        failed_but_short = CandidateQuality(1, 1.0, 1)

        self.assertEqual(
            metaheuristics._annealing_delta(successful_but_slow, failed_but_short),
            -float("inf"),
        )
        self.assertEqual(
            metaheuristics._annealing_delta(failed_but_short, successful_but_slow),
            float("inf"),
        )
        self.assertEqual(
            metaheuristics._annealing_delta(
                CandidateQuality(0, 12.5, 3), CandidateQuality(0, 10.0, 1)
            ),
            2.5,
        )
        self.assertEqual(
            metaheuristics._annealing_delta(
                CandidateQuality(1, 10.0, 1), CandidateQuality(1, 10.0, 3)
            ),
            -2.0,
        )



    def test_simulated_annealing_reports_budget_and_candidate_quality(self):
        result = run_simulated_annealing(
            LATE_MATERIAL.domain,
            exit_location=LATE_MATERIAL.exit_location,
            run_id="sa",
            instance_id="late-material",
            split=DatasetSplit.TEST,
            max_steps=50,
            config=SimulatedAnnealingConfig(iterations=8, seed=23),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.method, "simulated_annealing")
        self.assertEqual(result.metadata["seed"], 23)
        self.assertLessEqual(result.metadata["iterations_completed"], 8)
        self.assertEqual(result.metadata["best_quality"][0], 0)
        self.assertEqual(result.metadata["best_quality"][1], result.makespan)


if __name__ == "__main__":
    unittest.main()
