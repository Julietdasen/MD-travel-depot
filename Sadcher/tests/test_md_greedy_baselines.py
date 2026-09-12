import unittest

from experiments.protocol import DatasetSplit
from schedulers.md_greedy_baselines import (
    MDTransportGreedy,
    TransportGreedyStrategy,
    run_greedy_distance,
    run_greedy_eta,
    run_greedy_unlock,
    run_material_solo_greedy,
)
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


def _choice_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=True),
        tasks=(
            TransportTask(1, (1, 0), (10, 0), 1, 0, 0),
            ProcessTask(2, (10, 0), 1, (True,)),
            TransportTask(3, (3, 0), (4, 0), 1, 0, 0),
            ProcessTask(4, (4, 0), 1, (True,)),
        ),
        robots=(
            ProcessRobot(0, (0, 0), (True,)),
            TransportRobot(1, (0, 0), True, 1, 1, loaded_speed=1),
        ),
        material_edges=((1, 2), (3, 4)),
    )


def _strategy_diagnostic_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=True),
        tasks=(
            TransportTask(1, (1, 0), (10, 0), 1, 0, 0),
            ProcessTask(2, (10, 0), 1, (True,)),
            TransportTask(3, (3, 0), (4, 0), 1, 0, 0),
            ProcessTask(4, (4, 0), 1, (True,)),
            TransportTask(5, (20, 0), (20, 0), 1, 0, 0),
            ProcessTask(6, (20, 0), 1, (True,)),
            ProcessTask(7, (10, 0), 1, (True,)),
        ),
        robots=(
            ProcessRobot(0, (0, 0), (True,)),
            TransportRobot(1, (0, 0), True, 1, 1, loaded_speed=1),
        ),
        normal_edges=((2, 7),),
        material_edges=((1, 2), (3, 4), (5, 6)),
    )


class MDGreedyPolicyTests(unittest.TestCase):
    def test_distance_and_eta_use_distinct_auditable_scores(self):
        distance_sim = MDDiscreteSimulator(_choice_domain())
        eta_sim = MDDiscreteSimulator(_choice_domain())

        distance = MDTransportGreedy(TransportGreedyStrategy.DISTANCE).assign(distance_sim)
        eta = MDTransportGreedy(TransportGreedyStrategy.ETA).assign(eta_sim)

        self.assertEqual(distance[0].task_id, 1)
        self.assertEqual(eta[0].task_id, 3)
        self.assertEqual(distance[0].tie_break, (1, 1))
        self.assertGreater(distance[0].transport_eta, eta[0].transport_eta)

    def test_all_independent_entries_share_terminal_protocol(self):
        entries = (
            (run_greedy_distance, "greedy_distance"),
            (run_greedy_eta, "greedy_eta"),
            (run_greedy_unlock, "greedy_unlock"),
            (run_material_solo_greedy, "material_solo_greedy"),
        )
        for entry, method in entries:
            with self.subTest(method=method):
                result = entry(
                    MDDiscreteSimulator(_choice_domain()),
                    run_id=method,
                    instance_id="choice",
                    seed=9,
                    split=DatasetSplit.TEST,
                    max_steps=50,
                )
                self.assertTrue(result.success)
                self.assertEqual(result.method, method)
                self.assertEqual(result.metadata["tie_break"], ("robot_id", "task_id"))
                self.assertGreater(len(result.metadata["decisions"]), 0)

    def test_each_strategy_has_an_explainable_diagnostic_choice(self):
        selected = {}
        for strategy in TransportGreedyStrategy:
            simulator = MDDiscreteSimulator(_strategy_diagnostic_domain())
            decision = MDTransportGreedy(strategy).assign(simulator)[0]
            selected[strategy] = decision.task_id

        self.assertEqual(selected[TransportGreedyStrategy.DISTANCE], 1)
        self.assertEqual(selected[TransportGreedyStrategy.ETA], 3)
        self.assertEqual(selected[TransportGreedyStrategy.UNLOCK], 1)
        self.assertEqual(selected[TransportGreedyStrategy.MATERIAL_SOLO], 5)



if __name__ == "__main__":
    unittest.main()
