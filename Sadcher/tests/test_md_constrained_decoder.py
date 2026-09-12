import unittest
from unittest.mock import patch

import torch

from schedulers.md_constrained_decoder import (
    FastAssignmentRepair,
    LearnedConstrainedDecoder,
    MaskedGreedyDecoder,
    apply_decoder_result,
    simulator_hard_mask,
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


def _domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=True),
        tasks=(
            ProcessTask(1, (2, 0), 2, (True, True)),
            ProcessTask(2, (1, 0), 1, (True, False)),
            TransportTask(3, (0, 0), (3, 0), 2, 1, 1),
            ProcessTask(4, (3, 0), 1, (True, False)),
        ),
        robots=(
            ProcessRobot(0, (0, 0), (True, False)),
            ProcessRobot(1, (0, 0), (False, True)),
            TransportRobot(2, (0, 0), True, 3, 1, loaded_speed=1),
            TransportRobot(3, (0, 0), True, 3, 1, loaded_speed=1),
        ),
        material_edges=((3, 4),),
    )


def _scores() -> torch.Tensor:
    # Rows use sorted robot IDs; columns use sorted task IDs.
    return torch.tensor(
        [
            [9.0, 8.0, -5.0, 100.0],
            [7.0, -2.0, -5.0, 100.0],
            [-5.0, -5.0, 6.0, -5.0],
            [-5.0, -5.0, 5.0, -5.0],
        ]
    )


class MDConstrainedDecoderTests(unittest.TestCase):
    def test_learned_decoder_forms_process_coalition_and_transport_singleton(self):
        simulator = MDDiscreteSimulator(_domain())
        decoder = LearnedConstrainedDecoder()

        with patch("pulp.LpProblem.solve") as solver:
            result = decoder.decode(_scores(), simulator)

        solver.assert_not_called()
        self.assertEqual(result.assignments, ((0, 1), (1, 1), (2, 3)))
        self.assertEqual(result.illegal_assignment_count, 0)
        self.assertEqual(sum(task_id == 3 for _, task_id in result.assignments), 1)
        self.assertEqual(
            {robot_id for robot_id, task_id in result.assignments if task_id == 1},
            {0, 1},
        )
        applied = apply_decoder_result(simulator, result)
        self.assertEqual(applied, 3)
        self.assertEqual(simulator.task_state(1).assigned_robot_ids, {0, 1})
        self.assertEqual(simulator.task_state(3).assigned_robot_ids, {2})

    def test_decoder_rejects_any_mask_not_equal_to_centralized_source(self):
        simulator = MDDiscreteSimulator(_domain())
        wrong = torch.ones_like(simulator_hard_mask(simulator))

        with self.assertRaisesRegex(ValueError, "centralized hard-feasibility"):
            LearnedConstrainedDecoder().decode(
                _scores(), simulator, hard_feasibility_mask=wrong
            )

    def test_fast_repair_removes_conflicts_and_completes_coalition(self):
        simulator = MDDiscreteSimulator(_domain())
        proposal = (
            (0, 1),
            (0, 2),
            (2, 3),
            (3, 3),
            (1, 4),
        )

        result = FastAssignmentRepair().repair(proposal, _scores(), simulator)

        self.assertEqual(result.assignments, ((0, 1), (1, 1), (2, 3)))
        self.assertTrue(result.repaired)
        self.assertEqual(result.illegal_assignment_count, 0)
        codes = {event.code for event in result.repair_events}
        self.assertIn("duplicate_robot", codes)
        self.assertIn("transport_singleton", codes)
        self.assertIn("illegal_pair", codes)
        self.assertIn("coalition_completed", codes)
        apply_decoder_result(simulator, result)

    def test_masked_greedy_has_comparable_legal_result_contract(self):
        simulator = MDDiscreteSimulator(_domain())

        learned = LearnedConstrainedDecoder().decode(_scores(), simulator)
        greedy = MaskedGreedyDecoder().decode(_scores(), simulator)

        self.assertEqual(type(learned), type(greedy))
        self.assertEqual(greedy.illegal_assignment_count, 0)
        self.assertEqual(greedy.robot_ids, learned.robot_ids)
        self.assertEqual(greedy.task_ids, learned.task_ids)
        self.assertEqual(
            sum(task_id == 3 for _, task_id in greedy.assignments), 1
        )


if __name__ == "__main__":
    unittest.main()
