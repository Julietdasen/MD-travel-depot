import unittest

from simulation_environment.domain_model import (
    ProcessRobot,
    ProcessTask,
    TransportRobot,
    TransportTask,
)
from simulation_environment.hard_feasibility import (
    FeasibilityCode,
    RobotFeasibilityContext,
    TaskFeasibilityContext,
    TaskStatus,
    hard_feasibility_mask,
    is_assignment_feasible,
    is_process_coalition_start_feasible,
)


class HardFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.process_task = ProcessTask(1, (0, 0), 2, (True, True))
        self.transport_task = TransportTask(2, (0, 0), (1, 0), 5, 1, 1)
        self.skill_a = ProcessRobot(0, (0, 0), (True, False))
        self.skill_b = ProcessRobot(1, (0, 0), (False, True))
        self.transport_robot = TransportRobot(2, (0, 0), True, 6, 1, 1)

    def test_type_capability_capacity_readiness_pending_and_occupancy_reasons(self):
        cases = (
            (
                "unready",
                self.skill_a,
                TaskFeasibilityContext(self.process_task, ready=False),
                FeasibilityCode.TASK_NOT_READY,
            ),
            (
                "complete",
                self.skill_a,
                TaskFeasibilityContext(
                    self.process_task, status=TaskStatus.COMPLETE
                ),
                FeasibilityCode.TASK_NOT_PENDING,
            ),
            (
                "occupied",
                RobotFeasibilityContext(self.skill_a, available=False),
                self.process_task,
                FeasibilityCode.ROBOT_OCCUPIED,
            ),
            (
                "wrong type",
                self.transport_robot,
                self.process_task,
                FeasibilityCode.ROBOT_TYPE_MISMATCH,
            ),
            (
                "no transport capability",
                TransportRobot(2, (0, 0), False, 6, 1, 1),
                self.transport_task,
                FeasibilityCode.TRANSPORT_CAPABILITY_REQUIRED,
            ),
            (
                "capacity",
                TransportRobot(2, (0, 0), True, 4, 1, 1),
                self.transport_task,
                FeasibilityCode.INSUFFICIENT_CAPACITY,
            ),
        )

        for name, robot, task, expected in cases:
            with self.subTest(name=name):
                result = is_assignment_feasible(robot, task)
                self.assertFalse(result.is_feasible)
                self.assertIs(result.reason_code, expected)

    def test_process_contribution_is_separate_from_coalition_start(self):
        contribution = is_assignment_feasible(self.skill_a, self.process_task)
        partial = is_process_coalition_start_feasible(
            self.process_task, (self.skill_a,)
        )
        complete = is_process_coalition_start_feasible(
            self.process_task, (self.skill_a, self.skill_b)
        )

        self.assertTrue(contribution.is_feasible)
        self.assertFalse(partial.is_feasible)
        self.assertIs(partial.reason_code, FeasibilityCode.COALITION_INCOMPLETE)
        self.assertTrue(complete.is_feasible)

        duplicate_skill = is_assignment_feasible(
            self.skill_a,
            TaskFeasibilityContext(
                self.process_task, covered_skills=(True, False)
            ),
        )
        self.assertFalse(duplicate_skill.is_feasible)
        self.assertIs(
            duplicate_skill.reason_code, FeasibilityCode.NO_SKILL_CONTRIBUTION
        )

    def test_hard_mask_excludes_every_illegal_pair(self):
        tasks = (
            TaskFeasibilityContext(self.process_task, ready=False),
            self.transport_task,
        )
        robots = (self.skill_a, self.transport_robot)

        mask = hard_feasibility_mask(robots, tasks)

        self.assertEqual(mask, ((False, False), (False, True)))


if __name__ == "__main__":
    unittest.main()
