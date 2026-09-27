import unittest

from experiments.protocol import DatasetSplit, FailureReason
from schedulers.process_greedy_md import (
    ProcessGreedyMDAdapter,
    run_process_greedy_md,
)
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)
from simulation_environment.hard_feasibility import TaskStatus
from simulation_environment.md_discrete_simulator import (
    MDDiscreteSimulator,
    RobotActivity,
)


class ProcessGreedyMDAdapterTests(unittest.TestCase):
    def test_forms_a_complete_process_only_coalition(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=True),
            tasks=(ProcessTask(1, (3, 0), 2, (True, True)),),
            robots=(
                ProcessRobot(0, (0, 0), (True, False)),
                ProcessRobot(1, (0, 0), (False, True)),
                TransportRobot(2, (0, 0), True, 10, 1, loaded_speed=1),
            ),
        )
        simulator = MDDiscreteSimulator(domain)

        assignments = ProcessGreedyMDAdapter().assign(simulator)

        self.assertEqual(
            [(assignment.robot_id, assignment.task_id) for assignment in assignments],
            [(0, 1), (1, 1)],
        )
        self.assertIs(simulator.task_state(1).status, TaskStatus.IN_PROGRESS)
        self.assertEqual(simulator.task_state(1).assigned_robot_ids, {0, 1})
        self.assertIs(simulator.robot_state(2).activity, RobotActivity.AVAILABLE)
        self.assertIsNone(simulator.robot_state(2).task_id)

    def test_preserves_skill_coverage_priority_and_distance_tie_break(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=True),
            tasks=(
                ProcessTask(1, (10, 0), 1, (True, False)),
                ProcessTask(2, (1, 0), 1, (True, True)),
                ProcessTask(3, (2, 0), 1, (True, False)),
            ),
            robots=(
                ProcessRobot(0, (0, 0), (True, False)),
                ProcessRobot(1, (0, 0), (False, True)),
            ),
        )
        simulator = MDDiscreteSimulator(domain)

        assignments = ProcessGreedyMDAdapter().assign(simulator)

        # Tasks 1 and 3 both become fully covered, so distance breaks the tie.
        # Task 2 is nearer than task 1 but still has one uncovered skill.
        self.assertEqual(assignments[0].task_id, 3)

    def test_waits_for_both_normal_and_material_predecessors(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=True),
            tasks=(
                ProcessTask(1, (0, 0), 1, (True,)),
                TransportTask(2, (0, 0), (0, 0), 1, 0, 0),
                ProcessTask(3, (0, 0), 1, (True,)),
            ),
            robots=(
                ProcessRobot(0, (0, 0), (True,)),
                TransportRobot(1, (0, 0), True, 1, 1, loaded_speed=1),
            ),
            normal_edges=((1, 3),),
            material_edges=((2, 3),),
        )
        simulator = MDDiscreteSimulator(domain)
        scheduler = ProcessGreedyMDAdapter()

        first = scheduler.assign(simulator)
        self.assertEqual([assignment.task_id for assignment in first], [1])
        simulator.step()

        self.assertIs(simulator.task_state(1).status, TaskStatus.COMPLETE)
        # Under the arrival contract, task 3 is not yet ready (material
        # predecessor 2 has not completed) but ProcessGreedy may still commit
        # to it; simulator.is_task_ready still reports False until precursors
        # complete, and the coalition waits for the material.
        self.assertFalse(simulator.is_task_ready(3))
        second = scheduler.assign(simulator)
        self.assertEqual([assignment.task_id for assignment in second], [3])
        self.assertIs(simulator.task_state(3).status, TaskStatus.PENDING)

        simulator.assign(robot_id=1, task_id=2)
        simulator.step()
        self.assertIs(simulator.task_state(2).status, TaskStatus.COMPLETE)
        # The next step launches the already-assigned coalition once every
        # precursor is COMPLETE, and its service (duration 1) also finishes
        # inside that step.
        simulator.step()
        self.assertIs(simulator.task_state(3).status, TaskStatus.COMPLETE)


class ProcessGreedyMDRolloutTests(unittest.TestCase):
    def test_success_uses_the_unified_terminal_result_and_records(self):
        simulator = MDDiscreteSimulator(
            SchedulingDomain.create(
                config=MaterialDeliveryConfig(enabled=True),
                tasks=(ProcessTask(1, (2, 0), 1, (True,)),),
                robots=(ProcessRobot(0, (2, 0), (True,)),),
            )
        )

        result = run_process_greedy_md(
            simulator,
            run_id="process-greedy-success",
            instance_id="process-only-1",
            seed=7,
            split=DatasetSplit.TEST,
            max_steps=5,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.method, "process_greedy_md")
        # The process robot travels zero to task (same location) and returns
        # to its home location (also (2, 0)); only service duration counts.
        self.assertEqual(result.makespan, 1)
        self.assertTrue(result.all_real_tasks_completed)
        self.assertTrue(result.all_robots_at_exit)
        self.assertEqual(result.process_execution_records[0]["robot_ids"], (0,))

    def test_material_work_is_an_explicit_auxiliary_policy_integration_point(self):
        simulator = MDDiscreteSimulator(_material_domain())

        def assign_transport_once(sim: MDDiscreteSimulator) -> None:
            if sim.assignment_feasibility(robot_id=1, task_id=1).is_feasible:
                sim.assign(robot_id=1, task_id=1)

        result = run_process_greedy_md(
            simulator,
            run_id="process-greedy-with-transport",
            instance_id="material-1",
            seed=0,
            split=DatasetSplit.TEST,
            max_steps=5,
            auxiliary_policy=assign_transport_once,
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.transport_execution_records), 1)
        self.assertEqual(len(result.process_execution_records), 1)

    def test_missing_transport_policy_is_reported_as_deadlock(self):
        result = run_process_greedy_md(
            MDDiscreteSimulator(_material_domain()),
            run_id="process-greedy-deadlock",
            instance_id="material-1",
            seed=0,
            split=DatasetSplit.TEST,
            max_steps=5,
        )

        self.assertFalse(result.success)
        self.assertIs(result.failure_reason, FailureReason.DEADLOCK)
        self.assertIsNone(result.makespan)

    def test_step_budget_exhaustion_is_reported_as_timeout(self):
        simulator = MDDiscreteSimulator(
            SchedulingDomain.create(
                config=MaterialDeliveryConfig(enabled=True),
                tasks=(ProcessTask(1, (0, 0), 5, (True,)),),
                robots=(ProcessRobot(0, (0, 0), (True,)),),
            )
        )

        decision_times: list[int] = []

        def record_decision_time(sim: MDDiscreteSimulator) -> None:
            decision_times.append(sim.time)

        result = run_process_greedy_md(
            simulator,
            run_id="process-greedy-timeout",
            instance_id="long-process-1",
            seed=0,
            split=DatasetSplit.TEST,
            max_steps=1,
            auxiliary_policy=record_decision_time,
        )

        self.assertFalse(result.success)
        self.assertIs(result.failure_reason, FailureReason.TIMEOUT)
        self.assertIsNone(result.makespan)
        self.assertEqual(decision_times, [0])


def _material_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=True),
        tasks=(
            TransportTask(1, (0, 0), (0, 0), 1, 0, 0),
            ProcessTask(2, (0, 0), 1, (True,)),
        ),
        robots=(
            ProcessRobot(0, (0, 0), (True,)),
            TransportRobot(1, (0, 0), True, 1, 1, loaded_speed=1),
        ),
        material_edges=((1, 2),),
    )


if __name__ == "__main__":
    unittest.main()
