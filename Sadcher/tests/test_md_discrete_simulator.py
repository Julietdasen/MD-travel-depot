import unittest

from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)
from simulation_environment.hard_feasibility import FeasibilityCode
from simulation_environment.md_discrete_simulator import (
    MDDiscreteSimulator,
    RobotActivity,
    TaskStatus,
    TransportPhase,
)


def simulator_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=True),
        tasks=(
            ProcessTask(1, (4, 0), 2, (True, True)),
            TransportTask(2, (2, 0), (4, 0), 5, 1, 1),
        ),
        robots=(
            ProcessRobot(0, (4, 0), (True, False), speed=1),
            ProcessRobot(1, (4, 0), (False, True), speed=1),
            TransportRobot(2, (0, 0), True, 6, 2, loaded_speed=1),
        ),
        material_edges=((2, 1),),
    )


class MDDiscreteSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.sim = MDDiscreteSimulator(simulator_domain(), exit_location=(0, 0))

    def test_transport_uses_each_phase_and_unlocks_only_after_unloading(self):
        self.assertIs(self.sim.task_state(2).transport_phase, TransportPhase.WAITING)
        self.assertFalse(self.sim.is_task_ready(1))

        result = self.sim.assign(robot_id=2, task_id=2)

        self.assertTrue(result.is_feasible)
        self.assertIs(self.sim.task_state(2).transport_phase, TransportPhase.TO_PICKUP)
        self.assertIs(self.sim.robot_state(2).activity, RobotActivity.TRANSPORT)

        self.sim.step()
        transport = self.sim.task_state(2)
        self.assertIs(transport.transport_phase, TransportPhase.LOADING)
        self.assertEqual(transport.started_at, 1)
        self.assertFalse(self.sim.is_task_ready(1))

        self.sim.step()
        self.assertIs(
            self.sim.task_state(2).transport_phase, TransportPhase.TO_DELIVERY
        )
        self.sim.step()
        self.assertIs(
            self.sim.task_state(2).transport_phase, TransportPhase.TO_DELIVERY
        )
        self.sim.step()
        self.assertIs(self.sim.task_state(2).transport_phase, TransportPhase.UNLOADING)
        self.assertFalse(self.sim.is_task_ready(1))

        self.sim.step()
        transport = self.sim.task_state(2)
        self.assertIs(transport.transport_phase, TransportPhase.COMPLETE)
        self.assertIs(transport.status, TaskStatus.COMPLETE)
        self.assertEqual(transport.completed_at, 5)
        self.assertEqual(self.sim.robot_state(2).location, (4.0, 0.0))
        self.assertTrue(self.sim.is_task_ready(1))
        record = self.sim.transport_execution_records[0]
        self.assertEqual(record["empty_travel_duration"], 1)
        self.assertEqual(record["loaded_travel_duration"], 2)
        self.assertEqual(record["service_duration"], 4)
        self.assertEqual(record["occupied_duration"], 5)

    def test_transport_assignment_is_non_preemptive(self):
        self.sim.assign(robot_id=2, task_id=2)

        result = self.sim.assignment_feasibility(robot_id=2, task_id=2)

        self.assertFalse(result.is_feasible)
        self.assertIs(result.reason_code, FeasibilityCode.ROBOT_OCCUPIED)
        with self.assertRaisesRegex(ValueError, "robot_occupied"):
            self.sim.assign(robot_id=2, task_id=2)

    def test_process_starts_only_with_full_coalition_and_all_robots_return(self):
        self.sim.assign(robot_id=2, task_id=2)
        for _ in range(5):
            self.sim.step()

        first = self.sim.assign(robot_id=0, task_id=1)
        self.assertTrue(first.is_feasible)
        self.assertIs(self.sim.task_state(1).status, TaskStatus.PENDING)

        second = self.sim.assign(robot_id=1, task_id=1)
        self.assertTrue(second.is_feasible)
        process = self.sim.task_state(1)
        self.assertIs(process.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(process.started_at, 5)

        self.sim.step()
        self.assertIs(self.sim.task_state(1).status, TaskStatus.IN_PROGRESS)
        self.sim.step()
        self.assertIs(self.sim.task_state(1).status, TaskStatus.COMPLETE)
        self.assertFalse(self.sim.done)

        while not self.sim.done:
            self.sim.step()

        # Process robots return to their home location (default: initial
        # location (4, 0)) — zero travel — while the transport robot returns
        # to the shared exit (0, 0) over distance 4 at unloaded speed 2.
        self.assertEqual(self.sim.time, 9)
        self.assertTrue(self.sim.all_real_tasks_completed)
        self.assertTrue(self.sim.all_robots_at_exit)
        self.assertTrue(
            all(
                state.activity is RobotActivity.AT_EXIT
                for state in self.sim.robot_states.values()
            )
        )

    def test_movement_durations_are_ceil_distance_over_phase_speed(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=True),
            tasks=(
                ProcessTask(1, (3, 0), 0, (True,)),
                TransportTask(2, (3, 0), (3, 0), 1, 0, 0),
            ),
            robots=(
                ProcessRobot(0, (3, 0), (True,)),
                TransportRobot(1, (0, 0), True, 1, 2, loaded_speed=2),
            ),
            material_edges=((2, 1),),
        )
        sim = MDDiscreteSimulator(domain)

        sim.assign(robot_id=1, task_id=2)
        self.assertEqual(sim.task_state(2).phase_remaining, 2)
        sim.step()
        self.assertIs(sim.task_state(2).transport_phase, TransportPhase.TO_PICKUP)
        sim.step()

        self.assertIs(sim.task_state(2).transport_phase, TransportPhase.COMPLETE)
        self.assertEqual(sim.task_state(2).started_at, 2)
        self.assertEqual(sim.task_state(2).completed_at, 2)


if __name__ == "__main__":
    unittest.main()
