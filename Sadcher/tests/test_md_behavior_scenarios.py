import unittest

from experiments.protocol import FailureReason
from simulation_environment.hard_feasibility import FeasibilityCode
from simulation_environment.md_discrete_simulator import (
    MDDiscreteSimulator,
    RobotActivity,
    TaskStatus,
    TransportPhase,
)
from simulation_environment.md_static_validator import (
    StaticValidationCode,
    validate_md_domain,
)
from tests.fixtures.md_diagnostic_scenarios import (
    CAPACITY_FAILURE,
    COALITION,
    DEADLOCK,
    LATE_MATERIAL,
    LOCATION_MISMATCH,
    TIMEOUT,
    TYPE_MISMATCH,
    UNLOCK,
    DiagnosticScenario,
)


class MDDiagnosticScenarioTests(unittest.TestCase):
    EXPECTED_SEEDS = {
        "unlock-after-unloading": 1009,
        "late-material-after-normal-predecessor": 1010,
        "insufficient-transport-capacity": 1011,
        "process-robot-on-transport-task": 1012,
        "material-delivery-location-mismatch": 1013,
        "two-skill-process-coalition": 1014,
        "split-incomplete-process-coalitions": 1015,
        "long-process-timeout": 1016,
    }

    def assert_seed_is_fixed(self, scenario: DiagnosticScenario) -> None:
        self.assertEqual(scenario.seed, self.EXPECTED_SEEDS[scenario.name])

    def test_transport_completion_unlocks_process_at_the_expected_tick(self):
        self.assert_seed_is_fixed(UNLOCK)
        sim = MDDiscreteSimulator(UNLOCK.domain, exit_location=UNLOCK.exit_location)

        self.assertFalse(sim.is_task_ready(UNLOCK.process_task_id))
        sim.assign(robot_id=UNLOCK.transport_robot_id, task_id=UNLOCK.transport_task_id)
        while sim.time < UNLOCK.material_ready_at:
            sim.step()
            if sim.time < UNLOCK.material_ready_at:
                self.assertFalse(sim.is_task_ready(UNLOCK.process_task_id))

        transport = sim.task_state(UNLOCK.transport_task_id)
        self.assertIs(transport.status, TaskStatus.COMPLETE)
        self.assertIs(transport.transport_phase, TransportPhase.COMPLETE)
        self.assertEqual(transport.completed_at, UNLOCK.material_ready_at)
        self.assertTrue(sim.is_task_ready(UNLOCK.process_task_id))
        sim.assign(
            robot_id=UNLOCK.process_robot_id,
            task_id=UNLOCK.process_task_id,
        )
        while not sim.done:
            sim.step()

        metrics = sim.metrics(inference_time_seconds=0.0)
        self.assertTrue(metrics.success)
        self.assertIsNone(metrics.failure_reason)
        self.assertEqual(metrics.makespan, 10)
        self.assertEqual(metrics.material_starvation, {"1": 5.0})
        self.assertEqual(metrics.robot_utilization, {"0": 0.2, "1": 0.5})

    def test_late_material_has_a_fixed_normal_and_material_ready_gap(self):
        self.assert_seed_is_fixed(LATE_MATERIAL)
        sim = MDDiscreteSimulator(
            LATE_MATERIAL.domain, exit_location=LATE_MATERIAL.exit_location
        )

        sim.assign(
            robot_id=LATE_MATERIAL.normal_robot_id,
            task_id=LATE_MATERIAL.normal_predecessor_id,
        )
        sim.assign(
            robot_id=LATE_MATERIAL.transport_robot_id,
            task_id=LATE_MATERIAL.transport_task_id,
        )
        while sim.time < LATE_MATERIAL.material_ready_at:
            sim.step()

        normal = sim.task_state(LATE_MATERIAL.normal_predecessor_id)
        material = sim.task_state(LATE_MATERIAL.transport_task_id)
        self.assertEqual(normal.completed_at, LATE_MATERIAL.normal_ready_at)
        assert normal.completed_at is not None
        assert material.completed_at is not None
        self.assertEqual(material.completed_at, LATE_MATERIAL.material_ready_at)
        self.assertEqual(
            material.completed_at - normal.completed_at,
            LATE_MATERIAL.expected_material_starvation,
        )
        self.assertTrue(sim.is_task_ready(LATE_MATERIAL.process_task_id))
        sim.assign(
            robot_id=LATE_MATERIAL.normal_robot_id,
            task_id=LATE_MATERIAL.process_task_id,
        )
        while not sim.done:
            sim.step()

        metrics = sim.metrics()
        self.assertTrue(metrics.success)
        self.assertEqual(metrics.makespan, 13)
        self.assertEqual(metrics.material_starvation, {"1": 6.0, "2": 0.0})
        self.assertAlmostEqual(metrics.robot_utilization["0"], 3 / 13)
        self.assertAlmostEqual(metrics.robot_utilization["1"], 7 / 13)

    def test_capacity_failure_is_rejected_without_mutating_runtime_state(self):
        self.assert_seed_is_fixed(CAPACITY_FAILURE)
        sim = MDDiscreteSimulator(
            CAPACITY_FAILURE.domain, exit_location=CAPACITY_FAILURE.exit_location
        )

        result = sim.assignment_feasibility(
            robot_id=CAPACITY_FAILURE.rejected_robot_id,
            task_id=CAPACITY_FAILURE.transport_task_id,
        )

        self.assertFalse(result.is_feasible)
        self.assertFalse(CAPACITY_FAILURE.terminal_metrics_expected)
        self.assertIs(result.reason_code, FeasibilityCode.INSUFFICIENT_CAPACITY)
        self.assertIs(result.reason_code, CAPACITY_FAILURE.expected_feasibility_code)
        self.assertIs(
            sim.task_state(CAPACITY_FAILURE.transport_task_id).status,
            TaskStatus.PENDING,
        )
        self.assertIs(
            sim.robot_state(CAPACITY_FAILURE.rejected_robot_id).activity,
            RobotActivity.AVAILABLE,
        )

    def test_robot_type_mismatch_has_a_stable_reason(self):
        self.assert_seed_is_fixed(TYPE_MISMATCH)
        sim = MDDiscreteSimulator(
            TYPE_MISMATCH.domain, exit_location=TYPE_MISMATCH.exit_location
        )

        result = sim.assignment_feasibility(
            robot_id=TYPE_MISMATCH.rejected_robot_id,
            task_id=TYPE_MISMATCH.transport_task_id,
        )

        self.assertFalse(result.is_feasible)
        self.assertFalse(TYPE_MISMATCH.terminal_metrics_expected)
        self.assertIs(result.reason_code, FeasibilityCode.ROBOT_TYPE_MISMATCH)
        self.assertIs(TYPE_MISMATCH.failure_reason, FailureReason.ROBOT_TYPE_MISMATCH)
        self.assertIs(result.reason_code, TYPE_MISMATCH.expected_feasibility_code)

    def test_delivery_location_mismatch_fails_static_validation(self):
        self.assert_seed_is_fixed(LOCATION_MISMATCH)

        result = validate_md_domain(LOCATION_MISMATCH.domain)

        self.assertFalse(result.is_valid)
        self.assertFalse(LOCATION_MISMATCH.terminal_metrics_expected)
        self.assertIs(result.reason_code, StaticValidationCode.INVALID_LOCATION)
        self.assertIs(result.failure_reason, FailureReason.INVALID_GRAPH)
        self.assertIs(LOCATION_MISMATCH.failure_reason, FailureReason.INVALID_GRAPH)

    def test_process_starts_only_when_the_fixed_coalition_is_complete(self):
        self.assert_seed_is_fixed(COALITION)
        sim = MDDiscreteSimulator(COALITION.domain, exit_location=COALITION.exit_location)

        sim.assign(robot_id=COALITION.first_robot_id, task_id=COALITION.process_task_id)
        state = sim.task_state(COALITION.process_task_id)
        self.assertIs(state.status, TaskStatus.PENDING)
        self.assertEqual(state.assigned_robot_ids, {COALITION.first_robot_id})
        sim.step()
        self.assertIs(sim.task_state(COALITION.process_task_id).status, TaskStatus.PENDING)

        sim.assign(robot_id=COALITION.second_robot_id, task_id=COALITION.process_task_id)
        state = sim.task_state(COALITION.process_task_id)
        self.assertIs(state.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(state.started_at, COALITION.expected_started_at)
        self.assertEqual(
            state.assigned_robot_ids,
            {COALITION.first_robot_id, COALITION.second_robot_id},
        )

        while not sim.done:
            sim.step()

        self.assertEqual(sim.process_execution_records[0]["waiting_duration"], 1)
        metrics = sim.metrics()
        self.assertEqual(metrics.makespan, 3)
        self.assertTrue(metrics.success)
        self.assertIsNone(metrics.failure_reason)
        self.assertEqual(metrics.material_starvation, {"1": 0.0})
        self.assertEqual(metrics.robot_utilization, {"0": 1.0, "1": 2 / 3})

    def test_split_coalitions_form_a_reproducible_deadlock_state(self):
        self.assert_seed_is_fixed(DEADLOCK)
        sim = MDDiscreteSimulator(DEADLOCK.domain, exit_location=DEADLOCK.exit_location)
        for robot_id, task_id in DEADLOCK.initial_assignments:
            sim.assign(robot_id=robot_id, task_id=task_id)
        sim.step()

        self.assertFalse(sim.done)
        self.assertFalse(sim.all_real_tasks_completed)
        self.assertTrue(
            all(state.status is TaskStatus.PENDING for state in sim.task_states.values())
        )
        self.assertTrue(
            all(
                state.activity is RobotActivity.PROCESS
                for state in sim.robot_states.values()
            )
        )
        self.assertIs(DEADLOCK.failure_reason, FailureReason.DEADLOCK)

        metrics = sim.metrics(failure_reason=DEADLOCK.failure_reason)
        self.assertFalse(metrics.success)
        self.assertIsNone(metrics.makespan)
        self.assertIs(metrics.failure_reason, FailureReason.DEADLOCK)
        self.assertEqual(metrics.robot_utilization, {"0": 1.0, "1": 1.0})

    def test_long_running_task_exposes_a_reproducible_timeout_boundary(self):
        self.assert_seed_is_fixed(TIMEOUT)
        sim = MDDiscreteSimulator(TIMEOUT.domain, exit_location=TIMEOUT.exit_location)
        robot_id, task_id = TIMEOUT.initial_assignments[0]
        sim.assign(robot_id=robot_id, task_id=task_id)
        for _ in range(TIMEOUT.max_steps):
            sim.step()

        self.assertEqual(sim.time, TIMEOUT.max_steps)
        self.assertFalse(sim.done)
        self.assertIs(sim.task_state(task_id).status, TaskStatus.IN_PROGRESS)
        self.assertIs(TIMEOUT.failure_reason, FailureReason.TIMEOUT)

        metrics = sim.metrics(failure_reason=TIMEOUT.failure_reason)
        self.assertFalse(metrics.success)
        self.assertIsNone(metrics.makespan)
        self.assertIs(metrics.failure_reason, FailureReason.TIMEOUT)
        self.assertEqual(metrics.robot_utilization, {"0": 1.0})


if __name__ == "__main__":
    unittest.main()
