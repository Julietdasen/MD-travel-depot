import json
import unittest

from experiments.protocol import DatasetSplit, FailureReason
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from simulation_environment.md_metrics import MDMetrics


def metrics_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=True),
        tasks=(
            ProcessTask(3, (1, 0), 1, (True,)),
            ProcessTask(1, (4, 0), 2, (True,)),
            TransportTask(2, (2, 0), (4, 0), 5, 1, 1),
        ),
        robots=(
            ProcessRobot(0, (0, 0), (True,), speed=1),
            TransportRobot(1, (0, 0), True, 5, 2, loaded_speed=1),
        ),
        normal_edges=((3, 1),),
        material_edges=((2, 1),),
    )


class MDMetricsTests(unittest.TestCase):
    def test_success_requires_boolean_type(self):
        with self.assertRaisesRegex(ValueError, "success must be boolean"):
            MDMetrics(1, None, 0, {}, {})

    def test_transport_record_contains_complete_auditable_timeline(self):
        sim = MDDiscreteSimulator(metrics_domain())

        sim.assign(robot_id=1, task_id=2)
        for _ in range(5):
            sim.step()

        self.assertEqual(
            sim.transport_execution_records,
            (
                {
                    "task_id": 2,
                    "robot_id": 1,
                    "assigned_at": 0,
                    "arrived_pickup_at": 1,
                    "loading_started_at": 1,
                    "loading_completed_at": 2,
                    "arrived_delivery_at": 4,
                    "unloading_started_at": 4,
                    "unloading_completed_at": 5,
                    "completed_at": 5,
                    "empty_travel_duration": 1,
                    "loading_duration": 1,
                    "loaded_travel_duration": 2,
                    "unloading_duration": 1,
                    "service_duration": 4,
                    "occupied_duration": 5,
                },
            ),
        )
        json.dumps(sim.transport_execution_records)

    def test_success_metrics_require_tasks_complete_and_robots_at_exit(self):
        sim = MDDiscreteSimulator(metrics_domain())
        sim.assign(robot_id=0, task_id=3)
        sim.assign(robot_id=1, task_id=2)
        for _ in range(5):
            sim.step()
        sim.assign(robot_id=0, task_id=1)
        for _ in range(2):
            sim.step()

        self.assertTrue(sim.all_real_tasks_completed)
        self.assertFalse(sim.all_robots_at_exit)
        with self.assertRaisesRegex(ValueError, "unfinished simulator.*failure reason"):
            sim.metrics()

        while not sim.done:
            sim.step()

        metrics = sim.metrics(inference_time_seconds=0.125)
        self.assertTrue(metrics.success)
        self.assertIsNone(metrics.failure_reason)
        self.assertEqual(metrics.makespan, sim.time)
        self.assertEqual(
            metrics.material_starvation, {"1": 4.0, "3": 0.0}
        )
        self.assertAlmostEqual(metrics.robot_utilization["0"], 3 / sim.time)
        self.assertAlmostEqual(metrics.robot_utilization["1"], 5 / sim.time)
        self.assertEqual(metrics.inference_time_seconds, 0.125)
        self.assertEqual(metrics.total_material_starvation, 4.0)

    def test_failed_metrics_have_null_makespan_and_partial_utilization(self):
        sim = MDDiscreteSimulator(metrics_domain())
        sim.assign(robot_id=1, task_id=2)
        sim.step()
        sim.step()

        metrics = sim.metrics(failure_reason=FailureReason.TIMEOUT)

        self.assertFalse(metrics.success)
        self.assertIs(metrics.failure_reason, FailureReason.TIMEOUT)
        self.assertIsNone(metrics.makespan)
        self.assertEqual(metrics.robot_utilization, {"0": 0.0, "1": 1.0})
        self.assertEqual(metrics.material_starvation, {"3": 0.0})
        with self.assertRaisesRegex(ValueError, "completed simulator.*failure reason"):
            completed_domain = SchedulingDomain.create(
                config=MaterialDeliveryConfig(enabled=False),
                tasks=(ProcessTask(1, (0, 0), 0, (True,)),),
                robots=(ProcessRobot(0, (0, 0), (True,)),),
            )
            completed = MDDiscreteSimulator(completed_domain)
            completed.assign(robot_id=0, task_id=1)
            completed.step()
            self.assertTrue(completed.done)
            completed.metrics(failure_reason=FailureReason.DEADLOCK)

    def test_staggered_process_coalition_records_waiting_and_utilization(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=True),
            tasks=(ProcessTask(1, (0, 0), 1, (True, True)),),
            robots=(
                ProcessRobot(0, (0, 0), (True, False)),
                ProcessRobot(1, (0, 0), (False, True)),
            ),
        )
        sim = MDDiscreteSimulator(domain)
        sim.assign(robot_id=0, task_id=1)
        sim.step()
        sim.assign(robot_id=1, task_id=1)
        sim.step()

        self.assertTrue(sim.done)
        self.assertEqual(sim.process_execution_records[0]["waiting_duration"], 1)
        self.assertEqual(sim.metrics().robot_utilization, {"0": 1.0, "1": 0.5})

    def test_process_record_uses_discrete_service_duration(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=False),
            tasks=(ProcessTask(1, (0, 0), 1.2, (True,)),),
            robots=(ProcessRobot(0, (0, 0), (True,)),),
        )
        sim = MDDiscreteSimulator(domain)
        sim.assign(robot_id=0, task_id=1)
        sim.step()
        sim.step()

        self.assertEqual(sim.process_execution_records[0]["service_duration"], 2)
        self.assertEqual(sim.process_execution_records[0]["completed_at"], 2)

    def test_disabled_material_delivery_preserves_legacy_waiting_record(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=False),
            tasks=(ProcessTask(1, (0, 0), 1, (True, True)),),
            robots=(
                ProcessRobot(0, (0, 0), (True, False)),
                ProcessRobot(1, (0, 0), (False, True)),
            ),
        )
        sim = MDDiscreteSimulator(domain)
        sim.assign(robot_id=0, task_id=1)
        sim.step()
        sim.assign(robot_id=1, task_id=1)
        sim.step()

        self.assertEqual(sim.process_execution_records[0]["waiting_duration"], 0)

    def test_build_experiment_result_uses_unified_metrics_and_records(self):
        sim = MDDiscreteSimulator(metrics_domain())
        sim.assign(robot_id=1, task_id=2)
        sim.step()

        result = sim.build_experiment_result(
            run_id="greedy-instance-1-seed-0",
            method="greedy",
            instance_id="instance-1",
            seed=0,
            split=DatasetSplit.TEST,
            failure_reason=FailureReason.DEADLOCK,
            inference_time_seconds=0.01,
        )

        payload = result.to_dict()
        self.assertFalse(payload["termination"]["success"])
        self.assertEqual(payload["termination"]["failure_reason"], "deadlock")
        self.assertIsNone(payload["metrics"]["makespan"])
        self.assertEqual(payload["metrics"]["inference_time_seconds"], 0.01)
        self.assertEqual(payload["execution_records"]["transport"][0]["task_id"], 2)
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
