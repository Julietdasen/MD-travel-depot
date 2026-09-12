import unittest
from dataclasses import FrozenInstanceError, replace

from data_generation.md_instance_generator import (
    GeneratedMDInstance,
    MDGeneratorConfig,
    generate_md_instance,
)
from simulation_environment.domain_model import (
    ProcessTask,
    TransportRobot,
    TransportTask,
)


def longest_process_path(instance: GeneratedMDInstance) -> int:
    process_ids = {
        task.task_id
        for task in instance.domain.tasks
        if isinstance(task, ProcessTask)
    }
    incoming = {task_id: [] for task_id in process_ids}
    for source_id, target_id in instance.domain.normal_edges:
        incoming[target_id].append(source_id)

    lengths = {}
    remaining = set(process_ids)
    while remaining:
        ready = sorted(
            task_id
            for task_id in remaining
            if all(predecessor in lengths for predecessor in incoming[task_id])
        )
        if not ready:
            raise AssertionError("generated normal graph contains a cycle")
        for task_id in ready:
            lengths[task_id] = 1 + max(
                (lengths[predecessor] for predecessor in incoming[task_id]),
                default=0,
            )
            remaining.remove(task_id)
    return max(lengths.values(), default=0)


class MDSyntheticInstanceGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.config = MDGeneratorConfig(
            seed=17,
            task_count=12,
            transport_ratio=0.25,
            precedence_density=0.4,
            critical_path_length=4,
            capacity_slack=0.2,
            speed_ratio=0.6,
            process_robot_count=3,
            transport_robot_count=2,
            skill_count=3,
        )

    def test_fixed_seed_reproduces_the_complete_instance_and_records_config(self):
        first = generate_md_instance(self.config)
        second = generate_md_instance(self.config)

        self.assertEqual(first, second)
        self.assertIs(first.generation_config, self.config)
        self.assertEqual(first.generation_config.seed, 17)

        different_seed = generate_md_instance(replace(self.config, seed=18))
        self.assertNotEqual(first.domain, different_seed.domain)

    def test_task_ratio_critical_path_density_slack_and_speed_are_controlled(self):
        instance = generate_md_instance(self.config)
        process_tasks = [
            task for task in instance.domain.tasks if isinstance(task, ProcessTask)
        ]
        transport_tasks = [
            task for task in instance.domain.tasks if isinstance(task, TransportTask)
        ]
        transport_robots = [
            robot
            for robot in instance.domain.robots
            if isinstance(robot, TransportRobot)
        ]

        self.assertEqual(len(instance.domain.tasks), self.config.task_count)
        self.assertEqual(
            len(transport_tasks), self.config.transport_task_count
        )
        self.assertEqual(
            len(process_tasks), self.config.process_task_count
        )
        self.assertEqual(longest_process_path(instance), 4)
        self.assertEqual(
            len(instance.domain.normal_edges), self.config.normal_edge_count
        )

        sparse = generate_md_instance(
            replace(self.config, precedence_density=0.2)
        )
        dense = generate_md_instance(
            replace(self.config, precedence_density=0.8)
        )
        self.assertLess(len(sparse.domain.normal_edges), len(dense.domain.normal_edges))

        maximum_load = max(task.load for task in transport_tasks)
        minimum_capacity = min(robot.capacity for robot in transport_robots)
        self.assertAlmostEqual(
            (minimum_capacity - maximum_load) / maximum_load,
            self.config.capacity_slack,
        )
        for robot in transport_robots:
            self.assertAlmostEqual(
                robot.loaded_speed / robot.unloaded_speed,
                self.config.speed_ratio,
            )

    def test_material_edges_are_one_to_one_and_delivery_matches_process_location(self):
        instance = generate_md_instance(self.config)
        tasks = {task.task_id: task for task in instance.domain.tasks}

        self.assertEqual(
            len(instance.domain.material_edges),
            self.config.transport_task_count,
        )
        downstream_ids = [
            process_id for _, process_id in instance.domain.material_edges
        ]
        self.assertEqual(len(downstream_ids), len(set(downstream_ids)))

        for transport_id, process_id in instance.domain.material_edges:
            transport = tasks[transport_id]
            process = tasks[process_id]
            self.assertIsInstance(transport, TransportTask)
            self.assertIsInstance(process, ProcessTask)
            self.assertEqual(transport.downstream_process_task_id, process_id)
            self.assertEqual(process.material_predecessor, transport_id)
            self.assertEqual(transport.delivery_location, process.location)

    def test_transport_ratio_changes_the_number_of_transport_tasks(self):
        low = generate_md_instance(replace(self.config, transport_ratio=0.0))
        high = generate_md_instance(replace(self.config, transport_ratio=0.5))

        low_transport_count = sum(
            isinstance(task, TransportTask) for task in low.domain.tasks
        )
        high_transport_count = sum(
            isinstance(task, TransportTask) for task in high.domain.tasks
        )
        self.assertEqual(low_transport_count, 0)
        self.assertEqual(high_transport_count, 6)

    def test_config_rejects_invalid_or_incompatible_controls_clearly(self):
        invalid_configs = (
            (
                "transport ratio",
                {"transport_ratio": 0.6},
                "transport_ratio must be between 0 and 0.5",
            ),
            (
                "critical path",
                {"critical_path_length": 10},
                "critical_path_length cannot exceed process_task_count",
            ),
            (
                "capacity slack",
                {"capacity_slack": -0.1},
                "capacity_slack must be a non-negative finite number",
            ),
            (
                "speed ratio",
                {"speed_ratio": 0},
                r"speed_ratio must be between 0 \(exclusive\) and 1",
            ),
            (
                "density",
                {"precedence_density": 1.1},
                "precedence_density must be between 0 and 1",
            ),
            (
                "missing transport robot",
                {"transport_robot_count": 0},
                "transport tasks require at least one transport robot",
            ),
            (
                "incompatible density and critical path",
                {"precedence_density": 0},
                "precedence_density is too low for critical_path_length",
            ),
        )
        for name, changes, message in invalid_configs:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    replace(self.config, **changes)

    def test_config_and_generated_record_are_immutable(self):
        instance = generate_md_instance(self.config)

        with self.assertRaises(FrozenInstanceError):
            self.config.seed = 99
        with self.assertRaises(FrozenInstanceError):
            instance.domain = None


if __name__ == "__main__":
    unittest.main()
