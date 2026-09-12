import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    RobotType,
    SchedulingDomain,
    TaskType,
    TransportRobot,
    TransportTask,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "legacy_8t3r3s"
    / "problem_instances"
    / "problem_instance_1p_000000.json"
)


def md_entities():
    tasks = (
        ProcessTask(10, (4, 0), 7, (True, False)),
        ProcessTask(20, (8, 0), 5, (False, True)),
        TransportTask(30, (0, 0), (4, 0), 6, 2, 3),
    )
    robots = (
        ProcessRobot(0, (0, 0), (True, True), speed=1),
        TransportRobot(1, (1, 0), True, 8, 2),
    )
    return tasks, robots


class MaterialDeliveryDomainTests(unittest.TestCase):
    def setUp(self):
        self.config = MaterialDeliveryConfig(
            enabled=True, loaded_speed_factor=0.5
        )

    def create_domain(
        self, *, tasks=None, robots=None, normal_edges=None, material_edges=None
    ):
        default_tasks, default_robots = md_entities()
        return SchedulingDomain.create(
            config=self.config,
            tasks=default_tasks if tasks is None else tasks,
            robots=default_robots if robots is None else robots,
            normal_edges=((10, 20),) if normal_edges is None else normal_edges,
            material_edges=((30, 10),) if material_edges is None else material_edges,
        )

    def test_md_objects_expose_types_transport_attributes_and_predecessors(self):
        domain = self.create_domain()
        tasks = {task.task_id: task for task in domain.tasks}
        robots = {robot.robot_id: robot for robot in domain.robots}

        self.assertFalse(domain.is_legacy)
        self.assertEqual(domain.normal_edges, ((10, 20),))
        self.assertEqual(domain.material_edges, ((30, 10),))
        self.assertIs(tasks[10].task_type, TaskType.PROCESS)
        self.assertEqual(tasks[10].material_predecessor, 30)
        self.assertEqual(tasks[20].normal_predecessors, (10,))
        self.assertIs(tasks[30].task_type, TaskType.TRANSPORT)
        self.assertEqual(tasks[30].pickup_location, (0, 0))
        self.assertEqual(tasks[30].delivery_location, (4, 0))
        self.assertEqual(tasks[30].load, 6)
        self.assertEqual(tasks[30].loading_duration, 2)
        self.assertEqual(tasks[30].unloading_duration, 3)
        self.assertEqual(tasks[30].downstream_process_task_id, 10)

        self.assertIs(robots[0].robot_type, RobotType.PROCESS_ROBOT)
        self.assertEqual(robots[0].capabilities, (True, True))
        self.assertIs(robots[1].robot_type, RobotType.TRANSPORT_ROBOT)
        self.assertTrue(robots[1].transport_capable)
        self.assertEqual(robots[1].capacity, 8)
        self.assertEqual(robots[1].unloaded_speed, 2)
        self.assertEqual(robots[1].loaded_speed, 1.0)

    def test_explicit_loaded_speed_overrides_the_configured_factor(self):
        tasks, robots = md_entities()
        transport = robots[1]
        robots = (
            robots[0],
            TransportRobot(
                transport.robot_id,
                transport.location,
                True,
                transport.capacity,
                transport.unloaded_speed,
                loaded_speed=1.75,
            ),
        )

        domain = self.create_domain(tasks=tasks, robots=robots)

        self.assertEqual(domain.robots[1].loaded_speed, 1.75)

    def test_legacy_schema_loads_as_process_only_when_disabled(self):
        legacy = json.loads(LEGACY_FIXTURE.read_text())
        domain = SchedulingDomain.from_legacy_mapping(
            legacy, MaterialDeliveryConfig(enabled=False)
        )

        self.assertTrue(domain.is_legacy)
        self.assertEqual(len(domain.tasks), 8)
        self.assertEqual(len(domain.robots), 3)
        self.assertTrue(all(isinstance(task, ProcessTask) for task in domain.tasks))
        self.assertTrue(all(isinstance(robot, ProcessRobot) for robot in domain.robots))
        self.assertEqual(domain.normal_edges, ((7, 3),))
        self.assertEqual(domain.material_edges, ())
        self.assertEqual(domain.tasks[2].normal_predecessors, (7,))
        self.assertIsNone(domain.tasks[2].material_predecessor)
        self.assertEqual(domain.legacy_travel_times[0][1], legacy["T_t"][0][1])

    def test_disabled_mode_accepts_process_only_but_rejects_transport(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=False),
            tasks=(ProcessTask(1, (0, 0), 1, (True,)),),
            robots=(ProcessRobot(0, (0, 0), (True,)),),
        )
        self.assertFalse(domain.config.enabled)

        tasks, robots = md_entities()
        with self.assertRaisesRegex(
            ValueError, "transport entities require material_delivery.enabled=true"
        ):
            SchedulingDomain.create(
                config=MaterialDeliveryConfig(enabled=False),
                tasks=tasks,
                robots=robots,
                material_edges=((30, 10),),
            )

    def test_material_edge_references_and_endpoint_types_fail_clearly(self):
        with self.assertRaisesRegex(
            ValueError, "material edge references unknown task 99"
        ):
            self.create_domain(material_edges=((30, 99),))

        with self.assertRaisesRegex(
            ValueError, "material edge source 10 must be TRANSPORT"
        ):
            self.create_domain(material_edges=((10, 20),))

    def test_material_relation_is_one_to_one_and_location_consistent(self):
        tasks, robots = md_entities()
        tasks += (TransportTask(40, (2, 0), (4, 0), 1, 1, 1),)
        with self.assertRaisesRegex(
            ValueError, "PROCESS task 10 has more than one material predecessor"
        ):
            self.create_domain(
                tasks=tasks,
                robots=robots,
                material_edges=((30, 10), (40, 10)),
            )

        tasks, robots = md_entities()
        tasks = tasks[:2] + (TransportTask(30, (0, 0), (5, 0), 6, 2, 3),)
        with self.assertRaisesRegex(
            ValueError, "delivery location must match PROCESS task 10 location"
        ):
            self.create_domain(tasks=tasks, robots=robots)

    def test_normal_edges_and_typed_constructors_reject_type_mixing(self):
        with self.assertRaisesRegex(
            ValueError, "normal edge source 30 must be PROCESS"
        ):
            self.create_domain(normal_edges=((30, 20),))

        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'capacity'"):
            ProcessRobot(0, (0, 0), (True,), capacity=10)

        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'load'"):
            ProcessTask(1, (0, 0), 1, (True,), load=3)

    def test_every_transport_has_exactly_one_downstream_process(self):
        with self.assertRaisesRegex(
            ValueError, "TRANSPORT task 30 must have exactly one downstream PROCESS task"
        ):
            self.create_domain(material_edges=())

    def test_duplicate_entity_ids_fail_clearly(self):
        tasks, robots = md_entities()
        with self.assertRaisesRegex(ValueError, "duplicate task_id 10"):
            self.create_domain(tasks=tasks + (tasks[0],), robots=robots)

        with self.assertRaisesRegex(ValueError, "duplicate robot_id 0"):
            self.create_domain(tasks=tasks, robots=robots + (robots[0],))

    def test_legacy_schema_is_rejected_when_material_delivery_is_enabled(self):
        legacy = json.loads(LEGACY_FIXTURE.read_text())
        with self.assertRaisesRegex(
            ValueError, "legacy schema requires material_delivery.enabled=false"
        ):
            SchedulingDomain.from_legacy_mapping(legacy, self.config)

    def test_prefilled_relation_fields_must_match_typed_edges(self):
        tasks, robots = md_entities()
        tasks = tasks[:2] + (
            TransportTask(
                30,
                (0, 0),
                (4, 0),
                6,
                2,
                3,
                downstream_process_task_id=20,
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "TRANSPORT task 30 downstream_process_task_id 20 "
            "conflicts with material edge target 10",
        ):
            self.create_domain(tasks=tasks, robots=robots)

        tasks, robots = md_entities()
        tasks = (
            ProcessTask(
                10,
                (4, 0),
                7,
                (True, False),
                material_predecessor=40,
            ),
        ) + tasks[1:]
        with self.assertRaisesRegex(
            ValueError,
            "PROCESS task 10 material_predecessor conflicts with material edges",
        ):
            self.create_domain(tasks=tasks, robots=robots)

        tasks, robots = md_entities()
        tasks = tasks[:1] + (
            ProcessTask(
                20,
                (8, 0),
                5,
                (False, True),
                normal_predecessors=(99,),
            ),
        ) + tasks[2:]
        with self.assertRaisesRegex(
            ValueError,
            "PROCESS task 20 normal_predecessors conflict with normal edges",
        ):
            self.create_domain(tasks=tasks, robots=robots)

    def test_entity_values_reject_invalid_numbers_and_boolean_vectors(self):
        invalid_entities = (
            (
                "non-finite location",
                lambda: ProcessTask(1, (float("nan"), 0), 1, (True,)),
                "location coordinate must be a finite number",
            ),
            (
                "non-boolean requirement",
                lambda: ProcessTask(1, (0, 0), 1, (1,)),
                "requirements must contain only boolean values",
            ),
            (
                "negative duration",
                lambda: ProcessTask(1, (0, 0), -1, (True,)),
                "duration must be a non-negative finite number",
            ),
            (
                "zero load",
                lambda: TransportTask(1, (0, 0), (1, 0), 0, 1, 1),
                "load must be a positive finite number",
            ),
            (
                "negative loading duration",
                lambda: TransportTask(1, (0, 0), (1, 0), 1, -1, 1),
                "loading_duration must be a non-negative finite number",
            ),
            (
                "non-boolean transport capability",
                lambda: TransportRobot(1, (0, 0), 1, 1, 1),
                "transport_capable must be boolean",
            ),
            (
                "zero capacity",
                lambda: TransportRobot(1, (0, 0), True, 0, 1),
                "capacity must be a positive finite number",
            ),
            (
                "zero speed",
                lambda: ProcessRobot(1, (0, 0), (True,), speed=0),
                "speed must be a positive finite number",
            ),
        )
        for name, constructor, message in invalid_entities:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    constructor()

    def test_domain_values_are_immutable(self):
        domain = self.create_domain()

        with self.assertRaises(FrozenInstanceError):
            domain.config.enabled = False
        with self.assertRaises(FrozenInstanceError):
            domain.tasks[0].duration = 99


if __name__ == "__main__":
    unittest.main()
