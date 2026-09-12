"""Immutable runtime domain objects for material-delivery scheduling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, TypeAlias


LEGACY_SCHEMA_FIELDS = frozenset(
    {"Q", "R", "T_e", "T_t", "task_locations", "precedence_constraints"}
)


class TaskType(str, Enum):
    PROCESS = "PROCESS"
    TRANSPORT = "TRANSPORT"


class RobotType(str, Enum):
    PROCESS_ROBOT = "PROCESS_ROBOT"
    TRANSPORT_ROBOT = "TRANSPORT_ROBOT"


@dataclass(frozen=True, slots=True)
class MaterialDeliveryConfig:
    enabled: bool = False
    use_typed_edges: bool = True
    use_downstream_encoding: bool = True
    use_transport_eta: bool = True
    use_capacity_features: bool = True
    loaded_speed_factor: float = 0.8

    def __post_init__(self) -> None:
        boolean_fields = (
            "enabled",
            "use_typed_edges",
            "use_downstream_encoding",
            "use_transport_eta",
            "use_capacity_features",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"material_delivery.{name} must be boolean")
        factor = self.loaded_speed_factor
        if (
            isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not math.isfinite(factor)
            or factor <= 0
        ):
            raise ValueError("loaded_speed_factor must be a positive finite number")
        object.__setattr__(self, "loaded_speed_factor", float(factor))


Location: TypeAlias = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ProcessTask:
    task_id: int
    location: Location
    duration: float
    requirements: tuple[bool, ...]
    normal_predecessors: tuple[int, ...] = ()
    material_predecessor: int | None = None
    task_type: TaskType = field(default=TaskType.PROCESS, init=False)

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task_id")
        object.__setattr__(self, "location", _location(self.location, "location"))
        object.__setattr__(
            self, "duration", _nonnegative_number(self.duration, "duration")
        )
        object.__setattr__(
            self, "requirements", _booleans(self.requirements, "requirements")
        )
        object.__setattr__(
            self,
            "normal_predecessors",
            tuple(
                _identifier(task_id, "normal predecessor")
                for task_id in self.normal_predecessors
            ),
        )
        if self.material_predecessor is not None:
            _identifier(self.material_predecessor, "material_predecessor")


@dataclass(frozen=True, slots=True)
class TransportTask:
    task_id: int
    pickup_location: Location
    delivery_location: Location
    load: float
    loading_duration: float
    unloading_duration: float
    downstream_process_task_id: int | None = None
    task_type: TaskType = field(default=TaskType.TRANSPORT, init=False)

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task_id")
        object.__setattr__(
            self,
            "pickup_location",
            _location(self.pickup_location, "pickup_location"),
        )
        object.__setattr__(
            self,
            "delivery_location",
            _location(self.delivery_location, "delivery_location"),
        )
        object.__setattr__(self, "load", _positive_number(self.load, "load"))
        object.__setattr__(
            self,
            "loading_duration",
            _nonnegative_number(self.loading_duration, "loading_duration"),
        )
        object.__setattr__(
            self,
            "unloading_duration",
            _nonnegative_number(self.unloading_duration, "unloading_duration"),
        )
        if self.downstream_process_task_id is not None:
            _identifier(
                self.downstream_process_task_id, "downstream_process_task_id"
            )


@dataclass(frozen=True, slots=True)
class ProcessRobot:
    robot_id: int
    location: Location
    capabilities: tuple[bool, ...]
    speed: float = 1.0
    robot_type: RobotType = field(default=RobotType.PROCESS_ROBOT, init=False)

    def __post_init__(self) -> None:
        _identifier(self.robot_id, "robot_id")
        object.__setattr__(self, "location", _location(self.location, "location"))
        object.__setattr__(
            self, "capabilities", _booleans(self.capabilities, "capabilities")
        )
        object.__setattr__(self, "speed", _positive_number(self.speed, "speed"))


@dataclass(frozen=True, slots=True)
class TransportRobot:
    robot_id: int
    location: Location
    transport_capable: bool
    capacity: float
    unloaded_speed: float
    loaded_speed: float | None = None
    robot_type: RobotType = field(default=RobotType.TRANSPORT_ROBOT, init=False)

    def __post_init__(self) -> None:
        _identifier(self.robot_id, "robot_id")
        object.__setattr__(self, "location", _location(self.location, "location"))
        if not isinstance(self.transport_capable, bool):
            raise ValueError("transport_capable must be boolean")
        object.__setattr__(
            self, "capacity", _positive_number(self.capacity, "capacity")
        )
        object.__setattr__(
            self,
            "unloaded_speed",
            _positive_number(self.unloaded_speed, "unloaded_speed"),
        )
        if self.loaded_speed is not None:
            object.__setattr__(
                self,
                "loaded_speed",
                _positive_number(self.loaded_speed, "loaded_speed"),
            )


TaskEntity: TypeAlias = ProcessTask | TransportTask
RobotEntity: TypeAlias = ProcessRobot | TransportRobot
Edge: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class SchedulingDomain:
    config: MaterialDeliveryConfig
    tasks: tuple[TaskEntity, ...]
    robots: tuple[RobotEntity, ...]
    normal_edges: tuple[Edge, ...]
    material_edges: tuple[Edge, ...]
    is_legacy: bool = False
    legacy_travel_times: tuple[tuple[float, ...], ...] | None = None

    @classmethod
    def create(
        cls,
        *,
        config: MaterialDeliveryConfig,
        tasks: Sequence[TaskEntity],
        robots: Sequence[RobotEntity],
        normal_edges: Sequence[Sequence[int]] = (),
        material_edges: Sequence[Sequence[int]] = (),
    ) -> SchedulingDomain:
        if not isinstance(config, MaterialDeliveryConfig):
            raise TypeError("config must be a MaterialDeliveryConfig")

        frozen_tasks = tuple(tasks)
        frozen_robots = tuple(robots)
        task_by_id = _index_entities(frozen_tasks, "task", "task_id")
        _index_entities(frozen_robots, "robot", "robot_id")
        normal = _edges(normal_edges, "normal")
        material = _edges(material_edges, "material")

        has_transport = any(
            isinstance(entity, (TransportTask, TransportRobot))
            for entity in (*frozen_tasks, *frozen_robots)
        )
        if not config.enabled and (has_transport or material):
            raise ValueError(
                "transport entities require material_delivery.enabled=true"
            )

        normal_predecessors: dict[int, list[int]] = {
            task.task_id: []
            for task in frozen_tasks
            if isinstance(task, ProcessTask)
        }
        material_predecessors: dict[int, int] = {}
        downstream: dict[int, int] = {}

        for source_id, target_id in normal:
            source = _edge_task(task_by_id, source_id, "normal")
            target = _edge_task(task_by_id, target_id, "normal")
            if not isinstance(source, ProcessTask):
                raise ValueError(f"normal edge source {source_id} must be PROCESS")
            if not isinstance(target, ProcessTask):
                raise ValueError(f"normal edge target {target_id} must be PROCESS")
            normal_predecessors[target_id].append(source_id)

        for source_id, target_id in material:
            source = _edge_task(task_by_id, source_id, "material")
            target = _edge_task(task_by_id, target_id, "material")
            if not isinstance(source, TransportTask):
                raise ValueError(f"material edge source {source_id} must be TRANSPORT")
            if not isinstance(target, ProcessTask):
                raise ValueError(f"material edge target {target_id} must be PROCESS")
            if source_id in downstream:
                raise ValueError(
                    f"TRANSPORT task {source_id} must have exactly one downstream PROCESS task"
                )
            if target_id in material_predecessors:
                raise ValueError(
                    f"PROCESS task {target_id} has more than one material predecessor"
                )
            if source.delivery_location != target.location:
                raise ValueError(
                    "TRANSPORT task "
                    f"{source_id} delivery location must match PROCESS task "
                    f"{target_id} location"
                )
            downstream[source_id] = target_id
            material_predecessors[target_id] = source_id

        _validate_declared_relations(
            frozen_tasks,
            normal_predecessors,
            material_predecessors,
            downstream,
        )

        derived_tasks: list[TaskEntity] = []
        for task in frozen_tasks:
            if isinstance(task, ProcessTask):
                derived_tasks.append(
                    replace(
                        task,
                        normal_predecessors=tuple(normal_predecessors[task.task_id]),
                        material_predecessor=material_predecessors.get(task.task_id),
                    )
                )
            else:
                derived_tasks.append(
                    replace(
                        task,
                        downstream_process_task_id=downstream[task.task_id],
                    )
                )

        derived_robots = tuple(
            replace(
                robot,
                loaded_speed=robot.unloaded_speed * config.loaded_speed_factor,
            )
            if isinstance(robot, TransportRobot) and robot.loaded_speed is None
            else robot
            for robot in frozen_robots
        )
        return cls(
            config=config,
            tasks=tuple(derived_tasks),
            robots=derived_robots,
            normal_edges=normal,
            material_edges=material,
        )

    @classmethod
    def from_legacy_mapping(
        cls, data: Mapping[str, Any], config: MaterialDeliveryConfig
    ) -> SchedulingDomain:
        if not isinstance(config, MaterialDeliveryConfig):
            raise TypeError("config must be a MaterialDeliveryConfig")
        if config.enabled:
            raise ValueError("legacy schema requires material_delivery.enabled=false")
        if not isinstance(data, Mapping):
            raise TypeError("legacy domain must be a mapping")
        missing = sorted(LEGACY_SCHEMA_FIELDS - set(data))
        unexpected = sorted(set(data) - LEGACY_SCHEMA_FIELDS)
        if missing or unexpected:
            raise ValueError(
                "legacy schema mismatch; "
                f"missing fields={missing}, unexpected fields={unexpected}"
            )

        requirements = _rows(data["R"], "R")
        durations = _sequence(data["T_e"], "T_e")
        locations = _rows(data["task_locations"], "task_locations")
        travel_times = _rows(data["T_t"], "T_t")
        node_count = len(requirements)
        if node_count < 2:
            raise ValueError("legacy R must include start and exit rows")
        if len(durations) != node_count or len(locations) != node_count:
            raise ValueError("legacy task arrays must have matching lengths")
        if len(travel_times) != node_count or any(
            len(row) != node_count for row in travel_times
        ):
            raise ValueError("legacy T_t must be square and match the task count")

        real_task_ids = set(range(1, node_count - 1))
        normal = _edges(
            ()
            if data["precedence_constraints"] is None
            else data["precedence_constraints"],
            "normal",
        )
        predecessors: dict[int, list[int]] = {
            task_id: [] for task_id in real_task_ids
        }
        for source_id, target_id in normal:
            if source_id not in real_task_ids:
                raise ValueError(
                    f"normal edge references unknown legacy task {source_id}"
                )
            if target_id not in real_task_ids:
                raise ValueError(
                    f"normal edge references unknown legacy task {target_id}"
                )
            predecessors[target_id].append(source_id)

        tasks = tuple(
            ProcessTask(
                task_id,
                _location(locations[task_id], f"task {task_id} location"),
                durations[task_id],
                _flags(requirements[task_id], f"task {task_id} requirements"),
                tuple(predecessors[task_id]),
            )
            for task_id in range(1, node_count - 1)
        )
        start_location = _location(locations[0], "legacy start location")
        robots = tuple(
            ProcessRobot(
                robot_id,
                start_location,
                _flags(row, f"robot {robot_id} capabilities"),
            )
            for robot_id, row in enumerate(_rows(data["Q"], "Q"))
        )
        return cls(
            config=config,
            tasks=tasks,
            robots=robots,
            normal_edges=normal,
            material_edges=(),
            is_legacy=True,
            legacy_travel_times=tuple(tuple(row) for row in travel_times),
        )


def _validate_declared_relations(
    tasks: tuple[TaskEntity, ...],
    normal_predecessors: Mapping[int, list[int]],
    material_predecessors: Mapping[int, int],
    downstream: Mapping[int, int],
) -> None:
    for task in tasks:
        if isinstance(task, ProcessTask):
            expected_normal = tuple(normal_predecessors[task.task_id])
            if task.normal_predecessors and task.normal_predecessors != expected_normal:
                raise ValueError(
                    f"PROCESS task {task.task_id} normal_predecessors "
                    "conflict with normal edges"
                )
            expected_material = material_predecessors.get(task.task_id)
            if (
                task.material_predecessor is not None
                and task.material_predecessor != expected_material
            ):
                raise ValueError(
                    f"PROCESS task {task.task_id} material_predecessor "
                    "conflicts with material edges"
                )
            continue

        expected_downstream = downstream.get(task.task_id)
        if expected_downstream is None:
            raise ValueError(
                f"TRANSPORT task {task.task_id} must have exactly one "
                "downstream PROCESS task"
            )
        if (
            task.downstream_process_task_id is not None
            and task.downstream_process_task_id != expected_downstream
        ):
            raise ValueError(
                f"TRANSPORT task {task.task_id} downstream_process_task_id "
                f"{task.downstream_process_task_id} conflicts with material "
                f"edge target {expected_downstream}"
            )


def _index_entities(
    entities: tuple[Any, ...], kind: str, id_attribute: str
) -> dict[int, Any]:
    expected = (
        (ProcessTask, TransportTask)
        if kind == "task"
        else (ProcessRobot, TransportRobot)
    )
    indexed: dict[int, Any] = {}
    for entity in entities:
        if not isinstance(entity, expected):
            raise TypeError(f"{kind}s must contain typed {kind} entities")
        entity_id = getattr(entity, id_attribute)
        if entity_id in indexed:
            raise ValueError(f"duplicate {id_attribute} {entity_id}")
        indexed[entity_id] = entity
    return indexed


def _edges(value: Sequence[Sequence[int]], name: str) -> tuple[Edge, ...]:
    try:
        rows = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} edges must be a sequence of task ID pairs") from error
    parsed: list[Edge] = []
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise ValueError(f"{name} edge must contain exactly two task IDs")
        try:
            pair = tuple(row)
        except TypeError as error:
            raise ValueError(
                f"{name} edge must contain exactly two task IDs"
            ) from error
        if len(pair) != 2:
            raise ValueError(f"{name} edge must contain exactly two task IDs")
        source_id = _identifier(pair[0], f"{name} edge source")
        target_id = _identifier(pair[1], f"{name} edge target")
        parsed.append((source_id, target_id))
    return tuple(parsed)


def _edge_task(
    task_by_id: dict[int, TaskEntity], task_id: int, edge_name: str
) -> TaskEntity:
    try:
        return task_by_id[task_id]
    except KeyError as error:
        raise ValueError(
            f"{edge_name} edge references unknown task {task_id}"
        ) from error


def _identifier(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_number(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _nonnegative_number(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return number


def _location(value: Any, name: str) -> Location:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly two coordinates")
    try:
        coordinates = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain exactly two coordinates") from error
    if len(coordinates) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    return (
        _finite_number(coordinates[0], f"{name} coordinate"),
        _finite_number(coordinates[1], f"{name} coordinate"),
    )


def _booleans(value: Any, name: str) -> tuple[bool, ...]:
    flags = _sequence(value, name)
    if any(not isinstance(item, bool) for item in flags):
        raise ValueError(f"{name} must contain only boolean values")
    return flags


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"legacy {name} must be a sequence")
    try:
        return tuple(value)
    except TypeError as error:
        raise ValueError(f"legacy {name} must be a sequence") from error


def _rows(value: Any, name: str) -> tuple[tuple[Any, ...], ...]:
    rows = _sequence(value, name)
    try:
        return tuple(tuple(row) for row in rows)
    except TypeError as error:
        raise ValueError(f"legacy {name} must contain sequences") from error


def _flags(value: Any, name: str) -> tuple[bool, ...]:
    flags = _sequence(value, name)
    if any(item not in (0, 1, False, True) for item in flags):
        raise ValueError(f"legacy {name} must contain only boolean values")
    return tuple(bool(item) for item in flags)
