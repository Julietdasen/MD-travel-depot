"""Static validation gate for material-delivery runtime domains."""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from experiments.protocol import FailureReason
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    RobotType,
    SchedulingDomain,
    TaskEntity,
    TaskType,
    TransportRobot,
    TransportTask,
)


class StaticValidationCode(str, Enum):
    """Stable detail codes emitted before an MD rollout starts."""

    INVALID_ID = "invalid_id"
    INVALID_ENTITY_TYPE = "invalid_entity_type"
    INVALID_REFERENCE = "invalid_reference"
    INVALID_MATERIAL_RELATION = "invalid_material_relation"
    INVALID_LOCATION = "invalid_location"
    INVALID_NUMERIC_VALUE = "invalid_numeric_value"
    INVALID_VALUE = "invalid_value"
    INCOMPATIBLE_FEATURE_DIMENSION = "incompatible_feature_dimension"
    INVALID_GRAPH = "invalid_graph"
    EMPTY_DOMAIN = "empty_domain"
    NO_CAPABLE_TRANSPORT_ROBOT = "no_capable_transport_robot"
    INSUFFICIENT_SOLO_CAPACITY = "insufficient_solo_capacity"
    NO_CAPABLE_PROCESS_COALITION = "no_capable_process_coalition"


_FAILURE_REASON_BY_CODE = MappingProxyType(
    {
        code: FailureReason.INVALID_GRAPH
        for code in (
            StaticValidationCode.INVALID_ID,
            StaticValidationCode.INVALID_ENTITY_TYPE,
            StaticValidationCode.INVALID_REFERENCE,
            StaticValidationCode.INVALID_MATERIAL_RELATION,
            StaticValidationCode.INVALID_LOCATION,
            StaticValidationCode.INVALID_NUMERIC_VALUE,
            StaticValidationCode.INVALID_VALUE,
            StaticValidationCode.INCOMPATIBLE_FEATURE_DIMENSION,
            StaticValidationCode.INVALID_GRAPH,
        )
    }
    | {
        StaticValidationCode.EMPTY_DOMAIN: FailureReason.STATIC_INFEASIBLE,
        StaticValidationCode.NO_CAPABLE_TRANSPORT_ROBOT: (
            FailureReason.NO_CAPABLE_TRANSPORT_ROBOT
        ),
        StaticValidationCode.INSUFFICIENT_SOLO_CAPACITY: (
            FailureReason.STATIC_INFEASIBLE
        ),
        StaticValidationCode.NO_CAPABLE_PROCESS_COALITION: (
            FailureReason.STATIC_INFEASIBLE
        ),
    }
)


@dataclass(frozen=True, slots=True)
class StaticValidationResult:
    is_valid: bool
    reason_code: StaticValidationCode | None = None
    failure_reason: FailureReason | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.is_valid:
            if any(
                value is not None
                for value in (self.reason_code, self.failure_reason, self.message)
            ):
                raise ValueError("valid static result cannot contain failure details")
            return
        if not isinstance(self.reason_code, StaticValidationCode):
            raise ValueError("invalid static result requires a reason_code")
        if not isinstance(self.failure_reason, FailureReason):
            raise ValueError("invalid static result requires a failure_reason")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("invalid static result requires a message")


class StaticValidationError(ValueError):
    """Raised by the rollout gate when static validation fails."""

    def __init__(self, result: StaticValidationResult):
        if not isinstance(result, StaticValidationResult) or result.is_valid:
            raise ValueError("StaticValidationError requires an invalid result")
        self.result = result
        reason_code = result.reason_code
        if reason_code is None:
            raise ValueError("invalid result requires a reason_code")
        super().__init__(f"{reason_code.value}: {result.message}")


def validate_md_domain(domain: SchedulingDomain) -> StaticValidationResult:
    """Return the first deterministic structural or feasibility failure."""

    if not isinstance(domain, SchedulingDomain):
        return _invalid(
            StaticValidationCode.INVALID_ENTITY_TYPE,
            "domain must be a SchedulingDomain",
        )

    failure = _validate_config(domain.config)
    if failure is not None:
        return failure

    tasks = domain.tasks
    robots = domain.robots
    if not isinstance(tasks, tuple) or not isinstance(robots, tuple):
        return _invalid(
            StaticValidationCode.INVALID_ENTITY_TYPE,
            "domain tasks and robots must be immutable tuples",
        )
    if not tasks:
        return _invalid(
            StaticValidationCode.EMPTY_DOMAIN,
            "domain must contain at least one real task",
        )

    failure = _validate_entity_types(tasks, robots)
    if failure is not None:
        return failure
    failure = _validate_continuous_ids(tasks, robots)
    if failure is not None:
        return failure
    failure = _validate_entity_values(tasks, robots)
    if failure is not None:
        return failure

    task_by_id = {task.task_id: task for task in tasks}
    relation_result = _validate_relations(domain, task_by_id)
    if isinstance(relation_result, StaticValidationResult):
        return relation_result
    graph_edges = relation_result
    if not _is_dag(task_by_id, graph_edges):
        return _invalid(
            StaticValidationCode.INVALID_GRAPH,
            "combined normal and material task graph must be acyclic",
        )

    failure = _validate_feature_dimensions(tasks, robots)
    if failure is not None:
        return failure
    failure = _validate_static_feasibility(tasks, robots)
    if failure is not None:
        return failure
    return StaticValidationResult(is_valid=True)


def require_valid_md_domain(domain: SchedulingDomain) -> SchedulingDomain:
    """Return a validated domain or prevent it from entering rollout."""

    result = validate_md_domain(domain)
    if not result.is_valid:
        raise StaticValidationError(result)
    return domain


def _validate_config(
    config: MaterialDeliveryConfig,
) -> StaticValidationResult | None:
    if not isinstance(config, MaterialDeliveryConfig):
        return _invalid(
            StaticValidationCode.INVALID_ENTITY_TYPE,
            "domain config must be a MaterialDeliveryConfig",
        )
    boolean_fields = (
        "enabled",
        "use_typed_edges",
        "use_downstream_encoding",
        "use_transport_eta",
        "use_capacity_features",
    )
    if any(not isinstance(getattr(config, name), bool) for name in boolean_fields):
        return _value_failure("material-delivery flags must be boolean")
    if not _positive_finite(config.loaded_speed_factor):
        return _numeric_failure("loaded_speed_factor must be positive and finite")
    return None


def _validate_entity_types(
    tasks: tuple[Any, ...], robots: tuple[Any, ...]
) -> StaticValidationResult | None:
    for task in tasks:
        if not isinstance(task, (ProcessTask, TransportTask)):
            return _entity_type_failure("tasks must contain typed task entities")
        expected_task_type = (
            TaskType.PROCESS if isinstance(task, ProcessTask) else TaskType.TRANSPORT
        )
        if task.task_type is not expected_task_type:
            return _entity_type_failure(
                f"task {task.task_id} has a mismatched task_type"
            )
    for robot in robots:
        if not isinstance(robot, (ProcessRobot, TransportRobot)):
            return _entity_type_failure("robots must contain typed robot entities")
        expected_robot_type = (
            RobotType.PROCESS_ROBOT
            if isinstance(robot, ProcessRobot)
            else RobotType.TRANSPORT_ROBOT
        )
        if robot.robot_type is not expected_robot_type:
            return _entity_type_failure(
                f"robot {robot.robot_id} has a mismatched robot_type"
            )
    return None


def _validate_continuous_ids(
    tasks: tuple[TaskEntity, ...],
    robots: tuple[ProcessRobot | TransportRobot, ...],
) -> StaticValidationResult | None:
    task_ids = [task.task_id for task in tasks]
    robot_ids = [robot.robot_id for robot in robots]
    if any(not _is_integer(identifier) for identifier in (*task_ids, *robot_ids)):
        return _id_failure("task and robot IDs must be integers")
    if sorted(task_ids) != list(range(1, len(task_ids) + 1)):
        return _id_failure("task IDs must be unique and continuous from 1")
    if sorted(robot_ids) != list(range(len(robot_ids))):
        return _id_failure("robot IDs must be unique and continuous from 0")
    return None


def _validate_entity_values(
    tasks: tuple[TaskEntity, ...],
    robots: tuple[ProcessRobot | TransportRobot, ...],
) -> StaticValidationResult | None:
    for task in tasks:
        if isinstance(task, ProcessTask):
            if not _valid_location(task.location):
                return _location_failure(f"PROCESS task {task.task_id} has invalid location")
            if not _nonnegative_finite(task.duration):
                return _numeric_failure(
                    f"PROCESS task {task.task_id} duration must be non-negative and finite"
                )
            if not _boolean_sequence(task.requirements):
                return _value_failure(
                    f"PROCESS task {task.task_id} requirements must be boolean"
                )
            relation_ids = (*task.normal_predecessors, task.material_predecessor)
            if any(
                identifier is not None and not _is_integer(identifier)
                for identifier in relation_ids
            ):
                return _id_failure(
                    f"PROCESS task {task.task_id} predecessor IDs must be integers"
                )
            continue
        if not _valid_location(task.pickup_location):
            return _location_failure(
                f"TRANSPORT task {task.task_id} has invalid pickup_location"
            )
        if not _valid_location(task.delivery_location):
            return _location_failure(
                f"TRANSPORT task {task.task_id} has invalid delivery_location"
            )
        if not _positive_finite(task.load):
            return _numeric_failure(
                f"TRANSPORT task {task.task_id} load must be positive and finite"
            )
        if not _nonnegative_finite(task.loading_duration) or not _nonnegative_finite(
            task.unloading_duration
        ):
            return _numeric_failure(
                f"TRANSPORT task {task.task_id} service durations must be non-negative and finite"
            )
        if task.downstream_process_task_id is not None and not _is_integer(
            task.downstream_process_task_id
        ):
            return _id_failure(
                f"TRANSPORT task {task.task_id} downstream ID must be an integer"
            )

    for robot in robots:
        if not _valid_location(robot.location):
            return _location_failure(f"robot {robot.robot_id} has invalid location")
        if isinstance(robot, ProcessRobot):
            if not _boolean_sequence(robot.capabilities):
                return _value_failure(
                    f"PROCESS_ROBOT {robot.robot_id} capabilities must be boolean"
                )
            if not _positive_finite(robot.speed):
                return _numeric_failure(
                    f"PROCESS_ROBOT {robot.robot_id} speed must be positive and finite"
                )
            continue
        if not isinstance(robot.transport_capable, bool):
            return _value_failure(
                f"TRANSPORT_ROBOT {robot.robot_id} transport_capable must be boolean"
            )
        if not all(
            _positive_finite(value)
            for value in (robot.capacity, robot.unloaded_speed, robot.loaded_speed)
        ):
            return _numeric_failure(
                f"TRANSPORT_ROBOT {robot.robot_id} capacity and speeds must be positive and finite"
            )
    return None


def _validate_relations(
    domain: SchedulingDomain, task_by_id: dict[int, TaskEntity]
) -> tuple[tuple[int, int], ...] | StaticValidationResult:
    parsed_normal = _parse_edges(domain.normal_edges, "normal")
    if isinstance(parsed_normal, StaticValidationResult):
        return parsed_normal
    parsed_material = _parse_edges(domain.material_edges, "material")
    if isinstance(parsed_material, StaticValidationResult):
        return parsed_material

    if not domain.config.enabled and (
        parsed_material
        or any(isinstance(task, TransportTask) for task in domain.tasks)
        or any(isinstance(robot, TransportRobot) for robot in domain.robots)
    ):
        return _entity_type_failure(
            "transport entities require material delivery to be enabled"
        )

    normal_predecessors: dict[int, list[int]] = {
        task.task_id: []
        for task in domain.tasks
        if isinstance(task, ProcessTask)
    }
    for source_id, target_id in parsed_normal:
        reference_failure = _unknown_reference(task_by_id, source_id, target_id)
        if reference_failure is not None:
            return reference_failure
        if not isinstance(task_by_id[source_id], ProcessTask) or not isinstance(
            task_by_id[target_id], ProcessTask
        ):
            return _entity_type_failure("normal edges must connect PROCESS tasks")
        normal_predecessors[target_id].append(source_id)

    downstream: dict[int, int] = {}
    material_predecessors: dict[int, int] = {}
    for source_id, target_id in parsed_material:
        reference_failure = _unknown_reference(task_by_id, source_id, target_id)
        if reference_failure is not None:
            return reference_failure
        source = task_by_id[source_id]
        target = task_by_id[target_id]
        if not isinstance(source, TransportTask) or not isinstance(target, ProcessTask):
            return _entity_type_failure(
                "material edges must connect TRANSPORT to PROCESS"
            )
        if source_id in downstream or target_id in material_predecessors:
            return _invalid(
                StaticValidationCode.INVALID_MATERIAL_RELATION,
                "material edges must be one-to-one",
            )
        if source.delivery_location != target.location:
            return _location_failure(
                f"TRANSPORT task {source_id} delivery location must match PROCESS task {target_id} location"
            )
        downstream[source_id] = target_id
        material_predecessors[target_id] = source_id

    for task in domain.tasks:
        if isinstance(task, ProcessTask):
            if task.normal_predecessors != tuple(normal_predecessors[task.task_id]):
                return _invalid(
                    StaticValidationCode.INVALID_REFERENCE,
                    f"PROCESS task {task.task_id} normal predecessors conflict with normal edges",
                )
            if task.material_predecessor != material_predecessors.get(task.task_id):
                return _invalid(
                    StaticValidationCode.INVALID_MATERIAL_RELATION,
                    f"PROCESS task {task.task_id} material predecessor conflicts with material edges",
                )
            continue
        if downstream.get(task.task_id) != task.downstream_process_task_id:
            return _invalid(
                StaticValidationCode.INVALID_MATERIAL_RELATION,
                f"TRANSPORT task {task.task_id} must have exactly one matching downstream PROCESS task",
            )
    return parsed_normal + parsed_material


def _parse_edges(
    edges: Any, name: str
) -> tuple[tuple[int, int], ...] | StaticValidationResult:
    if not isinstance(edges, tuple):
        return _invalid(
            StaticValidationCode.INVALID_GRAPH,
            f"{name} edges must be an immutable tuple",
        )
    parsed: list[tuple[int, int]] = []
    for edge in edges:
        if not isinstance(edge, tuple) or len(edge) != 2:
            return _invalid(
                StaticValidationCode.INVALID_GRAPH,
                f"{name} edges must contain task ID pairs",
            )
        if not all(_is_integer(identifier) for identifier in edge):
            return _id_failure(f"{name} edge IDs must be integers")
        parsed.append(edge)
    if len(set(parsed)) != len(parsed):
        return _invalid(
            StaticValidationCode.INVALID_GRAPH,
            f"{name} edges must not contain duplicates",
        )
    return tuple(parsed)


def _unknown_reference(
    task_by_id: dict[int, TaskEntity], source_id: int, target_id: int
) -> StaticValidationResult | None:
    unknown = next(
        (task_id for task_id in (source_id, target_id) if task_id not in task_by_id),
        None,
    )
    if unknown is None:
        return None
    return _invalid(
        StaticValidationCode.INVALID_REFERENCE,
        f"edge references unknown task {unknown}",
    )


def _is_dag(
    task_by_id: dict[int, TaskEntity], edges: Sequence[tuple[int, int]]
) -> bool:
    successors: dict[int, list[int]] = {task_id: [] for task_id in task_by_id}
    indegree = Counter({task_id: 0 for task_id in task_by_id})
    for source_id, target_id in edges:
        successors[source_id].append(target_id)
        indegree[target_id] += 1
    ready = deque(task_id for task_id in task_by_id if indegree[task_id] == 0)
    visited = 0
    while ready:
        task_id = ready.popleft()
        visited += 1
        for successor in successors[task_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    return visited == len(task_by_id)


def _validate_feature_dimensions(
    tasks: tuple[TaskEntity, ...],
    robots: tuple[ProcessRobot | TransportRobot, ...],
) -> StaticValidationResult | None:
    process_tasks = tuple(task for task in tasks if isinstance(task, ProcessTask))
    process_robots = tuple(
        robot for robot in robots if isinstance(robot, ProcessRobot)
    )
    dimensions = {
        len(vector)
        for vector in (
            *(task.requirements for task in process_tasks),
            *(robot.capabilities for robot in process_robots),
        )
    }
    if len(dimensions) > 1:
        return _invalid(
            StaticValidationCode.INCOMPATIBLE_FEATURE_DIMENSION,
            "process requirement and capability vectors must share one dimension",
        )
    return None


def _validate_static_feasibility(
    tasks: tuple[TaskEntity, ...],
    robots: tuple[ProcessRobot | TransportRobot, ...],
) -> StaticValidationResult | None:
    transport_tasks = tuple(task for task in tasks if isinstance(task, TransportTask))
    capable_transport_robots = tuple(
        robot
        for robot in robots
        if isinstance(robot, TransportRobot) and robot.transport_capable
    )
    if transport_tasks and not capable_transport_robots:
        return _invalid(
            StaticValidationCode.NO_CAPABLE_TRANSPORT_ROBOT,
            "transport tasks require at least one transport-capable robot",
        )
    for transport_task in transport_tasks:
        if not any(
            robot.capacity >= transport_task.load
            for robot in capable_transport_robots
        ):
            return _invalid(
                StaticValidationCode.INSUFFICIENT_SOLO_CAPACITY,
                f"TRANSPORT task {transport_task.task_id} "
                "cannot be carried by one robot",
            )

    process_tasks = tuple(task for task in tasks if isinstance(task, ProcessTask))
    process_robots = tuple(
        robot for robot in robots if isinstance(robot, ProcessRobot)
    )
    for process_task in process_tasks:
        if not process_robots or any(
            required
            and not any(robot.capabilities[index] for robot in process_robots)
            for index, required in enumerate(process_task.requirements)
        ):
            return _invalid(
                StaticValidationCode.NO_CAPABLE_PROCESS_COALITION,
                f"PROCESS task {process_task.task_id} requirements cannot be covered",
            )
    return None


def _invalid(code: StaticValidationCode, message: str) -> StaticValidationResult:
    return StaticValidationResult(
        is_valid=False,
        reason_code=code,
        failure_reason=_FAILURE_REASON_BY_CODE[code],
        message=message,
    )


def _id_failure(message: str) -> StaticValidationResult:
    return _invalid(StaticValidationCode.INVALID_ID, message)


def _entity_type_failure(message: str) -> StaticValidationResult:
    return _invalid(StaticValidationCode.INVALID_ENTITY_TYPE, message)


def _location_failure(message: str) -> StaticValidationResult:
    return _invalid(StaticValidationCode.INVALID_LOCATION, message)


def _numeric_failure(message: str) -> StaticValidationResult:
    return _invalid(StaticValidationCode.INVALID_NUMERIC_VALUE, message)


def _value_failure(message: str) -> StaticValidationResult:
    return _invalid(StaticValidationCode.INVALID_VALUE, message)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _positive_finite(value: Any) -> bool:
    return _finite_number(value) and value > 0


def _nonnegative_finite(value: Any) -> bool:
    return _finite_number(value) and value >= 0


def _valid_location(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(_finite_number(coordinate) for coordinate in value)
    )


def _boolean_sequence(value: Any) -> bool:
    return isinstance(value, tuple) and all(isinstance(item, bool) for item in value)
