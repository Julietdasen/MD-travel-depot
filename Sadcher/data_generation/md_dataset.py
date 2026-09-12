"""Versioned JSON envelope and task-level loader for synthetic MD instances."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from data_generation.md_instance_generator import (
    GeneratedMDInstance,
    MDGeneratorConfig,
)
from experiments.protocol import DatasetSplit, task_level_split
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


MD_DATASET_SCHEMA_VERSION = "1.0.0"
_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "instance_id",
        "task_group_id",
        "seed",
        "generation_config",
        "domain",
        "split",
        "solution",
        "execution_records",
        "metrics",
    }
)


@dataclass(frozen=True, slots=True)
class MDInstanceRecord:
    instance_id: str
    task_group_id: str
    seed: int
    generated: GeneratedMDInstance
    solution: Mapping[str, Any] | None = None
    execution_records: Mapping[str, Any] | None = None
    metrics: Mapping[str, Any] | None = None
    split: DatasetSplit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be a non-empty string")
        if not isinstance(self.task_group_id, str) or not self.task_group_id.strip():
            raise ValueError("task_group_id must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.generated, GeneratedMDInstance):
            raise TypeError("generated must be a GeneratedMDInstance")
        if self.generated.generation_config.seed != self.seed:
            raise ValueError("seed must match generation_config.seed")
        expected_split = task_level_split(self.task_group_id)
        if self.split is not None and self.split is not expected_split:
            raise ValueError("split must match task_level_split(task_group_id)")

        object.__setattr__(self, "split", expected_split)
        object.__setattr__(
            self,
            "solution",
            _freeze_json_mapping("solution", {} if self.solution is None else self.solution),
        )
        object.__setattr__(
            self,
            "execution_records",
            _freeze_execution_records(
                {"process": [], "transport": []}
                if self.execution_records is None
                else self.execution_records
            ),
        )
        object.__setattr__(
            self,
            "metrics",
            _freeze_json_mapping("metrics", {} if self.metrics is None else self.metrics),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.split is None:
            raise RuntimeError("record split was not initialized")
        return {
            "schema_version": MD_DATASET_SCHEMA_VERSION,
            "instance_id": self.instance_id,
            "task_group_id": self.task_group_id,
            "seed": self.seed,
            "split": self.split.value,
            "generation_config": _config_to_dict(
                self.generated.generation_config
            ),
            "domain": _domain_to_dict(self.generated.domain),
            "solution": _plain_json(self.solution),
            "execution_records": _plain_json(self.execution_records),
            "metrics": _plain_json(self.metrics),
        }


class MDDatasetLoader:
    """Load versioned MD records and filter by task-level split/group."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise ValueError(f"MD dataset root does not exist: {self.root}")
        if not self.root.is_dir():
            raise ValueError(f"MD dataset root must be a directory: {self.root}")

    def load(
        self,
        *,
        split: DatasetSplit | None = None,
        task_group_id: str | None = None,
    ) -> tuple[MDInstanceRecord, ...]:
        if split is not None and not isinstance(split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit")
        if task_group_id is not None and (
            not isinstance(task_group_id, str) or not task_group_id.strip()
        ):
            raise ValueError("task_group_id must be a non-empty string")

        records = []
        seen_instance_ids: set[str] = set()
        for path in sorted(self.root.glob("*.json")):
            record = load_md_instance(path)
            if record.instance_id in seen_instance_ids:
                raise ValueError(f"duplicate MD instance_id: {record.instance_id}")
            seen_instance_ids.add(record.instance_id)
            if split is not None and record.split is not split:
                continue
            if task_group_id is not None and record.task_group_id != task_group_id:
                continue
            records.append(record)
        return tuple(records)


def save_md_instance(path: str | Path, record: MDInstanceRecord) -> None:
    if not isinstance(record, MDInstanceRecord):
        raise TypeError("record must be an MDInstanceRecord")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        record.to_dict(),
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    destination.write_text(serialized, encoding="utf-8")


def load_md_instance(path: str | Path) -> MDInstanceRecord:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read MD instance JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("MD instance document must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != MD_DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported MD schema_version: {schema_version!r}")
    missing = sorted(_REQUIRED_FIELDS - set(payload))
    if missing:
        raise ValueError(f"missing required MD instance fields: {missing}")

    record = MDInstanceRecord(
        instance_id=_string(payload["instance_id"], "instance_id"),
        task_group_id=_string(payload["task_group_id"], "task_group_id"),
        seed=_integer(payload["seed"], "seed"),
        generated=_generated_from_dict(payload["generation_config"], payload["domain"]),
        solution=_freeze_json_mapping("solution", payload["solution"]),
        execution_records=_freeze_execution_records(payload["execution_records"]),
        metrics=_freeze_json_mapping("metrics", payload["metrics"]),
        split=_split(payload.get("split")),
    )
    return record


def load_md_domain(path: str | Path) -> SchedulingDomain:
    """Load one synthetic MD instance and return its shared runtime domain."""
    return load_md_instance(path).generated.domain


def _config_to_dict(config: MDGeneratorConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "task_count": config.task_count,
        "transport_ratio": config.transport_ratio,
        "precedence_density": config.precedence_density,
        "critical_path_length": config.critical_path_length,
        "capacity_slack": config.capacity_slack,
        "speed_ratio": config.speed_ratio,
        "process_robot_count": config.process_robot_count,
        "transport_robot_count": config.transport_robot_count,
        "skill_count": config.skill_count,
    }


def _generated_from_dict(
    config_payload: Any, domain_payload: Any
) -> GeneratedMDInstance:
    if not isinstance(config_payload, Mapping):
        raise ValueError("generation_config must be a JSON object")
    config_fields = (
        "seed",
        "task_count",
        "transport_ratio",
        "precedence_density",
        "critical_path_length",
        "capacity_slack",
        "speed_ratio",
        "process_robot_count",
        "transport_robot_count",
        "skill_count",
    )
    missing = [field for field in config_fields if field not in config_payload]
    if missing:
        raise ValueError(f"generation_config missing fields: {missing}")
    config = MDGeneratorConfig(
        **{field: config_payload[field] for field in config_fields}
    )
    domain = _domain_from_dict(domain_payload)
    if domain.config.loaded_speed_factor != config.speed_ratio:
        raise ValueError("domain loaded_speed_factor must match generation speed_ratio")
    return GeneratedMDInstance(domain=domain, generation_config=config)


def _domain_to_dict(domain: SchedulingDomain) -> dict[str, Any]:
    return {
        "config": {
            "enabled": domain.config.enabled,
            "use_typed_edges": domain.config.use_typed_edges,
            "use_downstream_encoding": domain.config.use_downstream_encoding,
            "use_transport_eta": domain.config.use_transport_eta,
            "use_capacity_features": domain.config.use_capacity_features,
            "loaded_speed_factor": domain.config.loaded_speed_factor,
        },
        "tasks": [_task_to_dict(task) for task in domain.tasks],
        "robots": [_robot_to_dict(robot) for robot in domain.robots],
        "normal_edges": [list(edge) for edge in domain.normal_edges],
        "material_edges": [list(edge) for edge in domain.material_edges],
    }


def _domain_from_dict(payload: Any) -> SchedulingDomain:
    if not isinstance(payload, Mapping):
        raise ValueError("domain must be a JSON object")
    try:
        config_payload = payload["config"]
        task_payload = payload["tasks"]
        robot_payload = payload["robots"]
        normal_edges = payload["normal_edges"]
        material_edges = payload["material_edges"]
    except KeyError as error:
        raise ValueError(f"domain missing field: {error.args[0]}") from error
    if not isinstance(config_payload, Mapping):
        raise ValueError("domain.config must be a JSON object")
    config = MaterialDeliveryConfig(**dict(config_payload))
    tasks = tuple(_task_from_dict(item) for item in _sequence(task_payload, "domain.tasks"))
    robots = tuple(_robot_from_dict(item) for item in _sequence(robot_payload, "domain.robots"))
    return SchedulingDomain.create(
        config=config,
        tasks=tasks,
        robots=robots,
        normal_edges=normal_edges,
        material_edges=material_edges,
    )


def _task_to_dict(task: ProcessTask | TransportTask) -> dict[str, Any]:
    if isinstance(task, ProcessTask):
        return {
            "task_type": TaskType.PROCESS.value,
            "task_id": task.task_id,
            "location": list(task.location),
            "duration": task.duration,
            "requirements": list(task.requirements),
        }
    return {
        "task_type": TaskType.TRANSPORT.value,
        "task_id": task.task_id,
        "pickup_location": list(task.pickup_location),
        "delivery_location": list(task.delivery_location),
        "load": task.load,
        "loading_duration": task.loading_duration,
        "unloading_duration": task.unloading_duration,
    }


def _task_from_dict(payload: Any) -> ProcessTask | TransportTask:
    if not isinstance(payload, Mapping):
        raise ValueError("domain task must be a JSON object")
    task_type = payload.get("task_type")
    if task_type == TaskType.PROCESS.value:
        process_required_fields = (
            "task_id",
            "location",
            "duration",
            "requirements",
        )
        _require_mapping_fields(payload, process_required_fields, "PROCESS task")
        return ProcessTask(
            task_id=_integer(payload["task_id"], "task_id"),
            location=payload["location"],
            duration=payload["duration"],
            requirements=payload["requirements"],
        )
    if task_type == TaskType.TRANSPORT.value:
        transport_required_fields = (
            "task_id",
            "pickup_location",
            "delivery_location",
            "load",
            "loading_duration",
            "unloading_duration",
        )
        _require_mapping_fields(
            payload, transport_required_fields, "TRANSPORT task"
        )
        return TransportTask(
            task_id=_integer(payload["task_id"], "task_id"),
            pickup_location=payload["pickup_location"],
            delivery_location=payload["delivery_location"],
            load=payload["load"],
            loading_duration=payload["loading_duration"],
            unloading_duration=payload["unloading_duration"],
        )
    raise ValueError(f"domain task has unsupported task_type: {task_type!r}")


def _robot_to_dict(robot: ProcessRobot | TransportRobot) -> dict[str, Any]:
    if isinstance(robot, ProcessRobot):
        return {
            "robot_type": RobotType.PROCESS_ROBOT.value,
            "robot_id": robot.robot_id,
            "location": list(robot.location),
            "capabilities": list(robot.capabilities),
            "speed": robot.speed,
        }
    return {
        "robot_type": RobotType.TRANSPORT_ROBOT.value,
        "robot_id": robot.robot_id,
        "location": list(robot.location),
        "transport_capable": robot.transport_capable,
        "capacity": robot.capacity,
        "unloaded_speed": robot.unloaded_speed,
        "loaded_speed": robot.loaded_speed,
    }


def _robot_from_dict(payload: Any) -> ProcessRobot | TransportRobot:
    if not isinstance(payload, Mapping):
        raise ValueError("domain robot must be a JSON object")
    robot_type = payload.get("robot_type")
    if robot_type == RobotType.PROCESS_ROBOT.value:
        _require_mapping_fields(
            payload, ("robot_id", "location", "capabilities", "speed"), "PROCESS_ROBOT"
        )
        return ProcessRobot(
            robot_id=_integer(payload["robot_id"], "robot_id"),
            location=payload["location"],
            capabilities=payload["capabilities"],
            speed=payload["speed"],
        )
    if robot_type == RobotType.TRANSPORT_ROBOT.value:
        _require_mapping_fields(
            payload,
            (
                "robot_id",
                "location",
                "transport_capable",
                "capacity",
                "unloaded_speed",
                "loaded_speed",
            ),
            "TRANSPORT_ROBOT",
        )
        return TransportRobot(
            robot_id=_integer(payload["robot_id"], "robot_id"),
            location=payload["location"],
            transport_capable=payload["transport_capable"],
            capacity=payload["capacity"],
            unloaded_speed=payload["unloaded_speed"],
            loaded_speed=payload["loaded_speed"],
        )
    raise ValueError(f"domain robot has unsupported robot_type: {robot_type!r}")


def _freeze_execution_records(value: Any) -> Mapping[str, Any]:
    frozen = _freeze_json_mapping("execution_records", value)
    if set(frozen) != {"process", "transport"}:
        raise ValueError(
            "execution_records must contain exactly process and transport arrays"
        )
    for name in ("process", "transport"):
        records = frozen[name]
        if not isinstance(records, tuple):
            raise ValueError(f"execution_records.{name} must be a JSON array")
        if any(not isinstance(record, Mapping) for record in records):
            raise ValueError(
                f"execution_records.{name} must contain JSON objects"
            )
    return frozen


def _freeze_json_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    frozen = _freeze_json_value(name, value)
    if not isinstance(frozen, Mapping):
        raise AssertionError("mapping freezer returned a non-mapping")
    return frozen


def _freeze_json_value(name: str, value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _string(key, f"{name} key"): _freeze_json_value(
                    f"{name}.{key}", item
                )
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json_value(f"{name}[{index}]", item)
            for index, item in enumerate(value)
        )
    raise ValueError(f"{name} contains a non-JSON value")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _require_mapping_fields(
    payload: Mapping[str, Any], fields: Sequence[str], name: str
) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"{name} missing fields: {missing}")


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return tuple(value)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _split(value: Any) -> DatasetSplit | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("split must be a string")
    try:
        return DatasetSplit(value)
    except ValueError as error:
        raise ValueError(f"unsupported split: {value!r}") from error
