"""Offline MD expert decision samples produced by schedule replay."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from baselines.md_oracle_types import (
    GurobiOracleResult,
    GurobiOracleStatus,
    OracleAction,
)
from data_generation.md_dataset import MDInstanceRecord
from experiments.protocol import DatasetSplit
from simulation_environment.domain_model import (
    ProcessRobot,
    ProcessTask,
    TransportRobot,
    TransportTask,
)
from simulation_environment.hard_feasibility import TaskStatus
from simulation_environment.md_discrete_simulator import (
    MDDiscreteSimulator,
    RobotActivity,
)


MD_EXPERT_SCHEMA_VERSION = "1.0.0"


class ExpertQuality(str, Enum):
    OPTIMAL = "optimal"
    TIME_LIMITED_FEASIBLE = "time_limited_feasible"
    HEURISTIC = "heuristic"


@dataclass(frozen=True, slots=True)
class TypedTaskGraph:
    task_types: tuple[tuple[int, str], ...]
    normal_edges: tuple[tuple[int, int], ...]
    material_edges: tuple[tuple[int, int], ...]
    downstream_links: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_types": [list(item) for item in self.task_types],
            "normal_edges": [list(item) for item in self.normal_edges],
            "material_edges": [list(item) for item in self.material_edges],
            "downstream_links": [list(item) for item in self.downstream_links],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypedTaskGraph":
        return cls(
            task_types=tuple(
                (_integer(row[0], "task type id"), _string(row[1], "task type"))
                for row in _pairs(payload.get("task_types"), "task_types")
            ),
            normal_edges=_integer_pairs(payload.get("normal_edges"), "normal_edges"),
            material_edges=_integer_pairs(
                payload.get("material_edges"), "material_edges"
            ),
            downstream_links=_integer_pairs(
                payload.get("downstream_links"), "downstream_links"
            ),
        )


@dataclass(frozen=True, slots=True)
class RobotDecisionState:
    robot_id: int
    robot_type: str
    location: tuple[float, float]
    activity: str
    available: bool
    task_id: int | None
    remaining: int
    capabilities: tuple[bool, ...]
    speed: float
    loaded_speed: float | None
    capacity: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "robot_type": self.robot_type,
            "location": list(self.location),
            "activity": self.activity,
            "available": self.available,
            "task_id": self.task_id,
            "remaining": self.remaining,
            "capabilities": list(self.capabilities),
            "speed": self.speed,
            "loaded_speed": self.loaded_speed,
            "capacity": self.capacity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RobotDecisionState":
        return cls(
            robot_id=_integer(payload.get("robot_id"), "robot_id"),
            robot_type=_string(payload.get("robot_type"), "robot_type"),
            location=_location(payload.get("location"), "robot location"),
            activity=_string(payload.get("activity"), "activity"),
            available=_boolean(payload.get("available"), "available"),
            task_id=_optional_integer(payload.get("task_id"), "task_id"),
            remaining=_integer(payload.get("remaining"), "remaining"),
            capabilities=_booleans(payload.get("capabilities"), "capabilities"),
            speed=_number(payload.get("speed"), "speed"),
            loaded_speed=_optional_number(
                payload.get("loaded_speed"), "loaded_speed"
            ),
            capacity=_optional_number(payload.get("capacity"), "capacity"),
        )


@dataclass(frozen=True, slots=True)
class TaskDecisionState:
    task_id: int
    task_type: str
    status: str
    ready: bool
    material_ready: bool
    location: tuple[float, float]
    pickup_location: tuple[float, float] | None
    delivery_location: tuple[float, float] | None
    duration: float
    load: float
    loading_duration: float
    unloading_duration: float
    requirements: tuple[bool, ...]
    normal_predecessors: tuple[int, ...]
    material_predecessor: int | None
    downstream_process_task_id: int | None
    transport_phase: str | None
    assigned_robot_ids: tuple[int, ...]
    phase_remaining: int
    started_at: int | None
    completed_at: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "ready": self.ready,
            "material_ready": self.material_ready,
            "location": list(self.location),
            "pickup_location": (
                list(self.pickup_location) if self.pickup_location else None
            ),
            "delivery_location": (
                list(self.delivery_location) if self.delivery_location else None
            ),
            "duration": self.duration,
            "load": self.load,
            "loading_duration": self.loading_duration,
            "unloading_duration": self.unloading_duration,
            "requirements": list(self.requirements),
            "normal_predecessors": list(self.normal_predecessors),
            "material_predecessor": self.material_predecessor,
            "downstream_process_task_id": self.downstream_process_task_id,
            "transport_phase": self.transport_phase,
            "assigned_robot_ids": list(self.assigned_robot_ids),
            "phase_remaining": self.phase_remaining,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskDecisionState":
        pickup = payload.get("pickup_location")
        delivery = payload.get("delivery_location")
        return cls(
            task_id=_integer(payload.get("task_id"), "task_id"),
            task_type=_string(payload.get("task_type"), "task_type"),
            status=_string(payload.get("status"), "status"),
            ready=_boolean(payload.get("ready"), "ready"),
            material_ready=_boolean(
                payload.get("material_ready"), "material_ready"
            ),
            location=_location(payload.get("location"), "task location"),
            pickup_location=(
                None if pickup is None else _location(pickup, "pickup_location")
            ),
            delivery_location=(
                None
                if delivery is None
                else _location(delivery, "delivery_location")
            ),
            duration=_number(payload.get("duration"), "duration"),
            load=_number(payload.get("load"), "load"),
            loading_duration=_number(
                payload.get("loading_duration"), "loading_duration"
            ),
            unloading_duration=_number(
                payload.get("unloading_duration"), "unloading_duration"
            ),
            requirements=_booleans(payload.get("requirements"), "requirements"),
            normal_predecessors=tuple(
                _integer(value, "normal predecessor")
                for value in _sequence(
                    payload.get("normal_predecessors"), "normal_predecessors"
                )
            ),
            material_predecessor=_optional_integer(
                payload.get("material_predecessor"), "material_predecessor"
            ),
            downstream_process_task_id=_optional_integer(
                payload.get("downstream_process_task_id"),
                "downstream_process_task_id",
            ),
            transport_phase=(
                None
                if payload.get("transport_phase") is None
                else _string(payload.get("transport_phase"), "transport_phase")
            ),
            assigned_robot_ids=tuple(
                _integer(value, "assigned robot id")
                for value in _sequence(
                    payload.get("assigned_robot_ids"), "assigned_robot_ids"
                )
            ),
            phase_remaining=_integer(
                payload.get("phase_remaining"), "phase_remaining"
            ),
            started_at=_optional_integer(payload.get("started_at"), "started_at"),
            completed_at=_optional_integer(
                payload.get("completed_at"), "completed_at"
            ),
        )


@dataclass(frozen=True, slots=True)
class MDDecisionSample:
    sample_id: str
    instance_id: str
    task_group_id: str
    seed: int
    split: DatasetSplit
    quality: ExpertQuality
    decision_index: int
    decision_time: int
    robot_ids: tuple[int, ...]
    task_ids: tuple[int, ...]
    typed_graph: TypedTaskGraph
    robots: tuple[RobotDecisionState, ...]
    tasks: tuple[TaskDecisionState, ...]
    hard_feasibility_mask: tuple[tuple[bool, ...], ...]
    capacity_feasibility: tuple[tuple[bool, ...], ...]
    expert_actions: tuple[tuple[int, int], ...]
    expert_assignment: tuple[tuple[bool, ...], ...]
    expert_reward: tuple[tuple[float, ...], ...]
    post_assignment_robots: tuple[RobotDecisionState, ...]
    post_assignment_tasks: tuple[TaskDecisionState, ...]
    next_decision_time: int | None
    remaining_makespan: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "instance_id": self.instance_id,
            "task_group_id": self.task_group_id,
            "seed": self.seed,
            "split": self.split.value,
            "quality": self.quality.value,
            "decision_index": self.decision_index,
            "decision_time": self.decision_time,
            "robot_ids": list(self.robot_ids),
            "task_ids": list(self.task_ids),
            "typed_graph": self.typed_graph.to_dict(),
            "execution_state": {
                "robots": [state.to_dict() for state in self.robots],
                "tasks": [state.to_dict() for state in self.tasks],
            },
            "hard_feasibility_mask": _matrix_to_lists(
                self.hard_feasibility_mask
            ),
            "capacity_feasibility": _matrix_to_lists(
                self.capacity_feasibility
            ),
            "expert_action": {
                "assignments": [list(item) for item in self.expert_actions],
                "matrix": _matrix_to_lists(self.expert_assignment),
                "reward": _matrix_to_lists(self.expert_reward),
            },
            "transition": {
                "post_assignment_robots": [
                    state.to_dict() for state in self.post_assignment_robots
                ],
                "post_assignment_tasks": [
                    state.to_dict() for state in self.post_assignment_tasks
                ],
                "next_decision_time": self.next_decision_time,
            },
            "remaining_makespan": self.remaining_makespan,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MDDecisionSample":
        execution = _mapping(payload.get("execution_state"), "execution_state")
        action = _mapping(payload.get("expert_action"), "expert_action")
        transition = _mapping(payload.get("transition"), "transition")
        robot_ids = tuple(
            _integer(value, "robot id")
            for value in _sequence(payload.get("robot_ids"), "robot_ids")
        )
        task_ids = tuple(
            _integer(value, "task id")
            for value in _sequence(payload.get("task_ids"), "task_ids")
        )
        shape = (len(robot_ids), len(task_ids))
        return cls(
            sample_id=_string(payload.get("sample_id"), "sample_id"),
            instance_id=_string(payload.get("instance_id"), "instance_id"),
            task_group_id=_string(
                payload.get("task_group_id"), "task_group_id"
            ),
            seed=_integer(payload.get("seed"), "seed"),
            split=DatasetSplit(_string(payload.get("split"), "split")),
            quality=ExpertQuality(_string(payload.get("quality"), "quality")),
            decision_index=_integer(
                payload.get("decision_index"), "decision_index"
            ),
            decision_time=_integer(payload.get("decision_time"), "decision_time"),
            robot_ids=robot_ids,
            task_ids=task_ids,
            typed_graph=TypedTaskGraph.from_dict(
                _mapping(payload.get("typed_graph"), "typed_graph")
            ),
            robots=tuple(
                RobotDecisionState.from_dict(_mapping(item, "robot state"))
                for item in _sequence(execution.get("robots"), "robots")
            ),
            tasks=tuple(
                TaskDecisionState.from_dict(_mapping(item, "task state"))
                for item in _sequence(execution.get("tasks"), "tasks")
            ),
            hard_feasibility_mask=_boolean_matrix(
                payload.get("hard_feasibility_mask"),
                "hard_feasibility_mask",
                shape,
            ),
            capacity_feasibility=_boolean_matrix(
                payload.get("capacity_feasibility"),
                "capacity_feasibility",
                shape,
            ),
            expert_actions=_integer_pairs(
                action.get("assignments"), "expert assignments"
            ),
            expert_assignment=_boolean_matrix(
                action.get("matrix"), "expert assignment matrix", shape
            ),
            expert_reward=_number_matrix(
                action.get("reward"), "expert reward", shape
            ),
            post_assignment_robots=tuple(
                RobotDecisionState.from_dict(_mapping(item, "post robot state"))
                for item in _sequence(
                    transition.get("post_assignment_robots"),
                    "post_assignment_robots",
                )
            ),
            post_assignment_tasks=tuple(
                TaskDecisionState.from_dict(_mapping(item, "post task state"))
                for item in _sequence(
                    transition.get("post_assignment_tasks"),
                    "post_assignment_tasks",
                )
            ),
            next_decision_time=_optional_integer(
                transition.get("next_decision_time"), "next_decision_time"
            ),
            remaining_makespan=_number(
                payload.get("remaining_makespan"), "remaining_makespan"
            ),
        )


@dataclass(frozen=True, slots=True)
class MDExpertRecord:
    instance_id: str
    task_group_id: str
    seed: int
    split: DatasetSplit
    quality: ExpertQuality
    samples: tuple[MDDecisionSample, ...]
    terminal_makespan: float
    terminal_result: Mapping[str, Any]
    expert_metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MD_EXPERT_SCHEMA_VERSION,
            "instance_id": self.instance_id,
            "task_group_id": self.task_group_id,
            "seed": self.seed,
            "split": self.split.value,
            "expert_quality": self.quality.value,
            "expert_metadata": _plain_json(self.expert_metadata),
            "decision_samples": [sample.to_dict() for sample in self.samples],
            "terminal_makespan": self.terminal_makespan,
            "terminal_result": _plain_json(self.terminal_result),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MDExpertRecord":
        if payload.get("schema_version") != MD_EXPERT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported MD expert schema_version: {payload.get('schema_version')!r}"
            )
        record = cls(
            instance_id=_string(payload.get("instance_id"), "instance_id"),
            task_group_id=_string(
                payload.get("task_group_id"), "task_group_id"
            ),
            seed=_integer(payload.get("seed"), "seed"),
            split=DatasetSplit(_string(payload.get("split"), "split")),
            quality=ExpertQuality(
                _string(payload.get("expert_quality"), "expert_quality")
            ),
            samples=tuple(
                MDDecisionSample.from_dict(_mapping(item, "decision sample"))
                for item in _sequence(
                    payload.get("decision_samples"), "decision_samples"
                )
            ),
            terminal_makespan=_number(
                payload.get("terminal_makespan"), "terminal_makespan"
            ),
            terminal_result=_mapping(
                payload.get("terminal_result"), "terminal_result"
            ),
            expert_metadata=_mapping(
                payload.get("expert_metadata"), "expert_metadata"
            ),
        )
        for sample in record.samples:
            if (
                sample.instance_id != record.instance_id
                or sample.task_group_id != record.task_group_id
                or sample.seed != record.seed
                or sample.split is not record.split
                or sample.quality is not record.quality
            ):
                raise ValueError("decision sample metadata must match its expert record")
        return record


class MDExpertDatasetLoader:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"MD expert dataset root must be a directory: {self.root}")

    def load_records(
        self,
        *,
        split: DatasetSplit | None = None,
        quality: ExpertQuality | None = None,
    ) -> tuple[MDExpertRecord, ...]:
        records: list[MDExpertRecord] = []
        seen: set[str] = set()
        for path in sorted(self.root.glob("*.json")):
            record = load_md_expert_record(path)
            if record.instance_id in seen:
                raise ValueError(f"duplicate MD expert instance_id: {record.instance_id}")
            seen.add(record.instance_id)
            if split is not None and record.split is not split:
                continue
            if quality is not None and record.quality is not quality:
                continue
            records.append(record)
        return tuple(records)

    def load_samples(
        self,
        *,
        split: DatasetSplit | None = None,
        quality: ExpertQuality | None = None,
    ) -> tuple[MDDecisionSample, ...]:
        return tuple(
            sample
            for record in self.load_records(split=split, quality=quality)
            for sample in record.samples
        )


@dataclass(slots=True)
class _SampleDraft:
    decision_time: int
    robots: tuple[RobotDecisionState, ...]
    tasks: tuple[TaskDecisionState, ...]
    hard_mask: tuple[tuple[bool, ...], ...]
    capacity_mask: tuple[tuple[bool, ...], ...]
    actions: tuple[tuple[int, int], ...]
    assignment: tuple[tuple[bool, ...], ...]
    post_robots: tuple[RobotDecisionState, ...]
    post_tasks: tuple[TaskDecisionState, ...]


def generate_md_expert_record(
    instance: MDInstanceRecord,
    actions: Sequence[OracleAction],
    *,
    quality: ExpertQuality,
    exit_location: tuple[float, float] = (0.0, 0.0),
    max_steps: int,
    expected_makespan: float | None = None,
    expert_metadata: Mapping[str, Any] | None = None,
) -> MDExpertRecord:
    """Replay fixed expert actions without invoking a solver."""

    if not isinstance(instance, MDInstanceRecord):
        raise TypeError("instance must be an MDInstanceRecord")
    if not instance.generated.domain.config.enabled:
        raise ValueError("expert generation requires an enabled MD instance")
    if not isinstance(quality, ExpertQuality):
        raise TypeError("quality must be an ExpertQuality")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    ordered = tuple(sorted(actions, key=lambda action: action.order))
    if not ordered or any(not isinstance(action, OracleAction) for action in ordered):
        raise ValueError("actions must contain at least one OracleAction")
    if len({action.order for action in ordered}) != len(ordered):
        raise ValueError("OracleAction.order values must be unique")

    simulator = MDDiscreteSimulator(
        instance.generated.domain, exit_location=exit_location
    )
    robot_ids = tuple(sorted(simulator.robot_states))
    task_ids = tuple(sorted(simulator.task_states))
    typed_graph = _typed_graph(instance)
    pending = list(ordered)
    drafts: list[_SampleDraft] = []
    steps = 0

    while not simulator.done and steps < max_steps:
        before_robots, before_tasks = _snapshot(simulator)
        hard_mask = _hard_mask(simulator, robot_ids, task_ids)
        capacity_mask = _capacity_mask(simulator, robot_ids, task_ids)
        assigned: list[tuple[int, int]] = []
        for action in tuple(pending):
            if action.planned_assignment > simulator.time:
                continue
            if not simulator.assignment_feasibility(
                robot_id=action.robot_id, task_id=action.task_id
            ).is_feasible:
                continue
            simulator.assign(robot_id=action.robot_id, task_id=action.task_id)
            assigned.append((action.robot_id, action.task_id))
            pending.remove(action)

        if assigned:
            post_robots, post_tasks = _snapshot(simulator)
            assignment = _assignment_matrix(
                robot_ids, task_ids, tuple(assigned)
            )
            drafts.append(
                _SampleDraft(
                    decision_time=simulator.time,
                    robots=before_robots,
                    tasks=before_tasks,
                    hard_mask=hard_mask,
                    capacity_mask=capacity_mask,
                    actions=tuple(assigned),
                    assignment=assignment,
                    post_robots=post_robots,
                    post_tasks=post_tasks,
                )
            )

        if simulator.done:
            break
        if not simulator.has_advancing_work and not any(
            action.planned_assignment > simulator.time for action in pending
        ):
            raise RuntimeError(
                "expert replay deadlocked before consuming every expert action"
            )
        simulator.step()
        steps += 1

    if not simulator.done:
        raise RuntimeError("expert replay exceeded max_steps")
    if pending:
        raise RuntimeError("expert replay terminated with pending expert actions")

    terminal = simulator.build_experiment_result(
        run_id=f"expert-{instance.instance_id}-{instance.seed}",
        method="offline_md_expert_replay",
        instance_id=instance.instance_id,
        seed=instance.seed,
        split=instance.split,
        metadata={"expert_quality": quality.value},
    )
    assert terminal.makespan is not None
    if expected_makespan is not None and not math.isclose(
        terminal.makespan, expected_makespan, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(
            "expert replay makespan does not match the source schedule: "
            f"{terminal.makespan} != {expected_makespan}"
        )

    samples = tuple(
        MDDecisionSample(
            sample_id=f"{instance.instance_id}:decision:{index}",
            instance_id=instance.instance_id,
            task_group_id=instance.task_group_id,
            seed=instance.seed,
            split=instance.split,
            quality=quality,
            decision_index=index,
            decision_time=draft.decision_time,
            robot_ids=robot_ids,
            task_ids=task_ids,
            typed_graph=typed_graph,
            robots=draft.robots,
            tasks=draft.tasks,
            hard_feasibility_mask=draft.hard_mask,
            capacity_feasibility=draft.capacity_mask,
            expert_actions=draft.actions,
            expert_assignment=draft.assignment,
            expert_reward=tuple(
                tuple(float(value) for value in row) for row in draft.assignment
            ),
            post_assignment_robots=draft.post_robots,
            post_assignment_tasks=draft.post_tasks,
            next_decision_time=(
                drafts[index + 1].decision_time
                if index + 1 < len(drafts)
                else None
            ),
            remaining_makespan=terminal.makespan - draft.decision_time,
        )
        for index, draft in enumerate(drafts)
    )
    metadata = {
        "source": "offline_schedule_replay",
        "action_count": len(ordered),
        "actions": [
            {
                "order": action.order,
                "robot_id": action.robot_id,
                "task_id": action.task_id,
                "planned_assignment": action.planned_assignment,
                "planned_start": action.planned_start,
            }
            for action in ordered
        ],
        **({} if expert_metadata is None else dict(expert_metadata)),
    }
    return MDExpertRecord(
        instance_id=instance.instance_id,
        task_group_id=instance.task_group_id,
        seed=instance.seed,
        split=instance.split,
        quality=quality,
        samples=samples,
        terminal_makespan=terminal.makespan,
        terminal_result=terminal.to_dict(),
        expert_metadata=_plain_json(metadata),
    )


def generate_md_expert_record_from_oracle(
    instance: MDInstanceRecord,
    oracle: GurobiOracleResult,
    *,
    exit_location: tuple[float, float] = (0.0, 0.0),
    max_steps: int,
) -> MDExpertRecord:
    if oracle.status is GurobiOracleStatus.OPTIMAL:
        quality = ExpertQuality.OPTIMAL
    elif oracle.status is GurobiOracleStatus.FEASIBLE and oracle.feasible:
        quality = ExpertQuality.TIME_LIMITED_FEASIBLE
    else:
        raise ValueError("oracle result must contain an optimal or feasible schedule")
    if oracle.objective is None:
        raise ValueError("replayable oracle result requires an objective")
    return generate_md_expert_record(
        instance,
        oracle.action_order,
        quality=quality,
        exit_location=exit_location,
        max_steps=max_steps,
        expected_makespan=oracle.objective,
        expert_metadata={
            "source": "gurobi_md_oracle",
            "oracle_schedule": [
                {
                    "robot_id": entry.robot_id,
                    "task_id": entry.task_id,
                    "start": entry.start,
                    "completion": entry.completion,
                }
                for entry in oracle.schedule
            ],
            "oracle_status": oracle.status.value,
            "solve_time_seconds": oracle.solve_time_seconds,
            "timeout": oracle.timeout,
            "optimality_gap": oracle.optimality_gap,
            "oracle_message": oracle.message,
        },
    )


def save_md_expert_record(path: str | Path, record: MDExpertRecord) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record.to_dict(), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_md_expert_record(path: str | Path) -> MDExpertRecord:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read MD expert JSON: {source}") from error
    return MDExpertRecord.from_dict(_mapping(payload, "MD expert document"))


def _typed_graph(instance: MDInstanceRecord) -> TypedTaskGraph:
    domain = instance.generated.domain
    return TypedTaskGraph(
        task_types=tuple(
            (task.task_id, task.task_type.value)
            for task in sorted(domain.tasks, key=lambda item: item.task_id)
        ),
        normal_edges=domain.normal_edges,
        material_edges=domain.material_edges,
        downstream_links=tuple(domain.material_edges),
    )


def _snapshot(
    simulator: MDDiscreteSimulator,
) -> tuple[tuple[RobotDecisionState, ...], tuple[TaskDecisionState, ...]]:
    robots: list[RobotDecisionState] = []
    for robot in sorted(simulator.domain.robots, key=lambda item: item.robot_id):
        runtime = simulator.robot_state(robot.robot_id)
        if isinstance(robot, ProcessRobot):
            capabilities = robot.capabilities
            speed = robot.speed
            loaded_speed = None
            capacity = None
        else:
            capabilities = ()
            speed = robot.unloaded_speed
            loaded_speed = robot.loaded_speed
            capacity = robot.capacity
        robots.append(
            RobotDecisionState(
                robot_id=robot.robot_id,
                robot_type=robot.robot_type.value,
                location=runtime.location,
                activity=runtime.activity.value,
                available=runtime.activity is RobotActivity.AVAILABLE,
                task_id=runtime.task_id,
                remaining=runtime.remaining,
                capabilities=capabilities,
                speed=speed,
                loaded_speed=loaded_speed,
                capacity=capacity,
            )
        )

    tasks: list[TaskDecisionState] = []
    for task in sorted(simulator.domain.tasks, key=lambda item: item.task_id):
        runtime = simulator.task_state(task.task_id)
        if isinstance(task, ProcessTask):
            material_ready = (
                task.material_predecessor is None
                or simulator.task_state(task.material_predecessor).status
                is TaskStatus.COMPLETE
            )
            location = task.location
            pickup = None
            delivery = None
            duration = task.duration
            load = loading = unloading = 0.0
            requirements = task.requirements
            normal_predecessors = task.normal_predecessors
            material_predecessor = task.material_predecessor
            downstream = None
        else:
            material_ready = True
            location = task.pickup_location
            pickup = task.pickup_location
            delivery = task.delivery_location
            duration = 0.0
            load = task.load
            loading = task.loading_duration
            unloading = task.unloading_duration
            requirements = ()
            normal_predecessors = ()
            material_predecessor = None
            downstream = task.downstream_process_task_id
        tasks.append(
            TaskDecisionState(
                task_id=task.task_id,
                task_type=task.task_type.value,
                status=runtime.status.value,
                ready=simulator.is_task_ready(task.task_id),
                material_ready=material_ready,
                location=location,
                pickup_location=pickup,
                delivery_location=delivery,
                duration=duration,
                load=load,
                loading_duration=loading,
                unloading_duration=unloading,
                requirements=requirements,
                normal_predecessors=normal_predecessors,
                material_predecessor=material_predecessor,
                downstream_process_task_id=downstream,
                transport_phase=(
                    runtime.transport_phase.value
                    if runtime.transport_phase is not None
                    else None
                ),
                assigned_robot_ids=tuple(sorted(runtime.assigned_robot_ids)),
                phase_remaining=runtime.phase_remaining,
                started_at=runtime.started_at,
                completed_at=runtime.completed_at,
            )
        )
    return tuple(robots), tuple(tasks)


def snapshot_md_simulator(
    simulator: MDDiscreteSimulator,
) -> tuple[tuple[RobotDecisionState, ...], tuple[TaskDecisionState, ...]]:
    """Return the canonical present-state snapshot used by offline features."""

    if not isinstance(simulator, MDDiscreteSimulator):
        raise TypeError("simulator must be an MDDiscreteSimulator")
    return _snapshot(simulator)


def _hard_mask(
    simulator: MDDiscreteSimulator,
    robot_ids: tuple[int, ...],
    task_ids: tuple[int, ...],
) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        tuple(
            simulator.assignment_feasibility(
                robot_id=robot_id, task_id=task_id
            ).is_feasible
            for task_id in task_ids
        )
        for robot_id in robot_ids
    )


def _capacity_mask(
    simulator: MDDiscreteSimulator,
    robot_ids: tuple[int, ...],
    task_ids: tuple[int, ...],
) -> tuple[tuple[bool, ...], ...]:
    robots = {robot.robot_id: robot for robot in simulator.domain.robots}
    tasks = {task.task_id: task for task in simulator.domain.tasks}
    return tuple(
        tuple(
            isinstance(robots[robot_id], TransportRobot)
            and isinstance(tasks[task_id], TransportTask)
            and robots[robot_id].transport_capable
            and robots[robot_id].capacity >= tasks[task_id].load
            for task_id in task_ids
        )
        for robot_id in robot_ids
    )


def _assignment_matrix(
    robot_ids: tuple[int, ...],
    task_ids: tuple[int, ...],
    actions: tuple[tuple[int, int], ...],
) -> tuple[tuple[bool, ...], ...]:
    selected = set(actions)
    return tuple(
        tuple((robot_id, task_id) in selected for task_id in task_ids)
        for robot_id in robot_ids
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a JSON array")
    try:
        return tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a JSON array") from error


def _pairs(value: Any, name: str) -> tuple[tuple[Any, Any], ...]:
    pairs = []
    for row in _sequence(value, name):
        values = _sequence(row, name)
        if len(values) != 2:
            raise ValueError(f"{name} rows must contain two values")
        pairs.append((values[0], values[1]))
    return tuple(pairs)


def _integer_pairs(value: Any, name: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (_integer(left, name), _integer(right, name))
        for left, right in _pairs(value, name)
    )


def _boolean_matrix(
    value: Any, name: str, shape: tuple[int, int]
) -> tuple[tuple[bool, ...], ...]:
    rows = tuple(_booleans(row, name) for row in _sequence(value, name))
    _validate_shape(rows, name, shape)
    return rows


def _number_matrix(
    value: Any, name: str, shape: tuple[int, int]
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(
        tuple(_number(item, name) for item in _sequence(row, name))
        for row in _sequence(value, name)
    )
    _validate_shape(rows, name, shape)
    return rows


def _validate_shape(rows: Sequence[Sequence[Any]], name: str, shape: tuple[int, int]) -> None:
    if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
        raise ValueError(f"{name} must have shape {shape}")


def _matrix_to_lists(value: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [list(row) for row in value]


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_integer(value: Any, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")
    return float(value)


def _optional_number(value: Any, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _booleans(value: Any, name: str) -> tuple[bool, ...]:
    result = _sequence(value, name)
    if any(not isinstance(item, bool) for item in result):
        raise ValueError(f"{name} must contain booleans")
    return result


def _location(value: Any, name: str) -> tuple[float, float]:
    coordinates = _sequence(value, name)
    if len(coordinates) != 2:
        raise ValueError(f"{name} must contain two coordinates")
    return (
        _signed_number(coordinates[0], name),
        _signed_number(coordinates[1], name),
    )


def _signed_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _plain_json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


__all__ = [
    "ExpertQuality",
    "MDDecisionSample",
    "MDExpertDatasetLoader",
    "MDExpertRecord",
    "MD_EXPERT_SCHEMA_VERSION",
    "RobotDecisionState",
    "TaskDecisionState",
    "TypedTaskGraph",
    "generate_md_expert_record",
    "generate_md_expert_record_from_oracle",
    "load_md_expert_record",
    "save_md_expert_record",
    "snapshot_md_simulator",
]
