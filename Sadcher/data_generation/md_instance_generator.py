"""Deterministic synthetic instance generation for MD-SADCHER experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)


GRID_LIMIT = 100
MIN_PROCESS_DURATION = 5
MAX_PROCESS_DURATION = 20
MIN_TRANSPORT_LOAD = 1
MAX_TRANSPORT_LOAD = 10
MIN_SERVICE_DURATION = 1
MAX_SERVICE_DURATION = 4
TRANSPORT_UNLOADED_SPEED = 2.0


@dataclass(frozen=True, slots=True)
class MDGeneratorConfig:
    """Controls for one reproducible synthetic runtime instance.

    task_count includes process and transport tasks. transport_ratio is
    converted to a count with floor(task_count * transport_ratio).
    precedence_density is the fraction of admissible edges between the
    levels used to enforce the exact process-task critical path length.
    capacity_slack sets every transport robot capacity relative to the
    maximum generated load. speed_ratio is loaded speed divided by
    unloaded speed.
    """

    seed: int
    task_count: int = 12
    transport_ratio: float = 0.25
    precedence_density: float = 0.25
    critical_path_length: int = 3
    capacity_slack: float = 0.2
    speed_ratio: float = 0.8
    process_robot_count: int = 3
    transport_robot_count: int = 2
    skill_count: int = 3
    scarce_skill_count: int = 0
    material_downstream_stratified: bool = False
    process_duration_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _nonnegative_integer(self.seed, "seed")
        _positive_integer(self.task_count, "task_count")
        _positive_integer(self.process_robot_count, "process_robot_count")
        _nonnegative_integer(
            self.transport_robot_count, "transport_robot_count"
        )
        _positive_integer(self.skill_count, "skill_count")
        _positive_integer(self.critical_path_length, "critical_path_length")

        object.__setattr__(
            self,
            "transport_ratio",
            _bounded_ratio(self.transport_ratio, "transport_ratio", maximum=0.5),
        )
        object.__setattr__(
            self,
            "precedence_density",
            _bounded_ratio(self.precedence_density, "precedence_density"),
        )
        object.__setattr__(
            self,
            "capacity_slack",
            _nonnegative_number(self.capacity_slack, "capacity_slack"),
        )
        object.__setattr__(
            self,
            "speed_ratio",
            _bounded_ratio(
                self.speed_ratio,
                "speed_ratio",
                minimum_exclusive=True,
            ),
        )

        if self.critical_path_length > self.process_task_count:
            raise ValueError(
                "critical_path_length cannot exceed process_task_count"
            )
        if self.transport_task_count and not self.transport_robot_count:
            raise ValueError(
                "transport tasks require at least one transport robot"
            )
        _nonnegative_integer(self.scarce_skill_count, "scarce_skill_count")
        if self.scarce_skill_count > self.skill_count:
            raise ValueError(
                "scarce_skill_count cannot exceed skill_count"
            )
        if self.process_duration_range is not None:
            lo, hi = self.process_duration_range
            if not isinstance(lo, int) or not isinstance(hi, int):
                raise ValueError("process_duration_range entries must be int")
            if lo < 1 or hi < lo:
                raise ValueError(
                    "process_duration_range must satisfy 1 <= lo <= hi"
                )

        candidate_count = self.precedence_candidate_count
        required_count = self.critical_path_length - 1
        if candidate_count == 0 and self.precedence_density:
            raise ValueError(
                "precedence_density must be 0 when critical_path_length is 1"
            )
        if self.normal_edge_count < required_count:
            raise ValueError(
                "precedence_density is too low for critical_path_length"
            )

    @property
    def transport_task_count(self) -> int:
        return int(self.task_count * self.transport_ratio)

    @property
    def process_task_count(self) -> int:
        return self.task_count - self.transport_task_count

    @property
    def precedence_candidate_count(self) -> int:
        level_counts = _level_counts(
            self.process_task_count, self.critical_path_length
        )
        return sum(
            source_count * target_count
            for index, source_count in enumerate(level_counts)
            for target_count in level_counts[index + 1 :]
        )

    @property
    def normal_edge_count(self) -> int:
        return _round_half_up(
            self.precedence_density * self.precedence_candidate_count
        )


@dataclass(frozen=True, slots=True)
class GeneratedMDInstance:
    domain: SchedulingDomain
    generation_config: MDGeneratorConfig


def generate_md_instance(config: MDGeneratorConfig) -> GeneratedMDInstance:
    """Generate one deterministic, feasible-by-construction MD runtime domain."""

    if not isinstance(config, MDGeneratorConfig):
        raise TypeError("config must be an MDGeneratorConfig")

    generator = random.Random(config.seed)
    process_ids = list(range(1, config.process_task_count + 1))
    transport_ids = list(
        range(config.process_task_count + 1, config.task_count + 1)
    )

    normal_edges, level_by_id = _generate_normal_edges(process_ids, config, generator)
    process_locations = {
        task_id: _random_location(generator) for task_id in process_ids
    }
    process_requirements = _random_nonempty_boolean_rows(
        config.process_task_count, config.skill_count, generator
    )

    if config.material_downstream_stratified and config.transport_task_count:
        downstream_ids = _stratified_downstream_sample(
            process_ids,
            level_by_id,
            config.critical_path_length,
            config.transport_task_count,
            generator,
        )
    else:
        downstream_ids = generator.sample(
            process_ids, config.transport_task_count
        )
    material_edges = tuple(zip(transport_ids, downstream_ids, strict=True))

    if config.process_duration_range is not None:
        duration_lo, duration_hi = config.process_duration_range
    else:
        duration_lo, duration_hi = MIN_PROCESS_DURATION, MAX_PROCESS_DURATION

    process_tasks = tuple(
        ProcessTask(
            task_id=task_id,
            location=process_locations[task_id],
            duration=generator.randint(duration_lo, duration_hi),
            requirements=process_requirements[index],
        )
        for index, task_id in enumerate(process_ids)
    )

    transport_loads = [
        float(generator.randint(MIN_TRANSPORT_LOAD, MAX_TRANSPORT_LOAD))
        for _ in transport_ids
    ]
    transport_tasks = tuple(
        TransportTask(
            task_id=task_id,
            pickup_location=_pickup_location(
                generator, process_locations[downstream_id]
            ),
            delivery_location=process_locations[downstream_id],
            load=transport_loads[index],
            loading_duration=generator.randint(
                MIN_SERVICE_DURATION, MAX_SERVICE_DURATION
            ),
            unloading_duration=generator.randint(
                MIN_SERVICE_DURATION, MAX_SERVICE_DURATION
            ),
        )
        for index, (task_id, downstream_id) in enumerate(material_edges)
    )

    process_capabilities = _random_covering_boolean_rows(
        config.process_robot_count, config.skill_count, generator
    )
    if config.scarce_skill_count > 0:
        process_capabilities = _apply_scarce_skills(
            process_capabilities,
            config.scarce_skill_count,
            generator,
        )
    depot = _random_location(generator)
    process_robots = tuple(
        ProcessRobot(
            robot_id=robot_id,
            location=depot,
            capabilities=process_capabilities[robot_id],
            home_location=depot,
        )
        for robot_id in range(config.process_robot_count)
    )

    reference_load = max(transport_loads, default=1.0)
    transport_capacity = reference_load * (1.0 + config.capacity_slack)
    transport_robots = tuple(
        TransportRobot(
            robot_id=config.process_robot_count + index,
            location=_random_location(generator),
            transport_capable=True,
            capacity=transport_capacity,
            unloaded_speed=TRANSPORT_UNLOADED_SPEED,
        )
        for index in range(config.transport_robot_count)
    )

    domain = SchedulingDomain.create(
        config=MaterialDeliveryConfig(
            enabled=True,
            loaded_speed_factor=config.speed_ratio,
        ),
        tasks=process_tasks + transport_tasks,
        robots=process_robots + transport_robots,
        normal_edges=normal_edges,
        material_edges=material_edges,
    )
    return GeneratedMDInstance(domain=domain, generation_config=config)


def _generate_normal_edges(
    process_ids: list[int],
    config: MDGeneratorConfig,
    generator: random.Random,
) -> tuple[tuple[tuple[int, int], ...], dict[int, int]]:
    ordered_ids = process_ids.copy()
    generator.shuffle(ordered_ids)
    level_by_id = {
        task_id: index % config.critical_path_length
        for index, task_id in enumerate(ordered_ids)
    }
    anchors = ordered_ids[: config.critical_path_length]
    required_edges = tuple(
        (anchors[index], anchors[index + 1])
        for index in range(len(anchors) - 1)
    )
    required_set = set(required_edges)

    optional_edges = [
        (source_id, target_id)
        for source_id in process_ids
        for target_id in process_ids
        if level_by_id[source_id] < level_by_id[target_id]
        and (source_id, target_id) not in required_set
    ]
    generator.shuffle(optional_edges)
    optional_count = config.normal_edge_count - len(required_edges)
    selected_edges = required_edges + tuple(optional_edges[:optional_count])
    return tuple(sorted(selected_edges)), level_by_id


def _stratified_downstream_sample(
    process_ids: list[int],
    level_by_id: dict[int, int],
    critical_path_length: int,
    transport_task_count: int,
    generator: random.Random,
) -> list[int]:
    """Sample downstream process ids for transports split between upstream / downstream halves."""
    threshold = critical_path_length / 2.0
    upper = [tid for tid in process_ids if level_by_id[tid] < threshold]
    lower = [tid for tid in process_ids if level_by_id[tid] >= threshold]
    upper_quota = transport_task_count // 2
    lower_quota = transport_task_count - upper_quota
    if len(upper) < upper_quota:
        overflow = upper_quota - len(upper)
        upper_quota -= overflow
        lower_quota += overflow
    if len(lower) < lower_quota:
        overflow = lower_quota - len(lower)
        lower_quota -= overflow
        upper_quota += overflow
    picked_upper = generator.sample(upper, upper_quota) if upper_quota else []
    picked_lower = generator.sample(lower, lower_quota) if lower_quota else []
    combined = picked_upper + picked_lower
    generator.shuffle(combined)
    return combined


def _apply_scarce_skills(
    capabilities: tuple[tuple[bool, ...], ...],
    scarce_skill_count: int,
    generator: random.Random,
) -> tuple[tuple[bool, ...], ...]:
    """Force the last k skill columns to be owned by exactly one robot each."""
    row_count = len(capabilities)
    column_count = len(capabilities[0]) if capabilities else 0
    scarce_start = column_count - scarce_skill_count
    mutable = [list(row) for row in capabilities]
    for column in range(scarce_start, column_count):
        chosen = generator.randrange(row_count)
        for row_index in range(row_count):
            mutable[row_index][column] = row_index == chosen
    for row_index, row in enumerate(mutable):
        if not any(row):
            row[generator.randrange(scarce_start) if scarce_start else 0] = True
    return tuple(tuple(row) for row in mutable)


def _random_nonempty_boolean_rows(
    row_count: int, column_count: int, generator: random.Random
) -> tuple[tuple[bool, ...], ...]:
    rows: list[tuple[bool, ...]] = []
    for _ in range(row_count):
        row = [generator.random() < 0.5 for _ in range(column_count)]
        if not any(row):
            row[generator.randrange(column_count)] = True
        rows.append(tuple(row))
    return tuple(rows)


def _random_covering_boolean_rows(
    row_count: int, column_count: int, generator: random.Random
) -> tuple[tuple[bool, ...], ...]:
    rows = [
        [generator.random() < 0.5 for _ in range(column_count)]
        for _ in range(row_count)
    ]
    for row in rows:
        if not any(row):
            row[generator.randrange(column_count)] = True
    for column in range(column_count):
        if not any(row[column] for row in rows):
            rows[generator.randrange(row_count)][column] = True
    return tuple(tuple(row) for row in rows)


def _random_location(generator: random.Random) -> tuple[float, float]:
    return (
        float(generator.randint(0, GRID_LIMIT)),
        float(generator.randint(0, GRID_LIMIT)),
    )


def _pickup_location(
    generator: random.Random, delivery_location: tuple[float, float]
) -> tuple[float, float]:
    pickup = _random_location(generator)
    while pickup == delivery_location:
        pickup = _random_location(generator)
    return pickup


def _level_counts(task_count: int, level_count: int) -> tuple[int, ...]:
    quotient, remainder = divmod(task_count, level_count)
    return tuple(
        quotient + (1 if level < remainder else 0)
        for level in range(level_count)
    )


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    integer = _nonnegative_integer(value, name)
    if integer == 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _finite_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _nonnegative_number(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return number


def _bounded_ratio(
    value: object,
    name: str,
    *,
    maximum: float = 1.0,
    minimum_exclusive: bool = False,
) -> float:
    ratio = _finite_number(value, name)
    below_minimum = ratio <= 0 if minimum_exclusive else ratio < 0
    if below_minimum or ratio > maximum:
        lower_bound = "0 (exclusive)" if minimum_exclusive else "0"
        raise ValueError(
            f"{name} must be between {lower_bound} and {maximum:g}"
        )
    return ratio
