"""Data-only relational candidate generation frozen for Ticket 42."""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass, replace
from typing import Sequence


ROBOT_COUNT = 3
CANDIDATE_TASK_COUNT = 4
RELATIONAL_FAMILIES = (
    "competitor_robot_position",
    "competitor_robot_speed",
    "alternative_task_pickup",
    "alternative_downstream_priority",
)
Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class RelationalState:
    state_id: str
    pair_id: str
    family: str
    robot_positions: tuple[Point, ...]
    robot_loaded_speeds: tuple[float, ...]
    task_pickups: tuple[Point, ...]
    task_deliveries: tuple[Point, ...]
    downstream_priorities: tuple[float, ...]
    eta: tuple[tuple[float, ...], ...]
    action_values: tuple[float, ...]
    oracle_action: int
    hard_mask: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True, slots=True)
class RelationalTwin:
    family: str
    pair_id: str
    changed_entity_index: int
    before: RelationalState
    after: RelationalState


@dataclass(frozen=True, slots=True)
class _StateSpec:
    robot_positions: tuple[Point, ...]
    robot_loaded_speeds: tuple[float, ...]
    task_pickups: tuple[Point, ...]
    task_deliveries: tuple[Point, ...]
    downstream_priorities: tuple[float, ...]


def build_unconditioned_relational_candidates(
    *,
    candidates_per_family: int,
    seed: int,
    perturbation_seed: int | None = None,
) -> tuple[RelationalTwin, ...]:
    """Generate the unconditioned candidates without importing model code."""
    _validate_generation_arguments(
        candidates_per_family,
        seed,
        perturbation_seed,
    )
    state_generator = random.Random(seed)
    intervention_generator = (
        state_generator
        if perturbation_seed is None
        else random.Random(perturbation_seed)
    )
    twins: list[RelationalTwin] = []
    signatures: set[str] = set()
    for family in RELATIONAL_FAMILIES:
        for index in range(candidates_per_family):
            pair_id = f"unconditioned-{family}-{index:06d}"
            for _attempt in range(100):
                before_spec = _random_state_spec(state_generator)
                changed_index, after_spec = _relational_intervention(
                    family,
                    before_spec,
                    intervention_generator,
                )
                signature = _structural_template_signature(
                    family,
                    changed_index,
                    before_spec,
                    after_spec,
                )
                if signature in signatures:
                    continue
                twins.append(
                    RelationalTwin(
                        family=family,
                        pair_id=pair_id,
                        changed_entity_index=changed_index,
                        before=_state_from_spec(
                            state_id=f"{pair_id}-before",
                            pair_id=pair_id,
                            family=family,
                            spec=before_spec,
                        ),
                        after=_state_from_spec(
                            state_id=f"{pair_id}-after",
                            pair_id=pair_id,
                            family=family,
                            spec=after_spec,
                        ),
                    )
                )
                signatures.add(signature)
                break
            else:
                raise RuntimeError(f"could not build unique candidate {pair_id}")
    return tuple(twins)


def state_completion_margin(state: RelationalState) -> float:
    values = sorted(state.action_values, reverse=True)
    return values[0] - values[1]


def template_signature(twin: RelationalTwin) -> str:
    return json.dumps(
        {
            "changed": twin.changed_entity_index,
            "before": {
                "robot_positions": twin.before.robot_positions,
                "robot_speeds": twin.before.robot_loaded_speeds,
                "task_pickups": twin.before.task_pickups,
                "deliveries": twin.before.task_deliveries,
                "priorities": twin.before.downstream_priorities,
            },
            "after": {
                "robot_positions": twin.after.robot_positions,
                "robot_speeds": twin.after.robot_loaded_speeds,
                "task_pickups": twin.after.task_pickups,
                "deliveries": twin.after.task_deliveries,
                "priorities": twin.after.downstream_priorities,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _random_state_spec(generator: random.Random) -> _StateSpec:
    return _StateSpec(
        robot_positions=tuple(_point(generator) for _ in range(ROBOT_COUNT)),
        robot_loaded_speeds=tuple(
            generator.uniform(0.65, 1.6) for _ in range(ROBOT_COUNT)
        ),
        task_pickups=tuple(
            _point(generator) for _ in range(CANDIDATE_TASK_COUNT)
        ),
        task_deliveries=tuple(
            _point(generator) for _ in range(CANDIDATE_TASK_COUNT)
        ),
        downstream_priorities=tuple(
            generator.uniform(0.25, 1.0)
            for _ in range(CANDIDATE_TASK_COUNT)
        ),
    )


def _point(generator: random.Random) -> Point:
    return generator.uniform(0.0, 2.0), generator.uniform(0.0, 2.0)


def _relational_intervention(
    family: str,
    spec: _StateSpec,
    generator: random.Random,
) -> tuple[int, _StateSpec]:
    if family == "competitor_robot_position":
        index = generator.randrange(ROBOT_COUNT)
        positions = list(spec.robot_positions)
        positions[index] = _point(generator)
        return index, replace(spec, robot_positions=tuple(positions))
    if family == "competitor_robot_speed":
        index = generator.randrange(ROBOT_COUNT)
        speeds = list(spec.robot_loaded_speeds)
        speeds[index] = generator.uniform(0.45, 2.0)
        return index, replace(spec, robot_loaded_speeds=tuple(speeds))
    if family == "alternative_task_pickup":
        index = generator.randrange(CANDIDATE_TASK_COUNT)
        pickups = list(spec.task_pickups)
        pickups[index] = _point(generator)
        return index, replace(spec, task_pickups=tuple(pickups))
    if family == "alternative_downstream_priority":
        index = generator.randrange(CANDIDATE_TASK_COUNT)
        priorities = list(spec.downstream_priorities)
        priorities[index] = generator.uniform(0.05, 1.25)
        return index, replace(spec, downstream_priorities=tuple(priorities))
    raise ValueError(f"unknown relational family: {family}")


def _state_from_spec(
    *,
    state_id: str,
    pair_id: str,
    family: str,
    spec: _StateSpec,
) -> RelationalState:
    eta = tuple(
        tuple(
            _transport_eta(
                spec.robot_positions[robot],
                spec.robot_loaded_speeds[robot],
                spec.task_pickups[task],
                spec.task_deliveries[task],
            )
            for task in range(CANDIDATE_TASK_COUNT)
        )
        for robot in range(ROBOT_COUNT)
    )
    action_values = _exact_action_values(eta, spec.downstream_priorities)
    oracle_action = max(
        range(len(action_values)),
        key=lambda action: (action_values[action], -action),
    )
    hard_mask = tuple(
        (True,) * CANDIDATE_TASK_COUNT
        + (False,) * CANDIDATE_TASK_COUNT
        for _ in range(ROBOT_COUNT)
    )
    return RelationalState(
        state_id=state_id,
        pair_id=pair_id,
        family=family,
        robot_positions=spec.robot_positions,
        robot_loaded_speeds=spec.robot_loaded_speeds,
        task_pickups=spec.task_pickups,
        task_deliveries=spec.task_deliveries,
        downstream_priorities=spec.downstream_priorities,
        eta=eta,
        action_values=action_values,
        oracle_action=oracle_action,
        hard_mask=hard_mask,
    )


def _transport_eta(
    robot_position: Point,
    loaded_speed: float,
    pickup: Point,
    delivery: Point,
) -> float:
    unloaded_leg = math.dist(robot_position, pickup) / 2.0
    loaded_leg = math.dist(pickup, delivery) / loaded_speed
    return unloaded_leg + loaded_leg


def _exact_action_values(
    eta: Sequence[Sequence[float]],
    priorities: Sequence[float],
) -> tuple[float, ...]:
    edge_utility = tuple(
        tuple(
            1.25 * priorities[task] - eta[robot][task]
            for task in range(CANDIDATE_TASK_COUNT)
        )
        for robot in range(ROBOT_COUNT)
    )
    assignments = tuple(
        (
            tasks,
            sum(edge_utility[robot][task] for robot, task in enumerate(tasks)),
        )
        for tasks in itertools.permutations(
            range(CANDIDATE_TASK_COUNT),
            ROBOT_COUNT,
        )
    )
    values = []
    for robot in range(ROBOT_COUNT):
        for task in range(CANDIDATE_TASK_COUNT):
            completion = max(
                total
                for assigned_tasks, total in assignments
                if assigned_tasks[robot] == task
            )
            values.append(completion + 0.05 * edge_utility[robot][task])
    return tuple(values)


def _structural_template_signature(
    family: str,
    changed_index: int,
    before: _StateSpec,
    after: _StateSpec,
) -> str:
    def payload(spec: _StateSpec) -> dict[str, object]:
        return {
            "robot_positions": spec.robot_positions,
            "robot_loaded_speeds": spec.robot_loaded_speeds,
            "task_pickups": spec.task_pickups,
            "task_deliveries": spec.task_deliveries,
            "downstream_priorities": spec.downstream_priorities,
        }

    return json.dumps(
        {
            "family": family,
            "changed_entity_index": changed_index,
            "before": payload(before),
            "after": payload(after),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_generation_arguments(
    candidates_per_family: int,
    seed: int,
    perturbation_seed: int | None,
) -> None:
    if (
        isinstance(candidates_per_family, bool)
        or not isinstance(candidates_per_family, int)
        or candidates_per_family < 2
    ):
        raise ValueError(
            "candidates_per_family must be an integer of at least two"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if perturbation_seed is not None and (
        isinstance(perturbation_seed, bool)
        or not isinstance(perturbation_seed, int)
        or perturbation_seed < 0
    ):
        raise ValueError("perturbation_seed must be a non-negative integer")
