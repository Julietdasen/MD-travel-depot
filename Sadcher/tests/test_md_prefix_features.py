from __future__ import annotations

import pytest
import torch

from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from experiments.md_prefix_features import (
    EDGE_FEATURE_DIM,
    PREFIX_FEATURE_DIM,
    PrefixAction,
    action_orders,
    canonical_action,
    prefix_sequence,
    validate_complete_action,
)
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
)


def _coalition_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=True),
        tasks=(ProcessTask(1, (0, 0), 2, (True, True)),),
        robots=(
            ProcessRobot(0, (0, 0), (True, False)),
            ProcessRobot(1, (0, 0), (False, True)),
        ),
    )


def test_prefix_tracks_remaining_skills_and_changes_by_prefix() -> None:
    domain = _coalition_domain()
    assignments = ((0, 1), (1, 1))
    action = PrefixAction(assignments, torch.zeros(2, EDGE_FEATURE_DIM))
    sequence = prefix_sequence(domain, action, assignments)
    prefix = sequence[:, EDGE_FEATURE_DIM:]
    assert prefix.shape == (2, PREFIX_FEATURE_DIM)
    assert prefix[0, 4].item() == 1.0
    assert prefix[0, 6].item() == 0.5
    assert prefix[0, 7].item() == 0.0
    assert prefix[1, 4].item() == 0.5
    assert prefix[1, 6].item() == 0.0
    assert prefix[1, 7].item() == 1.0
    assert not torch.equal(prefix[0], prefix[1])


def test_duplicate_robot_and_incomplete_coalition_are_rejected() -> None:
    domain = _coalition_domain()
    with pytest.raises(ValueError, match="robot may appear only once"):
        validate_complete_action(domain, ((0, 1), (0, 1)))
    with pytest.raises(ValueError, match="does not complete"):
        validate_complete_action(domain, ((0, 1),))


def test_orders_are_reproducible_permutations() -> None:
    assignments = ((2, 4), (0, 1), (1, 1))
    first = action_orders(assignments, identity="fixed", random_count=3)
    second = action_orders(assignments, identity="fixed", random_count=3)
    assert first == second
    expected = set(canonical_action(assignments))
    assert all(set(order) == expected and len(order) == len(expected) for order in first)


def test_generated_cached_shape_is_supported() -> None:
    domain = generate_md_instance(MDGeneratorConfig(seed=75000)).domain
    process = next(task for task in domain.tasks if isinstance(task, ProcessTask))
    capable = [
        robot
        for robot in domain.robots
        if isinstance(robot, ProcessRobot)
        and any(a and b for a, b in zip(robot.capabilities, process.requirements))
    ]
    assert capable
