"""Canonical online feature bridge for MD simulator rollouts."""

from __future__ import annotations

import torch

from data_generation.md_expert_dataset import TypedTaskGraph, snapshot_md_simulator
from models.md_legacy_features import legacy_policy_inputs_from_state_views
from models.md_policy import MDPolicyInputs
from models.md_policy_features import (
    MDPolicyStateView,
    md_policy_inputs_from_state_views,
)
from schedulers.md_constrained_decoder import simulator_hard_mask
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


def build_md_policy_state_from_simulator(
    simulator: MDDiscreteSimulator,
) -> MDPolicyStateView:
    """Snapshot one runnable simulator state without consulting future work."""

    if not isinstance(simulator, MDDiscreteSimulator):
        raise TypeError("simulator must be an MDDiscreteSimulator")
    robots, tasks = snapshot_md_simulator(simulator)
    hard_mask = simulator_hard_mask(simulator)
    return MDPolicyStateView(
        robot_ids=tuple(robot.robot_id for robot in robots),
        task_ids=tuple(task.task_id for task in tasks),
        typed_graph=TypedTaskGraph(
            task_types=tuple(
                (task.task_id, task.task_type.value)
                for task in sorted(simulator.domain.tasks, key=lambda item: item.task_id)
            ),
            normal_edges=simulator.domain.normal_edges,
            material_edges=simulator.domain.material_edges,
            downstream_links=simulator.domain.material_edges,
        ),
        robots=robots,
        tasks=tasks,
        hard_feasibility_mask=tuple(
            tuple(bool(value) for value in row) for row in hard_mask.tolist()
        ),
    )


def build_md_policy_inputs_from_simulator(
    simulator: MDDiscreteSimulator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MDPolicyInputs]:
    """Build legacy and MD tensors that a real checkpoint consumes online."""

    state = build_md_policy_state_from_simulator(simulator)
    legacy = legacy_policy_inputs_from_state_views((state,))
    md_inputs = md_policy_inputs_from_state_views((state,))
    return (
        legacy.robot_features,
        legacy.task_features,
        legacy.task_adjacency,
        md_inputs,
    )
