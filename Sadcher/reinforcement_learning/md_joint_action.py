"""Autoregressive legality for fixed-size MD joint actions."""

from __future__ import annotations

from typing import Mapping

import torch


def autoregressive_action_masks(
    observation: Mapping[str, torch.Tensor], actions: torch.Tensor
) -> torch.Tensor:
    """Return each robot's legacy legal mask conditioned on prior actions."""
    return _action_masks(
        observation,
        actions,
        optional_idle=False,
        require_nonempty=False,
        require_minimal_process=False,
    )


def oracle_aligned_action_masks(
    observation: Mapping[str, torch.Tensor],
    actions: torch.Tensor,
    *,
    _completion_cache: dict[int, dict[tuple[object, ...], bool]] | None = None,
) -> torch.Tensor:
    """Return masks that cover the complete exact-action support.

    Unlike the legacy RL mask, this support permits a robot to remain idle when
    the remaining robots can still complete a non-empty joint action.  This is
    the action support used by the supervised prefix prototype; the production
    mask above intentionally keeps its historical behavior unchanged.
    """
    return _action_masks(
        observation,
        actions,
        optional_idle=True,
        require_nonempty=True,
        require_minimal_process=True,
        completion_cache=_completion_cache,
    )


def _action_masks(
    observation: Mapping[str, torch.Tensor],
    actions: torch.Tensor,
    *,
    optional_idle: bool,
    require_nonempty: bool,
    require_minimal_process: bool,
    completion_cache: dict[int, dict[tuple[object, ...], bool]] | None = None,
) -> torch.Tensor:
    """Shared constrained-prefix search used by both mask contracts."""
    base_mask = observation["action_mask"].bool()
    if base_mask.ndim != 3:
        raise ValueError("batched action_mask must have shape [B, R, T+1]")
    batch_size, robot_count, action_count = base_mask.shape
    if tuple(actions.shape) != (batch_size, robot_count):
        raise ValueError("actions must have shape [B, R]")
    task_count = action_count - 1
    required_keys = ("robot_features", "task_features", "task_is_transport")
    if any(key not in observation for key in required_keys):
        raise ValueError("observation is missing joint-action metadata")

    masks = torch.zeros_like(base_mask)
    for batch_index in range(batch_size):
        static = base_mask[batch_index, :, 1:].detach().cpu().bool()
        transport = observation["task_is_transport"][batch_index].detach().cpu().bool()
        robot_skills = observation["robot_features"][batch_index, :, 3:6].detach().cpu() > 0.5
        task_skills = observation["task_features"][batch_index, :, 3:6].detach().cpu() > 0.5
        covered_source = observation.get("process_covered_skills")
        if covered_source is None:
            covered_skills = torch.zeros_like(task_skills, dtype=torch.bool)
        else:
            covered_skills = covered_source[batch_index].detach().cpu() > 0.5
        assigned = observation["task_features"][batch_index, :, 7].detach().cpu() > 0.5

        requirements = tuple(_bits(row) for row in task_skills)
        covers = tuple(_bits(row) for row in covered_skills)
        robot_capabilities = tuple(_bits(row) for row in robot_skills)
        transport_flags = tuple(bool(value) for value in transport)
        initial_covers = covers
        active = 0
        completed = 0
        for task_index in range(task_count):
            if transport_flags[task_index]:
                continue
            if covers[task_index] & requirements[task_index] == requirements[task_index]:
                if assigned[task_index]:
                    completed |= 1 << task_index
            elif assigned[task_index]:
                active |= 1 << task_index

        def completion_exists(
            robot_index,
            used_transport,
            active_process,
            completed_process,
            skill_covers,
            process_members,
            selected_any,
        ):
            state_key = (
                robot_index,
                used_transport,
                active_process,
                completed_process,
                skill_covers,
                process_members,
                selected_any,
            )
            cache = None if completion_cache is None else completion_cache.setdefault(batch_index, {})
            if cache is not None and state_key in cache:
                return cache[state_key]
            if robot_index == robot_count:
                result = active_process == 0 and (
                    selected_any or not require_nonempty
                )
                if result and require_minimal_process:
                    result = _process_coalitions_are_minimal(
                        requirements,
                        initial_covers,
                        robot_capabilities,
                        process_members,
                    )
                if cache is not None:
                    cache[state_key] = result
                return result
            for task_index in range(task_count):
                transition = _transition(
                    robot_index, task_index, static, transport_flags, requirements,
                    robot_capabilities, used_transport, active_process,
                    completed_process, skill_covers, process_members,
                )
                if transition is not None and completion_exists(
                    robot_index + 1, *transition, True
                ):
                    if cache is not None:
                        cache[state_key] = True
                    return True
            result = completion_exists(
                robot_index + 1, used_transport, active_process,
                completed_process, skill_covers, process_members, selected_any,
            )
            if cache is not None:
                cache[state_key] = result
            return result

        state = (
            0,
            active,
            completed,
            covers,
            tuple(0 for _ in range(task_count)),
            False,
        )
        for robot_index in range(robot_count):
            options = []
            for task_index in range(task_count):
                transition = _transition(
                    robot_index, task_index, static, transport_flags, requirements,
                    robot_capabilities, *state[:5],
                )
                if transition is not None and completion_exists(
                    robot_index + 1, *transition, True
                ):
                    options.append((task_index + 1, transition))
            if optional_idle:
                if completion_exists(robot_index + 1, *state):
                    masks[batch_index, robot_index, 0] = True
                for choice, _ in options:
                    masks[batch_index, robot_index, choice] = True
            elif options:
                for choice, _ in options:
                    masks[batch_index, robot_index, choice] = True
            elif completion_exists(robot_index + 1, *state):
                masks[batch_index, robot_index, 0] = True
            else:
                raise RuntimeError("joint action prefix has no legal completion")

            selected = int(actions[batch_index, robot_index].detach().cpu())
            if selected < 0:
                break
            selected_transition = next((transition for choice, transition in options if choice == selected), None)
            if selected_transition is not None:
                state = (*selected_transition, True)
    return masks


def joint_action_is_legal(observation, actions: torch.Tensor) -> bool:
    masks = autoregressive_action_masks(observation, actions)
    selected = masks.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    return bool(selected.all())


def oracle_aligned_action_is_legal(observation, actions: torch.Tensor) -> bool:
    """Return whether a complete action is legal under optional-idle support."""
    masks = oracle_aligned_action_masks(observation, actions)
    selected = masks.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    return bool(selected.all())


def _bits(values: torch.Tensor) -> int:
    return sum(1 << index for index, value in enumerate(values.tolist()) if value)


def _transition(
    robot_index, task_index, static, transport, requirements,
    robot_capabilities, used_transport, active_process, completed_process, covers,
    process_members,
):
    if not bool(static[robot_index, task_index]):
        return None
    task_bit = 1 << task_index
    if transport[task_index]:
        if used_transport & task_bit:
            return None
        return (
            used_transport | task_bit,
            active_process,
            completed_process,
            covers,
            process_members,
        )
    if completed_process & task_bit:
        return None
    uncovered = requirements[task_index] & ~covers[task_index]
    if robot_capabilities[robot_index] & uncovered == 0 and requirements[task_index]:
        return None
    updated_covers = list(covers)
    updated_covers[task_index] |= robot_capabilities[robot_index]
    updated_members = list(process_members)
    updated_members[task_index] |= 1 << robot_index
    if updated_covers[task_index] & requirements[task_index] == requirements[task_index]:
        active_process &= ~task_bit
        completed_process |= task_bit
    else:
        active_process |= task_bit
    return (
        used_transport,
        active_process,
        completed_process,
        tuple(updated_covers),
        tuple(updated_members),
    )


def _process_coalitions_are_minimal(
    requirements,
    initial_covers,
    robot_capabilities,
    process_members,
):
    """Match the exact oracle's minimal process-coalition convention."""
    for task_index, members in enumerate(process_members):
        if not members:
            continue
        requirement = requirements[task_index]
        if requirement == 0:
            if members & (members - 1):
                return False
            continue
        for robot_index, capability in enumerate(robot_capabilities):
            if not members & (1 << robot_index):
                continue
            without_robot = initial_covers[task_index]
            for other_index, other_capability in enumerate(robot_capabilities):
                if other_index != robot_index and members & (1 << other_index):
                    without_robot |= other_capability
            if without_robot & requirement == requirement:
                return False
    return True


__all__ = [
    "autoregressive_action_masks",
    "joint_action_is_legal",
    "oracle_aligned_action_masks",
    "oracle_aligned_action_is_legal",
]
