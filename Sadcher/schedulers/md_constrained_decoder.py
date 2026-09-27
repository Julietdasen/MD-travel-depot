"""Fast, solver-free constrained decoding and repair for MD assignments."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import torch

from simulation_environment.domain_model import (
    ProcessRobot,
    ProcessTask,
    TransportTask,
)
from simulation_environment.md_discrete_simulator import (
    MDDiscreteSimulator,
    TaskStatus,
)


Assignment = tuple[int, int]


@dataclass(frozen=True, slots=True)
class RepairEvent:
    code: str
    robot_id: int | None
    task_id: int | None
    message: str


@dataclass(frozen=True, slots=True)
class DecoderResult:
    assignments: tuple[Assignment, ...]
    robot_ids: tuple[int, ...]
    task_ids: tuple[int, ...]
    decoder: str
    repaired: bool = False
    repair_events: tuple[RepairEvent, ...] = ()
    illegal_assignment_count: int = 0
    repair_time_seconds: float = 0.0


def simulator_hard_mask(simulator: MDDiscreteSimulator) -> torch.Tensor:
    """Return the canonical pair mask in sorted runtime ID order."""

    robot_ids = tuple(sorted(simulator.robot_states))
    task_ids = tuple(sorted(simulator.task_states))
    mask = torch.tensor(
        [
            [
                simulator.assignment_feasibility(
                    robot_id=robot_id, task_id=task_id
                ).is_feasible
                for task_id in task_ids
            ]
            for robot_id in robot_ids
        ],
        dtype=torch.bool,
    )
    _apply_ready_preference_guard(mask, simulator, robot_ids, task_ids)
    return mask


def _apply_ready_preference_guard(
    mask: torch.Tensor,
    simulator: MDDiscreteSimulator,
    robot_ids: tuple[int, ...],
    task_ids: tuple[int, ...],
) -> None:
    """Reject process-task columns whose precursors are not yet committed.

    ``is_task_assignable`` allows a robot to depart toward a process task
    while its precursors are still IN_PROGRESS (OR-Tools "travel overlap"
    contract): the precursor is expected to finish while the robot travels.
    The same contract holds while a precursor is PENDING with a complete
    coalition already committed — those robots are on their way and service
    starts as soon as the precursor's own precursors clear. The contract
    only breaks when a precursor is PENDING with nobody committed yet, so
    a robot departing to the successor could wait forever while its peers
    get consumed on other blocked tasks, producing a deadlock.

    Classify each process task by precursor state (both normal_predecessors
    and material_predecessor):

    - precursor COMPLETE or IN_PROGRESS → safe (leave mask alone)
    - precursor PENDING with non-empty assigned_robot_ids → safe (coalition
      committed and traveling; matches the OR-Tools departure-during-
      precursor pattern)
    - precursor PENDING with no assigned robots → unsafe (zero out
      process-robot entries so the decoder cannot commit a robot to the
      successor yet)
    """
    tasks = {task.task_id: task for task in simulator.domain.tasks}
    task_states = simulator.task_states

    def precursor_ids(task: ProcessTask) -> tuple[int, ...]:
        ids = list(task.normal_predecessors)
        if task.material_predecessor is not None:
            ids.append(task.material_predecessor)
        return tuple(ids)

    def is_uncommitted_pending(task_id: int) -> bool:
        state = task_states[task_id]
        return (
            state.status is TaskStatus.PENDING and not state.assigned_robot_ids
        )

    unsafe_columns: list[int] = []
    for column, task_id in enumerate(task_ids):
        task = tasks[task_id]
        if not isinstance(task, ProcessTask):
            continue
        precursors = precursor_ids(task)
        if not precursors:
            continue
        if any(is_uncommitted_pending(predecessor) for predecessor in precursors):
            unsafe_columns.append(column)
    if not unsafe_columns:
        return
    robots = {robot.robot_id: robot for robot in simulator.domain.robots}
    process_robot_rows = [
        row
        for row, robot_id in enumerate(robot_ids)
        if isinstance(robots[robot_id], ProcessRobot)
    ]
    if not process_robot_rows:
        return
    for row in process_robot_rows:
        for column in unsafe_columns:
            mask[row, column] = False


class LearnedConstrainedDecoder:
    """Turn learned pair scores into legal task-level assignments."""

    name = "learned_constrained"

    def decode(
        self,
        scores: torch.Tensor,
        simulator: MDDiscreteSimulator,
        *,
        hard_feasibility_mask: torch.Tensor | None = None,
    ) -> DecoderResult:
        context = _DecoderContext.create(
            scores, simulator, hard_feasibility_mask
        )
        used_robots: set[int] = set()
        proposal: list[Assignment] = []
        task_order = sorted(
            context.task_ids,
            key=lambda task_id: (
                -context.best_score(task_id, excluded=used_robots),
                task_id,
            ),
        )
        for task_id in task_order:
            task = context.tasks[task_id]
            if isinstance(task, TransportTask):
                candidates = context.legal_robots(task_id, excluded=used_robots)
                if not candidates:
                    continue
                robot_id = max(
                    candidates,
                    key=lambda candidate: (
                        context.score(candidate, task_id), -candidate
                    ),
                )
                proposal.append((robot_id, task_id))
                used_robots.add(robot_id)
                continue
            coalition = context.process_coalition(
                task_id, preferred=(), excluded=used_robots
            )
            if coalition is None:
                continue
            proposal.extend((robot_id, task_id) for robot_id in coalition)
            used_robots.update(coalition)

        repaired = FastAssignmentRepair().repair(
            tuple(proposal), scores, simulator,
            hard_feasibility_mask=context.hard_mask,
        )
        return replace(repaired, decoder=self.name)


class MaskedGreedyDecoder(LearnedConstrainedDecoder):
    """Deterministic task-ID greedy baseline with the same constraints."""

    name = "masked_greedy"

    def decode(
        self,
        scores: torch.Tensor,
        simulator: MDDiscreteSimulator,
        *,
        hard_feasibility_mask: torch.Tensor | None = None,
    ) -> DecoderResult:
        context = _DecoderContext.create(
            scores, simulator, hard_feasibility_mask
        )
        used_robots: set[int] = set()
        proposal: list[Assignment] = []
        for task_id in context.task_ids:
            task = context.tasks[task_id]
            if isinstance(task, TransportTask):
                candidates = context.legal_robots(task_id, excluded=used_robots)
                if candidates:
                    robot_id = max(
                        candidates,
                        key=lambda candidate: (
                            context.score(candidate, task_id), -candidate
                        ),
                    )
                    proposal.append((robot_id, task_id))
                    used_robots.add(robot_id)
                continue
            coalition = context.process_coalition(
                task_id, preferred=(), excluded=used_robots
            )
            if coalition is not None:
                proposal.extend((robot_id, task_id) for robot_id in coalition)
                used_robots.update(coalition)
        repaired = FastAssignmentRepair().repair(
            tuple(proposal), scores, simulator,
            hard_feasibility_mask=context.hard_mask,
        )
        return replace(repaired, decoder=self.name)


class FastAssignmentRepair:
    def repair(
        self,
        proposal: Sequence[Assignment],
        scores: torch.Tensor,
        simulator: MDDiscreteSimulator,
        *,
        hard_feasibility_mask: torch.Tensor | None = None,
    ) -> DecoderResult:
        started = time.perf_counter()
        context = _DecoderContext.create(
            scores, simulator, hard_feasibility_mask
        )
        events: list[RepairEvent] = []
        legal: list[Assignment] = []
        seen: set[Assignment] = set()
        for item in proposal:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(value, int) for value in item)
            ):
                events.append(
                    RepairEvent("illegal_pair", None, None, "malformed assignment")
                )
                continue
            robot_id, task_id = item
            if item in seen:
                events.append(
                    RepairEvent(
                        "duplicate_pair", robot_id, task_id, "duplicate pair removed"
                    )
                )
                continue
            seen.add(item)
            if not context.is_legal(robot_id, task_id):
                events.append(
                    RepairEvent(
                        "illegal_pair",
                        robot_id,
                        task_id,
                        "pair rejected by centralized hard-feasibility",
                    )
                )
                continue
            legal.append(item)

        by_robot: dict[int, list[Assignment]] = {}
        for assignment in legal:
            by_robot.setdefault(assignment[0], []).append(assignment)
        robot_unique: list[Assignment] = []
        for robot_id in sorted(by_robot):
            choices = by_robot[robot_id]
            keep = max(
                choices,
                key=lambda item: (context.score(*item), -item[1]),
            )
            robot_unique.append(keep)
            for dropped in choices:
                if dropped != keep:
                    events.append(
                        RepairEvent(
                            "duplicate_robot",
                            dropped[0],
                            dropped[1],
                            "robot kept its highest-scoring assignment",
                        )
                    )

        by_task: dict[int, list[Assignment]] = {}
        for assignment in robot_unique:
            by_task.setdefault(assignment[1], []).append(assignment)
        singleton_safe: list[Assignment] = []
        for task_id in sorted(by_task):
            choices = by_task[task_id]
            if not isinstance(context.tasks[task_id], TransportTask):
                singleton_safe.extend(choices)
                continue
            keep = max(
                choices,
                key=lambda item: (context.score(*item), -item[0]),
            )
            singleton_safe.append(keep)
            for dropped in choices:
                if dropped != keep:
                    events.append(
                        RepairEvent(
                            "transport_singleton",
                            dropped[0],
                            task_id,
                            "extra transport assignment removed",
                        )
                    )

        accepted: list[Assignment] = [
            assignment
            for assignment in singleton_safe
            if isinstance(context.tasks[assignment[1]], TransportTask)
        ]
        used = {robot_id for robot_id, _ in accepted}
        process_task_ids = sorted(
            {
                task_id
                for _, task_id in singleton_safe
                if isinstance(context.tasks[task_id], ProcessTask)
            },
            key=lambda task_id: (
                -max(
                    context.score(robot_id, task_id)
                    for robot_id, selected_task in singleton_safe
                    if selected_task == task_id
                ),
                task_id,
            ),
        )
        for task_id in process_task_ids:
            preferred = tuple(
                robot_id
                for robot_id, selected_task in singleton_safe
                if selected_task == task_id
            )
            coalition = context.process_coalition(
                task_id, preferred=preferred, excluded=used
            )
            if coalition is None:
                events.append(
                    RepairEvent(
                        "coalition_incomplete",
                        None,
                        task_id,
                        "incomplete process coalition rejected",
                    )
                )
                continue
            added = set(coalition) - set(preferred)
            if added:
                events.append(
                    RepairEvent(
                        "coalition_completed",
                        None,
                        task_id,
                        f"added robots {sorted(added)} to cover required skills",
                    )
                )
            accepted.extend((robot_id, task_id) for robot_id in coalition)
            used.update(coalition)

        assignments = tuple(sorted(accepted, key=lambda item: (item[1], item[0])))
        _validate_joint(assignments, context)
        return DecoderResult(
            assignments=assignments,
            robot_ids=context.robot_ids,
            task_ids=context.task_ids,
            decoder="fast_repair",
            repaired=bool(events) or tuple(proposal) != assignments,
            repair_events=tuple(events),
            illegal_assignment_count=0,
            repair_time_seconds=time.perf_counter() - started,
        )


def apply_decoder_result(
    simulator: MDDiscreteSimulator, result: DecoderResult
) -> int:
    context = _DecoderContext.create(
        torch.zeros(len(result.robot_ids), len(result.task_ids)),
        simulator,
        None,
    )
    if result.robot_ids != context.robot_ids or result.task_ids != context.task_ids:
        raise ValueError("decoder result IDs do not match simulator state")
    _validate_joint(result.assignments, context)
    for robot_id, task_id in result.assignments:
        simulator.assign(robot_id=robot_id, task_id=task_id)
    return len(result.assignments)


@dataclass(frozen=True, slots=True)
class _DecoderContext:
    simulator: MDDiscreteSimulator
    scores: torch.Tensor
    hard_mask: torch.Tensor
    robot_ids: tuple[int, ...]
    task_ids: tuple[int, ...]
    robots: dict[int, object]
    tasks: dict[int, object]
    robot_index: dict[int, int]
    task_index: dict[int, int]

    @classmethod
    def create(
        cls,
        scores: torch.Tensor,
        simulator: MDDiscreteSimulator,
        supplied_mask: torch.Tensor | None,
    ) -> "_DecoderContext":
        if not isinstance(simulator, MDDiscreteSimulator):
            raise TypeError("simulator must be an MDDiscreteSimulator")
        robot_ids = tuple(sorted(simulator.robot_states))
        task_ids = tuple(sorted(simulator.task_states))
        if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != (
            len(robot_ids), len(task_ids)
        ):
            raise ValueError("scores must have robot-by-task shape")
        if not torch.isfinite(scores).all():
            raise ValueError("decoder scores must be finite")
        central = simulator_hard_mask(simulator)
        if supplied_mask is not None:
            candidate = supplied_mask.detach().cpu().to(dtype=torch.bool)
            if tuple(candidate.shape) != tuple(central.shape) or not torch.equal(
                candidate, central
            ):
                raise ValueError(
                    "supplied mask must equal centralized hard-feasibility source"
                )
        return cls(
            simulator=simulator,
            scores=scores.detach().cpu(),
            hard_mask=central,
            robot_ids=robot_ids,
            task_ids=task_ids,
            robots={robot.robot_id: robot for robot in simulator.domain.robots},
            tasks={task.task_id: task for task in simulator.domain.tasks},
            robot_index={value: index for index, value in enumerate(robot_ids)},
            task_index={value: index for index, value in enumerate(task_ids)},
        )

    def score(self, robot_id: int, task_id: int) -> float:
        return float(
            self.scores[self.robot_index[robot_id], self.task_index[task_id]]
        )

    def is_legal(self, robot_id: int, task_id: int) -> bool:
        if robot_id not in self.robot_index or task_id not in self.task_index:
            return False
        return bool(
            self.hard_mask[
                self.robot_index[robot_id], self.task_index[task_id]
            ]
        )

    def legal_robots(
        self, task_id: int, *, excluded: set[int]
    ) -> tuple[int, ...]:
        return tuple(
            robot_id
            for robot_id in self.robot_ids
            if robot_id not in excluded and self.is_legal(robot_id, task_id)
        )

    def best_score(self, task_id: int, *, excluded: set[int]) -> float:
        candidates = self.legal_robots(task_id, excluded=excluded)
        return max(
            (self.score(robot_id, task_id) for robot_id in candidates),
            default=float("-inf"),
        )

    def process_coalition(
        self,
        task_id: int,
        *,
        preferred: Sequence[int],
        excluded: set[int],
    ) -> tuple[int, ...] | None:
        task = self.tasks[task_id]
        assert isinstance(task, ProcessTask)
        coverage = [False] * len(task.requirements)
        selected: list[int] = []

        def add_if_useful(robot_id: int) -> None:
            if robot_id in excluded or robot_id in selected:
                return
            if not self.is_legal(robot_id, task_id):
                return
            robot = self.robots[robot_id]
            if not isinstance(robot, ProcessRobot):
                return
            contributes = any(
                required and capability and not coverage[index]
                for index, (required, capability) in enumerate(
                    zip(task.requirements, robot.capabilities, strict=True)
                )
            )
            if not any(task.requirements) or contributes:
                selected.append(robot_id)
                for index, capability in enumerate(robot.capabilities):
                    coverage[index] = coverage[index] or capability

        for robot_id in preferred:
            add_if_useful(robot_id)
        candidates = sorted(
            self.legal_robots(task_id, excluded=excluded | set(selected)),
            key=lambda robot_id: (-self.score(robot_id, task_id), robot_id),
        )
        for robot_id in candidates:
            if selected and all(
                not required or coverage[index]
                for index, required in enumerate(task.requirements)
            ):
                break
            add_if_useful(robot_id)
        complete = bool(selected) and all(
            not required or coverage[index]
            for index, required in enumerate(task.requirements)
        )
        return tuple(selected) if complete else None


def _validate_joint(
    assignments: Iterable[Assignment], context: _DecoderContext
) -> None:
    assignments = tuple(assignments)
    robot_ids = [robot_id for robot_id, _ in assignments]
    if len(robot_ids) != len(set(robot_ids)):
        raise ValueError("decoder result assigns one robot more than once")
    by_task: dict[int, list[int]] = {}
    for robot_id, task_id in assignments:
        if not context.is_legal(robot_id, task_id):
            raise ValueError("decoder result contains an illegal pair")
        by_task.setdefault(task_id, []).append(robot_id)
    for task_id, assigned in by_task.items():
        task = context.tasks[task_id]
        if isinstance(task, TransportTask):
            if len(assigned) != 1:
                raise ValueError("transport task must have a singleton assignment")
            continue
        assert isinstance(task, ProcessTask)
        coverage = [False] * len(task.requirements)
        for robot_id in assigned:
            robot = context.robots[robot_id]
            assert isinstance(robot, ProcessRobot)
            coverage = [
                current or capability
                for current, capability in zip(
                    coverage, robot.capabilities, strict=True
                )
            ]
        if not all(
            not required or coverage[index]
            for index, required in enumerate(task.requirements)
        ):
            raise ValueError("process task has an incomplete coalition")


__all__ = [
    "DecoderResult",
    "FastAssignmentRepair",
    "LearnedConstrainedDecoder",
    "MaskedGreedyDecoder",
    "RepairEvent",
    "apply_decoder_result",
    "simulator_hard_mask",
]
