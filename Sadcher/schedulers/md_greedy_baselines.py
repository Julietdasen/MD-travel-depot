"""Auditable transport Greedy baselines for the unified MD runtime."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from experiments.protocol import DatasetSplit, ExperimentResult
from schedulers.process_greedy_md import run_process_greedy_md
from simulation_environment.domain_model import ProcessTask, TransportRobot, TransportTask
from simulation_environment.hard_feasibility import TaskStatus
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from simulation_environment.transport_timing import transport_durations


class TransportGreedyStrategy(str, Enum):
    DISTANCE = "distance"
    ETA = "eta"
    UNLOCK = "unlock"
    MATERIAL_SOLO = "material_solo"


@dataclass(frozen=True, slots=True)
class TransportGreedyDecision:
    strategy: str
    robot_id: int
    task_id: int
    empty_distance: float
    transport_eta: int
    material_service_duration: int
    downstream_ready: bool
    downstream_reach: int
    tie_break: tuple[int, int]


class MDTransportGreedy:
    """Assign feasible transport work using one explicit ranking rule."""

    def __init__(self, strategy: TransportGreedyStrategy) -> None:
        if not isinstance(strategy, TransportGreedyStrategy):
            raise ValueError("strategy must be a TransportGreedyStrategy")
        self.strategy = strategy

    def assign(
        self, simulator: MDDiscreteSimulator
    ) -> tuple[TransportGreedyDecision, ...]:
        decisions: list[TransportGreedyDecision] = []
        while True:
            candidates: list[
                tuple[tuple[float | int, ...], TransportGreedyDecision]
            ] = []
            robots = sorted(
                (
                    robot
                    for robot in simulator.domain.robots
                    if isinstance(robot, TransportRobot)
                ),
                key=lambda robot: robot.robot_id,
            )
            tasks = sorted(
                (
                    task
                    for task in simulator.domain.tasks
                    if isinstance(task, TransportTask)
                ),
                key=lambda task: task.task_id,
            )
            for robot in robots:
                for task in tasks:
                    if not simulator.assignment_feasibility(
                        robot_id=robot.robot_id, task_id=task.task_id
                    ).is_feasible:
                        continue
                    decision = _decision(simulator, robot, task, self.strategy)
                    candidates.append((_score(decision, self.strategy), decision))
            if not candidates:
                return tuple(decisions)
            _, selected = min(candidates, key=lambda item: item[0])
            simulator.assign(robot_id=selected.robot_id, task_id=selected.task_id)
            decisions.append(selected)


def run_greedy_distance(
    simulator: MDDiscreteSimulator, **kwargs: Any
) -> ExperimentResult:
    return _run(
        simulator, TransportGreedyStrategy.DISTANCE, "greedy_distance", **kwargs
    )


def run_greedy_eta(
    simulator: MDDiscreteSimulator, **kwargs: Any
) -> ExperimentResult:
    return _run(simulator, TransportGreedyStrategy.ETA, "greedy_eta", **kwargs)


def run_greedy_unlock(
    simulator: MDDiscreteSimulator, **kwargs: Any
) -> ExperimentResult:
    return _run(
        simulator, TransportGreedyStrategy.UNLOCK, "greedy_unlock", **kwargs
    )


def run_material_solo_greedy(
    simulator: MDDiscreteSimulator, **kwargs: Any
) -> ExperimentResult:
    return _run(
        simulator,
        TransportGreedyStrategy.MATERIAL_SOLO,
        "material_solo_greedy",
        **kwargs,
    )


def _run(
    simulator: MDDiscreteSimulator,
    strategy: TransportGreedyStrategy,
    method: str,
    *,
    run_id: str,
    instance_id: str,
    seed: int,
    split: DatasetSplit,
    max_steps: int,
) -> ExperimentResult:
    policy = MDTransportGreedy(strategy)
    decision_log: list[dict[str, Any]] = []

    def assign_transport(sim: MDDiscreteSimulator) -> None:
        decision_log.extend(asdict(decision) for decision in policy.assign(sim))

    return run_process_greedy_md(
        simulator,
        run_id=run_id,
        instance_id=instance_id,
        seed=seed,
        split=split,
        max_steps=max_steps,
        auxiliary_policy=assign_transport,
        method=method,
        metadata={
            "strategy": strategy.value,
            "selection_rule": _selection_rule(strategy),
            "tie_break": ("robot_id", "task_id"),
            "uses_hard_feasibility": True,
            "decisions": decision_log,
        },
    )


def _decision(
    simulator: MDDiscreteSimulator,
    robot: TransportRobot,
    task: TransportTask,
    strategy: TransportGreedyStrategy,
) -> TransportGreedyDecision:
    assert robot.loaded_speed is not None
    state = simulator.robot_state(robot.robot_id)
    empty_distance = math.dist(state.location, task.pickup_location)
    durations = transport_durations(state.location, robot, task)
    downstream = _downstream_task(simulator, task)
    downstream_ready = all(
        simulator.task_state(predecessor).status is TaskStatus.COMPLETE
        for predecessor in downstream.normal_predecessors
    )
    return TransportGreedyDecision(
        strategy=strategy.value,
        robot_id=robot.robot_id,
        task_id=task.task_id,
        empty_distance=empty_distance,
        transport_eta=durations.occupied,
        material_service_duration=durations.service,
        downstream_ready=downstream_ready,
        downstream_reach=_downstream_reach(simulator, downstream.task_id),
        tie_break=(robot.robot_id, task.task_id),
    )


def _score(
    decision: TransportGreedyDecision, strategy: TransportGreedyStrategy
) -> tuple[float | int, ...]:
    tie = decision.tie_break
    if strategy is TransportGreedyStrategy.DISTANCE:
        return (decision.empty_distance, decision.transport_eta, *tie)
    if strategy is TransportGreedyStrategy.ETA:
        return (decision.transport_eta, decision.empty_distance, *tie)
    if strategy is TransportGreedyStrategy.UNLOCK:
        return (
            0 if decision.downstream_ready else 1,
            -decision.downstream_reach,
            decision.transport_eta,
            decision.empty_distance,
            *tie,
        )
    return (
        decision.material_service_duration,
        decision.transport_eta,
        *tie,
    )


def _downstream_task(
    simulator: MDDiscreteSimulator, task: TransportTask
) -> ProcessTask:
    task_id = task.downstream_process_task_id
    assert task_id is not None
    downstream = next(
        entity for entity in simulator.domain.tasks if entity.task_id == task_id
    )
    assert isinstance(downstream, ProcessTask)
    return downstream


def _downstream_reach(simulator: MDDiscreteSimulator, root: int) -> int:
    successors: dict[int, list[int]] = {}
    for source, target in simulator.domain.normal_edges:
        successors.setdefault(source, []).append(target)
    reached: set[int] = set()
    frontier = [root]
    while frontier:
        current = frontier.pop()
        if current in reached:
            continue
        reached.add(current)
        frontier.extend(successors.get(current, ()))
    return len(reached)


def _selection_rule(strategy: TransportGreedyStrategy) -> str:
    return {
        TransportGreedyStrategy.DISTANCE: "minimum unloaded distance to pickup",
        TransportGreedyStrategy.ETA: "minimum full transport completion ETA",
        TransportGreedyStrategy.UNLOCK: (
            "normal-ready downstream first, then maximum downstream reach"
        ),
        TransportGreedyStrategy.MATERIAL_SOLO: (
            "minimum material service duration without downstream priority"
        ),
    }[strategy]


__all__ = [
    "MDTransportGreedy",
    "TransportGreedyDecision",
    "TransportGreedyStrategy",
    "run_greedy_distance",
    "run_greedy_eta",
    "run_greedy_unlock",
    "run_material_solo_greedy",
]
