"""Seeded GA and simulated-annealing baselines evaluated by the MD simulator."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace
from typing import Any

from experiments.protocol import DatasetSplit, ExperimentResult, FailureReason
from simulation_environment.domain_model import SchedulingDomain
from simulation_environment.hard_feasibility import TaskStatus
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


@dataclass(frozen=True, slots=True)
class GeneticAlgorithmConfig:
    population_size: int = 20
    generations: int = 50
    mutation_probability: float = 0.2
    elite_count: int = 2
    seed: int = 0
    time_budget_seconds: float | None = None

    def __post_init__(self) -> None:
        _positive_int("population_size", self.population_size, minimum=2)
        _positive_int("generations", self.generations, minimum=0)
        _probability("mutation_probability", self.mutation_probability)
        _positive_int("elite_count", self.elite_count, minimum=1)
        if self.elite_count > self.population_size:
            raise ValueError("elite_count cannot exceed population_size")
        _seed(self.seed)
        _optional_time_budget(self.time_budget_seconds)


@dataclass(frozen=True, slots=True)
class SimulatedAnnealingConfig:
    iterations: int = 100
    initial_temperature: float = 10.0
    cooling_rate: float = 0.95
    seed: int = 0
    time_budget_seconds: float | None = None

    def __post_init__(self) -> None:
        _positive_int("iterations", self.iterations, minimum=0)
        if (
            isinstance(self.initial_temperature, bool)
            or not isinstance(self.initial_temperature, (int, float))
            or not math.isfinite(self.initial_temperature)
            or self.initial_temperature <= 0
        ):
            raise ValueError("initial_temperature must be positive and finite")
        if (
            isinstance(self.cooling_rate, bool)
            or not isinstance(self.cooling_rate, (int, float))
            or not math.isfinite(self.cooling_rate)
            or not 0 < self.cooling_rate <= 1
        ):
            raise ValueError("cooling_rate must be in (0, 1]")
        _seed(self.seed)
        _optional_time_budget(self.time_budget_seconds)


@dataclass(frozen=True, slots=True, order=True)
class CandidateQuality:
    failure_rank: int
    objective: float
    incomplete_tasks: int

    def as_tuple(self) -> tuple[int, float, int]:
        return self.failure_rank, self.objective, self.incomplete_tasks


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    task_order: tuple[int, ...]
    robot_order: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate: SearchCandidate
    quality: CandidateQuality
    result: ExperimentResult


def run_genetic_algorithm(
    domain: SchedulingDomain,
    *,
    exit_location: tuple[float, float] = (0.0, 0.0),
    run_id: str,
    instance_id: str,
    split: DatasetSplit,
    max_steps: int,
    config: GeneticAlgorithmConfig | None = None,
) -> ExperimentResult:
    """Search task priorities with a fixed-seed genetic algorithm."""

    config = GeneticAlgorithmConfig() if config is None else config
    _max_steps(max_steps)
    started = time.perf_counter()
    rng = random.Random(config.seed)
    task_ids = tuple(sorted(task.task_id for task in domain.tasks))
    robot_ids = tuple(sorted(robot.robot_id for robot in domain.robots))
    cache: dict[SearchCandidate, CandidateEvaluation] = {}

    def evaluate(candidate: SearchCandidate) -> CandidateEvaluation:
        if candidate not in cache:
            cache[candidate] = _evaluate_candidate(
                domain,
                exit_location,
                candidate,
                instance_id=instance_id,
                split=split,
                max_steps=max_steps,
                seed=config.seed,
            )
        return cache[candidate]

    population = [SearchCandidate(task_ids, robot_ids)]
    while len(population) < config.population_size:
        shuffled_tasks = list(task_ids)
        shuffled_robots = list(robot_ids)
        rng.shuffle(shuffled_tasks)
        rng.shuffle(shuffled_robots)
        population.append(
            SearchCandidate(tuple(shuffled_tasks), tuple(shuffled_robots))
        )

    evaluated = [evaluate(population[0])]
    for candidate in population[1:]:
        if _budget_expired(started, config.time_budget_seconds):
            break
        evaluated.append(evaluate(candidate))
    ranked = sorted(evaluated, key=_evaluation_key)
    best = ranked[0]
    generations_completed = 0
    for generation in range(config.generations):
        if _budget_expired(started, config.time_budget_seconds):
            break
        parent_pool = ranked[: max(1, len(ranked) // 2)]
        next_population = [
            evaluation.candidate for evaluation in ranked[: config.elite_count]
        ]
        while len(next_population) < config.population_size:
            child = _crossover_candidate(
                rng.choice(parent_pool).candidate,
                rng.choice(parent_pool).candidate,
                rng,
            )
            if rng.random() < config.mutation_probability:
                child = _mutate_candidate(child, rng)
            next_population.append(child)

        evaluated = []
        for candidate in next_population:
            if evaluated and _budget_expired(started, config.time_budget_seconds):
                break
            evaluated.append(evaluate(candidate))
        ranked = sorted(evaluated, key=_evaluation_key)
        if _evaluation_key(ranked[0]) < _evaluation_key(best):
            best = ranked[0]
        generations_completed = generation + 1
    elapsed = time.perf_counter() - started
    return _finalize(
        best,
        run_id=run_id,
        method="genetic_algorithm",
        seed=config.seed,
        split=split,
        wall_time_seconds=elapsed,
        metadata={
            "algorithm": "genetic_algorithm",
            "seed": config.seed,
            "population_size": config.population_size,
            "generation_budget": config.generations,
            "generations_completed": generations_completed,
            "time_budget_seconds": config.time_budget_seconds,
            "stopped_by_time_budget": _budget_expired(
                started, config.time_budget_seconds
            ),
            "evaluations": len(cache),
            "best_order": best.candidate.task_order,
            "best_robot_order": best.candidate.robot_order,
            "best_quality": best.quality.as_tuple(),
            "uses_unified_simulator": True,
            "infeasible_candidate_policy": "finite penalty after canonical failure",
            "tie_break": "quality_then_lexicographic_order",
        },
    )


def run_simulated_annealing(
    domain: SchedulingDomain,
    *,
    exit_location: tuple[float, float] = (0.0, 0.0),
    run_id: str,
    instance_id: str,
    split: DatasetSplit,
    max_steps: int,
    config: SimulatedAnnealingConfig | None = None,
) -> ExperimentResult:
    """Search task priorities with fixed-seed simulated annealing."""

    config = SimulatedAnnealingConfig() if config is None else config
    _max_steps(max_steps)
    started = time.perf_counter()
    rng = random.Random(config.seed)
    initial = SearchCandidate(
        tuple(sorted(task.task_id for task in domain.tasks)),
        tuple(sorted(robot.robot_id for robot in domain.robots)),
    )
    cache: dict[SearchCandidate, CandidateEvaluation] = {}

    def evaluate(candidate: SearchCandidate) -> CandidateEvaluation:
        if candidate not in cache:
            cache[candidate] = _evaluate_candidate(
                domain,
                exit_location,
                candidate,
                instance_id=instance_id,
                split=split,
                max_steps=max_steps,
                seed=config.seed,
            )
        return cache[candidate]

    current = evaluate(initial)
    best = current
    temperature = float(config.initial_temperature)
    iterations_completed = 0
    for iteration in range(config.iterations):
        if _budget_expired(started, config.time_budget_seconds):
            break
        neighbor = evaluate(_mutate_candidate(current.candidate, rng))
        delta = _annealing_delta(neighbor.quality, current.quality)
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current = neighbor
        if _evaluation_key(current) < _evaluation_key(best):
            best = current
        temperature *= config.cooling_rate
        iterations_completed = iteration + 1

    elapsed = time.perf_counter() - started
    return _finalize(
        best,
        run_id=run_id,
        method="simulated_annealing",
        seed=config.seed,
        split=split,
        wall_time_seconds=elapsed,
        metadata={
            "algorithm": "simulated_annealing",
            "seed": config.seed,
            "iteration_budget": config.iterations,
            "iterations_completed": iterations_completed,
            "time_budget_seconds": config.time_budget_seconds,
            "stopped_by_time_budget": _budget_expired(
                started, config.time_budget_seconds
            ),
            "evaluations": len(cache),
            "best_order": best.candidate.task_order,
            "best_robot_order": best.candidate.robot_order,
            "best_quality": best.quality.as_tuple(),
            "final_temperature": temperature,
            "uses_unified_simulator": True,
            "infeasible_candidate_policy": "finite penalty after canonical failure",
            "tie_break": "quality_then_lexicographic_order",
        },
    )


def _evaluate_candidate(
    domain: SchedulingDomain,
    exit_location: tuple[float, float],
    candidate: SearchCandidate,
    *,
    instance_id: str,
    split: DatasetSplit,
    max_steps: int,
    seed: int,
) -> CandidateEvaluation:
    expected_tasks = {task.task_id for task in domain.tasks}
    expected_robots = {robot.robot_id for robot in domain.robots}
    if (
        len(candidate.task_order) != len(expected_tasks)
        or set(candidate.task_order) != expected_tasks
    ):
        raise ValueError("candidate order must be a permutation of all task IDs")
    if (
        len(candidate.robot_order) != len(expected_robots)
        or set(candidate.robot_order) != expected_robots
    ):
        raise ValueError("robot order must be a permutation of all robot IDs")

    simulator = MDDiscreteSimulator(domain, exit_location=exit_location)
    actions: list[dict[str, int]] = []
    failure_reason: FailureReason | None = None
    steps = 0
    while not simulator.done and steps < max_steps:
        _assign_in_priority_order(
            simulator, candidate.task_order, candidate.robot_order, actions
        )
        if not simulator.has_advancing_work:
            failure_reason = FailureReason.DEADLOCK
            break
        simulator.step()
        steps += 1
    if not simulator.done and failure_reason is None:
        failure_reason = FailureReason.TIMEOUT

    result = simulator.build_experiment_result(
        run_id=(
            f"candidate-{seed}-{'-'.join(map(str, candidate.task_order))}"
            f"-robots-{'-'.join(map(str, candidate.robot_order))}"
        ),
        method="metaheuristic_candidate",
        instance_id=instance_id,
        seed=seed,
        split=split,
        failure_reason=failure_reason,
        metadata={
            "candidate_order": candidate.task_order,
            "candidate_robot_order": candidate.robot_order,
            "assignment_order": actions,
            "uses_hard_feasibility": True,
        },
    )
    incomplete = sum(
        state.status is not TaskStatus.COMPLETE
        for state in simulator.task_states.values()
    )
    quality = CandidateQuality(
        failure_rank=0 if result.success else 1,
        objective=float(
            result.makespan
            if result.makespan is not None
            else max_steps + incomplete
        ),
        incomplete_tasks=incomplete,
    )
    return CandidateEvaluation(candidate, quality, result)


def _assign_in_priority_order(
    simulator: MDDiscreteSimulator,
    order: tuple[int, ...],
    robot_order: tuple[int, ...],
    actions: list[dict[str, int]],
) -> None:
    for task_id in order:
        while True:
            selected = next(
                (
                    robot_id
                    for robot_id in robot_order
                    if simulator.assignment_feasibility(
                        robot_id=robot_id, task_id=task_id
                    ).is_feasible
                ),
                None,
            )
            if selected is None:
                break
            simulator.assign(robot_id=selected, task_id=task_id)
            actions.append(
                {
                    "order": len(actions),
                    "time": simulator.time,
                    "robot_id": selected,
                    "task_id": task_id,
                }
            )




def _finalize(
    evaluation: CandidateEvaluation,
    *,
    run_id: str,
    method: str,
    seed: int,
    split: DatasetSplit,
    wall_time_seconds: float,
    metadata: dict[str, Any],
) -> ExperimentResult:
    combined = dict(evaluation.result.metadata)
    combined.update(metadata)
    return replace(
        evaluation.result,
        run_id=run_id,
        method=method,
        seed=seed,
        split=split,
        wall_time_seconds=wall_time_seconds,
        metadata=combined,
    )


def _crossover_candidate(
    left: SearchCandidate,
    right: SearchCandidate,
    rng: random.Random,
) -> SearchCandidate:
    return SearchCandidate(
        _ordered_crossover(left.task_order, right.task_order, rng),
        _ordered_crossover(left.robot_order, right.robot_order, rng),
    )


def _mutate_candidate(
    candidate: SearchCandidate, rng: random.Random
) -> SearchCandidate:
    mutable_task_order = len(candidate.task_order) >= 2
    mutable_robot_order = len(candidate.robot_order) >= 2
    if mutable_robot_order and (not mutable_task_order or rng.random() < 0.5):
        return SearchCandidate(
            candidate.task_order,
            _swap_neighbor(candidate.robot_order, rng),
        )
    return SearchCandidate(
        _swap_neighbor(candidate.task_order, rng),
        candidate.robot_order,
    )


def _ordered_crossover(
    left: tuple[int, ...], right: tuple[int, ...], rng: random.Random
) -> tuple[int, ...]:
    if len(left) < 2:
        return left
    start, stop = sorted(rng.sample(range(len(left)), 2))
    stop += 1
    child: list[int | None] = [None] * len(left)
    child[start:stop] = left[start:stop]
    remaining = (item for item in right if item not in child)
    for index, item in enumerate(child):
        if item is None:
            child[index] = next(remaining)
    return tuple(item for item in child if item is not None)


def _swap_neighbor(order: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    if len(order) < 2:
        return order
    left, right = rng.sample(range(len(order)), 2)
    candidate = list(order)
    candidate[left], candidate[right] = candidate[right], candidate[left]
    return tuple(candidate)


def _evaluation_key(
    evaluation: CandidateEvaluation,
) -> tuple[CandidateQuality, tuple[int, ...], tuple[int, ...]]:
    return (
        evaluation.quality,
        evaluation.candidate.task_order,
        evaluation.candidate.robot_order,
    )


def _annealing_delta(
    candidate: CandidateQuality, current: CandidateQuality
) -> float:
    if candidate.failure_rank != current.failure_rank:
        return -math.inf if candidate.failure_rank < current.failure_rank else math.inf
    if candidate.objective != current.objective:
        return candidate.objective - current.objective
    return float(candidate.incomplete_tasks - current.incomplete_tasks)


def _budget_expired(started: float, budget: float | None) -> bool:
    return budget is not None and time.perf_counter() - started >= budget


def _positive_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _probability(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be in [0, 1]")


def _seed(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("seed must be a non-negative integer")


def _optional_time_budget(value: float | None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("time_budget_seconds must be positive and finite")


def _max_steps(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_steps must be a non-negative integer")


__all__ = [
    "CandidateEvaluation",
    "CandidateQuality",
    "GeneticAlgorithmConfig",
    "SearchCandidate",
    "SimulatedAnnealingConfig",
    "run_genetic_algorithm",
    "run_simulated_annealing",
]
