"""Default solver-free online MD scheduling with explicit MIP fallback."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import pulp
import torch

from experiments.protocol import DatasetSplit, ExperimentResult, FailureReason
from schedulers.md_constrained_decoder import (
    DecoderResult,
    FastAssignmentRepair,
    LearnedConstrainedDecoder,
    apply_decoder_result,
    simulator_hard_mask,
)
from simulation_environment.domain_model import (
    ProcessRobot,
    ProcessTask,
    TransportRobot,
    TransportTask,
)
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from simulation_environment.transport_timing import travel_duration


@dataclass(frozen=True, slots=True)
class ScoreOutput:
    scores: torch.Tensor
    encoder_time_seconds: float
    scoring_time_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.scores, torch.Tensor):
            raise TypeError("scores must be a torch.Tensor")
        for name in ("encoder_time_seconds", "scoring_time_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")


class ScoreProvider(Protocol):
    def __call__(
        self, simulator: MDDiscreteSimulator
    ) -> torch.Tensor | ScoreOutput: ...


class OnlineNeuralScoreProvider:
    """Run a trained MD policy on canonical simulator-derived tensors."""

    def __init__(self, model, input_builder, *, device="cpu"):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model.to(torch.device(device)).eval()
        self.input_builder = input_builder
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, simulator: MDDiscreteSimulator) -> ScoreOutput:
        feature_started = time.perf_counter()
        robot_features, task_features, adjacency, md_inputs = self.input_builder(
            simulator
        )
        encoder_elapsed = time.perf_counter() - feature_started

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        scoring_started = time.perf_counter()
        scores = self.model(
            robot_features.to(self.device),
            task_features.to(self.device),
            adjacency.to(self.device),
            md_inputs=md_inputs.to(self.device),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        scoring_elapsed = time.perf_counter() - scoring_started
        if not isinstance(scores, torch.Tensor) or not bool(torch.isfinite(scores).all()):
            raise ValueError("neural policy produced non-finite scores")
        if scores.ndim != 3 or scores.shape[0] != 1:
            raise ValueError("neural policy must return one robot-by-task score matrix")
        return ScoreOutput(
            scores[0].detach().cpu(), encoder_elapsed, scoring_elapsed
        )


class TerminalReturnAwareScoreProvider:
    """Apply a fixed terminal-return penalty to an existing score provider."""

    def __init__(
        self,
        scorer: ScoreProvider,
        *,
        weight: float = 0.01,
        exit_location: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be non-negative and finite")
        self.scorer = scorer
        self.weight = float(weight)
        self.exit_location = exit_location

    def __call__(self, simulator: MDDiscreteSimulator) -> ScoreOutput | torch.Tensor:
        output = self.scorer(simulator)
        scores = output.scores if isinstance(output, ScoreOutput) else output
        robot_ids = tuple(sorted(simulator.robot_states))
        task_ids = tuple(sorted(simulator.task_states))
        robots = {robot.robot_id: robot for robot in simulator.domain.robots}
        tasks = {task.task_id: task for task in simulator.domain.tasks}
        penalty = torch.zeros_like(scores)
        for robot_index, robot_id in enumerate(robot_ids):
            robot = robots[robot_id]
            speed = robot.speed if isinstance(robot, ProcessRobot) else robot.unloaded_speed
            return_target = (
                robot.home_location
                if isinstance(robot, ProcessRobot)
                else self.exit_location
            )
            for task_index, task_id in enumerate(task_ids):
                task = tasks[task_id]
                location = task.delivery_location if isinstance(task, TransportTask) else task.location
                penalty[robot_index, task_index] = self.weight * travel_duration(
                    location, return_target, speed
                )
        adjusted = scores - penalty
        if isinstance(output, ScoreOutput):
            return ScoreOutput(adjusted, output.encoder_time_seconds, output.scoring_time_seconds)
        return adjusted


@dataclass(frozen=True, slots=True)
class FallbackOutcome:
    result: DecoderResult | None
    solver_time_seconds: float
    failure_reason: str | None

    @classmethod
    def succeeded(
        cls, result: DecoderResult, solver_time_seconds: float
    ) -> "FallbackOutcome":
        return cls(result, solver_time_seconds, None)

    @classmethod
    def failed(
        cls, reason: str, solver_time_seconds: float
    ) -> "FallbackOutcome":
        return cls(None, solver_time_seconds, reason)


@dataclass(frozen=True, slots=True)
class FallbackRecord:
    decision_time: int
    trigger_reason: str
    solver_time_seconds: float
    assignments: tuple[tuple[int, int], ...]
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class OnlineDecisionRecord:
    decision_time: int
    assignments: tuple[tuple[int, int], ...]
    confidence: float | None
    encoder_time_seconds: float
    scoring_time_seconds: float
    decoder_time_seconds: float
    repair_time_seconds: float
    fallback_time_seconds: float
    total_time_seconds: float
    fallback_used: bool
    repaired: bool = False
    repair_event_count: int = 0


@dataclass(frozen=True, slots=True)
class OnlineSchedulerResult:
    experiment: ExperimentResult
    decision_records: tuple[OnlineDecisionRecord, ...]
    fallback_records: tuple[FallbackRecord, ...]
    failure_reason: str | None

    @property
    def fallback_call_count(self) -> int:
        return len(self.fallback_records)


class ExplicitMIPFallback:
    """Exact coalition-aware MIP invoked only through this explicit object."""

    def __init__(self, *, time_limit_seconds: float = 5.0, threads: int = 1):
        if time_limit_seconds <= 0 or not math.isfinite(time_limit_seconds):
            raise ValueError("time_limit_seconds must be positive and finite")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
            raise ValueError("threads must be a positive integer")
        self.time_limit_seconds = float(time_limit_seconds)
        self.threads = threads

    def __call__(
        self, scores: torch.Tensor, simulator: MDDiscreteSimulator
    ) -> FallbackOutcome:
        started = time.perf_counter()
        robot_ids = tuple(sorted(simulator.robot_states))
        task_ids = tuple(sorted(simulator.task_states))
        hard_mask = simulator_hard_mask(simulator)
        if tuple(scores.shape) != tuple(hard_mask.shape):
            return FallbackOutcome.failed(
                "mip_score_shape_mismatch", time.perf_counter() - started
            )
        problem = pulp.LpProblem("ExplicitMIPFallback", pulp.LpMaximize)
        variables = {
            (robot_id, task_id): pulp.LpVariable(
                f"a_{robot_id}_{task_id}", cat=pulp.LpBinary
            )
            for robot_id in robot_ids
            for task_id in task_ids
        }
        task_active = {
            task_id: pulp.LpVariable(f"x_{task_id}", cat=pulp.LpBinary)
            for task_id in task_ids
        }
        robot_index = {value: index for index, value in enumerate(robot_ids)}
        task_index = {value: index for index, value in enumerate(task_ids)}
        robots = {robot.robot_id: robot for robot in simulator.domain.robots}
        tasks = {task.task_id: task for task in simulator.domain.tasks}

        for robot_id in robot_ids:
            problem += pulp.lpSum(
                variables[robot_id, task_id] for task_id in task_ids
            ) <= 1
        for robot_id in robot_ids:
            for task_id in task_ids:
                if not hard_mask[robot_index[robot_id], task_index[task_id]]:
                    problem += variables[robot_id, task_id] == 0
        for task_id in task_ids:
            task = tasks[task_id]
            assigned = pulp.lpSum(
                variables[robot_id, task_id] for robot_id in robot_ids
            )
            if isinstance(task, TransportTask):
                problem += assigned == task_active[task_id]
                continue
            assert isinstance(task, ProcessTask)
            problem += assigned >= task_active[task_id]
            for robot_id in robot_ids:
                problem += variables[robot_id, task_id] <= task_active[task_id]
            for skill, required in enumerate(task.requirements):
                if required:
                    problem += pulp.lpSum(
                        int(
                            isinstance(robots[robot_id], ProcessRobot)
                            and robots[robot_id].capabilities[skill]
                        )
                        * variables[robot_id, task_id]
                        for robot_id in robot_ids
                    ) >= task_active[task_id]
        if torch.any(hard_mask):
            problem += pulp.lpSum(task_active.values()) >= 1
        problem += pulp.lpSum(
            (float(scores[robot_index[robot_id], task_index[task_id]]) + 1e-6)
            * variables[robot_id, task_id]
            for robot_id in robot_ids
            for task_id in task_ids
        )
        problem.solve(
            pulp.PULP_CBC_CMD(
                msg=False,
                timeLimit=self.time_limit_seconds,
                threads=self.threads,
            )
        )
        elapsed = time.perf_counter() - started
        status = pulp.LpStatus.get(problem.status, "Unknown")
        if status not in {"Optimal", "Feasible"}:
            return FallbackOutcome.failed(
                f"mip_{status.lower().replace(' ', '_')}", elapsed
            )
        proposal = tuple(
            (robot_id, task_id)
            for robot_id in robot_ids
            for task_id in task_ids
            if (pulp.value(variables[robot_id, task_id]) or 0.0) > 0.5
        )
        if not proposal:
            return FallbackOutcome.failed("mip_no_incumbent", elapsed)
        repaired = FastAssignmentRepair().repair(
            proposal, scores, simulator, hard_feasibility_mask=hard_mask
        )
        if not repaired.assignments:
            return FallbackOutcome.failed("mip_repair_failed", elapsed)
        return FallbackOutcome.succeeded(repaired, elapsed)


def _has_dispatchable_assignment(
    simulator: MDDiscreteSimulator, hard_mask: torch.Tensor
) -> bool:
    """Return whether the current state admits at least one complete action.

    The canonical hard mask is pairwise: a PROCESS edge can be legal even when
    the available robots cannot yet form a complete skill coalition. Avoiding
    a neural forward is safe only when neither a transport assignment nor a
    complete process coalition can be applied.
    """

    robot_ids = tuple(sorted(simulator.robot_states))
    task_ids = tuple(sorted(simulator.task_states))
    robots = {robot.robot_id: robot for robot in simulator.domain.robots}
    tasks = {task.task_id: task for task in simulator.domain.tasks}
    for task_index, task_id in enumerate(task_ids):
        candidates = tuple(
            robot_id
            for robot_index, robot_id in enumerate(robot_ids)
            if bool(hard_mask[robot_index, task_index])
        )
        if not candidates:
            continue
        task = tasks[task_id]
        if isinstance(task, TransportTask):
            return True
        assert isinstance(task, ProcessTask)
        covered = [False] * len(task.requirements)
        for robot_id in simulator.task_states[task_id].assigned_robot_ids:
            robot = robots[robot_id]
            if isinstance(robot, ProcessRobot):
                covered = [
                    left or right for left, right in zip(covered, robot.capabilities)
                ]
        for robot_id in candidates:
            robot = robots[robot_id]
            assert isinstance(robot, ProcessRobot)
            covered = [
                left or right for left, right in zip(covered, robot.capabilities)
            ]
        if not any(task.requirements) or all(
            not required or is_covered
            for required, is_covered in zip(task.requirements, covered)
        ):
            return True
    return False


class OnlineMDScheduler:
    def __init__(
        self,
        *,
        scorer: ScoreProvider,
        decoder: LearnedConstrainedDecoder | None = None,
        fallback: Callable[[torch.Tensor, MDDiscreteSimulator], FallbackOutcome]
        | None = None,
        confidence_threshold: float | None = None,
        max_steps: int = 10_000,
        skip_non_dispatchable_states: bool = False,
    ):
        self.scorer = scorer
        self.decoder = LearnedConstrainedDecoder() if decoder is None else decoder
        self.fallback = fallback
        if confidence_threshold is not None and (
            not math.isfinite(confidence_threshold) or confidence_threshold < 0
        ):
            raise ValueError("confidence_threshold must be non-negative and finite")
        self.confidence_threshold = confidence_threshold
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        if not isinstance(skip_non_dispatchable_states, bool):
            raise TypeError("skip_non_dispatchable_states must be boolean")
        self.skip_non_dispatchable_states = skip_non_dispatchable_states

    def run(
        self,
        domain,
        *,
        run_id: str,
        instance_id: str,
        seed: int,
        split: DatasetSplit,
        exit_location: tuple[float, float] = (0.0, 0.0),
    ) -> OnlineSchedulerResult:
        wall_started = time.perf_counter()
        simulator = MDDiscreteSimulator(domain, exit_location=exit_location)
        decisions: list[OnlineDecisionRecord] = []
        fallbacks: list[FallbackRecord] = []
        failure_reason: str | None = None
        steps = 0
        inference_time = 0.0

        while not simulator.done and steps < self.max_steps:
            hard_mask = simulator_hard_mask(simulator)
            has_pairwise_candidate = bool(torch.any(hard_mask))
            should_score = has_pairwise_candidate
            if (
                self.skip_non_dispatchable_states
                and simulator.has_advancing_work
            ):
                should_score = _has_dispatchable_assignment(simulator, hard_mask)
            decision_started = time.perf_counter()
            if should_score:
                scoring_started = time.perf_counter()
                score_output = self.scorer(simulator)
                scoring_elapsed = time.perf_counter() - scoring_started
                if isinstance(score_output, ScoreOutput):
                    scores = score_output.scores
                    encoder_time = score_output.encoder_time_seconds
                    scoring_time = score_output.scoring_time_seconds
                else:
                    scores = score_output
                    encoder_time = 0.0
                    scoring_time = scoring_elapsed
                decoder_started = time.perf_counter()
                decoded = self.decoder.decode(
                    scores, simulator, hard_feasibility_mask=hard_mask
                )
                decoder_elapsed = time.perf_counter() - decoder_started
                repair_time = decoded.repair_time_seconds
                decoder_time = max(0.0, decoder_elapsed - repair_time)
                confidence = _assignment_confidence(
                    scores, hard_mask, decoded.assignments,
                    tuple(sorted(simulator.robot_states)),
                    tuple(sorted(simulator.task_states)),
                )
                wait_for_advancing_work = (
                    not decoded.assignments and simulator.has_advancing_work
                )
                trigger = None
                if not decoded.assignments and not wait_for_advancing_work:
                    trigger = "repair_failed"
                elif (
                    self.confidence_threshold is not None
                    and confidence is not None
                    and confidence < self.confidence_threshold
                ):
                    trigger = "low_confidence"
                fallback_time = 0.0
                fallback_used = False
                if trigger is not None and self.fallback is not None:
                    fallback_used = True
                    outcome = self.fallback(scores, simulator)
                    fallback_time = outcome.solver_time_seconds
                    assignments = (
                        () if outcome.result is None else outcome.result.assignments
                    )
                    fallbacks.append(
                        FallbackRecord(
                            simulator.time,
                            trigger,
                            fallback_time,
                            assignments,
                            outcome.failure_reason,
                        )
                    )
                    if outcome.result is None:
                        failure_reason = outcome.failure_reason or "mip_fallback_failed"
                        total_time = time.perf_counter() - decision_started
                        inference_time += total_time
                        decisions.append(OnlineDecisionRecord(
                            decision_time=simulator.time, assignments=(),
                            confidence=confidence, encoder_time_seconds=encoder_time,
                            scoring_time_seconds=scoring_time, decoder_time_seconds=decoder_time,
                            repair_time_seconds=repair_time, fallback_time_seconds=fallback_time,
                            total_time_seconds=total_time, fallback_used=True,
                            repaired=decoded.repaired,
                            repair_event_count=len(decoded.repair_events),
                        ))
                        break
                    decoded = outcome.result
                if not decoded.assignments:
                    if not wait_for_advancing_work:
                        failure_reason = "decoder_no_assignment"
                        break
                    total_time = time.perf_counter() - decision_started
                    inference_time += total_time
                    decisions.append(
                        OnlineDecisionRecord(
                            decision_time=simulator.time,
                            assignments=(),
                            confidence=confidence,
                            encoder_time_seconds=encoder_time,
                            scoring_time_seconds=scoring_time,
                            decoder_time_seconds=decoder_time,
                            repair_time_seconds=repair_time,
                            fallback_time_seconds=0.0,
                            total_time_seconds=total_time,
                            fallback_used=False,
                            repaired=decoded.repaired,
                            repair_event_count=len(decoded.repair_events),
                        )
                    )
                else:
                    apply_decoder_result(simulator, decoded)
                    total_time = time.perf_counter() - decision_started
                    inference_time += total_time
                    decisions.append(
                        OnlineDecisionRecord(
                            decision_time=simulator.time,
                            assignments=decoded.assignments,
                            confidence=confidence,
                            encoder_time_seconds=encoder_time,
                            scoring_time_seconds=scoring_time,
                            decoder_time_seconds=decoder_time,
                            repair_time_seconds=repair_time,
                            fallback_time_seconds=fallback_time,
                            total_time_seconds=total_time,
                            fallback_used=fallback_used,
                            repaired=decoded.repaired,
                            repair_event_count=len(decoded.repair_events),
                        )
                    )

            if simulator.done:
                break
            if not simulator.has_advancing_work and not has_pairwise_candidate:
                failure_reason = "deadlock"
                break
            simulator.step()
            steps += 1

        if not simulator.done and failure_reason is None:
            failure_reason = "timeout"
        if simulator.done:
            experiment = simulator.build_experiment_result(
                run_id=run_id,
                method="online_md_scheduler",
                instance_id=instance_id,
                seed=seed,
                split=split,
                inference_time_seconds=inference_time,
                wall_time_seconds=time.perf_counter() - wall_started,
                illegal_assignment_count=0,
                metadata={
                    "fallback_call_count": len(fallbacks),
                    "normal_path_uses_mip": False,
                    "hard_mask_source": "hard_feasibility",
                    "terminal_protocol": "canonical_md_simulator",
                    "decoder": self.decoder.name,
                },
            )
        else:
            experiment = simulator.build_experiment_result(
                run_id=run_id,
                method="online_md_scheduler",
                instance_id=instance_id,
                seed=seed,
                split=split,
                failure_reason=(
                    FailureReason.TIMEOUT
                    if failure_reason == "timeout"
                    else FailureReason.SCHEDULER_FAILURE
                ),
                inference_time_seconds=inference_time,
                illegal_assignment_count=0,
                wall_time_seconds=time.perf_counter() - wall_started,
                metadata={
                    "online_failure_reason": failure_reason,
                    "fallback_call_count": len(fallbacks),
                    "normal_path_uses_mip": False,
                    "hard_mask_source": "hard_feasibility",
                    "terminal_protocol": "canonical_md_simulator",
                    "decoder": self.decoder.name,
                },
            )
        return OnlineSchedulerResult(
            experiment=experiment,
            decision_records=tuple(decisions),
            fallback_records=tuple(fallbacks),
            failure_reason=None if simulator.done else failure_reason,
        )


def _assignment_confidence(
    scores: torch.Tensor,
    mask: torch.Tensor,
    assignments: tuple[tuple[int, int], ...],
    robot_ids: tuple[int, ...],
    task_ids: tuple[int, ...],
) -> float | None:
    if not assignments:
        return None
    robot_index = {value: index for index, value in enumerate(robot_ids)}
    task_index = {value: index for index, value in enumerate(task_ids)}
    margins = []
    for robot_id, task_id in assignments:
        row = robot_index[robot_id]
        column = task_index[task_id]
        alternatives = mask[row].clone()
        alternatives[column] = False
        if not torch.any(alternatives):
            margins.append(float("inf"))
            continue
        margins.append(
            float(scores[row, column] - torch.max(scores[row][alternatives]))
        )
    return min(margins)


__all__ = [
    "ExplicitMIPFallback",
    "FallbackOutcome",
    "FallbackRecord",
    "OnlineDecisionRecord",
    "OnlineMDScheduler",
    "OnlineSchedulerResult",
    "ScoreOutput",
    "OnlineNeuralScoreProvider",
    "TerminalReturnAwareScoreProvider",
]
