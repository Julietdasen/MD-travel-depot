"""Generate MILP-labelled residual snapshots from policy rollout states."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from baselines.md_oracle_types import ForcedAssignmentBatch, ResidualMDState, ResidualOracleResult, ResidualOracleStatus, enumerate_forced_batches
from baselines.md_ortools_residual_oracle import solve_residual_forced_batch
from data_generation.md_residual_dataset import (
    LEGACY_RESIDUAL_SPLIT_PLAN,
    ResidualDecisionSample,
    ResidualSplitPlan,
    build_batch_label,
    dump_samples,
    joint_projection_error,
    project_edge_labels,
    split_for_seed,
)
from simulation_environment.domain_model import SchedulingDomain
from simulation_environment.hard_feasibility import TaskStatus
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator, RobotActivity

@dataclass(frozen=True, slots=True)
class SnapshotCandidate:
    instance_id: str
    seed: int
    rollout_method: str
    domain: SchedulingDomain
    simulator: MDDiscreteSimulator
    training_features: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class ResidualExclusion:
    instance_id: str
    seed: int
    split: str
    current_time: int
    rollout_method: str
    reason: str
    batch: tuple[tuple[int, int], ...] | None = None
    attempted_batch_count: int = 0

def eligible_snapshot(simulator: MDDiscreteSimulator) -> bool:
    pending = sum(state.status is TaskStatus.PENDING for state in simulator.task_states.values())
    return all(state.activity is RobotActivity.AVAILABLE for state in simulator.robot_states.values()) and all(state.status is not TaskStatus.IN_PROGRESS for state in simulator.task_states.values()) and 1 <= pending <= 3

def snapshot_key(candidate: SnapshotCandidate) -> tuple[object, ...]:
    simulator = candidate.simulator
    completed = tuple(sorted(tid for tid, state in simulator.task_states.items() if state.status is TaskStatus.COMPLETE))
    pending = tuple(sorted(tid for tid, state in simulator.task_states.items() if state.status is TaskStatus.PENDING))
    locations = tuple((rid, tuple(state.location)) for rid, state in sorted(simulator.robot_states.items()))
    satisfied = tuple(
        edge for edge in candidate.domain.material_edges
        if simulator.task_states[edge[0]].status is TaskStatus.COMPLETE
    )
    return (
        candidate.instance_id,
        candidate.rollout_method,
        completed,
        pending,
        locations,
        satisfied,
    )

def residual_state(candidate: SnapshotCandidate) -> ResidualMDState:
    simulator = candidate.simulator
    completed = tuple(sorted(tid for tid, state in simulator.task_states.items() if state.status is TaskStatus.COMPLETE))
    pending = tuple(sorted(tid for tid, state in simulator.task_states.items() if state.status is TaskStatus.PENDING))
    satisfied = tuple(edge for edge in candidate.domain.material_edges if simulator.task_states[edge[0]].status is TaskStatus.COMPLETE)
    return ResidualMDState(simulator.time, {rid: state.location for rid, state in simulator.robot_states.items()}, completed, pending, satisfied, simulator.exit_location)

def _cached_result(path: Path, batch: ForcedAssignmentBatch) -> ResidualOracleResult | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(tuple(pair) for pair in payload["assignments"]) != batch.assignments:
        raise ValueError(f"batch cache does not match requested assignments: {path}")
    return ResidualOracleResult(
        status=ResidualOracleStatus(payload["status"]),
        makespan=payload.get("makespan"),
        remaining_task_horizon=payload.get("remaining_task_horizon"),
        terminal_return_tail=payload.get("terminal_return_tail"),
        latest_return_robot_id=payload.get("latest_return_robot_id"),
        optimality_gap=payload.get("optimality_gap"),
        solve_time_seconds=float(payload["solve_time_seconds"]),
        message=str(payload.get("message", "")),
    )

def _write_cached_result(
    path: Path,
    batch: ForcedAssignmentBatch,
    result: ResidualOracleResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "assignments": [list(pair) for pair in batch.assignments],
        "status": result.status.value,
        "makespan": result.makespan,
        "remaining_task_horizon": result.remaining_task_horizon,
        "terminal_return_tail": result.terminal_return_tail,
        "latest_return_robot_id": result.latest_return_robot_id,
        "optimality_gap": result.optimality_gap,
        "solve_time_seconds": result.solve_time_seconds,
        "message": result.message,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
def residual_replay_error(
    candidate: SnapshotCandidate,
    result: ResidualOracleResult,
    *,
    max_steps: int = 10_000,
) -> str | None:
    if result.makespan is None or result.remaining_task_horizon is None or result.terminal_return_tail is None:
        return "residual result has no terminal metrics"
    simulator = copy.deepcopy(candidate.simulator)
    pending = list(sorted(result.actions, key=lambda action: action.order))
    steps = 0
    while not simulator.done and steps < max_steps:
        for action in tuple(pending):
            if action.planned_assignment > simulator.time:
                continue
            feasibility = simulator.assignment_feasibility(
                robot_id=action.robot_id,
                task_id=action.task_id,
            )
            if not feasibility.is_feasible:
                return f"illegal replay action {(action.robot_id, action.task_id)} at {simulator.time}: {feasibility.message}"
            simulator.assign(robot_id=action.robot_id, task_id=action.task_id)
            pending.remove(action)
        if simulator.done:
            break
        if not simulator.has_advancing_work:
            if not any(action.planned_assignment > simulator.time for action in pending):
                return "residual replay deadlocked"
        simulator.step()
        steps += 1
    if not simulator.done:
        return "residual replay timed out"
    completions = [
        float(record["completed_at"])
        for record in (
            *simulator.process_execution_records,
            *simulator.transport_execution_records,
        )
        if record.get("completed_at") is not None
    ]
    latest_completion = max(completions)
    expected_completion = candidate.simulator.time + result.remaining_task_horizon
    observed_tail = float(simulator.time) - latest_completion
    comparisons = (
        ("task completion", latest_completion, expected_completion),
        ("terminal tail", observed_tail, result.terminal_return_tail),
        ("makespan", float(simulator.time), result.makespan),
    )
    for name, observed, expected in comparisons:
        if abs(observed - expected) > 1e-6:
            return f"{name} mismatch: replay={observed}, milp={expected}"
    return None

def label_snapshot(
    candidate: SnapshotCandidate,
    *,
    solver: Callable[..., ResidualOracleResult] = solve_residual_forced_batch,
    split_plan: ResidualSplitPlan = LEGACY_RESIDUAL_SPLIT_PLAN,
    time_limit_seconds: float = 10.0,
    threads: int = 1,
    batch_cache_dir: str | Path | None = None,
) -> ResidualDecisionSample | ResidualExclusion:
    split = split_for_seed(candidate.seed, split_plan)
    state = residual_state(candidate)
    batches = enumerate_forced_batches(candidate.domain, state)
    if not batches:
        return ResidualExclusion(candidate.instance_id, candidate.seed, split, candidate.simulator.time, candidate.rollout_method, "no_legal_batch")
    solved = []
    cache_dir = None if batch_cache_dir is None else Path(batch_cache_dir)
    for batch_index, batch in enumerate(batches):
        cache_path = None if cache_dir is None else cache_dir / f"batch_{batch_index:05d}.json"
        result = None if cache_path is None else _cached_result(cache_path, batch)
        if result is None:
            result = solver(
                candidate.domain,
                state,
                batch,
                time_limit_seconds=time_limit_seconds,
                threads=threads,
            )
            if result.status is not ResidualOracleStatus.OPTIMAL:
                if cache_path is not None:
                    _write_cached_result(cache_path, batch, result)
                return ResidualExclusion(candidate.instance_id, candidate.seed, split, candidate.simulator.time, candidate.rollout_method, result.status.value, batch.assignments, batch_index + 1)
            replay_error = residual_replay_error(candidate, result)
            if replay_error is not None:
                return ResidualExclusion(candidate.instance_id, candidate.seed, split, candidate.simulator.time, candidate.rollout_method, "schedule_replay_mismatch", batch.assignments, batch_index + 1)
            if cache_path is not None:
                _write_cached_result(cache_path, batch, result)
        elif result.status is not ResidualOracleStatus.OPTIMAL:
            return ResidualExclusion(candidate.instance_id, candidate.seed, split, candidate.simulator.time, candidate.rollout_method, result.status.value, batch.assignments, batch_index + 1)
        solved.append((batch, result))
    best_total = min(result.remaining_task_horizon + result.terminal_return_tail for _, result in solved if result.remaining_task_horizon is not None and result.terminal_return_tail is not None)
    labels = tuple(build_batch_label(batch, result, best_total) for batch, result in solved)
    edges = project_edge_labels(labels)
    snapshot = {**candidate.training_features, "rollout_method": candidate.rollout_method, "current_time": candidate.simulator.time, "robot_states": {str(rid): {**asdict(runtime), "activity": runtime.activity.value} for rid, runtime in candidate.simulator.robot_states.items()}, "task_states": {str(tid): {**asdict(runtime), "status": runtime.status.value, "transport_phase": None if runtime.transport_phase is None else runtime.transport_phase.value, "assigned_robot_ids": sorted(runtime.assigned_robot_ids)} for tid, runtime in candidate.simulator.task_states.items()}}
    legal_mask = tuple(tuple(bool(value) for value in row) for row in candidate.training_features["md_inputs"]["hard_feasibility_mask"])
    return ResidualDecisionSample(
        candidate.instance_id,
        candidate.seed,
        split,
        state,
        labels,
        edges,
        snapshot,
        legal_mask,
        joint_projection_error(labels, edges),
        len(labels),
    )

def generate_residual_dataset(
    candidates: Sequence[SnapshotCandidate],
    output_dir: str | Path,
    *,
    solver: Callable[..., ResidualOracleResult] = solve_residual_forced_batch,
    split_plan: ResidualSplitPlan = LEGACY_RESIDUAL_SPLIT_PLAN,
    time_limit_seconds: float = 10.0,
    threads: int = 1,
) -> Mapping[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[tuple[object, ...]] = set()
    samples, exclusions = [], []
    instance_splits: dict[str, str] = {}
    for candidate in candidates:
        split = split_for_seed(candidate.seed, split_plan)
        previous = instance_splits.setdefault(candidate.instance_id, split)
        if previous != split:
            raise ValueError(f"instance {candidate.instance_id} occurs in multiple splits")
        key = snapshot_key(candidate)
        if not eligible_snapshot(candidate.simulator) or key in seen:
            continue
        seen.add(key)
        result = label_snapshot(
            candidate,
            solver=solver,
            split_plan=split_plan,
            time_limit_seconds=time_limit_seconds,
            threads=threads,
        )
        (exclusions if isinstance(result, ResidualExclusion) else samples).append(result)
    paths = {name: destination / f"{name}.jsonl" for name in ("train", "development", "test")}
    for split, path in paths.items():
        dump_samples(tuple(sample for sample in samples if sample.split == split), path)
    exclusion_path = destination / "exclusions.jsonl"
    exclusion_path.write_text("".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in exclusions), encoding="utf-8")
    pending_distribution = {
        str(count): sum(len(sample.state.pending_task_ids) == count for sample in samples)
        for count in (1, 2, 3)
    }
    rollout_sources = {
        method: sum(sample.simulator_snapshot.get("rollout_method") == method for sample in samples)
        for method in sorted({candidate.rollout_method for candidate in candidates})
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "residual-il-2.0",
        "split_plan": split_plan.to_dict(),
        "sample_count": len(samples),
        "samples_by_split": {name: sum(sample.split == name for sample in samples) for name in paths},
        "exclusion_count": len(exclusions),
        "exclusions_by_reason": {reason: sum(row.reason == reason for row in exclusions) for reason in sorted({row.reason for row in exclusions})},
        "instance_count": len(instance_splits),
        "instance_splits": instance_splits,
        "forced_solve_count": sum(sample.batch_count for sample in samples) + sum(row.attempted_batch_count for row in exclusions),
        "optimal_solve_count": sum(sample.batch_count for sample in samples),
        "pending_count_distribution": pending_distribution,
        "rollout_sources": rollout_sources,
        "joint_projection_error_mean": sum(sample.joint_projection_error for sample in samples) / len(samples) if samples else 0.0,
        "solver": {"threads": threads, "time_limit_seconds": time_limit_seconds},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**paths, "exclusions": exclusion_path, "manifest": manifest_path}

def collect_rollout_snapshots(instance_id: str, seed: int, rollout_method: str, domain: SchedulingDomain, dispatch: Callable[[MDDiscreteSimulator], object], encode_features: Callable[[MDDiscreteSimulator], Mapping[str, Any]], *, max_steps: int = 10_000) -> tuple[SnapshotCandidate, ...]:
    """Collect immutable eligible states while a supplied policy drives a rollout."""
    simulator = MDDiscreteSimulator(domain)
    snapshots = []
    steps = 0
    while not simulator.done and steps < max_steps:
        if eligible_snapshot(simulator):
            frozen = copy.deepcopy(simulator)
            snapshots.append(SnapshotCandidate(instance_id, seed, rollout_method, domain, frozen, encode_features(frozen)))
        dispatch(simulator)
        if not simulator.has_advancing_work:
            break
        simulator.step()
        steps += 1
    return tuple(snapshots)
