"""Ticket 48 exact-action and candidate-coverage diagnostic."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from baselines.exact_online_action_oracle import (
    SCHEMA, CompleteOnlineAction, enumerate_complete_online_actions,
    label_to_json, solve_exact_action_continuation,
)
from baselines.gurobi_md_residual_oracle import (
    ForcedAssignmentBatch, ResidualOracleStatus, solve_residual_forced_batch,
)
from data_generation.md_residual_generation import SnapshotCandidate, eligible_snapshot, residual_state
from data_generation.md_residual_pipeline import collect_seed_snapshots
from imitation_learning.md_residual_train import load_residual_tail_checkpoint
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.md_constrained_decoder import LearnedConstrainedDecoder, MaskedGreedyDecoder, simulator_hard_mask
from schedulers.online_md_scheduler import OnlineNeuralScoreProvider
from simulation_environment.domain_model import TransportTask

SELECTION_SEED = 20260906
MODEL_SEEDS = (3101, 3102, 3103)
K_VALUES = (1, 2, 4, 8, 16)
OUTPUT_DIR = Path("reports/md_exact_action_candidate_diagnostic_2026-09-06")
C0 = {s: Path(f"reports/md_c0_end_to_end_diagnostic_pilot_2026-09-01/training/C0_seed{s}/checkpoints/best_checkpoint.pt") for s in MODEL_SEEDS}
RESIDUAL = {s: Path(f"reports/md_residual_scale10_2026-09-04/training/seed{s}/C0_residual_tail_seed{s}/best_checkpoint.pt") for s in MODEL_SEEDS}


def _state_key(candidate: SnapshotCandidate) -> tuple[object, ...]:
    state = residual_state(candidate)
    return (candidate.seed, state.current_time, state.completed_task_ids, state.pending_task_ids,
            tuple(sorted(state.robot_locations.items())), state.satisfied_material_dependencies)


def _quotas(split: str, pending: int) -> Mapping[str, int]:
    rotations = ((9, 8, 8), (8, 9, 8), (8, 8, 9)) if split == "train" else ((4, 3, 3), (3, 4, 3), (3, 3, 4))
    return {"masked_greedy": 25 if split == "train" else 10,
            **{f"legacy_c0_seed{s}": n for s, n in zip(MODEL_SEEDS, rotations[pending - 1], strict=True)}}


def select_snapshots(candidates: Sequence[SnapshotCandidate], split: str) -> tuple[SnapshotCandidate, ...]:
    rng = random.Random(SELECTION_SEED + (split == "development"))
    buckets: dict[tuple[int, str], list[SnapshotCandidate]] = {}
    for candidate in candidates:
        if eligible_snapshot(candidate.simulator):
            pending = len(residual_state(candidate).pending_task_ids)
            if pending in (1, 2, 3):
                buckets.setdefault((pending, candidate.rollout_method), []).append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (item.seed, item.simulator.time)); rng.shuffle(bucket)
    selected: list[SnapshotCandidate] = []; seen: set[tuple[object, ...]] = set()
    for pending in (1, 2, 3):
        for source, quota in _quotas(split, pending).items():
            accepted = 0
            for candidate in buckets.get((pending, source), ()):
                if _state_key(candidate) in seen: continue
                selected.append(candidate); seen.add(_state_key(candidate)); accepted += 1
                if accepted == quota: break
            if accepted != quota:
                raise RuntimeError(f"quota unavailable: {split=} {pending=} {source=} required={quota} observed={accepted}")
    return tuple(selected)


def _score(action: CompleteOnlineAction, scores: torch.Tensor, robots: Mapping[int, int], tasks: Mapping[int, int]) -> float:
    return float(sum(scores[robots[r], tasks[t]].item() for r, t in action.assignments))


def _atoms(action: CompleteOnlineAction) -> frozenset[tuple[int, tuple[int, ...]]]:
    grouped: dict[int, list[int]] = {}
    for robot, task in action.assignments: grouped.setdefault(task, []).append(robot)
    return frozenset((task, tuple(sorted(robots))) for task, robots in grouped.items())


def generate_candidates(candidate: SnapshotCandidate, legal: Sequence[CompleteOnlineAction], scores: torch.Tensor) -> tuple[CompleteOnlineAction, ...]:
    """Generate candidates without receiving or reading oracle values."""
    simulator = candidate.simulator; mask = simulator_hard_mask(simulator)
    robot_index = {value: index for index, value in enumerate(sorted(simulator.robot_states))}
    task_index = {value: index for index, value in enumerate(sorted(simulator.task_states))}
    legal_by_key = {action.key: action for action in legal}
    residual = LearnedConstrainedDecoder().decode(scores, simulator, hard_feasibility_mask=mask)
    greedy = MaskedGreedyDecoder().decode(torch.zeros_like(scores), simulator, hard_feasibility_mask=mask)
    ranked = sorted(legal, key=lambda action: (-_score(action, scores, robot_index, task_index), action.key))
    keys = {tuple(sorted(residual.assignments)), tuple(sorted(greedy.assignments))} & set(legal_by_key)
    keys.update(action.key for action in ranked[:16])
    base = legal_by_key.get(tuple(sorted(residual.assignments)))
    if base:
        keys.update(action.key for action in legal if len(_atoms(base).symmetric_difference(_atoms(action))) <= 2)
    return tuple(sorted((legal_by_key[key] for key in keys), key=lambda action: (-_score(action, scores, robot_index, task_index), action.key)))


def _cache_path(split: str, candidate: SnapshotCandidate) -> Path:
    return OUTPUT_DIR / "cache" / split / f"{candidate.seed}_{candidate.rollout_method}_t{candidate.simulator.time}.json"


def solve_snapshot(candidate: SnapshotCandidate, split: str) -> dict[str, Any]:
    state = residual_state(candidate); actions = enumerate_complete_online_actions(candidate.domain, state)
    path = _cache_path(split, candidate)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = tuple(tuple(tuple(pair) for pair in row["complete_first_action"]["assignments"]) for row in payload.get("labels", ()))
        if payload.get("schema") != SCHEMA or keys != tuple(action.key for action in actions): raise ValueError(f"cache identity/schema mismatch: {path}")
        if any(not row.get("schedule") or row.get("replay_audit") is None for row in payload["labels"]): raise ValueError(f"cache missing replay payload: {path}")
        return payload
    exact = []; old = []
    for action in actions:
        result = solve_exact_action_continuation(candidate.domain, state, action, time_limit_seconds=10, threads=1)
        forced = solve_residual_forced_batch(candidate.domain, state, ForcedAssignmentBatch(action.assignments), time_limit_seconds=10, threads=1)
        if result.status is not ResidualOracleStatus.OPTIMAL or forced.status is not ResidualOracleStatus.OPTIMAL: raise RuntimeError(f"non-optimal solve: exact={result.status.value}, old={forced.status.value}")
        exact.append((action, result)); old.append(forced)
    exact_min = min(result.continuation_cost for _, result in exact if result.continuation_cost is not None)
    old_costs = [result.makespan - state.current_time for result in old if result.makespan is not None]; old_min = min(old_costs)
    labels = []
    for (action, result), forced, old_cost in zip(exact, old, old_costs, strict=True):
        row = label_to_json(result, action, state_min_continuation_cost=exact_min, pending_task_ids=state.pending_task_ids)
        first_time = min(item.planned_assignment for item in forced.actions)
        old_first = tuple(sorted((item.robot_id, item.task_id) for item in forced.actions if item.planned_assignment == first_time))
        if result.continuation_cost is None:
            raise RuntimeError("optimal exact result lacks continuation cost")
        row.update(old_forced_continuation_cost=old_cost, old_forced_regret=old_cost-old_min,
                   old_forced_tolerance_optimal=old_cost-old_min <= 1, first_event_mismatch=old_first != action.assignments,
                   q_optimism_bias=result.continuation_cost-old_cost)
        labels.append(row)
    payload = {"schema": SCHEMA, "instance_id": candidate.instance_id, "seed": candidate.seed, "split": split,
               "rollout_method": candidate.rollout_method, "current_time": state.current_time, "pending_count": len(state.pending_task_ids), "labels": labels}
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True)+"\n", encoding="utf-8"); temporary.replace(path)
    return payload


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in rows), encoding="utf-8")


def _stratified_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    dimensions = ("model_seed", "pending_count", "action_cardinality", "action_type")
    for dimension in dimensions:
        for value in sorted({row[dimension] for row in records}, key=str):
            group = [row for row in records if row[dimension] == value]
            result[f"{dimension}:{value}"] = {
                "count": len(group),
                "recall_at_k": {
                    str(k): sum(bool(row["recall_at_k"][str(k)]) for row in group) / len(group)
                    for k in K_VALUES
                },
                "mean_value_gap_at_k": {
                    str(k): sum(float(row["value_gap_at_k"][str(k)]) for row in group) / len(group)
                    for k in K_VALUES
                },
                "mean_candidate_count": sum(int(row["candidate_count"]) for row in group) / len(group),
                "mean_complete_action_count": sum(int(row["complete_action_count"]) for row in group) / len(group),
                "mean_generation_time_seconds": sum(float(row["generation_time_seconds"]) for row in group) / len(group),
            }
    return result


def run(splits: Sequence[str]) -> Mapping[str, Any]:
    random.seed(SELECTION_SEED); np.random.seed(SELECTION_SEED); torch.manual_seed(SELECTION_SEED); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scorers = {}
    for seed, checkpoint in RESIDUAL.items():
        model, _ = load_residual_tail_checkpoint(checkpoint, device="cpu")
        scorers[seed] = OnlineNeuralScoreProvider(model, build_md_policy_inputs_from_simulator, device="cpu")
    selected: dict[str, tuple[SnapshotCandidate, ...]] = {}
    for split in splits:
        candidates: list[SnapshotCandidate] = []
        for seed in (range(75000, 75600) if split == "train" else range(75600, 75750)):
            candidates.extend(collect_seed_snapshots(seed, c0_checkpoints=C0, device="cpu", max_steps=10_000))
        selected[split] = select_snapshots(candidates, split)
    package: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for split, snapshots in selected.items():
        for candidate in snapshots:
            try:
                snapshot = solve_snapshot(candidate, split); state = residual_state(candidate)
                snapshot_id = f"{candidate.instance_id}:{candidate.rollout_method}:t{state.current_time}"
                labels = snapshot.get("labels")
                if not isinstance(labels, list):
                    raise ValueError("snapshot cache labels must be a list")
                package.extend({"snapshot_id": snapshot_id, "instance_id": candidate.instance_id, "seed": candidate.seed, "split": split,
                                "rollout_method": candidate.rollout_method, "current_time": state.current_time, **label} for label in labels)
                legal = enumerate_complete_online_actions(candidate.domain, state)
                regret = {tuple(tuple(pair) for pair in label["complete_first_action"]["assignments"]): float(label["regret"]) for label in labels}
                optimal = {key for key, value in regret.items() if value <= 1}
                tasks = {task.task_id: task for task in candidate.domain.tasks}
                for model_seed, scorer in scorers.items():
                    started = time.perf_counter(); generated = generate_candidates(candidate, legal, scorer(candidate.simulator).scores); elapsed = time.perf_counter()-started
                    recalls = {str(k): bool(optimal.intersection(action.key for action in generated[:k])) for k in K_VALUES}
                    gaps = {str(k): min((regret[action.key] for action in generated[:k]), default=None) for k in K_VALUES}
                    best = generated[0] if generated else None
                    kinds = set() if best is None else {"transport" if isinstance(tasks[t], TransportTask) else "process" for _, t in best.assignments}
                    records.append({"snapshot_id": snapshot_id, "instance_id": candidate.instance_id, "seed": candidate.seed, "split": split,
                                    "rollout_method": candidate.rollout_method, "current_time": state.current_time,
                                    "model_seed": model_seed, "pending_count": len(state.pending_task_ids), "action_cardinality": 0 if best is None else len(best.assignments),
                                    "action_type": "none" if not kinds else (next(iter(kinds)) if len(kinds)==1 else "mixed"), "candidate_count": len(generated),
                                    "complete_action_count": len(legal), "generation_time_seconds": elapsed, "recall_at_k": recalls, "value_gap_at_k": gaps,
                                    "candidate_action_keys": [[list(pair) for pair in action.key] for action in generated]})
            except Exception as error:
                failures.append({"instance_id": candidate.instance_id, "seed": candidate.seed, "split": split, "rollout_method": candidate.rollout_method,
                                 "error": f"{type(error).__name__}: {error}"})
    development = [row for row in records if row["split"] == "development"]
    by_model = {str(seed): sum(row["recall_at_k"]["8"] for row in development if row["model_seed"]==seed)/max(1,sum(row["model_seed"]==seed for row in development)) for seed in MODEL_SEEDS}
    by_stratum = {f"{seed}:{pending}": sum(row["recall_at_k"]["8"] for row in development if row["model_seed"]==seed and row["pending_count"]==pending)/max(1,sum(row["model_seed"]==seed and row["pending_count"]==pending for row in development)) for seed in MODEL_SEEDS for pending in (1,2,3)}
    expected = {"train":150,"development":60}; valid = not failures and all(len(selected.get(split,()))==expected[split] for split in splits)
    coverage = bool(development) and all(value>=.95 for value in by_model.values()) and all(value>=.90 for value in by_stratum.values())
    status = "candidate_coverage_supported_for_next_stage" if valid and coverage else "candidate_coverage_insufficient"
    source_quota = {f"{split}:{pending}:{source}": sum(
        candidate.rollout_method == source and len(residual_state(candidate).pending_task_ids) == pending
        for candidate in selected.get(split, ())
    ) for split in selected for pending in (1, 2, 3) for source in _quotas(split, pending)}
    mismatch_count = sum(bool(row["first_event_mismatch"]) for row in package)
    optimism_count = sum(float(row["q_optimism_bias"]) > 0 for row in package)
    disagreement_count = sum(bool(row["tolerance_optimal"]) != bool(row["old_forced_tolerance_optimal"]) for row in package)
    summary = {"schema":SCHEMA,"selection_seed":SELECTION_SEED,"status":status,"pilot_valid":valid,"snapshot_counts":{key:len(value) for key,value in selected.items()},
               "action_label_count":len(package),"candidate_record_count":len(records),"failure_count":len(failures),"development_recall_at_8_by_model_seed":by_model,
               "development_recall_at_8_by_model_seed_and_pending":by_stratum,"candidate_generation_reads_oracle_value":False,"confirmation_read":False}
    stratified_metrics = _stratified_metrics(development)
    summary.update(source_quota=source_quota, first_event_mismatch_count=mismatch_count,
                   q_optimism_bias_positive_count=optimism_count,
                   optimal_set_disagreement_count=disagreement_count,
                   development_stratified_metrics=stratified_metrics)
    protocol = {"schema":SCHEMA,"selection_seed":SELECTION_SEED,"train_seeds":[75000,75599],"development_seeds":[75600,75749],"regression_benchmark_seeds":[75800,75949],
                "confirmation_forbidden":[76000,76149],"solver":{"gurobi":"13.0.3","threads":1,"time_limit_seconds":10},"k_values":list(K_VALUES),
                "candidate_sources":["residual_decoder","masked_greedy","coalition_aware_beam_width_16","whole_atomic_swap_add_drop"]}
    _write_jsonl(OUTPUT_DIR/"exact_action_package.jsonl", package); _write_jsonl(OUTPUT_DIR/"candidate_records.jsonl", records); _write_jsonl(OUTPUT_DIR/"failures.jsonl", failures)
    for name,value in (("summary.json",summary),("protocol.json",protocol)): (OUTPUT_DIR/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    metrics = stratified_metrics
    lines=["# Ticket 48 Exact Action Candidate Diagnostic","",f"Final status: **{status}**.","",f"Pilot valid: **{str(valid).lower()}**; failures: **{len(failures)}**.","",
           f"Old forced comparison: {mismatch_count} first-event mismatches, {optimism_count} positive optimism biases, and {disagreement_count} tolerance-optimal-set disagreements.","",
           "## Development candidate coverage","","| Model seed | Recall@1 | @2 | @4 | @8 | @16 | Gap@1 | @2 | @4 | @8 | @16 |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
           *(f"| {seed} | " + " | ".join(f"{metrics[f'model_seed:{seed}']['recall_at_k'][str(k)]:.4f}" for k in K_VALUES) + " | " + " | ".join(f"{metrics[f'model_seed:{seed}']['mean_value_gap_at_k'][str(k)]:.4f}" for k in K_VALUES) + " |" for seed in MODEL_SEEDS),
           "","Full pending-count, action-cardinality, and action-type strata are stored in `summary.json`.","","Candidate generation did not read oracle values. Confirmation seeds were not read."]
    (OUTPUT_DIR/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); return summary


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--split",choices=("train","development","all"),default="all"); args=parser.parse_args(argv)
    print(json.dumps(run(("train","development") if args.split=="all" else (args.split,)),indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
