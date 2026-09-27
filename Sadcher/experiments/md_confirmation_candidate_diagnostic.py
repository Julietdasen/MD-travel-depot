"""Blind confirmation-split candidate diagnostic (seeds 76000-76149).

Reuses the exact protocol from md_exact_action_candidate_diagnostic.py, with
two intentional deviations documented up-front:

1. Confirmation split (seeds 76000-76149) instead of train/development.
2. Residual-tail scorer checkpoints (reports/md_residual_scale10_2026-09-04/...)
   are not present on this system, so the candidate-generation "scores" input
   is the zero tensor.  Because generate_candidates always includes the top-K
   legal actions plus decoder-preferred actions plus swap-add-drop neighbours,
   the resulting candidate set is a super-set that still contains every legal
   action when |legal| <= 16 (measured average legal count per selected state
   is 2-3 for pending in {1,2,3}).  The rerank job the physics scorer is asked
   to solve is therefore at least as hard as on development.

Nothing else about protocol, seed selection, or action-labelling changes.  The
Gurobi oracle in baselines.exact_online_action_oracle is used unmodified.
"""
from __future__ import annotations

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
from baselines.md_oracle_types import (
    ForcedAssignmentBatch, ResidualOracleStatus,
)
from baselines.md_ortools_residual_oracle import solve_residual_forced_batch
from data_generation.md_residual_generation import SnapshotCandidate, eligible_snapshot, residual_state
from data_generation.md_residual_pipeline import collect_seed_snapshots
from experiments.md_exact_action_candidate_diagnostic import (
    C0, MODEL_SEEDS, SELECTION_SEED, _atoms, _score, _state_key,
)
from schedulers.md_constrained_decoder import LearnedConstrainedDecoder, MaskedGreedyDecoder, simulator_hard_mask
from simulation_environment.domain_model import TransportTask

K_VALUES = (1, 2, 4, 8, 16)
OUTPUT_DIR = Path("reports/md_minimal_physics_scorer_confirmation_2026-09-17/candidate_diagnostic")
CONFIRMATION_SEEDS = range(76000, 76150)
SPLIT = "confirmation"


def _quotas(pending: int) -> Mapping[str, int]:
    rotations = ((4, 3, 3), (3, 4, 3), (3, 3, 4))
    return {"masked_greedy": 10,
            **{f"legacy_c0_seed{s}": n for s, n in zip(MODEL_SEEDS, rotations[pending - 1], strict=True)}}


def _select_confirmation(candidates: Sequence[SnapshotCandidate]) -> tuple[SnapshotCandidate, ...]:
    rng = random.Random(SELECTION_SEED + 2)
    buckets: dict[tuple[int, str], list[SnapshotCandidate]] = {}
    for candidate in candidates:
        if not eligible_snapshot(candidate.simulator):
            continue
        pending = len(residual_state(candidate).pending_task_ids)
        if pending in (1, 2, 3):
            buckets.setdefault((pending, candidate.rollout_method), []).append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (item.seed, item.simulator.time))
        rng.shuffle(bucket)
    selected: list[SnapshotCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for pending in (1, 2, 3):
        for source, quota in _quotas(pending).items():
            accepted = 0
            for candidate in buckets.get((pending, source), ()):
                if _state_key(candidate) in seen:
                    continue
                selected.append(candidate); seen.add(_state_key(candidate)); accepted += 1
                if accepted == quota:
                    break
            if accepted != quota:
                raise RuntimeError(f"quota unavailable: {pending=} {source=} required={quota} observed={accepted}")
    return tuple(selected)


def _generate_zero_score_candidates(
    candidate: SnapshotCandidate,
    legal: Sequence[CompleteOnlineAction],
) -> tuple[CompleteOnlineAction, ...]:
    """generate_candidates() from the original diagnostic, but with zero scores.

    Uses only the residual/greedy decoders' picks plus top-K by canonical
    key ordering, plus swap-add-drop neighbours.  No learned residual scorer.
    """
    simulator = candidate.simulator
    mask = simulator_hard_mask(simulator)
    scores = torch.zeros_like(mask, dtype=torch.float32)
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
    return tuple(sorted((legal_by_key[key] for key in keys),
                        key=lambda action: (-_score(action, scores, robot_index, task_index), action.key)))


def _cache_path(candidate: SnapshotCandidate) -> Path:
    return OUTPUT_DIR / "cache" / SPLIT / f"{candidate.seed}_{candidate.rollout_method}_t{candidate.simulator.time}.json"


def _solve_snapshot(candidate: SnapshotCandidate) -> dict[str, Any]:
    state = residual_state(candidate)
    actions = enumerate_complete_online_actions(candidate.domain, state)
    path = _cache_path(candidate)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload
    exact = []; old = []
    for action in actions:
        result = solve_exact_action_continuation(candidate.domain, state, action, time_limit_seconds=10, threads=1)
        forced = solve_residual_forced_batch(candidate.domain, state, ForcedAssignmentBatch(action.assignments), time_limit_seconds=10, threads=1)
        if result.status is not ResidualOracleStatus.OPTIMAL or forced.status is not ResidualOracleStatus.OPTIMAL:
            raise RuntimeError(f"non-optimal solve: exact={result.status.value}, old={forced.status.value}")
        exact.append((action, result)); old.append(forced)
    exact_min = min(result.continuation_cost for _, result in exact if result.continuation_cost is not None)
    old_costs = [result.makespan - state.current_time for result in old if result.makespan is not None]
    old_min = min(old_costs)
    labels = []
    for (action, result), forced, old_cost in zip(exact, old, old_costs, strict=True):
        row = label_to_json(result, action, state_min_continuation_cost=exact_min, pending_task_ids=state.pending_task_ids)
        first_time = min(item.planned_assignment for item in forced.actions)
        old_first = tuple(sorted((item.robot_id, item.task_id) for item in forced.actions if item.planned_assignment == first_time))
        if result.continuation_cost is None:
            raise RuntimeError("optimal exact result lacks continuation cost")
        row.update(old_forced_continuation_cost=old_cost, old_forced_regret=old_cost - old_min,
                   old_forced_tolerance_optimal=old_cost - old_min <= 1,
                   first_event_mismatch=old_first != action.assignments,
                   q_optimism_bias=result.continuation_cost - old_cost)
        labels.append(row)
    payload = {"schema": SCHEMA, "instance_id": candidate.instance_id, "seed": candidate.seed,
               "split": SPLIT, "rollout_method": candidate.rollout_method,
               "current_time": state.current_time, "pending_count": len(state.pending_task_ids), "labels": labels}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run() -> Mapping[str, Any]:
    random.seed(SELECTION_SEED); np.random.seed(SELECTION_SEED); torch.manual_seed(SELECTION_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started_collect = time.perf_counter()
    candidates: list[SnapshotCandidate] = []
    for seed in CONFIRMATION_SEEDS:
        candidates.extend(collect_seed_snapshots(seed, c0_checkpoints=C0, device="cpu", max_steps=10_000))
    collect_seconds = time.perf_counter() - started_collect
    selected = _select_confirmation(candidates)
    package: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_solve = time.perf_counter()
    for candidate in selected:
        try:
            snapshot = _solve_snapshot(candidate); state = residual_state(candidate)
            snapshot_id = f"{candidate.instance_id}:{candidate.rollout_method}:t{state.current_time}"
            labels = snapshot["labels"]
            package.extend({"snapshot_id": snapshot_id, "instance_id": candidate.instance_id, "seed": candidate.seed,
                             "split": SPLIT, "rollout_method": candidate.rollout_method,
                             "current_time": state.current_time, **label} for label in labels)
            legal = enumerate_complete_online_actions(candidate.domain, state)
            regret = {tuple(tuple(pair) for pair in label["complete_first_action"]["assignments"]): float(label["regret"]) for label in labels}
            optimal = {key for key, value in regret.items() if value <= 1}
            tasks = {task.task_id: task for task in candidate.domain.tasks}
            generated = _generate_zero_score_candidates(candidate, legal)
            recalls = {str(k): bool(optimal.intersection(action.key for action in generated[:k])) for k in K_VALUES}
            gaps = {str(k): min((regret[action.key] for action in generated[:k]), default=None) for k in K_VALUES}
            best = generated[0] if generated else None
            kinds = set() if best is None else {"transport" if isinstance(tasks[t], TransportTask) else "process" for _, t in best.assignments}
            # replicate per model seed for downstream consumers that expect per-seed rows
            for model_seed in MODEL_SEEDS:
                records.append({"snapshot_id": snapshot_id, "instance_id": candidate.instance_id, "seed": candidate.seed,
                                "split": SPLIT, "rollout_method": candidate.rollout_method,
                                "current_time": state.current_time, "model_seed": model_seed,
                                "pending_count": len(state.pending_task_ids),
                                "action_cardinality": 0 if best is None else len(best.assignments),
                                "action_type": "none" if not kinds else (next(iter(kinds)) if len(kinds) == 1 else "mixed"),
                                "candidate_count": len(generated),
                                "complete_action_count": len(legal), "generation_time_seconds": 0.0,
                                "recall_at_k": recalls, "value_gap_at_k": gaps,
                                "candidate_action_keys": [[list(pair) for pair in action.key] for action in generated]})
        except Exception as error:
            failures.append({"instance_id": candidate.instance_id, "seed": candidate.seed, "split": SPLIT,
                             "rollout_method": candidate.rollout_method,
                             "error": f"{type(error).__name__}: {error}"})
    solve_seconds = time.perf_counter() - started_solve
    _write_jsonl(OUTPUT_DIR / "exact_action_package.jsonl", package)
    _write_jsonl(OUTPUT_DIR / "candidate_records.jsonl", records)
    _write_jsonl(OUTPUT_DIR / "failures.jsonl", failures)
    summary = {"schema": SCHEMA, "split": SPLIT, "selection_seed": SELECTION_SEED,
               "snapshot_count": len(selected), "action_label_count": len(package),
               "candidate_record_count": len(records), "failure_count": len(failures),
               "collect_seconds": collect_seconds, "solve_seconds": solve_seconds,
               "residual_scorer_available": False,
               "candidate_generation_scores": "zero_tensor_placeholder",
               "seed_range": [76000, 76149]}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
