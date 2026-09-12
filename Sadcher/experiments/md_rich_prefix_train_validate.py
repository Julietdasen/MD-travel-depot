"""Train and validate prefix scorers on exact labels from rich initial MD states."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from baselines.exact_online_action_oracle import (
    CompleteOnlineAction,
    enumerate_complete_online_actions,
    solve_exact_action_continuation,
)
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_residual_generation import SnapshotCandidate, residual_state
from experiments.md_minimal_action_features import build_minimal_action_features
from experiments.md_prefix_conditioned_reranker import (
    MODEL_SEEDS,
    TRAINING_SEEDS,
    Candidate,
    ExperimentConfig,
    PrefixCostScorer,
    State,
    candidate_cost,
    evaluate,
    normalization,
    train_one,
)
from experiments.md_prefix_features import PrefixAction, canonical_action
from simulation_environment.domain_model import TransportTask
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


TRAIN_POOL = range(76300, 76380)
DEVELOPMENT_POOL = range(76400, 76460)
TEST_POOL = range(76500, 76560)
SPLIT_COUNTS = {"train": 8, "development": 8, "test": 8}
MIN_ACTIONS = 24
MAX_ACTIONS = 96
OUTPUT = Path("reports/md_rich_prefix_train_validate_2026-09-10")
MODES = ("prefix_zero", "prefix")


def _action_type(action: CompleteOnlineAction, tasks: dict[int, Any]) -> str:
    kinds = {
        "transport" if isinstance(tasks[task_id], TransportTask) else "process"
        for _, task_id in action.assignments
    }
    return "mixed" if len(kinds) > 1 else next(iter(kinds))


def _candidate(seed: int) -> tuple[SnapshotCandidate, Any, tuple[CompleteOnlineAction, ...]]:
    domain = generate_md_instance(MDGeneratorConfig(seed=seed)).domain
    simulator = MDDiscreteSimulator(domain)
    snapshot = SnapshotCandidate(f"md-rich-train-{seed}", seed, "initial", domain, simulator, {})
    state = residual_state(snapshot)
    legal = enumerate_complete_online_actions(domain, state)
    return snapshot, state, legal


def _select_seed_manifest() -> dict[str, list[dict[str, int]]]:
    pools = {
        "train": TRAIN_POOL,
        "development": DEVELOPMENT_POOL,
        "test": TEST_POOL,
    }
    manifest: dict[str, list[dict[str, int]]] = {}
    for split, pool in pools.items():
        selected: list[dict[str, int]] = []
        for seed in pool:
            _, state, legal = _candidate(seed)
            if MIN_ACTIONS <= len(legal) <= MAX_ACTIONS:
                selected.append(
                    {
                        "seed": seed,
                        "pending_count": len(state.pending_task_ids),
                        "legal_action_count": len(legal),
                    }
                )
            if len(selected) == SPLIT_COUNTS[split]:
                break
        if len(selected) != SPLIT_COUNTS[split]:
            raise RuntimeError(f"could not select {split} states: {len(selected)}")
        manifest[split] = selected
    return manifest


def _label_states(
    manifest: dict[str, list[dict[str, int]]], output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_path = output / "rich_exact_labels.jsonl"
    if label_path.exists():
        payload = [json.loads(line) for line in label_path.read_text().splitlines() if line.strip()]
        return payload, []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for split, entries in manifest.items():
        for entry in entries:
            seed = int(entry["seed"])
            snapshot, state, legal = _candidate(seed)
            tasks = {task.task_id: task for task in snapshot.domain.tasks}
            costs: dict[tuple[tuple[int, int], ...], float] = {}
            action_meta: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
            for index, action in enumerate(legal):
                result = solve_exact_action_continuation(
                    snapshot.domain, state, action, time_limit_seconds=10, threads=1
                )
                if result.status.value != "optimal" or result.continuation_cost is None:
                    failures.append(
                        {
                            "split": split,
                            "seed": seed,
                            "action_index": index,
                            "action": action.key,
                            "status": result.status.value,
                            "message": result.message,
                        }
                    )
                    break
                costs[action.key] = float(result.continuation_cost)
                action_meta[action.key] = {
                    "assignments": [list(pair) for pair in action.key],
                    "action_type": _action_type(action, tasks),
                    "action_cardinality": len(action.assignments),
                    "continuation_cost": float(result.continuation_cost),
                    "task_horizon": result.task_horizon,
                    "return_tail": result.return_tail,
                }
            if len(costs) != len(legal):
                continue
            minimum = min(costs.values())
            actions = []
            for action in legal:
                item = dict(action_meta[action.key])
                item["regret"] = costs[action.key] - minimum
                item["tolerance_optimal"] = item["regret"] <= 1.0
                actions.append(item)
            rows.append(
                {
                    "schema": "md-rich-exact-labels-1.0",
                    "split": split,
                    "seed": seed,
                    "pending_count": len(state.pending_task_ids),
                    "legal_action_count": len(legal),
                    "minimum_continuation_cost": minimum,
                    "actions": actions,
                }
            )
    label_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return rows, failures


def _build_states(
    rows: Iterable[dict[str, Any]], model_seed: int
) -> list[State]:
    states: list[State] = []
    for row in rows:
        seed = int(row["seed"])
        snapshot, _, _ = _candidate(seed)
        candidates: list[Candidate] = []
        for rank, item in enumerate(row["actions"]):
            action = CompleteOnlineAction(
                canonical_action(item["assignments"]), tuple()
            )
            edge = build_minimal_action_features(snapshot.simulator, action)
            candidates.append(
                Candidate(
                    PrefixAction(action.assignments, edge),
                    float(item["regret"]),
                    bool(item["tolerance_optimal"]),
                    rank,
                )
            )
        states.append(
            State(
                snapshot_id=f"rich:{seed}",
                split=str(row["split"]),
                seed=seed,
                model_seed=model_seed,
                pending_count=int(row["pending_count"]),
                action_type="rich",
                candidates=candidates,
            )
        )
    return states


def _load_model(mode: str, model_seed: int, output: Path) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    payload = torch.load(
        output / mode / f"seed{model_seed}" / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = PrefixCostScorer()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["normalization_mean"], payload["normalization_std"]


def _evaluate_test(
    states: list[State], output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_seed in MODEL_SEEDS:
        model_states = [state for state in states if state.model_seed == model_seed]
        for mode in MODES:
            model, mean, std = _load_model(mode, model_seed, output)
            metrics = evaluate(model, model_states, mean, std, mode=mode)
            order_metrics = {
                name: evaluate(
                    model,
                    model_states,
                    mean,
                    std,
                    mode=mode,
                    order_index=index,
                )
                for index, name in enumerate(("canonical", "reverse", "random"))
            }
            rows.append(
                {
                    "model_seed": model_seed,
                    "mode": mode,
                    "test": metrics,
                    "order_metrics": order_metrics,
                }
            )
    indexed = {(row["mode"], row["model_seed"]): row for row in rows}
    prefix_top1 = float(
        np.mean([indexed[("prefix", seed)]["test"]["tolerance_optimal_top1"] for seed in MODEL_SEEDS])
    )
    zero_top1 = float(
        np.mean([indexed[("prefix_zero", seed)]["test"]["tolerance_optimal_top1"] for seed in MODEL_SEEDS])
    )
    prefix_regret = float(
        np.mean([indexed[("prefix", seed)]["test"]["mean_regret"] for seed in MODEL_SEEDS])
    )
    zero_regret = float(
        np.mean([indexed[("prefix_zero", seed)]["test"]["mean_regret"] for seed in MODEL_SEEDS])
    )
    drops = [
        indexed[("prefix", seed)]["test"]["tolerance_optimal_top1"]
        - indexed[("prefix_zero", seed)]["test"]["tolerance_optimal_top1"]
        for seed in MODEL_SEEDS
    ]
    order_ranges = []
    for seed in MODEL_SEEDS:
        values = [
            indexed[("prefix", seed)]["order_metrics"][name]["tolerance_optimal_top1"]
            for name in ("canonical", "reverse", "random")
        ]
        order_ranges.append(max(values) - min(values))
    aggregate = {
        "prefix_top1": prefix_top1,
        "prefix_zero_top1": zero_top1,
        "prefix_top1_gain": prefix_top1 - zero_top1,
        "prefix_mean_regret": prefix_regret,
        "prefix_zero_mean_regret": zero_regret,
        "prefix_mean_regret_delta": prefix_regret - zero_regret,
        "per_seed_top1_deltas": drops,
        "max_prefix_order_top1_range": max(order_ranges),
    }
    checks = {
        "mean_top1_gain_at_least_0.02": prefix_top1 - zero_top1 >= 0.02,
        "mean_regret_lower": prefix_regret < zero_regret,
        "no_per_seed_drop_above_0.02": min(drops) >= -0.02,
        "order_range_at_most_0.03": max(order_ranges) <= 0.03,
    }
    return rows, {"aggregate": aggregate, "checks": checks}


def run() -> dict[str, Any]:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(f"completed output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "md-rich-prefix-train-validate-1.0",
        "created_before_labels_and_training": True,
        "selection": {
            "train_pool": [TRAIN_POOL.start, TRAIN_POOL.stop - 1],
            "development_pool": [DEVELOPMENT_POOL.start, DEVELOPMENT_POOL.stop - 1],
            "test_pool": [TEST_POOL.start, TEST_POOL.stop - 1],
            "first_states_with_24_to_96_legal_actions": True,
            "states_per_split": SPLIT_COUNTS,
        },
        "state": "initial_stationary",
        "action_labels": "exact continuation oracle; all legal actions",
        "inputs": ["16_observable_physics_features", "12_causal_prefix_features"],
        "oracle_values_used_as_inputs": False,
        "models": ["prefix_zero", "prefix"],
        "model_seeds": list(MODEL_SEEDS),
        "training_seeds": TRAINING_SEEDS,
        "confirmation_read": False,
        "regression_benchmark_read": False,
        "production_decoder_modified": False,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    manifest = _select_seed_manifest()
    (OUTPUT / "state_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    labels, failures = _label_states(manifest, OUTPUT)
    if failures:
        (OUTPUT / "failures.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures)
        )
        raise RuntimeError(f"exact labeling failures: {len(failures)}")
    if len(labels) != sum(SPLIT_COUNTS.values()):
        raise RuntimeError(f"expected {sum(SPLIT_COUNTS.values())} labeled states, observed {len(labels)}")

    train_states_by_seed = {seed: _build_states(labels, seed) for seed in MODEL_SEEDS}
    mean, std = normalization(next(iter(train_states_by_seed.values())))
    config = ExperimentConfig(epochs=60, evaluation_interval=5, patience=3, random_order_count=2)
    training_results = []
    for mode in MODES:
        for model_seed in MODEL_SEEDS:
            training_results.append(
                train_one(
                    model_seed,
                    train_states_by_seed[model_seed],
                    mean,
                    std,
                    config,
                    mode=mode,
                    output_dir=OUTPUT,
                )
            )
    test_states = [
        state
        for state in next(iter(train_states_by_seed.values()))
        if state.split == "test"
    ]
    test_states_by_seed = []
    for model_seed in MODEL_SEEDS:
        test_states_by_seed.extend(state for state in _build_states(labels, model_seed) if state.split == "test")
    evaluation_rows, evaluation = _evaluate_test(test_states_by_seed, OUTPUT)
    summary = {
        "schema": protocol["schema"],
        "status": "rich_prefix_supported_for_confirmation"
        if all(evaluation["checks"].values())
        else "rich_prefix_not_supported_for_confirmation",
        "state_counts": {split: len([row for row in labels if row["split"] == split]) for split in SPLIT_COUNTS},
        "label_action_count": sum(int(row["legal_action_count"]) for row in labels),
        "training_results": training_results,
        "evaluation": evaluation,
        "test_rows": evaluation_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "confirmation_read": False,
        "regression_benchmark_read": False,
        "production_decoder_modified": False,
    }
    (OUTPUT / "failures.jsonl").write_text("")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    aggregate = evaluation["aggregate"]
    lines = [
        "# Rich-State Prefix Train/Validate",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "Models were retrained on exact labels from rich initial states and evaluated on independent test states.",
        "",
        "| Metric | Prefix-zero | Prefix | Prefix minus zero |",
        "|---|---:|---:|---:|",
        f"| Top-1 tolerance-optimal | {aggregate['prefix_zero_top1']:.4f} | {aggregate['prefix_top1']:.4f} | {aggregate['prefix_top1_gain']:.4f} |",
        f"| Mean regret | {aggregate['prefix_zero_mean_regret']:.4f} | {aggregate['prefix_mean_regret']:.4f} | {aggregate['prefix_mean_regret_delta']:.4f} |",
        "",
        f"Maximum prefix order top-1 range: {aggregate['max_prefix_order_top1_range']:.4f}.",
        "",
        "This package is a development/test validation; confirmation and production regression were not read.",
    ]
    (OUTPUT / "final_report.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))
