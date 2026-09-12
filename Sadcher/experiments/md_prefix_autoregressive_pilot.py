"""Train and validate the supervised MD sequential prefix decoder.

This is a CPU-friendly research prototype.  It labels complete first actions
with the exact continuation oracle, trains a listwise regret objective, and
compares a causal prefix decoder with an otherwise identical prefix-zero
ablation.  It deliberately does not import RLlib or modify the production
decoder.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from baselines.exact_online_action_oracle import (
    CompleteOnlineAction,
    enumerate_complete_online_actions,
    solve_exact_action_continuation,
)
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_residual_generation import SnapshotCandidate, residual_state
from models.md_online_features import build_md_policy_inputs_from_simulator
from models.md_prefix_autoregressive import MDPrefixAutoregressivePolicy
from reinforcement_learning.md_joint_action import (
    oracle_aligned_action_is_legal,
    oracle_aligned_action_masks,
)
from simulation_environment.domain_model import TransportTask
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


TRAIN_POOL = range(76600, 76800)
DEVELOPMENT_POOL = range(76850, 77000)
TEST_POOL = range(77000, 77200)
SPLIT_COUNTS = {"train": 32, "development": 12, "test": 12}
MIN_ACTIONS = 24
MAX_ACTIONS = 96
MODEL_SEEDS = (11, 29, 47)
HIDDEN_DIM = 64
ATTENTION_HEADS = 4
LEARNING_RATE = 3e-4
MAX_EPOCHS = 100
BEAM_WIDTH = 16
LATENCY_REPEATS = 3
OUTPUT = Path("reports/md_prefix_autoregressive_pilot_2026-09-10")


@dataclass(frozen=True, slots=True)
class LabeledState:
    split: str
    seed: int
    observation: dict[str, torch.Tensor]
    actions: torch.Tensor
    regrets: torch.Tensor
    regret_by_action: dict[tuple[int, ...], float]
    legal_action_count: int
    mask_supported_count: int


def _observation(domain, simulator: MDDiscreteSimulator) -> dict[str, torch.Tensor]:
    robot, task, _adjacency, md = build_md_policy_inputs_from_simulator(simulator)
    hard = md.hard_feasibility_mask.bool()
    # The task channels stay raw because the independent oracle-aligned mask
    # reads skill bits and assignment state from these tensors.
    return {
        "robot_features": robot.float(),
        "task_features": task.float(),
        "robot_metadata": md.robot_metadata.float(),
        "task_metadata": md.task_metadata.float(),
        "pair_metadata": md.pair_metadata.float(),
        "task_is_transport": md.task_is_transport.bool(),
        "typed_adjacency": md.typed_adjacency.float(),
        "opportunity_context": md.opportunity_context.float(),
        "process_covered_skills": torch.zeros(
            1, hard.shape[2], 3, dtype=torch.float32
        ),
        "action_mask": torch.cat((~hard.any(dim=-1, keepdim=True), hard), dim=-1),
    }


def _candidate(seed: int) -> tuple[SnapshotCandidate, Any, tuple[CompleteOnlineAction, ...]]:
    domain = generate_md_instance(MDGeneratorConfig(seed=seed)).domain
    simulator = MDDiscreteSimulator(domain)
    candidate = SnapshotCandidate(
        f"md-prefix-{seed}", seed, "initial_stationary", domain, simulator, {}
    )
    state = residual_state(candidate)
    return candidate, state, enumerate_complete_online_actions(domain, state)


def _select_manifest() -> dict[str, list[dict[str, int]]]:
    pools: dict[str, Iterable[int]] = {
        "train": TRAIN_POOL,
        "development": DEVELOPMENT_POOL,
        "test": TEST_POOL,
    }
    manifest: dict[str, list[dict[str, int]]] = {}
    for split, pool in pools.items():
        selected: list[dict[str, int]] = []
        for seed in pool:
            _candidate_data, state, legal = _candidate(seed)
            if MIN_ACTIONS <= len(legal) <= MAX_ACTIONS:
                selected.append(
                    {
                        "seed": int(seed),
                        "pending_count": len(state.pending_task_ids),
                        "legal_action_count": len(legal),
                    }
                )
            if len(selected) == SPLIT_COUNTS[split]:
                break
        if len(selected) != SPLIT_COUNTS[split]:
            raise RuntimeError(
                f"could not select {split}: expected {SPLIT_COUNTS[split]}, "
                f"found {len(selected)}"
            )
        manifest[split] = selected
    return manifest


def _action_vector(action: CompleteOnlineAction, domain) -> list[int]:
    robot_ids = sorted(robot.robot_id for robot in domain.robots)
    task_index = {
        task_id: index + 1
        for index, task_id in enumerate(sorted(task.task_id for task in domain.tasks))
    }
    assignment = dict(action.assignments)
    return [task_index[assignment[robot_id]] if robot_id in assignment else 0 for robot_id in robot_ids]


def _action_type(action: CompleteOnlineAction, domain) -> str:
    tasks = {task.task_id: task for task in domain.tasks}
    kinds = {
        "transport" if isinstance(tasks[task_id], TransportTask) else "process"
        for _robot_id, task_id in action.assignments
    }
    return "mixed" if len(kinds) > 1 else next(iter(kinds))


def _label_state(split: str, seed: int) -> tuple[LabeledState, dict[str, Any]]:
    candidate, state, legal = _candidate(seed)
    observation = _observation(candidate.domain, candidate.simulator)
    costs: dict[tuple[tuple[int, int], ...], float] = {}
    action_rows: list[dict[str, Any]] = []
    unsupported: list[list[int]] = []
    for action in legal:
        vector = _action_vector(action, candidate.domain)
        if not oracle_aligned_action_is_legal(observation, torch.tensor([vector])):
            unsupported.append(vector)
        result = solve_exact_action_continuation(
            candidate.domain,
            state,
            action,
            time_limit_seconds=10.0,
            threads=1,
        )
        if result.status.value != "optimal" or result.continuation_cost is None:
            raise RuntimeError(
                f"exact label failed for {split}/{seed}/{action.key}: "
                f"{result.status.value}: {result.message}"
            )
        costs[action.key] = float(result.continuation_cost)
        action_rows.append(
            {
                "assignments": [list(pair) for pair in action.assignments],
                "idle_robot_ids": list(action.idle_robot_ids),
                "vector": vector,
                "action_type": _action_type(action, candidate.domain),
                "continuation_cost": float(result.continuation_cost),
                "task_horizon": result.task_horizon,
                "return_tail": result.return_tail,
            }
        )
    if unsupported:
        raise AssertionError(
            f"oracle-aligned mask rejected {len(unsupported)} exact actions for {split}/{seed}"
        )
    minimum = min(costs.values())
    regret_by_action: dict[tuple[int, ...], float] = {}
    for row, action in zip(action_rows, legal, strict=True):
        regret = costs[action.key] - minimum
        row["regret"] = regret
        row["tolerance_optimal"] = regret <= 1.0
        regret_by_action[tuple(row["vector"])] = regret
    actions = torch.tensor(list(regret_by_action), dtype=torch.long)
    regrets = torch.tensor(list(regret_by_action.values()), dtype=torch.float32)
    return (
        LabeledState(
            split,
            seed,
            observation,
            actions,
            regrets,
            regret_by_action,
            len(legal),
            len(legal),
        ),
        {
            "schema": "md-prefix-autoregressive-label-1.0",
            "split": split,
            "seed": seed,
            "pending_count": len(state.pending_task_ids),
            "legal_action_count": len(legal),
            "minimum_continuation_cost": minimum,
            "actions": action_rows,
        },
    )


def _label_split(
    manifest: dict[str, list[dict[str, int]]], split: str, output: Path
) -> tuple[list[LabeledState], dict[str, int]]:
    states: list[LabeledState] = []
    rows: list[dict[str, Any]] = []
    for entry in manifest[split]:
        state, row = _label_state(split, int(entry["seed"]))
        if state.legal_action_count != int(entry["legal_action_count"]):
            raise AssertionError("manifest action count changed during exact labeling")
        states.append(state)
        rows.append(row)
    (output / f"{split}_labels.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return states, {
        "states": len(states),
        "actions": sum(state.legal_action_count for state in states),
        "mask_supported": sum(state.mask_supported_count for state in states),
    }


def _repeat_observation(
    observation: dict[str, torch.Tensor], count: int
) -> dict[str, torch.Tensor]:
    return {
        key: value.repeat(count, *([1] * (value.ndim - 1)))
        for key, value in observation.items()
    }


def _train_step(model: MDPrefixAutoregressivePolicy, state: LabeledState) -> float:
    model.train()
    batch_observation = _repeat_observation(state.observation, state.actions.shape[0])
    log_probability = model.log_prob(batch_observation, state.actions)
    target = F.softmax(-state.regrets, dim=0)
    loss = -(target * log_probability).sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite prefix listwise loss")
    return loss


@torch.no_grad()
def _mean_regret(model: MDPrefixAutoregressivePolicy, states: list[LabeledState]) -> float:
    model.eval()
    values = []
    for state in states:
        decoded = model.beam_decode(state.observation, beam_width=BEAM_WIDTH)
        key = tuple(int(value) for value in decoded.actions[0].tolist())
        values.append(state.regret_by_action.get(key, float("inf")))
    if not values or not np.isfinite(values).all():
        return float("inf")
    return float(np.mean(values))


def _new_model(use_prefix: bool) -> MDPrefixAutoregressivePolicy:
    return MDPrefixAutoregressivePolicy(
        hidden_dim=HIDDEN_DIM,
        attention_heads=ATTENTION_HEADS,
        feedforward_dim=2 * HIDDEN_DIM,
        dropout=0.1,
        use_prefix=use_prefix,
    )


def _train_one(
    mode: str,
    model_seed: int,
    train_states: list[LabeledState],
    development_states: list[LabeledState],
    output: Path,
) -> dict[str, Any]:
    torch.manual_seed(model_seed)
    random.seed(model_seed)
    model = _new_model(mode == "prefix")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_value = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    gradient_checks = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = list(range(len(train_states)))
        random.Random(model_seed * 1000 + epoch).shuffle(order)
        losses = []
        for index in order:
            loss = _train_step(model, train_states[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError("non-finite prefix gradient")
            gradient_checks += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        development_regret = _mean_regret(model, development_states)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "development_mean_regret": development_regret,
            }
        )
        if development_regret < best_value - 1e-8:
            best_value = development_regret
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_dir = output / mode / f"seed{model_seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mode": mode,
                    "model_seed": model_seed,
                    "epoch": epoch,
                    "development_mean_regret": development_regret,
                    "model_config": {
                        "hidden_dim": HIDDEN_DIM,
                        "attention_heads": ATTENTION_HEADS,
                        "feedforward_dim": 2 * HIDDEN_DIM,
                        "dropout": 0.1,
                    },
                    "normalization": "identity; semantic skill/task fields kept raw for oracle mask",
                },
                checkpoint_dir / "best_checkpoint.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= 10:
                break
    if best_epoch == 0:
        raise RuntimeError(f"no finite development checkpoint for {mode}/seed{model_seed}")
    checkpoint = torch.load(
        output / mode / f"seed{model_seed}" / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    restored = _new_model(mode == "prefix")
    restored.load_state_dict(checkpoint["state_dict"])
    restored.eval()
    restore_ok = True
    if development_states:
        with torch.no_grad():
            original = _new_model(mode == "prefix")
            original.load_state_dict(checkpoint["state_dict"])
            original.eval()
            left = original.beam_decode(development_states[0].observation, beam_width=BEAM_WIDTH).actions
            right = restored.beam_decode(development_states[0].observation, beam_width=BEAM_WIDTH).actions
            restore_ok = bool(torch.equal(left, right))
    (output / mode / f"seed{model_seed}" / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "mode": mode,
        "model_seed": model_seed,
        "best_epoch": best_epoch,
        "best_development_mean_regret": best_value,
        "epochs_run": len(history),
        "gradient_checks": gradient_checks,
        "checkpoint_restore": restore_ok,
    }


@torch.no_grad()
def _evaluate(
    model: MDPrefixAutoregressivePolicy, states: list[LabeledState]
) -> dict[str, Any]:
    model.eval()
    outputs: dict[str, list[float]] = {"greedy": [], "beam16": []}
    optimal: dict[str, list[bool]] = {"greedy": [], "beam16": []}
    tolerance: dict[str, list[bool]] = {"greedy": [], "beam16": []}
    illegal: dict[str, int] = {"greedy": 0, "beam16": 0}
    timings: list[float] = []
    for state in states:
        for name, width in (("greedy", 1), ("beam16", BEAM_WIDTH)):
            started = time.perf_counter()
            decoded = model.beam_decode(state.observation, beam_width=width)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            key = tuple(int(value) for value in decoded.actions[0].tolist())
            legal = oracle_aligned_action_is_legal(state.observation, decoded.actions)
            if not legal:
                illegal[name] += 1
            if name == "beam16":
                timings.append(elapsed_ms)
                for _ in range(LATENCY_REPEATS - 1):
                    started = time.perf_counter()
                    model.beam_decode(state.observation, beam_width=width)
                    timings.append((time.perf_counter() - started) * 1000.0)
            regret = state.regret_by_action.get(key, float("inf"))
            outputs[name].append(regret)
            optimal[name].append(bool(np.isfinite(regret) and regret <= 1e-6))
            tolerance[name].append(bool(np.isfinite(regret) and regret <= 1.0))
    return {
        name: {
            "mean_regret": float(np.mean(values)) if values else float("inf"),
            "top1_optimal": float(np.mean(optimal[name])) if values else 0.0,
            "top1_tolerance_optimal": float(np.mean(tolerance[name])) if values else 0.0,
            "illegal_actions": illegal[name],
        }
        for name, values in outputs.items()
    } | {
        "beam16_latency_ms": {
            "p50": float(np.percentile(timings, 50)) if timings else float("inf"),
            "p95": float(np.percentile(timings, 95)) if timings else float("inf"),
            "samples": len(timings),
        }
    }


@torch.no_grad()
def _prefix_logit_probe(
    model: MDPrefixAutoregressivePolicy, states: list[LabeledState]
) -> dict[str, Any]:
    model.eval()
    different = 0
    checked = 0
    for state in states:
        encoding = model.encode(state.observation)
        prefix_state = model.initial_prefix_state(encoding)
        robot_count = encoding.robot_tokens.shape[1]
        if robot_count < 2:
            continue
        partial = torch.full((1, robot_count), -1, dtype=torch.long)
        first_mask = oracle_aligned_action_masks(state.observation, partial)[0, 0]
        task_choices = torch.nonzero(first_mask[1:], as_tuple=False).flatten()
        if task_choices.numel() == 0:
            continue
        task_choice = int(task_choices[0]) + 1
        idle_state = model.advance(encoding, prefix_state, 0, torch.tensor([0]))
        task_state = model.advance(encoding, prefix_state, 0, torch.tensor([task_choice]))
        left = model.step(encoding, idle_state, 1)
        right = model.step(encoding, task_state, 1)
        checked += 1
        different += int(not torch.allclose(left, right, atol=1e-7, rtol=1e-7))
    return {"checked": checked, "different": different, "all_checked_different": checked > 0 and different == checked}


def _write_protocol(output: Path, manifest: dict[str, list[dict[str, int]]]) -> None:
    protocol = {
        "schema": "md-prefix-autoregressive-pilot-1.0",
        "created_before_test_labels": True,
        "state_selection": {
            "train_pool": [TRAIN_POOL.start, TRAIN_POOL.stop - 1],
            "development_pool": [DEVELOPMENT_POOL.start, DEVELOPMENT_POOL.stop - 1],
            "test_pool": [TEST_POOL.start, TEST_POOL.stop - 1],
            "counts": SPLIT_COUNTS,
            "legal_action_range": [MIN_ACTIONS, MAX_ACTIONS],
            "stationary_initial_states": True,
        },
        "models": ["prefix", "prefix-zero"],
        "model_config": {
            "hidden_size": HIDDEN_DIM,
            "attention_heads": ATTENTION_HEADS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "beam_width": BEAM_WIDTH,
            "latency_repeats_per_test_state": LATENCY_REPEATS,
            "seeds": list(MODEL_SEEDS),
        },
        "objective": "q(a|s)=softmax(-regret(a)); loss=-sum(q log p)",
        "oracle_aligned_mask": True,
        "ppo_or_rllib_connected": False,
        "production_decoder_modified": False,
        "manifest": manifest,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run() -> dict[str, Any]:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(f"completed output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    manifest = _select_manifest()
    _write_protocol(OUTPUT, manifest)
    (OUTPUT / "state_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    train_states, train_coverage = _label_split(manifest, "train", OUTPUT)
    development_states, development_coverage = _label_split(manifest, "development", OUTPUT)
    training_results = []
    for mode in ("prefix", "prefix-zero"):
        for model_seed in MODEL_SEEDS:
            training_results.append(
                _train_one(
                    mode,
                    model_seed,
                    train_states,
                    development_states,
                    OUTPUT,
                )
            )
    (OUTPUT / "training_locked.json").write_text(
        json.dumps(
            {
                "model_and_hyperparameters_locked": True,
                "training_results": training_results,
                "test_labels_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # The test labels are deliberately generated only after all checkpoints
    # and hyperparameters have been locked above.
    test_states, test_coverage = _label_split(manifest, "test", OUTPUT)
    rows: list[dict[str, Any]] = []
    for mode in ("prefix", "prefix-zero"):
        for model_seed in MODEL_SEEDS:
            checkpoint = torch.load(
                OUTPUT / mode / f"seed{model_seed}" / "best_checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
            model = _new_model(mode == "prefix")
            model.load_state_dict(checkpoint["state_dict"])
            metrics = _evaluate(model, test_states)
            dependence = _prefix_logit_probe(model, test_states)
            rows.append(
                {
                    "mode": mode,
                    "model_seed": model_seed,
                    "metrics": metrics,
                    "prefix_dependence": dependence,
                }
            )

    indexed = {(row["mode"], row["model_seed"]): row for row in rows}
    prefix_beam = [indexed[("prefix", seed)]["metrics"]["beam16"] for seed in MODEL_SEEDS]
    zero_beam = [indexed[("prefix-zero", seed)]["metrics"]["beam16"] for seed in MODEL_SEEDS]
    top1_deltas = [
        prefix["top1_optimal"] - zero["top1_optimal"]
        for prefix, zero in zip(prefix_beam, zero_beam, strict=True)
    ]
    prefix_regret = float(np.mean([item["mean_regret"] for item in prefix_beam]))
    zero_regret = float(np.mean([item["mean_regret"] for item in zero_beam]))
    max_p95 = max(
        indexed[(mode, seed)]["metrics"]["beam16_latency_ms"]["p95"]
        for mode in ("prefix", "prefix-zero")
        for seed in MODEL_SEEDS
    )
    all_restore = all(item["checkpoint_restore"] for item in training_results)
    all_gradients = all(item["gradient_checks"] > 0 for item in training_results)
    expected_actions = sum(
        int(entry["legal_action_count"])
        for split_entries in manifest.values()
        for entry in split_entries
    )
    observed_actions = (
        train_coverage["actions"]
        + development_coverage["actions"]
        + test_coverage["actions"]
    )
    supported_actions = (
        train_coverage["mask_supported"]
        + development_coverage["mask_supported"]
        + test_coverage["mask_supported"]
    )
    checks = {
        "exact_label_generation_success_100": observed_actions == expected_actions,
        "oracle_aligned_mask_coverage_100": supported_actions == observed_actions,
        "prefix_top1_gain_at_least_0.05": float(np.mean(top1_deltas)) >= 0.05,
        "prefix_mean_regret_at_least_10pct_lower": (
            prefix_regret <= zero_regret * 0.9 if zero_regret > 0 else prefix_regret <= zero_regret
        ),
        "at_least_two_seeds_not_below_prefix_zero": sum(delta >= 0.0 for delta in top1_deltas) >= 2,
        "prefix_logits_depend_on_prefix": any(
            row["prefix_dependence"]["different"] > 0
            for row in rows
            if row["mode"] == "prefix"
        ),
        "decoder_illegal_actions_zero": all(
            row["metrics"]["beam16"]["illegal_actions"] == 0
            and row["metrics"]["greedy"]["illegal_actions"] == 0
            for row in rows
        ),
        "mac_cpu_beam16_p95_ms_at_most_20": max_p95 <= 20.0,
        "gradient_checks_passed": all_gradients,
        "checkpoint_restore_passed": all_restore,
    }
    summary = {
        "schema": "md-prefix-autoregressive-pilot-1.0",
        "status": "accepted_for_experimental_prototype"
        if all(checks.values())
        else "experimental_prototype_only",
        "state_counts": {split: len(manifest[split]) for split in SPLIT_COUNTS},
        "action_counts": {
            "train": train_coverage["actions"],
            "development": development_coverage["actions"],
            "test": test_coverage["actions"],
        },
        "label_coverage": {
            "expected_actions": expected_actions,
            "observed_actions": observed_actions,
            "oracle_mask_supported_actions": supported_actions,
        },
        "training_results": training_results,
        "evaluation": {
            "rows": rows,
            "prefix_mean_beam16_regret": prefix_regret,
            "prefix_zero_mean_beam16_regret": zero_regret,
            "beam16_top1_deltas": top1_deltas,
            "maximum_beam16_p95_ms": max_p95,
        },
        "checks": checks,
        "test_labels_read_after_lock": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "final_report.md").write_text(
        "\n".join(
            [
                "# MD-SADCHER Sequential Prefix Decoder Pilot",
                "",
                f"Status: **{summary['status']}**.",
                "",
                f"States: train={summary['state_counts']['train']}, development={summary['state_counts']['development']}, test={summary['state_counts']['test']}.",
                f"Exact labels: {sum(summary['action_counts'].values())}; test labels were generated after checkpoint lock.",
                "",
                "| Metric | prefix-zero | prefix | prefix minus zero |",
                "|---|---:|---:|---:|",
                f"| Beam-16 top-1 optimal | {float(np.mean([item['top1_optimal'] for item in zero_beam])):.4f} | {float(np.mean([item['top1_optimal'] for item in prefix_beam])):.4f} | {float(np.mean(top1_deltas)):.4f} |",
                f"| Beam-16 mean regret | {zero_regret:.4f} | {prefix_regret:.4f} | {prefix_regret - zero_regret:.4f} |",
                f"| Maximum beam-16 p95 CPU time (ms) | — | — | {max_p95:.3f} |",
                "",
                "If any acceptance check is false, this remains an experimental prototype and is not connected to production scheduling.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))
