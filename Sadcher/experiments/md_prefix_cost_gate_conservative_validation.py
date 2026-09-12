"""Validate a frozen confidence fallback for the prefix complete-action gate."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from experiments.md_prefix_autoregressive_pilot import MODEL_SEEDS, _candidate, _label_state
from experiments.md_prefix_cost_gate_large_validation import _load_frozen_gate, _metrics_for_rows
from experiments.md_prefix_cost_gate_pilot import (
    _action_features,
    _ensemble_cost,
    _label_from_saved_row,
    _load_existing_split,
    _load_prefix,
)
from reinforcement_learning.md_joint_action import oracle_aligned_action_is_legal


OUTPUT = Path("reports/md_prefix_cost_gate_conservative_validation_2026-09-10")
STATE_POOL = range(78250, 79250)
STATE_COUNT = 50
MIN_ACTIONS = 24
MAX_ACTIONS = 96
CONFIDENCE_THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
BOOTSTRAP_REPEATS = 10_000


def _manifest() -> list[dict[str, int]]:
    selected = []
    for seed in STATE_POOL:
        _candidate_data, state, legal = _candidate(seed)
        if MIN_ACTIONS <= len(legal) <= MAX_ACTIONS:
            selected.append(
                {
                    "seed": seed,
                    "pending_count": len(state.pending_task_ids),
                    "legal_action_count": len(legal),
                }
            )
        if len(selected) == STATE_COUNT:
            return selected
    raise RuntimeError(f"only found {len(selected)} eligible validation states")


@torch.no_grad()
def _candidate_record(model, gates, state, mean, std) -> dict[str, Any]:
    candidates = model.beam_candidates(state.observation, beam_width=16)
    features = torch.stack(
        [_action_features(state.observation, state.simulator, row) for row in candidates.actions.tolist()]
    )
    predicted = _ensemble_cost(gates, features, mean, std)
    best_index = int(torch.argmin(predicted))
    actions = [tuple(int(value) for value in row) for row in candidates.actions.tolist()]
    return {
        "actions": actions,
        "regrets": [state.regret_by_action.get(action, float("inf")) for action in actions],
        "best_index": best_index,
        "predicted_improvement": float(predicted[0] - predicted[best_index]),
    }


def _threshold_metrics(records: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    regrets = []
    switches = 0
    for record in records:
        selected = record["best_index"] if record["predicted_improvement"] >= threshold else 0
        switches += int(selected != 0)
        regrets.append(float(record["regrets"][selected]))
    values = np.asarray(regrets)
    return {
        "threshold": threshold,
        "mean_regret": float(values.mean()),
        "top1_optimal": float((values <= 1e-6).mean()),
        "max_regret": float(values.max()),
        "catastrophic_regret_at_least_20": int((values >= 20.0).sum()),
        "switches": switches,
    }


def _select_threshold(gates, mean, std) -> tuple[float, list[dict[str, Any]]]:
    states = _load_existing_split("development")
    models = [_load_prefix("prefix", seed) for seed in MODEL_SEEDS]
    records = [
        _candidate_record(model, gates, state, mean, std)
        for model in models
        for state in states
    ]
    results = [_threshold_metrics(records, threshold) for threshold in CONFIDENCE_THRESHOLDS]
    baseline = results[-1]
    safe = [
        row
        for row in results
        if row["top1_optimal"] >= baseline["top1_optimal"]
        and row["max_regret"] <= baseline["max_regret"]
        and row["catastrophic_regret_at_least_20"] <= baseline["catastrophic_regret_at_least_20"]
    ]
    selected = min(safe, key=lambda row: (row["mean_regret"], row["threshold"]))
    return float(selected["threshold"]), results


def _label_states(manifest: Sequence[dict[str, int]]) -> tuple[list[Any], int]:
    states = []
    rows = []
    action_count = 0
    for entry in manifest:
        labeled, row = _label_state("conservative_validation", int(entry["seed"]))
        if labeled.legal_action_count != int(entry["legal_action_count"]):
            raise AssertionError("validation action count changed during exact labeling")
        candidate, _residual, _legal = _candidate(int(entry["seed"]))
        states.append(_label_from_saved_row("conservative_validation", row, candidate))
        rows.append(row)
        action_count += labeled.legal_action_count
    (OUTPUT / "labels.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return states, action_count


@torch.no_grad()
def _evaluate(states, prefixes, zeros, gates, mean, std, threshold) -> dict[str, Any]:
    decisions = []
    timings = []
    for model_seed in MODEL_SEEDS:
        for state in states:
            started = time.perf_counter()
            record = _candidate_record(prefixes[model_seed], gates, state, mean, std)
            selected = record["best_index"] if record["predicted_improvement"] >= threshold else 0
            elapsed = (time.perf_counter() - started) * 1000.0
            base_action = record["actions"][0]
            conservative_action = record["actions"][selected]
            unrestricted_action = record["actions"][record["best_index"]]
            zero_action = tuple(
                int(value)
                for value in zeros[model_seed]
                .beam_decode(state.observation, beam_width=16)
                .actions[0]
                .tolist()
            )
            timings.append(elapsed)
            decisions.append(
                {
                    "model_seed": model_seed,
                    "state_seed": state.seed,
                    "base_action": list(base_action),
                    "conservative_action": list(conservative_action),
                    "unrestricted_action": list(unrestricted_action),
                    "prefix_zero_action": list(zero_action),
                    "base_regret": state.regret_by_action.get(base_action, float("inf")),
                    "conservative_regret": state.regret_by_action.get(conservative_action, float("inf")),
                    "unrestricted_regret": state.regret_by_action.get(unrestricted_action, float("inf")),
                    "prefix_zero_regret": state.regret_by_action.get(zero_action, float("inf")),
                    "minimum_continuation_cost": state.minimum_continuation_cost,
                    "switched": selected != 0,
                    "predicted_improvement": record["predicted_improvement"],
                    "legal": oracle_aligned_action_is_legal(
                        state.observation, torch.tensor([conservative_action], dtype=torch.long)
                    ),
                    "latency_ms": elapsed,
                }
            )
    result = {
        "state_count": len(states),
        "decision_count": len(decisions),
        "prefix": _metrics_for_rows(decisions, "base_regret"),
        "conservative_gate": _metrics_for_rows(decisions, "conservative_regret"),
        "unrestricted_gate": _metrics_for_rows(decisions, "unrestricted_regret"),
        "prefix_zero": _metrics_for_rows(decisions, "prefix_zero_regret"),
        "switches": sum(row["switched"] for row in decisions),
        "illegal_actions": sum(not row["legal"] for row in decisions),
        "unknown_actions": sum(
            not np.isfinite(float(row["conservative_regret"])) for row in decisions
        ),
        "latency_ms": {
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
        },
        "decisions": decisions,
    }
    return result


def _paired_analysis(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    delta = np.asarray(
        [float(row["base_regret"]) - float(row["conservative_regret"]) for row in decisions]
    )
    state_delta = np.asarray(
        [
            np.mean(
                [
                    float(row["base_regret"]) - float(row["conservative_regret"])
                    for row in decisions
                    if int(row["state_seed"]) == seed
                ]
            )
            for seed in sorted({int(row["state_seed"]) for row in decisions})
        ]
    )
    generator = np.random.default_rng(20260910)
    samples = generator.choice(
        state_delta, size=(BOOTSTRAP_REPEATS, len(state_delta)), replace=True
    ).mean(axis=1)
    return {
        "mean_regret_reduction": float(delta.mean()),
        "improved": int((delta > 0).sum()),
        "equal": int((delta == 0).sum()),
        "worse": int((delta < 0).sum()),
        "state_clustered_95pct_bootstrap_ci": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
    }


def _per_seed(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for seed in MODEL_SEEDS:
        rows = [row for row in decisions if int(row["model_seed"]) == seed]
        base = _metrics_for_rows(rows, "base_regret")
        gate = _metrics_for_rows(rows, "conservative_regret")
        result[str(seed)] = {
            "prefix": base,
            "conservative_gate": gate,
            "mean_regret_reduction": float(base["mean_regret"] - gate["mean_regret"]),
        }
    return result


def run() -> dict[str, Any]:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(f"completed output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    gates, mean, std, _unrestricted_margin = _load_frozen_gate()
    threshold, development_results = _select_threshold(gates, mean, std)
    manifest = _manifest()
    protocol = {
        "schema": "md-prefix-conservative-gate-protocol-1.0",
        "created_before_validation_labels": True,
        "selection_data": "original development split only",
        "threshold_candidates": list(CONFIDENCE_THRESHOLDS),
        "development_results": development_results,
        "selected_threshold": threshold,
        "state_pool": [STATE_POOL.start, STATE_POOL.stop - 1],
        "state_count": STATE_COUNT,
        "legal_action_range": [MIN_ACTIONS, MAX_ACTIONS],
        "model_seeds": list(MODEL_SEEDS),
        "manifest": manifest,
        "acceptance": {
            "mean_regret_lower": True,
            "top1_not_lower": True,
            "maximum_regret_not_higher": True,
            "catastrophic_count_not_higher": True,
            "at_least_two_model_seeds_not_worse": True,
            "illegal_and_unknown_zero": True,
            "mac_cpu_p95_ms_at_most_25": True,
        },
        "production_decoder_modified": False,
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    states, action_count = _label_states(manifest)
    prefixes = {seed: _load_prefix("prefix", seed) for seed in MODEL_SEEDS}
    zeros = {seed: _load_prefix("prefix-zero", seed) for seed in MODEL_SEEDS}
    evaluation = _evaluate(states, prefixes, zeros, gates, mean, std, threshold)
    per_seed = _per_seed(evaluation["decisions"])
    paired = _paired_analysis(evaluation["decisions"])
    base = evaluation["prefix"]
    gate = evaluation["conservative_gate"]
    checks = {
        "mean_regret_lower": gate["mean_regret"] < base["mean_regret"],
        "top1_not_lower": gate["top1_optimal"] >= base["top1_optimal"],
        "maximum_regret_not_higher": gate["max_regret"] <= base["max_regret"],
        "catastrophic_count_not_higher": (
            gate["catastrophic_regret_at_least_20"]
            <= base["catastrophic_regret_at_least_20"]
        ),
        "at_least_two_model_seeds_not_worse": sum(
            row["mean_regret_reduction"] >= 0 for row in per_seed.values()
        ) >= 2,
        "illegal_actions_zero": evaluation["illegal_actions"] == 0,
        "unknown_actions_zero": evaluation["unknown_actions"] == 0,
        "mac_cpu_p95_ms_at_most_25": evaluation["latency_ms"]["p95"] <= 25.0,
    }
    summary = {
        "schema": "md-prefix-conservative-gate-validation-1.0",
        "status": (
            "conservative_gate_validated"
            if all(checks.values())
            else "conservative_gate_not_validated"
        ),
        "production_readiness": (
            "eligible_for_separate_integration_review"
            if all(checks.values())
            else "experimental_only"
        ),
        "selected_threshold": threshold,
        "state_count": len(states),
        "exact_action_label_count": action_count,
        "decision_count": evaluation["decision_count"],
        "evaluation": evaluation,
        "per_model_seed": per_seed,
        "paired_analysis": paired,
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "production_decoder_modified": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ci = paired["state_clustered_95pct_bootstrap_ci"]
    (OUTPUT / "final_report.md").write_text(
        "\n".join(
            [
                "# MD Prefix Conservative Cost Gate Validation",
                "",
                f"Status: **{summary['status']}**.",
                f"Frozen confidence threshold: {threshold:.1f}.",
                f"Validation: {len(states)} new states, {action_count} exact actions, {evaluation['decision_count']} decisions.",
                "",
                "| Decoder | Mean regret | Mean duration | Top-1 | Max regret | Regret >= 20 |",
                "|---|---:|---:|---:|---:|---:|",
                *[
                    f"| {name} | {evaluation[name]['mean_regret']:.3f} | {evaluation[name]['mean_continuation_duration']:.3f} | {evaluation[name]['top1_optimal']:.3f} | {evaluation[name]['max_regret']:.1f} | {evaluation[name]['catastrophic_regret_at_least_20']} |"
                    for name in ("prefix", "conservative_gate", "unrestricted_gate", "prefix_zero")
                ],
                "",
                f"Gate switches: {evaluation['switches']} of {evaluation['decision_count']}.",
                f"Improved/equal/worse: {paired['improved']}/{paired['equal']}/{paired['worse']}.",
                f"Mean paired regret reduction: {paired['mean_regret_reduction']:.3f}.",
                f"State-clustered 95% bootstrap CI: [{ci[0]:.3f}, {ci[1]:.3f}].",
                f"Mac CPU Beam-16 + confidence gate p95: {evaluation['latency_ms']['p95']:.3f} ms.",
                "",
                "The confidence threshold was locked before validation labels were generated. Production scheduling remains unchanged.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))

