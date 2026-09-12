"""Frozen large-sample validation for the complete-action prefix cost gate."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from experiments.md_prefix_autoregressive_pilot import MODEL_SEEDS, _candidate, _label_state
from experiments.md_prefix_cost_gate_pilot import (
    GATE_SEEDS,
    MDCompleteActionCostGate,
    OUTPUT as PILOT_OUTPUT,
    _evaluate_confirmation,
    _label_from_saved_row,
    _load_prefix,
)


OUTPUT = Path("reports/md_prefix_cost_gate_large_validation_2026-09-10")
STATE_POOL = range(77350, 78250)
STATE_COUNT = 50
MIN_ACTIONS = 24
MAX_ACTIONS = 96
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
    raise RuntimeError(f"only found {len(selected)} eligible large-validation states")


def _load_frozen_gate() -> tuple[list[MDCompleteActionCostGate], torch.Tensor, torch.Tensor, float]:
    lock = json.loads((PILOT_OUTPUT / "training_locked.json").read_text(encoding="utf-8"))
    gates = []
    reference_mean = None
    reference_std = None
    for seed in GATE_SEEDS:
        checkpoint = torch.load(
            PILOT_OUTPUT / "cost_gate" / f"seed{seed}" / "best_checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        gate = MDCompleteActionCostGate()
        gate.load_state_dict(checkpoint["state_dict"])
        gate.eval()
        mean = checkpoint["feature_mean"]
        std = checkpoint["feature_std"]
        if reference_mean is None:
            reference_mean, reference_std = mean, std
        elif not torch.equal(reference_mean, mean) or not torch.equal(reference_std, std):
            raise AssertionError("frozen gate checkpoints use different normalization")
        gates.append(gate)
    if reference_mean is None or reference_std is None:
        raise RuntimeError("no frozen cost-gate checkpoint was loaded")
    return gates, reference_mean, reference_std, float(lock["selected_margin"])


def _label_states(manifest: Sequence[dict[str, int]]) -> tuple[list[Any], int]:
    states = []
    rows = []
    action_count = 0
    for entry in manifest:
        labeled, row = _label_state("large_validation", int(entry["seed"]))
        if labeled.legal_action_count != int(entry["legal_action_count"]):
            raise AssertionError("large-validation action count changed during labeling")
        candidate, _residual, _legal = _candidate(int(entry["seed"]))
        states.append(_label_from_saved_row("large_validation", row, candidate))
        rows.append(row)
        action_count += labeled.legal_action_count
    (OUTPUT / "labels.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return states, action_count


def _metrics_for_rows(rows: Sequence[dict[str, Any]], field: str) -> dict[str, float | int]:
    regrets = np.asarray([float(row[field]) for row in rows])
    minimums = np.asarray([float(row["minimum_continuation_cost"]) for row in rows])
    return {
        "mean_regret": float(regrets.mean()),
        "mean_continuation_duration": float((minimums + regrets).mean()),
        "top1_optimal": float((regrets <= 1e-6).mean()),
        "max_regret": float(regrets.max()),
        "catastrophic_regret_at_least_20": int((regrets >= 20.0).sum()),
    }


def _paired_analysis(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    differences = np.asarray(
        [float(row["base_regret"]) - float(row["gated_regret"]) for row in decisions]
    )
    state_differences = []
    for state_seed in sorted({int(row["state_seed"]) for row in decisions}):
        values = [
            float(row["base_regret"]) - float(row["gated_regret"])
            for row in decisions
            if int(row["state_seed"]) == state_seed
        ]
        state_differences.append(float(np.mean(values)))
    state_differences = np.asarray(state_differences)
    generator = np.random.default_rng(20260910)
    samples = generator.choice(
        state_differences,
        size=(BOOTSTRAP_REPEATS, len(state_differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean_regret_reduction": float(differences.mean()),
        "improved_decisions": int((differences > 0).sum()),
        "equal_decisions": int((differences == 0).sum()),
        "worse_decisions": int((differences < 0).sum()),
        "state_clustered_mean_reduction_95pct_bootstrap_ci": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
    }


def _per_model_seed(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for seed in MODEL_SEEDS:
        rows = [row for row in decisions if int(row["model_seed"]) == seed]
        base = _metrics_for_rows(rows, "base_regret")
        gated = _metrics_for_rows(rows, "gated_regret")
        result[str(seed)] = {
            "prefix": base,
            "prefix_gated": gated,
            "mean_regret_reduction": float(base["mean_regret"] - gated["mean_regret"]),
        }
    return result


def run() -> dict[str, Any]:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(f"completed output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    manifest = _manifest()
    gates, mean, std, margin = _load_frozen_gate()
    protocol = {
        "schema": "md-prefix-cost-gate-large-validation-protocol-1.0",
        "created_before_labels": True,
        "state_pool": [STATE_POOL.start, STATE_POOL.stop - 1],
        "state_count": STATE_COUNT,
        "legal_action_range": [MIN_ACTIONS, MAX_ACTIONS],
        "model_seeds": list(MODEL_SEEDS),
        "frozen_gate_seeds": list(GATE_SEEDS),
        "frozen_margin": margin,
        "manifest": manifest,
        "acceptance": {
            "mean_regret_reduction_at_least_20pct": True,
            "mean_continuation_duration_lower": True,
            "at_least_two_model_seeds_improve": True,
            "catastrophic_count_not_higher": True,
            "illegal_and_unknown_actions_zero": True,
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
    evaluation = _evaluate_confirmation(prefixes, zeros, gates, states, mean, std, margin)
    per_seed = _per_model_seed(evaluation["decisions"])
    paired = _paired_analysis(evaluation["decisions"])
    prefix_regret = float(evaluation["prefix"]["mean_regret"])
    gated_regret = float(evaluation["prefix_gated"]["mean_regret"])
    checks = {
        "mean_regret_reduction_at_least_20pct": gated_regret <= prefix_regret * 0.8,
        "mean_continuation_duration_lower": (
            evaluation["prefix_gated"]["mean_continuation_duration"]
            < evaluation["prefix"]["mean_continuation_duration"]
        ),
        "at_least_two_model_seeds_improve": sum(
            row["mean_regret_reduction"] > 0 for row in per_seed.values()
        ) >= 2,
        "catastrophic_count_not_higher": (
            evaluation["prefix_gated"]["catastrophic_regret_at_least_20"]
            <= evaluation["prefix"]["catastrophic_regret_at_least_20"]
        ),
        "illegal_actions_zero": evaluation["illegal_gated_actions"] == 0,
        "unknown_actions_zero": evaluation["unknown_gated_actions"] == 0,
        "mac_cpu_p95_ms_at_most_25": evaluation["online_latency_ms"]["p95"] <= 25.0,
    }
    summary = {
        "schema": "md-prefix-cost-gate-large-validation-1.0",
        "status": "large_validation_passed" if all(checks.values()) else "large_validation_failed",
        "state_count": len(states),
        "exact_action_label_count": action_count,
        "decision_count": evaluation["decision_count"],
        "frozen_margin": margin,
        "evaluation": evaluation,
        "per_model_seed": per_seed,
        "paired_analysis": paired,
        "checks": checks,
        "production_readiness": (
            "not_ready_when_top1_or_max_regret_worsens"
            if (
                evaluation["prefix_gated"]["top1_optimal"] < evaluation["prefix"]["top1_optimal"]
                or evaluation["prefix_gated"]["max_regret"] > evaluation["prefix"]["max_regret"]
            )
            else "eligible_for_separate_integration_review"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "production_decoder_modified": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ci = paired["state_clustered_mean_reduction_95pct_bootstrap_ci"]
    (OUTPUT / "final_report.md").write_text(
        "\n".join(
            [
                "# MD Prefix Cost Gate Large Validation",
                "",
                f"Status: **{summary['status']}**.",
                f"Frozen evaluation: {len(states)} new states, {action_count} exact actions, {evaluation['decision_count']} decoder decisions.",
                "",
                "| Decoder | Mean regret | Mean continuation duration | Top-1 | Max regret | Regret >= 20 |",
                "|---|---:|---:|---:|---:|---:|",
                *[
                    f"| {name} | {evaluation[name]['mean_regret']:.3f} | {evaluation[name]['mean_continuation_duration']:.3f} | {evaluation[name]['top1_optimal']:.3f} | {evaluation[name]['max_regret']:.1f} | {evaluation[name]['catastrophic_regret_at_least_20']} |"
                    for name in ("prefix", "prefix_gated", "prefix_zero")
                ],
                "",
                f"Mean paired regret reduction: {paired['mean_regret_reduction']:.3f}.",
                f"State-clustered 95% bootstrap interval: [{ci[0]:.3f}, {ci[1]:.3f}].",
                f"Improved/equal/worse decisions: {paired['improved_decisions']}/{paired['equal_decisions']}/{paired['worse_decisions']}.",
                f"Mac CPU Beam-16 + gate p95: {evaluation['online_latency_ms']['p95']:.3f} ms.",
                "",
                "## Limitation",
                "",
                "Passing the preregistered average-performance checks does not make this production-ready. "
                "If Top-1 accuracy or maximum regret worsens, the next experiment must add a conservative fallback before integration.",
                "",
                "The gate and threshold were frozen before these labels were generated. Production scheduling remains unchanged.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
