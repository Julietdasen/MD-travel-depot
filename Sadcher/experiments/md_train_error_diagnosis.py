"""Train-set error decomposition for the failed optimization diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import torch

from experiments.md_policy_relational_gate import (
    CANDIDATE_TASK_COUNT,
    DEFAULT_MODEL_SEEDS,
    RELATIONAL_FAMILIES,
    RelationalState,
    RelationalTwin,
    _build_model,
    _model_agreement,
    _model_logits,
    _train_model,
    _validated_model_seeds,
)
from experiments.md_train_only_optimization_diagnostic import (
    TRAINABLE_METHODS,
    _select_first_pairs_per_family,
)
from experiments.md_unconditioned_relational_audit import (
    build_unconditioned_relational_candidates,
)


FORMAL_PROTOCOL = {
    "candidate_pool_per_family": 5000,
    "train_pairs_per_family": 500,
    "epochs": 200,
    "data_seed": 3030,
    "model_seeds": DEFAULT_MODEL_SEEDS,
    "margin": 0.1,
}


@dataclass(frozen=True, slots=True)
class TrainErrorDiagnosisResult:
    summary_path: Path
    report_path: Path


def run_train_error_diagnosis(
    output_dir: str | Path,
    *,
    candidate_pool_per_family: int = 5000,
    train_pairs_per_family: int = 500,
    epochs: int = 200,
    data_seed: int = 3030,
    model_seeds: Sequence[int] = DEFAULT_MODEL_SEEDS,
    margin: float = 0.1,
) -> TrainErrorDiagnosisResult:
    """Reproduce Ticket 33 training and decompose its train errors."""
    _validate_arguments(
        candidate_pool_per_family,
        train_pairs_per_family,
        epochs,
        data_seed,
        margin,
    )
    resolved_model_seeds = _validated_model_seeds(model_seeds)
    destination = Path(output_dir)
    summary_path = destination / "train_error_diagnosis.json"
    report_path = destination / "train_error_diagnosis.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite train-error diagnosis: "
            + ", ".join(str(path) for path in existing)
        )

    candidates = build_unconditioned_relational_candidates(
        candidates_per_family=candidate_pool_per_family,
        seed=data_seed,
    )
    train_twins = _select_first_pairs_per_family(
        candidates,
        pairs_per_family=train_pairs_per_family,
    )
    train_states = tuple(
        state for twin in train_twins for state in (twin.before, twin.after)
    )

    methods: dict[str, dict[str, object]] = {}
    for method in TRAINABLE_METHODS:
        per_seed: list[dict[str, object]] = []
        for model_seed in resolved_model_seeds:
            model = _build_model(method, seed=model_seed)
            initial_agreement = _model_agreement(model, train_states)
            learning_curve = _train_model(
                model,
                train_states,
                epochs=epochs,
                training_objective="bounded_margin",
                margin=margin,
            )
            logits, diagnostics = _model_logits(model, train_states)
            predictions = cast(list[int], torch.argmax(logits, dim=-1).tolist())
            decomposition = _decompose_predictions(
                train_twins,
                train_states,
                predictions,
                logits,
            )
            per_seed.append(
                {
                    "model_seed": model_seed,
                    "initial_train_agreement": initial_agreement,
                    "overall_train_agreement": _state_agreement(
                        predictions, train_states
                    ),
                    "residual_saturation_rate": float(
                        diagnostics.residual_saturation_rate.mean()
                    ),
                    "loss_diagnostics": _loss_diagnostics(learning_curve),
                    **decomposition,
                }
            )
        agreements = [
            cast(float, row["overall_train_agreement"]) for row in per_seed
        ]
        methods[method] = {
            "parameter_count": sum(
                parameter.numel()
                for parameter in _build_model(
                    method, seed=resolved_model_seeds[0]
                ).parameters()
                if parameter.requires_grad
            ),
            "overall_train_agreement_range": [min(agreements), max(agreements)],
            "overall_train_agreement_seed_spread": max(agreements) - min(agreements),
            "per_seed": per_seed,
        }

    observations = _observations(methods)
    formal_protocol_run = {
        "candidate_pool_per_family": candidate_pool_per_family,
        "train_pairs_per_family": train_pairs_per_family,
        "epochs": epochs,
        "data_seed": data_seed,
        "model_seeds": resolved_model_seeds,
        "margin": margin,
    } == FORMAL_PROTOCOL
    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "train_error_diagnosis",
        "ticket": 34,
        "source_ticket": 33,
        "protocol_class": "exploratory_train_only_diagnosis",
        "formal_protocol_run": formal_protocol_run,
        "candidate_pool_per_family": candidate_pool_per_family,
        "train_pairs_per_family": train_pairs_per_family,
        "train_pair_count": len(train_twins),
        "train_state_count": len(train_states),
        "evaluation_pair_count": 0,
        "evaluation_state_count": 0,
        "data_seed": data_seed,
        "model_seeds": list(resolved_model_seeds),
        "training": {
            "objective": "bounded_margin",
            "margin": margin,
            "epochs": epochs,
            "learning_rate": 0.01,
            "batch_order": "single_full_batch_in_candidate_order",
        },
        "data_profile": _data_profile(train_twins, train_states),
        "methods": methods,
        "observations": observations,
        "next_protocol_recommendation": {
            "family_balanced_batches": True,
            "pair_structured_objective": True,
            "retain_state_bounded_margin": True,
            "change_model_capacity": False,
            "reason": (
                "Separate seed sensitivity from the observed gap between state "
                "agreement and exact correctness on both sides of relational pairs."
            ),
        },
        "controls": {
            "held_out_data_read": False,
            "held_out_model_metrics_read": False,
            "held_out_stage_authorized": False,
            "ticket_17_unfrozen": False,
            "ticket_20_unfrozen": False,
            "ticket_32_unblocked": False,
            "production_expert_dataset_written": False,
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return TrainErrorDiagnosisResult(summary_path, report_path)


def _validate_arguments(
    candidate_pool_per_family: int,
    train_pairs_per_family: int,
    epochs: int,
    data_seed: int,
    margin: float,
) -> None:
    for name, value in (
        ("candidate_pool_per_family", candidate_pool_per_family),
        ("train_pairs_per_family", train_pairs_per_family),
        ("epochs", epochs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if candidate_pool_per_family < train_pairs_per_family:
        raise ValueError(
            "candidate_pool_per_family must be >= train_pairs_per_family"
        )
    if isinstance(data_seed, bool) or not isinstance(data_seed, int) or data_seed < 0:
        raise ValueError("data_seed must be a non-negative integer")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not torch.isfinite(torch.tensor(float(margin)))
        or margin <= 0
    ):
        raise ValueError("margin must be positive and finite")


def _state_agreement(
    predictions: Sequence[int], states: Sequence[RelationalState]
) -> float:
    return sum(
        prediction == state.oracle_action
        for prediction, state in zip(predictions, states, strict=True)
    ) / len(states)


def _decompose_predictions(
    twins: Sequence[RelationalTwin],
    states: Sequence[RelationalState],
    predictions: Sequence[int],
    logits: torch.Tensor,
) -> dict[str, object]:
    state_index = {state.state_id: index for index, state in enumerate(states)}
    pair_rows: list[dict[str, object]] = []
    for twin in twins:
        before_index = state_index[twin.before.state_id]
        after_index = state_index[twin.after.state_id]
        before_correct = predictions[before_index] == twin.before.oracle_action
        after_correct = predictions[after_index] == twin.after.oracle_action
        logit_delta = float(
            (logits[before_index] - logits[after_index]).abs().max()
        )
        pair_rows.append(
            {
                "family": twin.family,
                "oracle_flip": twin.before.oracle_action != twin.after.oracle_action,
                "before_correct": before_correct,
                "after_correct": after_correct,
                "both_correct": before_correct and after_correct,
                "prediction_changed": (
                    predictions[before_index] != predictions[after_index]
                ),
                "maximum_absolute_logit_change": logit_delta,
            }
        )

    outcomes: Counter[str] = Counter()
    for row in pair_rows:
        before_correct = cast(bool, row["before_correct"])
        after_correct = cast(bool, row["after_correct"])
        if before_correct and after_correct:
            outcomes["both_correct"] += 1
        elif before_correct:
            outcomes["before_only"] += 1
        elif after_correct:
            outcomes["after_only"] += 1
        else:
            outcomes["neither"] += 1

    return {
        "side_accuracy": {
            "before": _mean_bool(
                [cast(bool, row["before_correct"]) for row in pair_rows]
            ),
            "after": _mean_bool(
                [cast(bool, row["after_correct"]) for row in pair_rows]
            ),
        },
        "pair_outcomes": {
            name: outcomes[name]
            for name in ("both_correct", "before_only", "after_only", "neither")
        },
        "flip_status_metrics": {
            "oracle_flip": _pair_metrics(
                [row for row in pair_rows if cast(bool, row["oracle_flip"])]
            ),
            "oracle_no_flip": _pair_metrics(
                [row for row in pair_rows if not cast(bool, row["oracle_flip"])]
            ),
        },
        "family_metrics": {
            family: _pair_metrics(
                [row for row in pair_rows if row["family"] == family]
            )
            for family in RELATIONAL_FAMILIES
        },
        "action_metrics": _action_metrics(states, predictions),
        "input_response": {
            "mean_maximum_absolute_logit_change": sum(
                cast(float, row["maximum_absolute_logit_change"])
                for row in pair_rows
            )
            / len(pair_rows),
            "nonzero_logit_response_rate": _mean_bool(
                [
                    cast(float, row["maximum_absolute_logit_change"]) > 1e-8
                    for row in pair_rows
                ]
            ),
            "prediction_change_rate": _mean_bool(
                [cast(bool, row["prediction_changed"]) for row in pair_rows]
            ),
        },
    }


def _pair_metrics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    before_accuracy = _mean_bool(
        [cast(bool, row["before_correct"]) for row in rows]
    )
    after_accuracy = _mean_bool(
        [cast(bool, row["after_correct"]) for row in rows]
    )
    return {
        "pair_count": len(rows),
        "before_accuracy": before_accuracy,
        "after_accuracy": after_accuracy,
        "exact_pair_accuracy": _mean_bool(
            [cast(bool, row["both_correct"]) for row in rows]
        ),
        "independent_correctness_reference": before_accuracy * after_accuracy,
        "prediction_change_rate": _mean_bool(
            [cast(bool, row["prediction_changed"]) for row in rows]
        ),
        "mean_maximum_absolute_logit_change": sum(
            cast(float, row["maximum_absolute_logit_change"]) for row in rows
        )
        / len(rows),
    }


def _action_metrics(
    states: Sequence[RelationalState], predictions: Sequence[int]
) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    for action in range(3 * CANDIDATE_TASK_COUNT):
        selected = [
            prediction == state.oracle_action
            for prediction, state in zip(predictions, states, strict=True)
            if state.oracle_action == action
        ]
        metrics[str(action)] = {
            "state_count": len(selected),
            "correct_state_count": sum(selected),
            "accuracy": _mean_bool(selected) if selected else None,
        }
    return metrics


def _loss_diagnostics(losses: Sequence[float]) -> dict[str, float]:
    final_window_start = max(0, len(losses) - 10)
    return {
        "epoch_1": losses[0],
        "midpoint": losses[(len(losses) - 1) // 2],
        "final": losses[-1],
        "final_window_improvement": losses[final_window_start] - losses[-1],
        "relative_total_reduction": (losses[0] - losses[-1]) / losses[0]
        if losses[0] > 0
        else 0.0,
    }


def _data_profile(
    twins: Sequence[RelationalTwin], states: Sequence[RelationalState]
) -> dict[str, object]:
    histogram = Counter(state.oracle_action for state in states)
    family_counts = Counter(twin.family for twin in twins)
    flip_counts = Counter(
        twin.family
        for twin in twins
        if twin.before.oracle_action != twin.after.oracle_action
    )
    return {
        "oracle_action_histogram": {
            str(action): histogram[action]
            for action in range(3 * CANDIDATE_TASK_COUNT)
        },
        "dominant_oracle_action_fraction": max(histogram.values()) / len(states),
        "family_pair_counts": {
            family: family_counts[family] for family in RELATIONAL_FAMILIES
        },
        "family_oracle_flip_pair_counts": {
            family: flip_counts[family] for family in RELATIONAL_FAMILIES
        },
    }


def _observations(methods: dict[str, dict[str, object]]) -> dict[str, object]:
    per_seed = [
        row
        for method in methods.values()
        for row in cast(list[dict[str, object]], method["per_seed"])
    ]
    saturation_values = [
        cast(float, row["residual_saturation_rate"]) for row in per_seed
    ]
    flip_exact = [
        cast(
            float,
            cast(dict[str, dict[str, object]], row["flip_status_metrics"])[
                "oracle_flip"
            ]["exact_pair_accuracy"],
        )
        for row in per_seed
    ]
    return {
        "seed_instability_observed": any(
            cast(float, method["overall_train_agreement_seed_spread"]) >= 0.10
            for method in methods.values()
        ),
        "pair_level_fit_gap_observed": max(flip_exact) < 0.60,
        "residual_saturation_is_primary_failure": max(saturation_values) > 0.25,
        "all_models_respond_to_relational_input": all(
            cast(
                float,
                cast(dict[str, object], row["input_response"])[
                    "nonzero_logit_response_rate"
                ],
            )
            == 1.0
            for row in per_seed
        ),
    }


def _mean_bool(values: Sequence[bool]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty boolean sequence")
    return sum(values) / len(values)


def _render_report(summary: dict[str, object]) -> str:
    methods = cast(dict[str, dict[str, object]], summary["methods"])
    observations = cast(dict[str, bool], summary["observations"])
    run_description = (
        "Formal reproduction of the frozen Ticket 33 training protocol."
        if cast(bool, summary["formal_protocol_run"])
        else "Reduced non-formal run for deterministic interface verification."
    )
    lines = [
        "# Ticket 34 Train Error Diagnosis",
        "",
        run_description,
        "",
        "## Observations",
        "",
        f"- Seed instability observed: **{str(observations['seed_instability_observed']).lower()}**.",
        f"- Pair-level fit gap observed: **{str(observations['pair_level_fit_gap_observed']).lower()}**.",
        f"- Residual saturation is the primary failure: **{str(observations['residual_saturation_is_primary_failure']).lower()}**.",
        f"- All models respond to relational inputs: **{str(observations['all_models_respond_to_relational_input']).lower()}**.",
        "",
        "## Per-Seed Decomposition",
        "",
        "| Method | Seed | Agreement | Before | After | Both correct | Flip exact | No-flip exact | Saturation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_name in TRAINABLE_METHODS:
        per_seed = cast(list[dict[str, object]], methods[method_name]["per_seed"])
        for row in per_seed:
            side = cast(dict[str, float], row["side_accuracy"])
            outcomes = cast(dict[str, int], row["pair_outcomes"])
            split = cast(dict[str, dict[str, float]], row["flip_status_metrics"])
            lines.append(
                f"| {method_name} | {row['model_seed']} | "
                f"{cast(float, row['overall_train_agreement']):.4f} | "
                f"{side['before']:.4f} | {side['after']:.4f} | "
                f"{outcomes['both_correct']} | "
                f"{split['oracle_flip']['exact_pair_accuracy']:.4f} | "
                f"{split['oracle_no_flip']['exact_pair_accuracy']:.4f} | "
                f"{cast(float, row['residual_saturation_rate']):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Next Protocol",
            "",
            "Pre-register a train-only comparison that retains the bounded state margin, uses deterministic family-balanced batches, and adds a pair-structured objective without changing model capacity.",
            "",
            "## Limitations",
            "",
            "This train-only diagnosis contains no held-out model result, does not authorize Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ticket 34 train-error diagnosis")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_train_error_diagnosis(args.output_dir)
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
