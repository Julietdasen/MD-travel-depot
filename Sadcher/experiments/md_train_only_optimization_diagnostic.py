"""Train-only optimization diagnostic over unconditioned relational pairs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast

import torch

from experiments.md_policy_relational_gate import (
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
from experiments.md_unconditioned_relational_audit import (
    build_unconditioned_relational_candidates,
)


TRAINABLE_METHODS: Final = (
    "matched_parameter_mlp",
    "pair_aware_attention",
)
FORMAL_PROTOCOL: Final = {
    "candidate_pool_per_family": 5000,
    "train_pairs_per_family": 500,
    "data_seed": 3030,
    "model_seeds": DEFAULT_MODEL_SEEDS,
    "epochs": 200,
    "training_objective": "bounded_margin",
    "margin": 0.1,
    "minimum_overall_train_agreement": 0.70,
    "minimum_family_flip_pair_exact": 0.60,
    "maximum_residual_saturation_rate": 0.25,
}


@dataclass(frozen=True, slots=True)
class TrainOnlyOptimizationDiagnosticResult:
    summary_path: Path
    report_path: Path


def run_train_only_optimization_diagnostic(
    output_dir: str | Path,
    *,
    candidate_pool_per_family: int = 5000,
    train_pairs_per_family: int = 500,
    epochs: int = 200,
    data_seed: int = 3030,
    model_seeds: Sequence[int] = DEFAULT_MODEL_SEEDS,
    margin: float = 0.1,
    minimum_overall_train_agreement: float = 0.70,
    minimum_family_flip_pair_exact: float = 0.60,
    maximum_residual_saturation_rate: float = 0.25,
) -> TrainOnlyOptimizationDiagnosticResult:
    """Train both probes without constructing or reading an evaluation split."""
    _validate_protocol(
        candidate_pool_per_family=candidate_pool_per_family,
        train_pairs_per_family=train_pairs_per_family,
        epochs=epochs,
        data_seed=data_seed,
        margin=margin,
        minimum_overall_train_agreement=minimum_overall_train_agreement,
        minimum_family_flip_pair_exact=minimum_family_flip_pair_exact,
        maximum_residual_saturation_rate=maximum_residual_saturation_rate,
    )
    resolved_model_seeds = _validated_model_seeds(model_seeds)
    destination = Path(output_dir)
    summary_path = destination / "train_only_optimization_diagnostic.json"
    report_path = destination / "train_only_optimization_diagnostic.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite train-only diagnostic: "
            + ", ".join(str(path) for path in existing)
        )

    candidate_pool = build_unconditioned_relational_candidates(
        candidates_per_family=candidate_pool_per_family,
        seed=data_seed,
    )
    train_twins = _select_first_pairs_per_family(
        candidate_pool,
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
            agreement = _agreement(predictions, train_states)
            saturation = float(diagnostics.residual_saturation_rate.mean())
            family_exact = _flip_pair_exact_by_family(
                train_twins,
                train_states,
                predictions,
                minimum_family_flip_pair_exact,
            )
            overall_passed = agreement >= minimum_overall_train_agreement
            saturation_passed = (
                saturation <= maximum_residual_saturation_rate
            )
            family_passed = all(
                cast(bool, row["criterion_met"])
                for row in family_exact.values()
            )
            per_seed.append(
                {
                    "model_seed": model_seed,
                    "initial_train_agreement": initial_agreement,
                    "overall_train_agreement": agreement,
                    "residual_saturation_rate": saturation,
                    "initial_train_loss": learning_curve[0],
                    "final_train_loss": learning_curve[-1],
                    "oracle_flip_pair_exact_by_family": family_exact,
                    "criteria": {
                        "overall_train_agreement_met": overall_passed,
                        "all_family_flip_pair_exact_met": family_passed,
                        "residual_saturation_rate_met": saturation_passed,
                        "all_training_conditions_met": (
                            overall_passed
                            and family_passed
                            and saturation_passed
                        ),
                    },
                }
            )
        parameter_count = sum(
            parameter.numel()
            for parameter in _build_model(method, seed=resolved_model_seeds[0]).parameters()
            if parameter.requires_grad
        )
        methods[method] = {
            "parameter_count": parameter_count,
            "mean_overall_train_agreement": _mean(
                [cast(float, row["overall_train_agreement"]) for row in per_seed]
            ),
            "mean_residual_saturation_rate": _mean(
                [cast(float, row["residual_saturation_rate"]) for row in per_seed]
            ),
            "all_seed_training_conditions_met": all(
                cast(
                    bool,
                    cast(dict[str, object], row["criteria"])[
                        "all_training_conditions_met"
                    ],
                )
                for row in per_seed
            ),
            "per_seed": per_seed,
        }

    parameter_counts = {
        cast(int, row["parameter_count"]) for row in methods.values()
    }
    parameter_budget_equal = len(parameter_counts) == 1
    all_training_conditions_met = parameter_budget_equal and all(
        cast(bool, row["all_seed_training_conditions_met"])
        for row in methods.values()
    )
    decision = (
        "optimization_feasible"
        if all_training_conditions_met
        else "optimization_not_feasible"
    )
    family_statistics = _selected_family_statistics(train_twins)
    formal_protocol = _matches_formal_protocol(
        candidate_pool_per_family=candidate_pool_per_family,
        train_pairs_per_family=train_pairs_per_family,
        epochs=epochs,
        data_seed=data_seed,
        model_seeds=resolved_model_seeds,
        margin=margin,
        minimum_overall_train_agreement=minimum_overall_train_agreement,
        minimum_family_flip_pair_exact=minimum_family_flip_pair_exact,
        maximum_residual_saturation_rate=maximum_residual_saturation_rate,
    )
    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "train_only_optimization_diagnostic",
        "report_phase": "train_only_optimization_diagnostic",
        "protocol_class": "exploratory",
        "formal_protocol_run": formal_protocol,
        "data_source": "ticket_30_unconditioned_relational_candidate_generator",
        "candidate_pool_per_family": candidate_pool_per_family,
        "train_pairs_per_family": train_pairs_per_family,
        "candidate_pool_pair_count": len(candidate_pool),
        "train_pair_count": len(train_twins),
        "train_state_count": len(train_states),
        "evaluation_pair_count": 0,
        "evaluation_state_count": 0,
        "data_seed": data_seed,
        "model_seeds": list(resolved_model_seeds),
        "relational_families": list(RELATIONAL_FAMILIES),
        "training": {
            "objective": "bounded_margin",
            "margin": margin,
            "epochs": epochs,
            "learning_rate": 0.01,
            "batch_order": "single_full_batch_in_candidate_order",
        },
        "thresholds": {
            "minimum_overall_train_agreement": minimum_overall_train_agreement,
            "minimum_family_flip_pair_exact": minimum_family_flip_pair_exact,
            "maximum_residual_saturation_rate": maximum_residual_saturation_rate,
        },
        "selection_controls": {
            "label_used_for_selection": False,
            "margin_used_for_selection": False,
            "model_output_used_for_selection": False,
            "oracle_flip_used_for_selection": False,
            "selection_order": "first_generated_pairs_per_family",
        },
        "family_statistics": family_statistics,
        "controls": {
            "held_out_data_read": False,
            "model_evaluation_run": False,
            "parameter_budget_equal": parameter_budget_equal,
            "ticket_17_unfrozen": False,
            "ticket_20_unfrozen": False,
            "ticket_32_unblocked": False,
            "production_expert_dataset_written": False,
        },
        "methods": methods,
        "all_training_conditions_met": all_training_conditions_met,
        "decision": decision,
        "interpretation": (
            "The fixed training conditions were met; this only supports running a separately frozen evaluation later."
            if all_training_conditions_met
            else "At least one fixed training condition failed; stop without viewing a new held-out evaluation."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return TrainOnlyOptimizationDiagnosticResult(summary_path, report_path)


def _validate_protocol(
    *,
    candidate_pool_per_family: int,
    train_pairs_per_family: int,
    epochs: int,
    data_seed: int,
    margin: float,
    minimum_overall_train_agreement: float,
    minimum_family_flip_pair_exact: float,
    maximum_residual_saturation_rate: float,
) -> None:
    for name, integer_value in (
        ("candidate_pool_per_family", candidate_pool_per_family),
        ("train_pairs_per_family", train_pairs_per_family),
        ("epochs", epochs),
    ):
        if (
            isinstance(integer_value, bool)
            or not isinstance(integer_value, int)
            or integer_value <= 0
        ):
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
        or not math.isfinite(margin)
        or margin <= 0
    ):
        raise ValueError("margin must be positive and finite")
    for name, rate in (
        ("minimum_overall_train_agreement", minimum_overall_train_agreement),
        ("minimum_family_flip_pair_exact", minimum_family_flip_pair_exact),
        ("maximum_residual_saturation_rate", maximum_residual_saturation_rate),
    ):
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or not 0 <= rate <= 1
        ):
            raise ValueError(f"{name} must be finite and within [0, 1]")


def _select_first_pairs_per_family(
    candidates: Sequence[RelationalTwin],
    *,
    pairs_per_family: int,
) -> tuple[RelationalTwin, ...]:
    selected: list[RelationalTwin] = []
    for family in RELATIONAL_FAMILIES:
        rows = [twin for twin in candidates if twin.family == family]
        if len(rows) < pairs_per_family:
            raise ValueError(
                f"not enough unconditioned candidates for {family}: "
                f"need {pairs_per_family}, found {len(rows)}"
            )
        selected.extend(rows[:pairs_per_family])
    return tuple(selected)


def _agreement(
    predictions: Sequence[int], states: Sequence[RelationalState]
) -> float:
    return sum(
        prediction == state.oracle_action
        for prediction, state in zip(predictions, states, strict=True)
    ) / len(states)


def _flip_pair_exact_by_family(
    twins: Sequence[RelationalTwin],
    states: Sequence[RelationalState],
    predictions: Sequence[int],
    threshold: float,
) -> dict[str, dict[str, object]]:
    state_index = {state.state_id: index for index, state in enumerate(states)}
    result: dict[str, dict[str, object]] = {}
    for family in RELATIONAL_FAMILIES:
        flip_twins = [
            twin
            for twin in twins
            if twin.family == family
            and twin.before.oracle_action != twin.after.oracle_action
        ]
        exact_count = sum(
            (
                predictions[state_index[twin.before.state_id]],
                predictions[state_index[twin.after.state_id]],
            )
            == (twin.before.oracle_action, twin.after.oracle_action)
            for twin in flip_twins
        )
        accuracy = exact_count / len(flip_twins) if flip_twins else None
        result[family] = {
            "oracle_flip_pair_count": len(flip_twins),
            "exact_pair_count": exact_count,
            "exact_pair_accuracy": accuracy,
            "criterion_met": accuracy is not None and accuracy >= threshold,
        }
    return result


def _selected_family_statistics(
    twins: Sequence[RelationalTwin],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for family in RELATIONAL_FAMILIES:
        rows = [twin for twin in twins if twin.family == family]
        flip_count = sum(
            twin.before.oracle_action != twin.after.oracle_action for twin in rows
        )
        result[family] = {
            "selected_pair_count": len(rows),
            "oracle_flip_pair_count": flip_count,
            "oracle_no_flip_pair_count": len(rows) - flip_count,
            "natural_oracle_flip_rate": flip_count / len(rows),
            "first_pair_id": rows[0].pair_id,
            "last_pair_id": rows[-1].pair_id,
        }
    return result


def _matches_formal_protocol(
    *,
    candidate_pool_per_family: int,
    train_pairs_per_family: int,
    epochs: int,
    data_seed: int,
    model_seeds: tuple[int, ...],
    margin: float,
    minimum_overall_train_agreement: float,
    minimum_family_flip_pair_exact: float,
    maximum_residual_saturation_rate: float,
) -> bool:
    return {
        "candidate_pool_per_family": candidate_pool_per_family,
        "train_pairs_per_family": train_pairs_per_family,
        "data_seed": data_seed,
        "model_seeds": model_seeds,
        "epochs": epochs,
        "training_objective": "bounded_margin",
        "margin": margin,
        "minimum_overall_train_agreement": minimum_overall_train_agreement,
        "minimum_family_flip_pair_exact": minimum_family_flip_pair_exact,
        "maximum_residual_saturation_rate": maximum_residual_saturation_rate,
    } == FORMAL_PROTOCOL


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _render_report(summary: dict[str, object]) -> str:
    methods = cast(dict[str, dict[str, object]], summary["methods"])
    thresholds = cast(dict[str, float], summary["thresholds"])
    lines = [
        "# Train-Only Optimization Diagnostic",
        "",
        f"Decision: **{summary['decision']}**.",
        "",
        "This exploratory run measures optimization on the selected training pool only. It contains no held-out model result and does not establish relational generalization.",
        "",
        "## Frozen Protocol",
        "",
        f"- Candidate pool per family: {summary['candidate_pool_per_family']}",
        f"- First training pairs per family: {summary['train_pairs_per_family']}",
        f"- Data seed: {summary['data_seed']}",
        f"- Model seeds: {', '.join(str(seed) for seed in cast(list[int], summary['model_seeds']))}",
        f"- Bounded-margin objective: {cast(dict[str, object], summary['training'])['margin']}",
        f"- Epochs: {cast(dict[str, object], summary['training'])['epochs']}",
        f"- Overall agreement threshold: {thresholds['minimum_overall_train_agreement']:.2f}",
        f"- Per-family flip-pair exact threshold: {thresholds['minimum_family_flip_pair_exact']:.2f}",
        f"- Saturation threshold: {thresholds['maximum_residual_saturation_rate']:.2f}",
        "",
        "## Per-Seed Results",
        "",
        "| Method | Seed | Overall agreement | Saturation | All conditions |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in TRAINABLE_METHODS:
        per_seed = cast(list[dict[str, object]], methods[method]["per_seed"])
        for row in per_seed:
            criteria = cast(dict[str, bool], row["criteria"])
            lines.append(
                f"| {method} | {row['model_seed']} | "
                f"{cast(float, row['overall_train_agreement']):.4f} | "
                f"{cast(float, row['residual_saturation_rate']):.4f} | "
                f"{str(criteria['all_training_conditions_met']).lower()} |"
            )
    lines.extend(
        [
            "",
            "## Oracle-Flip Pair Exact",
            "",
            "| Method | Seed | Family | Flip pairs | Exact accuracy | Pass |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for method in TRAINABLE_METHODS:
        per_seed = cast(list[dict[str, object]], methods[method]["per_seed"])
        for row in per_seed:
            family_rows = cast(
                dict[str, dict[str, object]],
                row["oracle_flip_pair_exact_by_family"],
            )
            for family in RELATIONAL_FAMILIES:
                family_row = family_rows[family]
                accuracy = family_row["exact_pair_accuracy"]
                rendered_accuracy = (
                    "n/a" if accuracy is None else f"{cast(float, accuracy):.4f}"
                )
                lines.append(
                    f"| {method} | {row['model_seed']} | {family} | "
                    f"{family_row['oracle_flip_pair_count']} | {rendered_accuracy} | "
                    f"{str(family_row['criterion_met']).lower()} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            cast(str, summary["interpretation"]),
            "",
            "Ticket 17, Ticket 20, and Ticket 32 remain unchanged. No production expert dataset was generated.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen train-only optimization diagnostic"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_train_only_optimization_diagnostic(args.output_dir)
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
