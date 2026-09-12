"""Evaluate relational architectures on the frozen Ticket 39 package."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast

import torch

from experiments.md_budgeted_saturation_curriculum_diagnostic import (
    FORMAL_PROTOCOL as TRAINING_PROTOCOL,
)
from experiments.md_flip_curriculum_train_diagnostic import (
    _family_balanced_batches,
    _prepare_batch,
    _train_curriculum,
)
from experiments.md_independent_held_out_package import (
    FORMAL_PROTOCOL as HELD_OUT_PROTOCOL,
    HELD_OUT_MARGIN_STRATA,
    _evaluation_record,
    _pair_margin_stratum,
    _select_evaluation_pairs,
)
from experiments.md_policy_relational_gate import (
    DEFAULT_MODEL_SEEDS,
    RELATIONAL_FAMILIES,
    RelationalState,
    RelationalTwin,
    _bootstrap_mean_ci,
    _build_model,
    _model_logits,
    _template_signature,
    _validated_model_seeds,
)
from experiments.md_relational_data_audit import (
    BASELINE_NAMES,
    _baseline_prediction,
)
from experiments.md_train_only_optimization_diagnostic import (
    TRAINABLE_METHODS,
    _select_first_pairs_per_family,
)
from experiments.md_unconditioned_relational_audit import (
    build_unconditioned_relational_candidates,
)


MODEL_METHODS: Final = (
    "matched_parameter_mlp",
    "pair_aware_attention",
)
COMPARISON_GAIN: Final = 0.05
MINIMUM_FAMILY_PAIR_ACCURACY: Final = 0.60
UTILITY_TOLERANCE: Final = 0.05
REPORT_CRITERIA: Final = (
    "state_agreement_gain_at_least_0.05",
    "exact_pair_accuracy_gain_at_least_0.05",
    "all_family_exact_pair_accuracy_at_least_0.60",
    "paired_bootstrap_state_agreement_ci_lower_above_zero",
    "mean_evaluation_residual_saturation_at_most_0.25",
)


@dataclass(frozen=True, slots=True)
class IndependentHeldOutEvaluationResult:
    summary_path: Path
    report_path: Path


def run_independent_held_out_evaluation(
    output_dir: str | Path,
    *,
    package_path: str | Path,
    pairs_per_family_per_batch: int = 100,
    pretrain_epochs: int = 200,
    fine_tune_epochs: int = 100,
    checkpoint_interval: int = 10,
    model_seeds: Sequence[int] = DEFAULT_MODEL_SEEDS,
    pretrain_learning_rate: float = 0.01,
    fine_tune_learning_rate: float = 0.002,
    state_margin: float = 0.1,
    flip_state_margin: float = 0.1,
    pair_margin: float = 0.1,
    flip_state_loss_weight: float = 0.5,
    pair_loss_weight: float = 0.25,
    saturation_loss_weight: float = 0.1,
    saturation_penalty_allowance: float = 0.25,
    minimum_overall_train_agreement: float = 0.70,
    minimum_family_flip_pair_exact: float = 0.60,
    maximum_residual_saturation_rate: float = 0.25,
) -> IndependentHeldOutEvaluationResult:
    """Replay the committed package and evaluate fixed trained methods."""
    destination = Path(output_dir)
    summary_path = destination / "independent_held_out_evaluation.json"
    report_path = destination / "independent_held_out_evaluation.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite independent held-out evaluation: "
            + ", ".join(str(path) for path in existing)
        )
    resolved_model_seeds = _validated_model_seeds(model_seeds)
    package = _load_package(Path(package_path))
    train_provenance = cast(dict[str, object], package["train_provenance"])
    evaluation_provenance = cast(
        dict[str, object], package["evaluation_provenance"]
    )
    candidate_pool_per_family = cast(
        int, train_provenance["candidate_pool_per_family"]
    )
    train_pairs_per_family = cast(int, train_provenance["pairs_per_family"])
    train_seed = cast(int, train_provenance["seed"])
    evaluation_seed = cast(int, evaluation_provenance["seed"])
    perturbation_seed = cast(int, package["perturbation_seed"])
    quota_per_family_stratum = cast(
        int, evaluation_provenance["quota_per_family_stratum"]
    )
    train_twins, evaluation_twins, mismatch_count, template_overlap = (
        _replay_package(
            package,
            candidate_pool_per_family=candidate_pool_per_family,
            train_pairs_per_family=train_pairs_per_family,
            train_seed=train_seed,
            evaluation_seed=evaluation_seed,
            perturbation_seed=perturbation_seed,
            quota_per_family_stratum=quota_per_family_stratum,
        )
    )
    if mismatch_count:
        raise ValueError("Ticket 39 package replay did not match frozen records")
    if template_overlap:
        raise ValueError("Ticket 39 package overlaps training templates")

    training_protocol = {
        "candidate_pool_per_family": candidate_pool_per_family,
        "train_pairs_per_family": train_pairs_per_family,
        "pairs_per_family_per_batch": pairs_per_family_per_batch,
        "pretrain_epochs": pretrain_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "checkpoint_interval": checkpoint_interval,
        "data_seed": train_seed,
        "model_seeds": resolved_model_seeds,
        "pretrain_learning_rate": pretrain_learning_rate,
        "fine_tune_learning_rate": fine_tune_learning_rate,
        "state_margin": state_margin,
        "flip_state_margin": flip_state_margin,
        "pair_margin": pair_margin,
        "flip_state_loss_weight": flip_state_loss_weight,
        "pair_loss_weight": pair_loss_weight,
        "saturation_loss_weight": saturation_loss_weight,
        "saturation_penalty_allowance": saturation_penalty_allowance,
        "minimum_overall_train_agreement": minimum_overall_train_agreement,
        "minimum_family_flip_pair_exact": minimum_family_flip_pair_exact,
        "maximum_residual_saturation_rate": maximum_residual_saturation_rate,
    }
    formal_protocol_run = training_protocol == TRAINING_PROTOCOL
    train_states = _states(train_twins)
    evaluation_states = _states(evaluation_twins)
    batches = _family_balanced_batches(
        train_twins,
        pairs_per_family_per_batch=pairs_per_family_per_batch,
    )
    prepared_batches = tuple(_prepare_batch(batch) for batch in batches)

    models: dict[str, dict[str, object]] = {}
    for method in MODEL_METHODS:
        per_seed: list[dict[str, object]] = []
        for model_seed in resolved_model_seeds:
            model = _build_model(method, seed=model_seed)
            training = _train_curriculum(
                model,
                prepared_batches,
                train_twins,
                train_states,
                pretrain_epochs=pretrain_epochs,
                fine_tune_epochs=fine_tune_epochs,
                checkpoint_interval=checkpoint_interval,
                pretrain_learning_rate=pretrain_learning_rate,
                fine_tune_learning_rate=fine_tune_learning_rate,
                state_margin=state_margin,
                flip_state_margin=flip_state_margin,
                pair_margin=pair_margin,
                flip_state_loss_weight=flip_state_loss_weight,
                pair_loss_weight=pair_loss_weight,
                saturation_loss_weight=saturation_loss_weight,
                saturation_penalty_allowance=saturation_penalty_allowance,
                minimum_overall_train_agreement=(
                    minimum_overall_train_agreement
                ),
                minimum_family_flip_pair_exact=(
                    minimum_family_flip_pair_exact
                ),
                maximum_residual_saturation_rate=(
                    maximum_residual_saturation_rate
                ),
            )
            logits, diagnostics = _model_logits(model, evaluation_states)
            predictions = cast(
                list[int], torch.argmax(logits, dim=-1).tolist()
            )
            state_saturation_rates = [
                float(value)
                for value in diagnostics.residual_saturation_rate.tolist()
            ]
            per_seed.append(
                {
                    "model_seed": model_seed,
                    "training": training,
                    "evaluation": _prediction_metrics(
                        evaluation_twins,
                        evaluation_states,
                        predictions,
                        state_saturation_rates=state_saturation_rates,
                    ),
                    "evaluation_residual_saturation_rate": float(
                        diagnostics.residual_saturation_rate.mean()
                    ),
                }
            )
        parameter_count = sum(
            parameter.numel()
            for parameter in _build_model(
                method, seed=resolved_model_seeds[0]
            ).parameters()
            if parameter.requires_grad
        )
        models[method] = {
            "parameter_count": parameter_count,
            "per_seed": per_seed,
            "aggregate": _aggregate_model(per_seed),
        }

    baselines = {
        name: _prediction_metrics(
            evaluation_twins,
            evaluation_states,
            [_baseline_prediction(name, state) for state in evaluation_states],
        )
        for name in BASELINE_NAMES
    }
    comparison = _comparison(models)
    training_valid = all(
        cast(
            bool,
            cast(
                dict[str, bool],
                cast(dict[str, object], row["training"])["criteria"],
            )["all_training_conditions_met"],
        )
        for model in models.values()
        for row in cast(list[dict[str, object]], model["per_seed"])
    )
    parameter_budget_equal = len(
        {cast(int, model["parameter_count"]) for model in models.values()}
    ) == 1
    criteria = cast(dict[str, bool], comparison["criteria"])
    comparison_valid = training_valid and parameter_budget_equal
    supported = comparison_valid and all(criteria.values())
    decision = (
        (
            "relational_generalization_supported"
            if supported
            else (
                "relational_generalization_not_supported"
                if comparison_valid
                else "comparison_invalid"
            )
        )
        if formal_protocol_run
        else "diagnostic_only"
    )
    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "independent_held_out_architecture_evaluation",
        "ticket": 40,
        "source_ticket": 39,
        "formal_protocol_run": formal_protocol_run,
        "decision": decision,
        "package_provenance": {
            "ticket": package["ticket"],
            "status": package["status"],
            "formal_protocol_run": package["formal_protocol_run"],
            "train_seed": train_seed,
            "evaluation_seed": evaluation_seed,
            "perturbation_seed": perturbation_seed,
        },
        "training_protocol": training_protocol,
        "model_seeds": list(resolved_model_seeds),
        "train_pair_count": len(train_twins),
        "train_state_count": len(train_states),
        "evaluation_pair_count": len(evaluation_twins),
        "evaluation_state_count": len(evaluation_states),
        "package_replay_mismatch_count": mismatch_count,
        "train_evaluation_template_overlap": template_overlap,
        "models": models,
        "baselines": baselines,
        "comparison": comparison,
        "comparison_valid": comparison_valid,
        "controls": {
            "parameter_budget_equal": parameter_budget_equal,
            "frozen_package_modified": False,
            "production_expert_dataset_written": False,
            "ticket_17_unfrozen": False,
            "ticket_20_unfrozen": False,
            "ticket_32_unblocked": False,
        },
        "next_step": {
            "relational_generalization_supported": (
                "The frozen held-out relational architecture gate is supported."
            ),
            "relational_generalization_not_supported": (
                "The frozen held-out relational architecture gate remains closed."
            ),
            "comparison_invalid": (
                "The held-out architecture comparison remains invalid."
            ),
            "diagnostic_only": (
                "This reduced run provides deterministic interface verification only."
            ),
        }[decision],
    }
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return IndependentHeldOutEvaluationResult(summary_path, report_path)


def _load_package(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("held-out package must contain a JSON object")
    required = {
        "status",
        "ticket",
        "formal_protocol_run",
        "train_provenance",
        "evaluation_provenance",
        "perturbation_seed",
        "pair_margin_definition",
        "pair_margin_strata",
        "evaluation_pair_count",
        "evaluation_state_count",
        "evaluation_records",
        "train_evaluation_template_overlap",
        "selection_controls",
        "controls",
    }
    if missing := sorted(required - payload.keys()):
        raise ValueError(f"held-out package is missing fields: {missing}")
    if payload["ticket"] != 39 or payload["status"] != (
        "frozen_independent_held_out_relational_package"
    ):
        raise ValueError("held-out package is not a Ticket 39 package")

    candidate_pool = cast(
        int, HELD_OUT_PROTOCOL["candidate_pool_per_family"]
    )
    train_pairs = cast(int, HELD_OUT_PROTOCOL["train_pairs_per_family"])
    quota = cast(int, HELD_OUT_PROTOCOL["quota_per_family_stratum"])
    expected_pair_count = (
        len(RELATIONAL_FAMILIES) * len(HELD_OUT_MARGIN_STRATA) * quota
    )
    expected_train_provenance = {
        "candidate_pool_per_family": candidate_pool,
        "pairs_per_family": train_pairs,
        "seed": HELD_OUT_PROTOCOL["train_seed"],
        "selection_order": "first_generated_pairs_per_family",
    }
    expected_evaluation_provenance = {
        "candidate_pool_per_family": candidate_pool,
        "quota_per_family_stratum": quota,
        "seed": HELD_OUT_PROTOCOL["evaluation_seed"],
        "selection_order": (
            "family_then_margin_stratum_then_candidate_generation_order"
        ),
    }
    expected_strata = {
        name: {
            "lower": lower,
            "upper": None if math.isinf(upper) else upper,
        }
        for name, lower, upper in HELD_OUT_MARGIN_STRATA
    }
    expected_selection_controls = {
        "family_used_for_selection": True,
        "pair_margin_used_for_selection": True,
        "oracle_action_used_for_selection": False,
        "oracle_flip_used_for_selection": False,
        "model_output_used_for_selection": False,
        "oracle_labels_observed_only_after_selection": True,
    }
    expected_controls = {
        "production_expert_dataset_written": False,
        "model_training_or_evaluation_run": False,
    }
    records = payload["evaluation_records"]
    protocol_matches = (
        payload["formal_protocol_run"] is True
        and payload["train_provenance"] == expected_train_provenance
        and payload["evaluation_provenance"]
        == expected_evaluation_provenance
        and payload["perturbation_seed"]
        == HELD_OUT_PROTOCOL["perturbation_seed"]
        and payload["pair_margin_definition"]
        == (
            "min(before_oracle_completion_margin, "
            "after_oracle_completion_margin)"
        )
        and payload["pair_margin_strata"] == expected_strata
        and payload["evaluation_pair_count"] == expected_pair_count
        and payload["evaluation_state_count"] == 2 * expected_pair_count
        and isinstance(records, list)
        and len(records) == expected_pair_count
        and payload["train_evaluation_template_overlap"] == 0
        and payload["selection_controls"] == expected_selection_controls
        and payload["controls"] == expected_controls
    )
    if not protocol_matches:
        raise ValueError(
            "held-out package does not match frozen Ticket 39 protocol"
        )
    return cast(dict[str, object], payload)


def _replay_package(
    package: dict[str, object],
    *,
    candidate_pool_per_family: int,
    train_pairs_per_family: int,
    train_seed: int,
    evaluation_seed: int,
    perturbation_seed: int,
    quota_per_family_stratum: int,
) -> tuple[
    tuple[RelationalTwin, ...],
    tuple[RelationalTwin, ...],
    int,
    int,
]:
    train_candidates = build_unconditioned_relational_candidates(
        candidates_per_family=candidate_pool_per_family,
        seed=train_seed,
    )
    train_twins = _select_first_pairs_per_family(
        train_candidates,
        pairs_per_family=train_pairs_per_family,
    )
    evaluation_candidates = build_unconditioned_relational_candidates(
        candidates_per_family=candidate_pool_per_family,
        seed=evaluation_seed,
        perturbation_seed=perturbation_seed,
    )
    expected_selection, _cells, insufficient = _select_evaluation_pairs(
        evaluation_candidates,
        quota_per_family_stratum=quota_per_family_stratum,
    )
    if insufficient:
        raise ValueError("frozen Ticket 39 quota is not available on replay")
    expected_records = [
        _evaluation_record(candidate_index, twin)
        for candidate_index, twin in expected_selection
    ]
    records = cast(list[dict[str, object]], package["evaluation_records"])
    mismatch_count = abs(len(records) - len(expected_records)) + sum(
        record != expected
        for record, expected in zip(records, expected_records)
    )
    selected = tuple(twin for _candidate_index, twin in expected_selection)
    overlap = {
        _template_signature(twin) for twin in train_twins
    } & {_template_signature(twin) for twin in selected}
    return train_twins, selected, mismatch_count, len(overlap)


def _states(twins: Sequence[RelationalTwin]) -> tuple[RelationalState, ...]:
    return tuple(
        state for twin in twins for state in (twin.before, twin.after)
    )


def _prediction_metrics(
    twins: Sequence[RelationalTwin],
    states: Sequence[RelationalState],
    predictions: Sequence[int],
    *,
    state_saturation_rates: Sequence[float] | None = None,
) -> dict[str, object]:
    saturation_rates = (
        state_saturation_rates
        if state_saturation_rates is not None
        else (0.0,) * len(states)
    )
    pair_rows: list[dict[str, object]] = []
    for pair_index, twin in enumerate(twins):
        before_index = 2 * pair_index
        after_index = before_index + 1
        before_prediction = predictions[before_index]
        after_prediction = predictions[after_index]
        before_correct = before_prediction == twin.before.oracle_action
        after_correct = after_prediction == twin.after.oracle_action
        before_regret = (
            max(twin.before.action_values)
            - twin.before.action_values[before_prediction]
        )
        after_regret = (
            max(twin.after.action_values)
            - twin.after.action_values[after_prediction]
        )
        oracle_flipped = twin.before.oracle_action != twin.after.oracle_action
        pair_rows.append(
            {
                "family": twin.family,
                "pair_margin_stratum": _pair_margin_stratum(
                    min(
                        _completion_margin(twin.before),
                        _completion_margin(twin.after),
                    )
                ),
                "oracle_flip_status": (
                    "oracle_flip" if oracle_flipped else "oracle_no_flip"
                ),
                "state_agreement": (before_correct + after_correct) / 2,
                "exact_pair": before_correct and after_correct,
                "residual_saturation_rate": (
                    saturation_rates[before_index]
                    + saturation_rates[after_index]
                )
                / 2,
                "prediction_flip_status_correct": (
                    (before_prediction != after_prediction) == oracle_flipped
                ),
                "mean_completion_proxy_regret": (
                    before_regret + after_regret
                )
                / 2,
                "utility_aware_accuracy_at_0.05": (
                    (before_regret <= UTILITY_TOLERANCE)
                    + (after_regret <= UTILITY_TOLERANCE)
                )
                / 2,
            }
        )
    return {
        "overall": _group_metrics(pair_rows),
        "by_family": {
            family: _group_metrics(
                [row for row in pair_rows if row["family"] == family]
            )
            for family in RELATIONAL_FAMILIES
        },
        "by_pair_margin_stratum": {
            stratum: _group_metrics(
                [
                    row
                    for row in pair_rows
                    if row["pair_margin_stratum"] == stratum
                ]
            )
            for stratum in (
                "near_tie_lt_0.01",
                "small_ge_0.01_lt_0.03",
                "medium_ge_0.03_lt_0.05",
                "high_ge_0.05",
            )
        },
        "by_oracle_flip_status": {
            status: _group_metrics(
                [
                    row
                    for row in pair_rows
                    if row["oracle_flip_status"] == status
                ]
            )
            for status in ("oracle_flip", "oracle_no_flip")
        },
        "pair_state_agreement_values": [
            cast(float, row["state_agreement"]) for row in pair_rows
        ],
        "pair_exact_values": [
            int(cast(bool, row["exact_pair"])) for row in pair_rows
        ],
    }


def _completion_margin(state: RelationalState) -> float:
    values = sorted(state.action_values, reverse=True)
    return values[0] - values[1]


def _group_metrics(rows: Sequence[dict[str, object]]) -> dict[str, int | float]:
    return {
        "pair_count": len(rows),
        "state_count": 2 * len(rows),
        "state_agreement": _mean_key(rows, "state_agreement"),
        "exact_pair_accuracy": _mean_bool_key(rows, "exact_pair"),
        "prediction_flip_status_accuracy": _mean_bool_key(
            rows, "prediction_flip_status_correct"
        ),
        "mean_completion_proxy_regret": _mean_key(
            rows, "mean_completion_proxy_regret"
        ),
        "utility_aware_accuracy_at_0.05": _mean_key(
            rows, "utility_aware_accuracy_at_0.05"
        ),
        "residual_saturation_rate": _mean_key(
            rows, "residual_saturation_rate"
        ),
    }


def _mean_key(rows: Sequence[dict[str, object]], key: str) -> float:
    return sum(cast(float, row[key]) for row in rows) / len(rows)


def _mean_bool_key(rows: Sequence[dict[str, object]], key: str) -> float:
    return sum(cast(bool, row[key]) for row in rows) / len(rows)


def _aggregate_model(per_seed: Sequence[dict[str, object]]) -> dict[str, object]:
    evaluations = [
        cast(dict[str, object], row["evaluation"]) for row in per_seed
    ]
    overall_rows = [
        cast(dict[str, int | float], evaluation["overall"])
        for evaluation in evaluations
    ]
    return {
        "overall": _aggregate_group(overall_rows),
        "by_family": _aggregate_grouping(evaluations, "by_family"),
        "by_pair_margin_stratum": _aggregate_grouping(
            evaluations, "by_pair_margin_stratum"
        ),
        "by_oracle_flip_status": _aggregate_grouping(
            evaluations, "by_oracle_flip_status"
        ),
        "mean_evaluation_residual_saturation_rate": sum(
            cast(float, row["evaluation_residual_saturation_rate"])
            for row in per_seed
        )
        / len(per_seed),
        "pair_state_agreement_values": _mean_vectors(
            [
                cast(list[float], evaluation["pair_state_agreement_values"])
                for evaluation in evaluations
            ]
        ),
        "pair_exact_values": _mean_vectors(
            [
                cast(list[float], evaluation["pair_exact_values"])
                for evaluation in evaluations
            ]
        ),
    }


def _aggregate_grouping(
    evaluations: Sequence[dict[str, object]], key: str
) -> dict[str, dict[str, int | float]]:
    first = cast(dict[str, object], evaluations[0][key])
    return {
        group: _aggregate_group(
            [
                cast(
                    dict[str, int | float],
                    cast(dict[str, object], evaluation[key])[group],
                )
                for evaluation in evaluations
            ]
        )
        for group in first
    }


def _aggregate_group(
    rows: Sequence[dict[str, int | float]],
) -> dict[str, int | float]:
    return {
        key: (
            cast(int, rows[0][key])
            if key in ("pair_count", "state_count")
            else sum(float(row[key]) for row in rows) / len(rows)
        )
        for key in rows[0]
    }


def _mean_vectors(rows: Sequence[Sequence[float]]) -> list[float]:
    return [
        sum(values) / len(values)
        for values in zip(*rows, strict=True)
    ]


def _comparison(models: dict[str, dict[str, object]]) -> dict[str, object]:
    matched = cast(
        dict[str, object], models["matched_parameter_mlp"]["aggregate"]
    )
    pair_aware = cast(
        dict[str, object], models["pair_aware_attention"]["aggregate"]
    )
    matched_overall = cast(dict[str, int | float], matched["overall"])
    pair_overall = cast(dict[str, int | float], pair_aware["overall"])
    agreement_gain = float(pair_overall["state_agreement"]) - float(
        matched_overall["state_agreement"]
    )
    pair_accuracy_gain = float(pair_overall["exact_pair_accuracy"]) - float(
        matched_overall["exact_pair_accuracy"]
    )
    differences = [
        pair_value - matched_value
        for pair_value, matched_value in zip(
            cast(list[float], pair_aware["pair_state_agreement_values"]),
            cast(list[float], matched["pair_state_agreement_values"]),
            strict=True,
        )
    ]
    difference_ci95 = list(_bootstrap_mean_ci(differences, seed=4040))
    pair_families = cast(
        dict[str, dict[str, int | float]], pair_aware["by_family"]
    )
    matched_strata = cast(
        dict[str, dict[str, int | float]],
        matched["by_pair_margin_stratum"],
    )
    pair_strata = cast(
        dict[str, dict[str, int | float]],
        pair_aware["by_pair_margin_stratum"],
    )
    stratum_gains = {
        stratum: float(pair_strata[stratum]["state_agreement"])
        - float(matched_strata[stratum]["state_agreement"])
        for stratum in pair_strata
    }
    criteria = {
        "state_agreement_gain_at_least_0.05": (
            agreement_gain >= COMPARISON_GAIN
        ),
        "exact_pair_accuracy_gain_at_least_0.05": (
            pair_accuracy_gain >= COMPARISON_GAIN
        ),
        "all_family_exact_pair_accuracy_at_least_0.60": all(
            float(row["exact_pair_accuracy"])
            >= MINIMUM_FAMILY_PAIR_ACCURACY
            for row in pair_families.values()
        ),
        "paired_bootstrap_state_agreement_ci_lower_above_zero": (
            difference_ci95[0] > 0
        ),
        "mean_evaluation_residual_saturation_at_most_0.25": (
            cast(
                float,
                pair_aware["mean_evaluation_residual_saturation_rate"],
            )
            <= 0.25
        ),
    }
    return {
        "pair_aware_minus_matched_state_agreement": agreement_gain,
        "pair_aware_minus_matched_exact_pair_accuracy": pair_accuracy_gain,
        "paired_bootstrap_state_agreement_difference_ci95": difference_ci95,
        "pair_aware_family_exact_pair_accuracy": {
            family: row["exact_pair_accuracy"]
            for family, row in pair_families.items()
        },
        "pair_aware_minus_matched_state_agreement_by_stratum": stratum_gains,
        "aggregate_advantage_only_near_tie": (
            agreement_gain > 0
            and stratum_gains["near_tie_lt_0.01"] > 0
            and max(
                gain
                for stratum, gain in stratum_gains.items()
                if stratum != "near_tie_lt_0.01"
            )
            <= 0
        ),
        "criteria": criteria,
    }


def _render_report(summary: dict[str, object]) -> str:
    models = cast(dict[str, dict[str, object]], summary["models"])
    baselines = cast(dict[str, dict[str, object]], summary["baselines"])
    comparison = cast(dict[str, object], summary["comparison"])
    lines = [
        "# Ticket 40 Independent Held-Out Architecture Evaluation",
        "",
        f"Decision: **{summary['decision']}**.",
        "",
        "## Aggregate Evaluation",
        "",
        "| Method | State agreement | Exact pair | Flip status | Regret | Utility@0.05 | Saturation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in MODEL_METHODS:
        aggregate = cast(dict[str, object], models[method]["aggregate"])
        overall = cast(dict[str, int | float], aggregate["overall"])
        lines.append(
            f"| {method} | {float(overall['state_agreement']):.4f} | "
            f"{float(overall['exact_pair_accuracy']):.4f} | "
            f"{float(overall['prediction_flip_status_accuracy']):.4f} | "
            f"{float(overall['mean_completion_proxy_regret']):.4f} | "
            f"{float(overall['utility_aware_accuracy_at_0.05']):.4f} | "
            f"{cast(float, aggregate['mean_evaluation_residual_saturation_rate']):.4f} |"
        )
    for baseline in BASELINE_NAMES:
        overall = cast(
            dict[str, int | float], baselines[baseline]["overall"]
        )
        lines.append(
            f"| {baseline} | {float(overall['state_agreement']):.4f} | "
            f"{float(overall['exact_pair_accuracy']):.4f} | "
            f"{float(overall['prediction_flip_status_accuracy']):.4f} | "
            f"{float(overall['mean_completion_proxy_regret']):.4f} | "
            f"{float(overall['utility_aware_accuracy_at_0.05']):.4f} | 0.0000 |"
        )
    lines.extend(
        [
            "",
            "## Pair-Aware Minus Matched",
            "",
            f"State agreement gain: **{cast(float, comparison['pair_aware_minus_matched_state_agreement']):.4f}**.",
            f"Exact-pair gain: **{cast(float, comparison['pair_aware_minus_matched_exact_pair_accuracy']):.4f}**.",
            f"Paired bootstrap CI95: **{comparison['paired_bootstrap_state_agreement_difference_ci95']}**.",
            "",
            "| Stratum | State-agreement gain |",
            "|---|---:|",
        ]
    )
    stratum_gains = cast(
        dict[str, float],
        comparison["pair_aware_minus_matched_state_agreement_by_stratum"],
    )
    for stratum, _, _ in HELD_OUT_MARGIN_STRATA:
        lines.append(f"| {stratum} | {stratum_gains[stratum]:.4f} |")
    lines.extend(
        [
            "",
            "## Criteria",
            "",
        ]
    )
    criteria = cast(dict[str, bool], comparison["criteria"])
    lines.extend(
        f"- {name}: **{str(criteria[name]).lower()}**."
        for name in REPORT_CRITERIA
    )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            cast(str, summary["next_step"]),
            "",
            "## Limitations",
            "",
            "This evaluation is limited to the frozen Ticket 39 relational package, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ticket 40 independent held-out architecture evaluation"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package", required=True)
    args = parser.parse_args()
    result = run_independent_held_out_evaluation(
        args.output_dir,
        package_path=args.package,
    )
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
