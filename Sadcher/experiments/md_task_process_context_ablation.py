"""Run the frozen Ticket 43 task/process context development comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast

import torch

from experiments.md_budgeted_saturation_curriculum_diagnostic import (
    FORMAL_PROTOCOL as TRAINING_PROTOCOL,
)
from experiments.md_context_ablation_development_package import (
    CORRECTED_HELD_OUT_PACKAGE,
    _replay_corrected_held_out,
    _replay_development_records,
    _template_signature,
)
from experiments.md_context_ablation_relational_data import (
    RELATIONAL_FAMILIES,
    RelationalState as DevelopmentRelationalState,
    RelationalTwin as DevelopmentRelationalTwin,
)
from experiments.md_flip_curriculum_train_diagnostic import (
    _family_balanced_batches,
    _prepare_batch,
    _train_curriculum,
)
from experiments.md_policy_relational_gate import (
    DEFAULT_MODEL_SEEDS,
    RelationalState,
    RelationalTwin,
    _bootstrap_mean_ci,
    _model_logits,
    _template_signature as _training_template_signature,
)
from experiments.md_train_only_optimization_diagnostic import (
    _select_first_pairs_per_family,
)
from experiments.md_unconditioned_relational_audit import (
    build_unconditioned_relational_candidates,
)
from experiments.md_task_process_context_models import (
    CONTEXT_ABLATION_VARIANTS,
    PARAMETER_BUDGET,
    build_context_ablation_model,
)
from models.md_enhanced_policy import MDEnhancedSchedulerNetwork


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEVELOPMENT_PACKAGE: Final = (
    REPOSITORY_ROOT
    / "reports"
    / "md_context_ablation_development_package_2026-08-30"
    / "context_ablation_development_package.json"
)
MARGIN_STRATA: Final = (
    "near_tie_lt_0.01",
    "small_ge_0.01_lt_0.03",
    "medium_ge_0.03_lt_0.05",
    "high_ge_0.05",
)
HIGH_MARGIN_STRATUM: Final = "high_ge_0.05"
BASELINE_VARIANT: Final = "current_pair_aware"
CANDIDATE_VARIANTS: Final = CONTEXT_ABLATION_VARIANTS[1:]
UTILITY_TOLERANCE: Final = 0.05
COMPARISON_BOOTSTRAP_SEED: Final = 4343


@dataclass(frozen=True, slots=True)
class TaskProcessContextAblationResult:
    summary_path: Path
    report_path: Path


def _resolve_device(
    device: str | torch.device | None,
) -> tuple[str, torch.device]:
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        resolved = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return requested, resolved
    if "," in requested or any(character.isspace() for character in requested):
        raise ValueError("device must identify exactly one CPU or CUDA device")
    try:
        parsed = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid device: {requested}") from error
    if parsed.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda:N")
    if parsed.type == "cpu":
        if parsed.index is not None:
            raise ValueError("CPU device must be specified as cpu")
        return requested, parsed
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {requested}")
    index = 0 if parsed.index is None else parsed.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {index} is unavailable; "
            f"visible device count is {torch.cuda.device_count()}"
        )
    return requested, torch.device("cuda", index)


def _runtime_metadata(
    requested: str, resolved: torch.device
) -> dict[str, object]:
    return {
        "requested_device": requested,
        "resolved_device": str(resolved),
        "single_device_execution": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(resolved)
            if resolved.type == "cuda"
            else None
        ),
    }


def run_task_process_context_ablation(
    output_dir: str | Path,
    *,
    package_path: str | Path = DEVELOPMENT_PACKAGE,
    corrected_held_out_package_path: str | Path = CORRECTED_HELD_OUT_PACKAGE,
    candidate_pool_per_family: int = 5000,
    train_pairs_per_family: int = 500,
    pairs_per_family_per_batch: int = 100,
    pretrain_epochs: int = 200,
    fine_tune_epochs: int = 100,
    checkpoint_interval: int = 10,
    data_seed: int = 3030,
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
    device: str | torch.device | None = None,
) -> TaskProcessContextAblationResult:
    """Train equal-budget variants and evaluate only on Ticket 42 records."""
    destination = Path(output_dir)
    summary_path = destination / "task_process_context_ablation.json"
    report_path = destination / "task_process_context_ablation.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite task/process context ablation: "
            + ", ".join(str(path) for path in existing)
        )

    requested_device, runtime_device = _resolve_device(device)
    seeds = _validated_ablation_seeds(model_seeds)
    protocol = {
        "candidate_pool_per_family": candidate_pool_per_family,
        "train_pairs_per_family": train_pairs_per_family,
        "pairs_per_family_per_batch": pairs_per_family_per_batch,
        "pretrain_epochs": pretrain_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "checkpoint_interval": checkpoint_interval,
        "data_seed": data_seed,
        "model_seeds": seeds,
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
    formal_protocol_run = protocol == TRAINING_PROTOCOL

    package = _load_development_package(Path(package_path))
    provenance = cast(dict[str, object], package["development_provenance"])
    package_records = cast(
        list[dict[str, object]], package["development_records"]
    )
    development_twins, package_replay_mismatch = _replay_development_records(
        package_records,
        development_candidates_per_family=cast(
            int, provenance["candidate_pool_per_family"]
        ),
        development_base_state_seed=cast(int, provenance["base_state_seed"]),
        development_perturbation_seed=cast(
            int, provenance["perturbation_seed"]
        ),
        quota_per_family_stratum=cast(
            int, provenance["quota_per_family_stratum"]
        ),
    )
    held_out_twins, held_out_replay_mismatch = _replay_corrected_held_out(
        Path(corrected_held_out_package_path)
    )
    training_candidates = build_unconditioned_relational_candidates(
        candidates_per_family=candidate_pool_per_family,
        seed=data_seed,
    )
    training_twins = _select_first_pairs_per_family(
        training_candidates,
        pairs_per_family=train_pairs_per_family,
    )
    training_templates = {
        _training_template_signature(twin) for twin in training_twins
    }
    development_templates = {
        _template_signature(twin) for twin in development_twins
    }
    held_out_templates = {_template_signature(twin) for twin in held_out_twins}
    training_overlap = len(training_templates & development_templates)
    held_out_overlap = len(held_out_templates & development_templates)
    development_cells = _development_cell_counts(package_records)
    unique_development_pairs = len(
        {cast(str, record["pair_id"]) for record in package_records}
    )

    training_states = _states(training_twins)
    development_states = _development_states(development_twins)
    balanced_batches = _family_balanced_batches(
        training_twins,
        pairs_per_family_per_batch=pairs_per_family_per_batch,
    )
    prepared_batches = tuple(
        _prepare_batch(batch, device=runtime_device)
        for batch in balanced_batches
    )
    variants: dict[str, dict[str, object]] = {}
    for variant in CONTEXT_ABLATION_VARIANTS:
        per_seed: list[dict[str, object]] = []
        for seed in seeds:
            model = cast(
                MDEnhancedSchedulerNetwork,
                build_context_ablation_model(
                    variant, seed=seed, device=runtime_device
                ),
            )
            training = _train_curriculum(
                model,
                prepared_batches,
                training_twins,
                training_states,
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
                minimum_overall_train_agreement=minimum_overall_train_agreement,
                minimum_family_flip_pair_exact=minimum_family_flip_pair_exact,
                maximum_residual_saturation_rate=(
                    maximum_residual_saturation_rate
                ),
            )
            records = _development_prediction_records(
                model, development_twins, development_states
            )
            per_seed.append(
                {
                    "model_seed": seed,
                    "training": training,
                    "development_records": records,
                    "metrics": _grouped_metrics(records),
                }
            )
        parameter_count = sum(
            parameter.numel()
            for parameter in build_context_ablation_model(
                variant, seed=seeds[0]
            ).parameters()
            if parameter.requires_grad
        )
        variants[variant] = {
            "parameter_count": parameter_count,
            "parameter_budget_evidence": {
                "trainable_parameter_count": parameter_count,
                "inert_padding_parameter_count": 0,
                "padding_parameters_used": False,
            },
            "per_seed": per_seed,
            "aggregate": _aggregate_variant(per_seed),
        }

    parameter_budget_equal = (
        {cast(int, row["parameter_count"]) for row in variants.values()}
        == {PARAMETER_BUDGET}
    )
    package_valid = (
        package_replay_mismatch == 0
        and held_out_replay_mismatch == 0
        and training_overlap == 0
        and held_out_overlap == 0
        and len(development_twins) == 800
        and unique_development_pairs == 800
        and len(development_cells) == 16
        and all(cell["pair_count"] == 50 for cell in development_cells)
    )
    training_valid = all(
        cast(
            bool,
            cast(
                dict[str, object],
                cast(dict[str, object], row["training"])["criteria"],
            )["all_training_conditions_met"],
        )
        for variant in variants.values()
        for row in cast(list[dict[str, object]], variant["per_seed"])
    )
    comparison_valid = package_valid and parameter_budget_equal and training_valid
    comparisons = {
        candidate: _candidate_comparison(
            variants[BASELINE_VARIANT], variants[candidate]
        )
        for candidate in CANDIDATE_VARIANTS
    }
    selection = select_context_ablation_outcome(
        comparisons, comparison_valid=comparison_valid
    )
    if not formal_protocol_run:
        status = "development_diagnostic_only"
        ticket_44_authorized = False
    elif not comparison_valid:
        status = "development_comparison_invalid"
        ticket_44_authorized = False
    else:
        status = cast(str, selection["outcome"])
        ticket_44_authorized = selection["selected_variant"] is not None

    summary: dict[str, object] = {
        "schema_version": "1.2.0",
        "status": status,
        "ticket": 43,
        "source_ticket": 42,
        "formal_protocol_run": formal_protocol_run,
        "training_protocol": _jsonable(protocol),
        "runtime": _runtime_metadata(requested_device, runtime_device),
        "development_package_path": _repository_relative(Path(package_path)),
        "development_package_status": package["status"],
        "model_seeds": list(seeds),
        "training_pair_count": len(training_twins),
        "training_state_count": len(training_states),
        "development_pair_count": len(development_twins),
        "development_state_count": len(development_states),
        "unique_development_pair_count": unique_development_pairs,
        "development_cells": development_cells,
        "package_replay_mismatch_count": package_replay_mismatch,
        "corrected_held_out_replay_mismatch_count": held_out_replay_mismatch,
        "training_development_template_overlap": training_overlap,
        "ticket_40_held_out_development_template_overlap": held_out_overlap,
        "variants": variants,
        "comparisons_to_current_pair_aware": comparisons,
        "selection": selection,
        "comparison_valid": comparison_valid,
        "ticket_44_authorized": ticket_44_authorized,
        "controls": {
            "ticket_42_records_reselected_or_modified": False,
            "ticket_40_per_pair_model_outputs_read": False,
            "ticket_40_held_out_used_for_candidate_selection": False,
            "same_training_and_development_records_for_all_variants": True,
            "same_batch_order_loss_optimizer_mask_and_decoder": True,
            "parameter_budget_equal": parameter_budget_equal,
            "inert_padding_parameters_used": False,
            "production_scorer_modified": False,
            "decoder_modified": False,
            "simulator_modified": False,
            "production_expert_dataset_written": False,
            "ticket_17_unfrozen": False,
            "ticket_20_unfrozen": False,
            "ticket_32_unblocked": False,
            "architecture_gate_reopened": False,
        },
        "next_step": _next_step(status, selection),
    }
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        render_task_process_context_ablation_report(summary), encoding="utf-8"
    )
    return TaskProcessContextAblationResult(summary_path, report_path)


def _load_development_package(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Ticket 42 package must be a JSON object")
    required = {
        "status", "ticket", "formal_protocol_run", "training_provenance",
        "development_provenance", "development_pair_count",
        "development_state_count", "development_records",
        "package_replay_mismatch_count",
        "training_development_template_overlap",
        "prior_held_out_development_template_overlap", "ticket_43_authorized",
    }
    if missing := sorted(required - payload.keys()):
        raise ValueError(f"Ticket 42 package is missing fields: {missing}")
    if not (
        payload["ticket"] == 42
        and payload["status"] == "development_package_frozen"
        and payload["formal_protocol_run"] is True
        and payload["development_pair_count"] == 800
        and payload["development_state_count"] == 1600
        and payload["package_replay_mismatch_count"] == 0
        and payload["training_development_template_overlap"] == 0
        and payload["prior_held_out_development_template_overlap"] == 0
        and payload["ticket_43_authorized"] is True
    ):
        raise ValueError("Ticket 42 package is not the frozen formal package")
    records = payload["development_records"]
    if not isinstance(records, list) or len(records) != 800:
        raise ValueError("Ticket 42 package must contain exactly 800 records")
    return cast(dict[str, object], payload)


def _validated_ablation_seeds(model_seeds: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(model_seeds)
    if not seeds:
        raise ValueError("model_seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("model_seeds must be unique")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("model_seeds must contain non-negative integers")
    return seeds


def _development_cell_counts(
    records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "family": family,
            "pair_margin_stratum": stratum,
            "pair_count": sum(
                record["family"] == family
                and record["pair_margin_stratum"] == stratum
                for record in records
            ),
        }
        for family in RELATIONAL_FAMILIES
        for stratum in MARGIN_STRATA
    ]


def _states(twins: Sequence[RelationalTwin]) -> tuple[RelationalState, ...]:
    return tuple(state for twin in twins for state in (twin.before, twin.after))


def _development_states(
    twins: Sequence[DevelopmentRelationalTwin],
) -> tuple[DevelopmentRelationalState, ...]:
    return tuple(state for twin in twins for state in (twin.before, twin.after))


def _development_prediction_records(
    model: MDEnhancedSchedulerNetwork,
    twins: Sequence[DevelopmentRelationalTwin],
    states: Sequence[DevelopmentRelationalState],
) -> list[dict[str, object]]:
    logits, diagnostics = _model_logits(
        model,
        cast(Sequence[RelationalState], cast(object, states)),
    )
    predictions = cast(list[int], torch.argmax(logits, dim=-1).tolist())
    residual = diagnostics.bounded_residual[:, :, :4]
    rows: list[dict[str, object]] = []
    for pair_index, twin in enumerate(twins):
        before_index = 2 * pair_index
        after_index = before_index + 1
        before = _state_prediction_record(
            twin.before,
            predictions[before_index],
            residual[before_index],
            float(diagnostics.residual_saturation_rate[before_index]),
        )
        after = _state_prediction_record(
            twin.after,
            predictions[after_index],
            residual[after_index],
            float(diagnostics.residual_saturation_rate[after_index]),
        )
        rows.append(
            {
                "pair_id": twin.pair_id,
                "family": twin.family,
                "changed_entity_index": twin.changed_entity_index,
                "pair_margin_stratum": _margin_stratum(twin),
                "oracle_flip_status": (
                    "oracle_flip"
                    if twin.before.oracle_action != twin.after.oracle_action
                    else "oracle_no_flip"
                ),
                "before": before,
                "after": after,
                "exact_pair_correct": bool(
                    before["correctness"] and after["correctness"]
                ),
            }
        )
    return rows


def _state_prediction_record(
    state: DevelopmentRelationalState,
    prediction: int,
    bounded_residual: torch.Tensor,
    saturation_rate: float,
) -> dict[str, object]:
    regret = max(state.action_values) - state.action_values[prediction]
    hard_mask = torch.tensor(state.hard_mask, dtype=torch.bool)[:, :4]
    values = bounded_residual[hard_mask]
    return {
        "state_id": state.state_id,
        "oracle_action": state.oracle_action,
        "prediction": prediction,
        "correctness": prediction == state.oracle_action,
        "completion_proxy_regret": regret,
        "utility_aware_accuracy_at_0.05": regret <= UTILITY_TOLERANCE,
        "residual_magnitude": float(values.abs().mean()),
        "residual_saturation_rate": saturation_rate,
    }


def _margin_stratum(twin: DevelopmentRelationalTwin) -> str:
    top_values = [
        sorted(state.action_values, reverse=True)[:2]
        for state in (twin.before, twin.after)
    ]
    margin = min(row[0] - row[1] for row in top_values)
    if margin < 0.01:
        return MARGIN_STRATA[0]
    if margin < 0.03:
        return MARGIN_STRATA[1]
    if margin < 0.05:
        return MARGIN_STRATA[2]
    return MARGIN_STRATA[3]


def _grouped_metrics(records: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "overall": _group_metrics(records),
        "by_family": {
            family: _group_metrics(
                [row for row in records if row["family"] == family]
            )
            for family in RELATIONAL_FAMILIES
        },
        "by_margin_stratum": {
            stratum: _group_metrics(
                [row for row in records if row["pair_margin_stratum"] == stratum]
            )
            for stratum in MARGIN_STRATA
        },
        "by_oracle_flip_status": {
            status: _group_metrics(
                [row for row in records if row["oracle_flip_status"] == status]
            )
            for status in ("oracle_flip", "oracle_no_flip")
        },
    }


def _group_metrics(
    records: Sequence[dict[str, object]],
) -> dict[str, int | float]:
    states = [
        cast(dict[str, object], row[side])
        for row in records
        for side in ("before", "after")
    ]
    state_count = len(states)
    pair_count = len(records)
    return {
        "pair_count": pair_count,
        "state_count": state_count,
        "state_agreement": sum(
            int(cast(bool, state["correctness"])) for state in states
        ) / state_count,
        "exact_pair_accuracy": sum(
            int(cast(bool, row["exact_pair_correct"])) for row in records
        ) / pair_count,
        "mean_completion_proxy_regret": sum(
            cast(float, state["completion_proxy_regret"]) for state in states
        ) / state_count,
        "utility_aware_accuracy_at_0.05": sum(
            int(cast(bool, state["utility_aware_accuracy_at_0.05"]))
            for state in states
        ) / state_count,
        "mean_residual_magnitude": sum(
            cast(float, state["residual_magnitude"]) for state in states
        ) / state_count,
        "mean_residual_saturation_rate": sum(
            cast(float, state["residual_saturation_rate"])
            for state in states
        ) / state_count,
    }


def _aggregate_variant(
    per_seed: Sequence[dict[str, object]],
) -> dict[str, object]:
    metric_sets = [cast(dict[str, object], row["metrics"]) for row in per_seed]
    return {
        "overall": _aggregate_groups(
            [cast(dict[str, int | float], row["overall"]) for row in metric_sets]
        ),
        "by_family": _aggregate_grouping(metric_sets, "by_family"),
        "by_margin_stratum": _aggregate_grouping(metric_sets, "by_margin_stratum"),
        "by_oracle_flip_status": _aggregate_grouping(
            metric_sets, "by_oracle_flip_status"
        ),
        "seed_range": _seed_range(metric_sets),
    }


def _aggregate_grouping(
    metric_sets: Sequence[dict[str, object]], key: str
) -> dict[str, dict[str, int | float]]:
    first = cast(dict[str, object], metric_sets[0][key])
    return {
        group: _aggregate_groups(
            [
                cast(
                    dict[str, int | float],
                    cast(dict[str, object], metrics[key])[group],
                )
                for metrics in metric_sets
            ]
        )
        for group in first
    }


def _aggregate_groups(
    groups: Sequence[dict[str, int | float]],
) -> dict[str, int | float]:
    return {
        key: (
            cast(int, groups[0][key])
            if key in ("pair_count", "state_count")
            else sum(float(group[key]) for group in groups) / len(groups)
        )
        for key in groups[0]
    }


def _seed_range(metric_sets: Sequence[dict[str, object]]) -> dict[str, object]:
    def metric_range(
        groups: Sequence[dict[str, int | float]],
    ) -> dict[str, dict[str, float]]:
        return {
            metric: {
                "minimum": min(float(group[metric]) for group in groups),
                "maximum": max(float(group[metric]) for group in groups),
            }
            for metric in groups[0]
            if metric not in ("pair_count", "state_count")
        }

    def grouping_range(key: str) -> dict[str, object]:
        first = cast(dict[str, object], metric_sets[0][key])
        return {
            group: metric_range(
                [
                    cast(
                        dict[str, int | float],
                        cast(dict[str, object], metrics[key])[group],
                    )
                    for metrics in metric_sets
                ]
            )
            for group in first
        }

    overall = [
        cast(dict[str, int | float], metrics["overall"])
        for metrics in metric_sets
    ]
    return {
        "overall": metric_range(overall),
        "by_family": grouping_range("by_family"),
        "by_margin_stratum": grouping_range("by_margin_stratum"),
        "by_oracle_flip_status": grouping_range("by_oracle_flip_status"),
    }


def _candidate_comparison(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    baseline_per_seed = cast(list[dict[str, object]], baseline["per_seed"])
    candidate_per_seed = cast(list[dict[str, object]], candidate["per_seed"])
    paired_differences: list[float] = []
    pickup_differences: list[float] = []
    pickup_by_seed: list[float] = []
    paired_comparison_by_seed: list[dict[str, object]] = []
    pooled_state_transitions = _empty_transitions()
    pooled_pair_transitions = _empty_transitions()
    pooled_pair_outcomes = {"win": 0, "tie": 0, "loss": 0}

    for baseline_seed, candidate_seed in zip(
        baseline_per_seed, candidate_per_seed, strict=True
    ):
        model_seed = cast(int, baseline_seed["model_seed"])
        if candidate_seed["model_seed"] != model_seed:
            raise ValueError("baseline and candidate model seeds must align")
        baseline_records = cast(
            list[dict[str, object]], baseline_seed["development_records"]
        )
        candidate_records = cast(
            list[dict[str, object]], candidate_seed["development_records"]
        )
        seed_comparison = _paired_record_comparison(
            baseline_records, candidate_records
        )
        seed_differences = cast(
            list[float], seed_comparison.pop("paired_differences")
        )
        seed_pickup = cast(
            list[float], seed_comparison.pop("pickup_differences")
        )
        paired_differences.extend(seed_differences)
        pickup_differences.extend(seed_pickup)
        pickup_by_seed.append(sum(seed_pickup) / len(seed_pickup))
        _add_counts(
            pooled_state_transitions,
            cast(dict[str, int], seed_comparison["state_transitions"]),
        )
        _add_counts(
            pooled_pair_transitions,
            cast(dict[str, int], seed_comparison["exact_pair_transitions"]),
        )
        _add_counts(
            pooled_pair_outcomes,
            cast(
                dict[str, int],
                seed_comparison["paired_pair_win_tie_loss"],
            ),
        )
        paired_comparison_by_seed.append(
            {
                "model_seed": model_seed,
                **seed_comparison,
                "paired_state_agreement_difference_ci95": list(
                    _bootstrap_mean_ci(
                        seed_differences,
                        seed=COMPARISON_BOOTSTRAP_SEED,
                    )
                ),
                "pickup_paired_state_agreement_difference_ci95": list(
                    _bootstrap_mean_ci(
                        seed_pickup,
                        seed=COMPARISON_BOOTSTRAP_SEED,
                    )
                ),
            }
        )

    baseline_aggregate = cast(dict[str, object], baseline["aggregate"])
    candidate_aggregate = cast(dict[str, object], candidate["aggregate"])
    baseline_overall = cast(
        dict[str, int | float], baseline_aggregate["overall"]
    )
    candidate_overall = cast(
        dict[str, int | float], candidate_aggregate["overall"]
    )
    baseline_families = cast(
        dict[str, dict[str, int | float]], baseline_aggregate["by_family"]
    )
    candidate_families = cast(
        dict[str, dict[str, int | float]], candidate_aggregate["by_family"]
    )
    baseline_strata = cast(
        dict[str, dict[str, int | float]],
        baseline_aggregate["by_margin_stratum"],
    )
    candidate_strata = cast(
        dict[str, dict[str, int | float]],
        candidate_aggregate["by_margin_stratum"],
    )
    pickup_exact_gain = (
        float(
            candidate_families["alternative_task_pickup"][
                "exact_pair_accuracy"
            ]
        )
        - float(
            baseline_families["alternative_task_pickup"][
                "exact_pair_accuracy"
            ]
        )
    )
    pickup_ci95 = list(
        _bootstrap_mean_ci(pickup_differences, seed=COMPARISON_BOOTSTRAP_SEED)
    )
    paired_ci95 = list(
        _bootstrap_mean_ci(paired_differences, seed=COMPARISON_BOOTSTRAP_SEED)
    )
    high_margin_gain = float(
        candidate_strata[HIGH_MARGIN_STRATUM]["state_agreement"]
    ) - float(baseline_strata[HIGH_MARGIN_STRATUM]["state_agreement"])
    overall_state_gain = float(candidate_overall["state_agreement"]) - float(
        baseline_overall["state_agreement"]
    )
    overall_exact_gain = float(
        candidate_overall["exact_pair_accuracy"]
    ) - float(baseline_overall["exact_pair_accuracy"])
    saturation = float(candidate_overall["mean_residual_saturation_rate"])
    training_conditions_met = all(
        cast(
            bool,
            cast(
                dict[str, object],
                cast(dict[str, object], row["training"])["criteria"],
            )["all_training_conditions_met"],
        )
        for row in candidate_per_seed
    )
    criteria = {
        "all_ticket_38_training_conditions_met": training_conditions_met,
        "parameter_budget_equal_to_current": (
            candidate["parameter_count"]
            == baseline["parameter_count"]
            == PARAMETER_BUDGET
        ),
        "pickup_exact_pair_gain_at_least_0.05": pickup_exact_gain >= 0.05,
        "pickup_paired_state_ci95_lower_above_zero": pickup_ci95[0] > 0,
        "high_margin_state_agreement_gain_at_least_0.03": (
            high_margin_gain >= 0.03
        ),
        "overall_state_agreement_rollback_at_most_0.01": (
            overall_state_gain >= -0.01
        ),
        "overall_exact_pair_rollback_at_most_0.01": (
            overall_exact_gain >= -0.01
        ),
        "all_seed_pickup_state_agreement_differences_positive": all(
            difference > 0 for difference in pickup_by_seed
        ),
        "mean_development_residual_saturation_at_most_0.25": (
            saturation <= 0.25
        ),
    }
    return {
        "pickup_exact_pair_gain": pickup_exact_gain,
        "pooled_pickup_paired_state_agreement_difference_ci95": pickup_ci95,
        "pickup_state_agreement_difference_by_seed": pickup_by_seed,
        "high_margin_state_agreement_gain": high_margin_gain,
        "overall_state_agreement_gain": overall_state_gain,
        "overall_exact_pair_accuracy_gain": overall_exact_gain,
        "mean_development_residual_saturation": saturation,
        "paired_comparison_by_seed": paired_comparison_by_seed,
        "pooled_paired_comparison": {
            "seed_count": len(paired_comparison_by_seed),
            "paired_pair_win_tie_loss": pooled_pair_outcomes,
            "state_transitions": pooled_state_transitions,
            "exact_pair_transitions": pooled_pair_transitions,
            "pooled_paired_state_agreement_difference_ci95": paired_ci95,
            "pooled_pickup_paired_state_agreement_difference_ci95": pickup_ci95,
        },
        "pooled_paired_state_agreement_difference_ci95": paired_ci95,
        "criteria": criteria,
        "all_acceptance_criteria_met": all(criteria.values()),
    }


def _paired_record_comparison(
    baseline_records: Sequence[dict[str, object]],
    candidate_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    paired_differences: list[float] = []
    pickup_differences: list[float] = []
    state_transitions = _empty_transitions()
    pair_transitions = _empty_transitions()
    pair_outcomes = {"win": 0, "tie": 0, "loss": 0}
    for baseline_row, candidate_row in zip(
        baseline_records, candidate_records, strict=True
    ):
        if baseline_row["pair_id"] != candidate_row["pair_id"]:
            raise ValueError("baseline and candidate pair records must align")
        baseline_pair = cast(bool, baseline_row["exact_pair_correct"])
        candidate_pair = cast(bool, candidate_row["exact_pair_correct"])
        _count_transition(pair_transitions, baseline_pair, candidate_pair)
        baseline_score = sum(
            int(
                cast(
                    bool,
                    cast(dict[str, object], baseline_row[side])["correctness"],
                )
            )
            for side in ("before", "after")
        )
        candidate_score = sum(
            int(
                cast(
                    bool,
                    cast(dict[str, object], candidate_row[side])["correctness"],
                )
            )
            for side in ("before", "after")
        )
        outcome = (
            "win"
            if candidate_score > baseline_score
            else ("loss" if candidate_score < baseline_score else "tie")
        )
        pair_outcomes[outcome] += 1
        for side in ("before", "after"):
            baseline_correct = cast(
                bool,
                cast(dict[str, object], baseline_row[side])["correctness"],
            )
            candidate_correct = cast(
                bool,
                cast(dict[str, object], candidate_row[side])["correctness"],
            )
            difference = float(candidate_correct) - float(baseline_correct)
            paired_differences.append(difference)
            _count_transition(
                state_transitions, baseline_correct, candidate_correct
            )
            if baseline_row["family"] == "alternative_task_pickup":
                pickup_differences.append(difference)
    return {
        "paired_differences": paired_differences,
        "pickup_differences": pickup_differences,
        "paired_pair_win_tie_loss": pair_outcomes,
        "state_transitions": state_transitions,
        "exact_pair_transitions": pair_transitions,
    }


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, count in source.items():
        target[key] += count


def _empty_transitions() -> dict[str, int]:
    return {"wrong_to_correct": 0, "correct_to_wrong": 0, "unchanged": 0}


def _count_transition(
    counts: dict[str, int], baseline_correct: bool, candidate_correct: bool
) -> None:
    if not baseline_correct and candidate_correct:
        counts["wrong_to_correct"] += 1
    elif baseline_correct and not candidate_correct:
        counts["correct_to_wrong"] += 1
    else:
        counts["unchanged"] += 1


def select_context_ablation_outcome(
    comparisons: dict[str, dict[str, object]], *, comparison_valid: bool
) -> dict[str, object]:
    passing = {
        name
        for name, comparison in comparisons.items()
        if cast(bool, comparison["all_acceptance_criteria_met"])
    }
    late = "transport_process_late_fusion"
    transport = "transport_context_only"
    process = "process_context_only"
    calibration = "transport_calibration_control"
    late_dominates_singles = (
        late in passing
        and cast(float, comparisons[late]["pickup_exact_pair_gain"])
        >= cast(float, comparisons[transport]["pickup_exact_pair_gain"]) + 0.02
        and cast(float, comparisons[late]["pickup_exact_pair_gain"])
        >= cast(float, comparisons[process]["pickup_exact_pair_gain"]) + 0.02
    )
    eligible = {name for name in (transport, process) if name in passing}
    if late_dominates_singles:
        eligible.add(late)
    if calibration in passing and not passing.intersection(
        {transport, process, late}
    ):
        eligible.add(calibration)
    selected = (
        max(
            eligible,
            key=lambda name: _selection_key(name, comparisons[name]),
        )
        if comparison_valid and eligible
        else None
    )
    outcomes = {
        late: "late_fusion_supported_for_confirmation",
        process: "process_context_supported_for_confirmation",
        transport: "transport_context_supported_for_confirmation",
        calibration: "transport_calibration_supported_for_confirmation",
    }
    outcome = (
        outcomes[selected]
        if selected is not None
        else (
            "current_architecture_retained"
            if comparison_valid
            else "development_comparison_invalid"
        )
    )
    return {
        "outcome": outcome,
        "selected_variant": selected,
        "passing_candidates": sorted(passing),
        "eligible_candidates": sorted(eligible),
        "late_fusion_pickup_gain_exceeds_each_single_by_at_least_0.02": (
            late_dominates_singles
        ),
        "ranking_order": [
            "pickup_exact_pair_gain",
            "high_margin_state_agreement_gain",
            "overall_exact_pair_accuracy_gain",
            "lower_mean_development_residual_saturation",
            "smaller_cross_seed_pickup_state_difference_range",
            "single_context_before_late_fusion_on_complete_tie",
        ],
    }


def _selection_key(name: str, comparison: dict[str, object]) -> tuple[float, ...]:
    seed_differences = cast(
        list[float], comparison["pickup_state_agreement_difference_by_seed"]
    )
    simplicity = float(
        name in ("transport_context_only", "process_context_only")
    )
    return (
        cast(float, comparison["pickup_exact_pair_gain"]),
        cast(float, comparison["high_margin_state_agreement_gain"]),
        cast(float, comparison["overall_exact_pair_accuracy_gain"]),
        -cast(float, comparison["mean_development_residual_saturation"]),
        -(max(seed_differences) - min(seed_differences)),
        simplicity,
    )


def render_task_process_context_ablation_report(
    summary: dict[str, object],
) -> str:
    variants = cast(dict[str, dict[str, object]], summary["variants"])
    comparisons = cast(
        dict[str, dict[str, object]],
        summary["comparisons_to_current_pair_aware"],
    )
    selection = cast(dict[str, object], summary["selection"])
    lines = [
        "# Ticket 43 Task And Process Context Ablation",
        "",
        f"Status: **{summary['status']}**.",
        f"Selection outcome: **{selection['outcome']}**.",
        f"Ticket 44 authorized: **{str(summary['ticket_44_authorized']).lower()}**.",
        "",
        "## Development Metrics",
        "",
        "| Variant | Parameters | State agreement | Exact pair | Regret | Utility@0.05 | Residual | Saturation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    runtime = summary.get("runtime")
    if runtime is not None:
        runtime_info = cast(dict[str, object], runtime)
        lines[6:6] = [
            "## Runtime",
            "",
            f"Requested device: **{runtime_info['requested_device']}**.",
            f"Resolved device: **{runtime_info['resolved_device']}**.",
            f"Single-device execution: **{str(runtime_info['single_device_execution']).lower()}**.",
            f"PyTorch: **{runtime_info['torch_version']}**.",
            f"CUDA runtime: **{runtime_info['cuda_version']}**.",
            f"GPU: **{runtime_info['gpu_name']}**.",
            "",
        ]
    for name in CONTEXT_ABLATION_VARIANTS:
        aggregate = cast(dict[str, object], variants[name]["aggregate"])
        overall = cast(dict[str, int | float], aggregate["overall"])
        lines.append(
            f"| {name} | {variants[name]['parameter_count']} | "
            f"{float(overall['state_agreement']):.4f} | "
            f"{float(overall['exact_pair_accuracy']):.4f} | "
            f"{float(overall['mean_completion_proxy_regret']):.4f} | "
            f"{float(overall['utility_aware_accuracy_at_0.05']):.4f} | "
            f"{float(overall['mean_residual_magnitude']):.4f} | "
            f"{float(overall['mean_residual_saturation_rate']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Gains Versus Current Pair-Aware",
            "",
            "| Candidate | Pickup exact gain | Pooled pickup state CI95 | High-margin state gain | Overall state gain | Overall exact gain | Saturation | Pass |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in CANDIDATE_VARIANTS:
        row = comparisons[name]
        lines.append(
            f"| {name} | {cast(float, row['pickup_exact_pair_gain']):.4f} | "
            f"{row['pooled_pickup_paired_state_agreement_difference_ci95']} | "
            f"{cast(float, row['high_margin_state_agreement_gain']):.4f} | "
            f"{cast(float, row['overall_state_agreement_gain']):.4f} | "
            f"{cast(float, row['overall_exact_pair_accuracy_gain']):.4f} | "
            f"{cast(float, row['mean_development_residual_saturation']):.4f} | "
            f"{str(row['all_acceptance_criteria_met']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"Package replay mismatches: **{summary['package_replay_mismatch_count']}**.",
            f"Training/development template overlap: **{summary['training_development_template_overlap']}**.",
            f"Corrected held-out/development template overlap: **{summary['ticket_40_held_out_development_template_overlap']}**.",
            "",
            "## Interpretation",
            "",
            _interpretation(summary),
            "",
            "## Next Step",
            "",
            cast(str, summary["next_step"]),
            "",
            "## Limitations",
            "",
            "This is a development-stage scorer comparison. It is not a confirmation result, a relational-generalization claim, or an architecture-gate decision.",
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation(summary: dict[str, object]) -> str:
    selection = cast(dict[str, object], summary["selection"])
    selected = selection["selected_variant"]
    if summary["status"] == "development_comparison_invalid":
        return (
            "The formal comparison is invalid because not all variants met the "
            "frozen Ticket 38 training conditions. No candidate or current-"
            "architecture development decision is authorized."
        )
    if selected is None:
        return (
            "No eligible context or calibration candidate was frozen. The "
            "current scorer remains the implementation choice and no confirmation "
            "package is authorized."
        )
    component = {
        "transport_context_only": "learned relational transport residual",
        "process_context_only": "learned task/process residual",
        "transport_process_late_fusion": (
            "separate transport/process residual branches"
        ),
        "transport_calibration_control": "positive transport physics calibration",
    }[cast(str, selected)]
    return (
        f"The pre-registered development criteria select {selected}; the observed "
        f"gain is attributed to the {component}. Ticket 44 may only freeze a new "
        "confirmation package for this exact candidate and protocol."
    )


def _next_step(status: str, selection: dict[str, object]) -> str:
    if status == "development_diagnostic_only":
        return "This reduced run cannot authorize Ticket 44."
    if status == "development_comparison_invalid":
        return "Stop: the formal development comparison is invalid."
    selected = selection["selected_variant"]
    if selected is None:
        return "Retain the current scorer; Ticket 44 remains unauthorized."
    return (
        "Ticket 44 is authorized only to freeze a fresh confirmation package "
        f"for {selected}; no confirmation model may run in Ticket 43."
    )


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ticket 43 task/process context ablation"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package", default=str(DEVELOPMENT_PACKAGE))
    parser.add_argument(
        "--device",
        default="auto",
        help="single device: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    args = parser.parse_args()
    result = run_task_process_context_ablation(
        args.output_dir, package_path=args.package, device=args.device
    )
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
