"""Family-balanced train-only comparison with a pair-structured objective."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast

import torch
import torch.nn.functional as F

from experiments.md_policy_relational_gate import (
    DEFAULT_MODEL_SEEDS,
    RELATIONAL_FAMILIES,
    RelationalTwin,
    _batch,
    _build_model,
    _flatten_transport_scores,
    _model_agreement,
    _model_logits,
    _validated_model_seeds,
)
from experiments.md_train_only_optimization_diagnostic import (
    TRAINABLE_METHODS,
    _agreement,
    _flip_pair_exact_by_family,
    _select_first_pairs_per_family,
)
from experiments.md_unconditioned_relational_audit import (
    build_unconditioned_relational_candidates,
)
from models.md_policy import MDPolicyInputs


TRAINING_VARIANTS: Final = (
    "family_balanced_state_only",
    "family_balanced_pair_structured",
)
FORMAL_PROTOCOL: Final = {
    "candidate_pool_per_family": 5000,
    "train_pairs_per_family": 500,
    "pairs_per_family_per_batch": 100,
    "data_seed": 3030,
    "model_seeds": DEFAULT_MODEL_SEEDS,
    "epochs": 200,
    "state_margin": 0.1,
    "pair_margin": 0.1,
    "pair_loss_weight": 1.0,
    "minimum_overall_train_agreement": 0.70,
    "minimum_family_flip_pair_exact": 0.60,
    "maximum_residual_saturation_rate": 0.25,
}


@dataclass(frozen=True, slots=True)
class PairStructuredTrainDiagnosticResult:
    summary_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    twins: tuple[RelationalTwin, ...]
    robot_features: torch.Tensor
    task_features: torch.Tensor
    task_adjacency: torch.Tensor
    md_inputs: object
    targets: torch.Tensor


def run_pair_structured_train_diagnostic(
    output_dir: str | Path,
    *,
    candidate_pool_per_family: int = 5000,
    train_pairs_per_family: int = 500,
    pairs_per_family_per_batch: int = 100,
    epochs: int = 200,
    data_seed: int = 3030,
    model_seeds: Sequence[int] = DEFAULT_MODEL_SEEDS,
    state_margin: float = 0.1,
    pair_margin: float = 0.1,
    pair_loss_weight: float = 1.0,
    minimum_overall_train_agreement: float = 0.70,
    minimum_family_flip_pair_exact: float = 0.60,
    maximum_residual_saturation_rate: float = 0.25,
) -> PairStructuredTrainDiagnosticResult:
    """Compare state-only and pair-structured objectives without evaluation data."""
    _validate_protocol(
        candidate_pool_per_family=candidate_pool_per_family,
        train_pairs_per_family=train_pairs_per_family,
        pairs_per_family_per_batch=pairs_per_family_per_batch,
        epochs=epochs,
        data_seed=data_seed,
        state_margin=state_margin,
        pair_margin=pair_margin,
        pair_loss_weight=pair_loss_weight,
        minimum_overall_train_agreement=minimum_overall_train_agreement,
        minimum_family_flip_pair_exact=minimum_family_flip_pair_exact,
        maximum_residual_saturation_rate=maximum_residual_saturation_rate,
    )
    resolved_model_seeds = _validated_model_seeds(model_seeds)
    destination = Path(output_dir)
    summary_path = destination / "pair_structured_train_diagnostic.json"
    report_path = destination / "pair_structured_train_diagnostic.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite pair-structured train diagnostic: "
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
    balanced_batches = _family_balanced_batches(
        train_twins,
        pairs_per_family_per_batch=pairs_per_family_per_batch,
    )
    prepared_batches = tuple(_prepare_batch(batch) for batch in balanced_batches)

    variants: dict[str, dict[str, object]] = {}
    for variant in TRAINING_VARIANTS:
        methods: dict[str, dict[str, object]] = {}
        for method in TRAINABLE_METHODS:
            per_seed: list[dict[str, object]] = []
            for model_seed in resolved_model_seeds:
                model = _build_model(method, seed=model_seed)
                initial_agreement = _model_agreement(model, train_states)
                loss_diagnostics = _train_model(
                    model,
                    prepared_batches,
                    epochs=epochs,
                    state_margin=state_margin,
                    pair_margin=pair_margin,
                    pair_loss_weight=(
                        pair_loss_weight
                        if variant == "family_balanced_pair_structured"
                        else 0.0
                    ),
                )
                logits, diagnostics = _model_logits(model, train_states)
                predictions = cast(
                    list[int], torch.argmax(logits, dim=-1).tolist()
                )
                agreement = _agreement(predictions, train_states)
                family_exact = _flip_pair_exact_by_family(
                    train_twins,
                    train_states,
                    predictions,
                    minimum_family_flip_pair_exact,
                )
                saturation = float(
                    diagnostics.residual_saturation_rate.mean()
                )
                overall_passed = agreement >= minimum_overall_train_agreement
                family_passed = all(
                    cast(bool, row["criterion_met"])
                    for row in family_exact.values()
                )
                saturation_passed = (
                    saturation <= maximum_residual_saturation_rate
                )
                per_seed.append(
                    {
                        "model_seed": model_seed,
                        "initial_train_agreement": initial_agreement,
                        "overall_train_agreement": agreement,
                        "residual_saturation_rate": saturation,
                        "oracle_flip_pair_exact_by_family": family_exact,
                        "loss_diagnostics": loss_diagnostics,
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
            methods[method] = {
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in _build_model(
                        method, seed=resolved_model_seeds[0]
                    ).parameters()
                    if parameter.requires_grad
                ),
                "mean_overall_train_agreement": _mean(
                    [
                        cast(float, row["overall_train_agreement"])
                        for row in per_seed
                    ]
                ),
                "mean_residual_saturation_rate": _mean(
                    [
                        cast(float, row["residual_saturation_rate"])
                        for row in per_seed
                    ]
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
        variants[variant] = {"methods": methods}

    candidate_methods = cast(
        dict[str, dict[str, object]],
        variants["family_balanced_pair_structured"]["methods"],
    )
    parameter_counts = {
        cast(int, row["parameter_count"]) for row in candidate_methods.values()
    }
    parameter_budget_equal = len(parameter_counts) == 1
    candidate_training_conditions_met = parameter_budget_equal and all(
        cast(bool, row["all_seed_training_conditions_met"])
        for row in candidate_methods.values()
    )
    formal_protocol_run = _matches_formal_protocol(
        candidate_pool_per_family=candidate_pool_per_family,
        train_pairs_per_family=train_pairs_per_family,
        pairs_per_family_per_batch=pairs_per_family_per_batch,
        epochs=epochs,
        data_seed=data_seed,
        model_seeds=resolved_model_seeds,
        state_margin=state_margin,
        pair_margin=pair_margin,
        pair_loss_weight=pair_loss_weight,
        minimum_overall_train_agreement=minimum_overall_train_agreement,
        minimum_family_flip_pair_exact=minimum_family_flip_pair_exact,
        maximum_residual_saturation_rate=maximum_residual_saturation_rate,
    )
    all_training_conditions_met = (
        formal_protocol_run and candidate_training_conditions_met
    )
    decision = (
        (
            "optimization_feasible"
            if all_training_conditions_met
            else "optimization_not_feasible"
        )
        if formal_protocol_run
        else "diagnostic_only"
    )
    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "pair_structured_train_diagnostic",
        "ticket": 35,
        "source_ticket": 34,
        "protocol_class": "exploratory_train_only_comparison",
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
            "epochs": epochs,
            "learning_rate": 0.01,
            "state_objective": "bounded_multi_margin",
            "state_margin": state_margin,
            "pair_objective": "flip_pair_cross_target_margin",
            "pair_margin": pair_margin,
            "pair_loss_weight": pair_loss_weight,
            "pairs_per_family_per_batch": pairs_per_family_per_batch,
            "batches_per_epoch": len(balanced_batches),
            "batch_order": "fixed_family_then_candidate_order",
            "family_pairs_per_batch": {
                family: pairs_per_family_per_batch
                for family in RELATIONAL_FAMILIES
            },
        },
        "thresholds": {
            "minimum_overall_train_agreement": minimum_overall_train_agreement,
            "minimum_family_flip_pair_exact": minimum_family_flip_pair_exact,
            "maximum_residual_saturation_rate": maximum_residual_saturation_rate,
        },
        "selection_controls": {
            "label_used_for_selection": False,
            "oracle_flip_used_for_selection": False,
            "margin_used_for_selection": False,
            "model_output_used_for_selection": False,
            "selection_order": "first_generated_pairs_per_family",
            "oracle_flip_used_only_after_selection_for_pair_loss": True,
        },
        "variants": variants,
        "candidate_variant": "family_balanced_pair_structured",
        "decision": decision,
        "candidate_training_conditions_met": candidate_training_conditions_met,
        "all_training_conditions_met": all_training_conditions_met,
        "controls": {
            "held_out_data_read": False,
            "held_out_model_metrics_read": False,
            "held_out_stage_authorized": all_training_conditions_met,
            "parameter_budget_equal": parameter_budget_equal,
            "ticket_17_unfrozen": False,
            "ticket_20_unfrozen": False,
            "ticket_32_unblocked": False,
            "production_expert_dataset_written": False,
        },
        "next_step": (
            "Freeze a new stability-based held-out protocol before generating its evaluation pool."
            if all_training_conditions_met
            else (
                "Stop without generating or viewing held-out evaluation data."
                if formal_protocol_run
                else "This reduced diagnostic run cannot authorize a next stage."
            )
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return PairStructuredTrainDiagnosticResult(summary_path, report_path)


def _validate_protocol(
    *,
    candidate_pool_per_family: int,
    train_pairs_per_family: int,
    pairs_per_family_per_batch: int,
    epochs: int,
    data_seed: int,
    state_margin: float,
    pair_margin: float,
    pair_loss_weight: float,
    minimum_overall_train_agreement: float,
    minimum_family_flip_pair_exact: float,
    maximum_residual_saturation_rate: float,
) -> None:
    for name, integer_value in (
        ("candidate_pool_per_family", candidate_pool_per_family),
        ("train_pairs_per_family", train_pairs_per_family),
        ("pairs_per_family_per_batch", pairs_per_family_per_batch),
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
    if train_pairs_per_family % pairs_per_family_per_batch:
        raise ValueError(
            "train_pairs_per_family must be divisible by pairs_per_family_per_batch"
        )
    if isinstance(data_seed, bool) or not isinstance(data_seed, int) or data_seed < 0:
        raise ValueError("data_seed must be a non-negative integer")
    for name, positive_value in (
        ("state_margin", state_margin),
        ("pair_margin", pair_margin),
        ("pair_loss_weight", pair_loss_weight),
    ):
        if (
            isinstance(positive_value, bool)
            or not isinstance(positive_value, (int, float))
            or not math.isfinite(positive_value)
            or positive_value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
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


def _family_balanced_batches(
    twins: Sequence[RelationalTwin],
    *,
    pairs_per_family_per_batch: int,
) -> tuple[tuple[RelationalTwin, ...], ...]:
    by_family = {
        family: [twin for twin in twins if twin.family == family]
        for family in RELATIONAL_FAMILIES
    }
    family_counts = {len(rows) for rows in by_family.values()}
    if len(family_counts) != 1:
        raise ValueError("training pairs must be balanced across families")
    pair_count = next(iter(family_counts))
    if pair_count % pairs_per_family_per_batch:
        raise ValueError(
            "family pair count must be divisible by pairs_per_family_per_batch"
        )
    batches: list[tuple[RelationalTwin, ...]] = []
    for start in range(0, pair_count, pairs_per_family_per_batch):
        batch: list[RelationalTwin] = []
        for family in RELATIONAL_FAMILIES:
            batch.extend(
                by_family[family][start : start + pairs_per_family_per_batch]
            )
        batches.append(tuple(batch))
    return tuple(batches)


def _prepare_batch(
    twins: tuple[RelationalTwin, ...],
    *,
    device: str | torch.device = "cpu",
) -> _PreparedBatch:
    states = tuple(
        state for twin in twins for state in (twin.before, twin.after)
    )
    robot_features, task_features, task_adjacency, md_inputs = _batch(states)
    return _PreparedBatch(
        twins=twins,
        robot_features=robot_features.to(device),
        task_features=task_features.to(device),
        task_adjacency=task_adjacency.to(device),
        md_inputs=cast(MDPolicyInputs, md_inputs).to(device),
        targets=torch.tensor(
            [state.oracle_action for state in states],
            dtype=torch.long,
            device=device,
        ),
    )


def _train_model(
    model: torch.nn.Module,
    batches: Sequence[_PreparedBatch],
    *,
    epochs: int,
    state_margin: float,
    pair_margin: float,
    pair_loss_weight: float,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    epoch_total_losses: list[float] = []
    epoch_state_losses: list[float] = []
    epoch_pair_losses: list[float] = []
    model.train()
    for _epoch in range(epochs):
        total_losses: list[float] = []
        state_losses: list[float] = []
        pair_losses: list[float] = []
        for batch in batches:
            diagnostics = model.forward_with_diagnostics(
                batch.robot_features,
                batch.task_features,
                batch.task_adjacency,
                md_inputs=batch.md_inputs,
            )
            logits = _flatten_transport_scores(diagnostics.scores)
            state_loss = F.multi_margin_loss(
                logits, batch.targets, margin=state_margin
            )
            pair_loss = _flip_pair_cross_target_loss(
                logits,
                batch.targets,
                margin=pair_margin,
            )
            total_loss = state_loss + pair_loss_weight * pair_loss
            optimizer.zero_grad()
            total_loss.backward()
            if not all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            ):
                raise RuntimeError("pair-structured probe produced non-finite gradients")
            optimizer.step()
            total_losses.append(float(total_loss.detach()))
            state_losses.append(float(state_loss.detach()))
            pair_losses.append(float(pair_loss.detach()))
        epoch_total_losses.append(_mean(total_losses))
        epoch_state_losses.append(_mean(state_losses))
        epoch_pair_losses.append(_mean(pair_losses))
    model.eval()
    return {
        "initial_total_loss": epoch_total_losses[0],
        "final_total_loss": epoch_total_losses[-1],
        "initial_state_loss": epoch_state_losses[0],
        "final_state_loss": epoch_state_losses[-1],
        "initial_pair_loss": epoch_pair_losses[0],
        "final_pair_loss": epoch_pair_losses[-1],
    }


def _flip_pair_cross_target_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    paired_logits = logits.reshape(-1, 2, logits.shape[-1])
    paired_targets = targets.reshape(-1, 2)
    flip_mask = paired_targets[:, 0] != paired_targets[:, 1]
    if not bool(flip_mask.any()):
        return logits.sum() * 0.0
    selected_logits = paired_logits[flip_mask]
    selected_targets = paired_targets[flip_mask]
    row = torch.arange(selected_logits.shape[0])
    before_correct = selected_logits[row, 0, selected_targets[:, 0]]
    before_other = selected_logits[row, 0, selected_targets[:, 1]]
    after_correct = selected_logits[row, 1, selected_targets[:, 1]]
    after_other = selected_logits[row, 1, selected_targets[:, 0]]
    return 0.5 * (
        F.relu(margin - (before_correct - before_other)).mean()
        + F.relu(margin - (after_correct - after_other)).mean()
    )


def _matches_formal_protocol(
    *,
    candidate_pool_per_family: int,
    train_pairs_per_family: int,
    pairs_per_family_per_batch: int,
    epochs: int,
    data_seed: int,
    model_seeds: tuple[int, ...],
    state_margin: float,
    pair_margin: float,
    pair_loss_weight: float,
    minimum_overall_train_agreement: float,
    minimum_family_flip_pair_exact: float,
    maximum_residual_saturation_rate: float,
) -> bool:
    return {
        "candidate_pool_per_family": candidate_pool_per_family,
        "train_pairs_per_family": train_pairs_per_family,
        "pairs_per_family_per_batch": pairs_per_family_per_batch,
        "data_seed": data_seed,
        "model_seeds": model_seeds,
        "epochs": epochs,
        "state_margin": state_margin,
        "pair_margin": pair_margin,
        "pair_loss_weight": pair_loss_weight,
        "minimum_overall_train_agreement": minimum_overall_train_agreement,
        "minimum_family_flip_pair_exact": minimum_family_flip_pair_exact,
        "maximum_residual_saturation_rate": maximum_residual_saturation_rate,
    } == FORMAL_PROTOCOL


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _render_report(summary: dict[str, object]) -> str:
    variants = cast(dict[str, dict[str, object]], summary["variants"])
    lines = [
        "# Ticket 35 Pair-Structured Train Diagnostic",
        "",
        f"Decision: **{summary['decision']}**.",
        "",
        "Comparison of deterministic family-balanced state-only and pair-structured training.",
        "",
        "## Per-Seed Results",
        "",
        "| Variant | Method | Seed | Agreement | Saturation | All conditions |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant_name in TRAINING_VARIANTS:
        methods = cast(
            dict[str, dict[str, object]], variants[variant_name]["methods"]
        )
        for method_name in TRAINABLE_METHODS:
            per_seed = cast(
                list[dict[str, object]], methods[method_name]["per_seed"]
            )
            for row in per_seed:
                criteria = cast(dict[str, bool], row["criteria"])
                lines.append(
                    f"| {variant_name} | {method_name} | {row['model_seed']} | "
                    f"{cast(float, row['overall_train_agreement']):.4f} | "
                    f"{cast(float, row['residual_saturation_rate']):.4f} | "
                    f"{str(criteria['all_training_conditions_met']).lower()} |"
                )
    lines.extend(
        [
            "",
            "## Candidate Family Flip-Pair Exact",
            "",
            "| Method | Seed | Family | Flip pairs | Exact accuracy | Pass |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    candidate_methods = cast(
        dict[str, dict[str, object]],
        variants["family_balanced_pair_structured"]["methods"],
    )
    for method_name in TRAINABLE_METHODS:
        per_seed = cast(
            list[dict[str, object]], candidate_methods[method_name]["per_seed"]
        )
        for row in per_seed:
            family_rows = cast(
                dict[str, dict[str, object]],
                row["oracle_flip_pair_exact_by_family"],
            )
            for family in RELATIONAL_FAMILIES:
                family_row = family_rows[family]
                lines.append(
                    f"| {method_name} | {row['model_seed']} | {family} | "
                    f"{family_row['oracle_flip_pair_count']} | "
                    f"{cast(float, family_row['exact_pair_accuracy']):.4f} | "
                    f"{str(family_row['criterion_met']).lower()} |"
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
            "This train-only comparison contains no held-out model result, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ticket 35 pair-structured train-only diagnostic"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_pair_structured_train_diagnostic(args.output_dir)
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
