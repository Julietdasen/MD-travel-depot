"""Two-stage train-only curriculum for relational flip optimization."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast

import torch
import torch.nn.functional as F

from experiments.md_pair_structured_train_diagnostic import (
    _PreparedBatch,
    _family_balanced_batches,
    _flip_pair_cross_target_loss,
    _mean,
    _prepare_batch,
)
from experiments.md_policy_relational_gate import (
    DEFAULT_MODEL_SEEDS,
    RELATIONAL_FAMILIES,
    RelationalState,
    RelationalTwin,
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
from models.md_enhanced_policy import (
    MDEnhancedSchedulerNetwork,
    MDPolicyDiagnostics,
)
from models.md_policy import MDPolicyInputs


FORMAL_PROTOCOL: Final = {
    "candidate_pool_per_family": 5000,
    "train_pairs_per_family": 500,
    "pairs_per_family_per_batch": 100,
    "pretrain_epochs": 200,
    "fine_tune_epochs": 100,
    "checkpoint_interval": 10,
    "data_seed": 3030,
    "model_seeds": DEFAULT_MODEL_SEEDS,
    "pretrain_learning_rate": 0.01,
    "fine_tune_learning_rate": 0.002,
    "state_margin": 0.1,
    "flip_state_margin": 0.1,
    "pair_margin": 0.1,
    "flip_state_loss_weight": 0.5,
    "pair_loss_weight": 0.25,
    "minimum_overall_train_agreement": 0.70,
    "minimum_family_flip_pair_exact": 0.60,
    "maximum_residual_saturation_rate": 0.25,
}


@dataclass(frozen=True, slots=True)
class FlipCurriculumTrainDiagnosticResult:
    summary_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class _TrainingCheckpoint:
    fine_tune_epoch: int
    state_dict: dict[str, torch.Tensor]
    overall_train_agreement: float
    residual_saturation_rate: float
    family_exact: dict[str, dict[str, object]]


def run_flip_curriculum_train_diagnostic(
    output_dir: str | Path,
    *,
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
    saturation_loss_weight: float = 0.0,
    saturation_penalty_allowance: float = 0.0,
    minimum_overall_train_agreement: float = 0.70,
    minimum_family_flip_pair_exact: float = 0.60,
    maximum_residual_saturation_rate: float = 0.25,
) -> FlipCurriculumTrainDiagnosticResult:
    """Run the frozen train-only curriculum without constructing evaluation data."""
    protocol = {
        "candidate_pool_per_family": candidate_pool_per_family,
        "train_pairs_per_family": train_pairs_per_family,
        "pairs_per_family_per_batch": pairs_per_family_per_batch,
        "pretrain_epochs": pretrain_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "checkpoint_interval": checkpoint_interval,
        "data_seed": data_seed,
        "model_seeds": _validated_model_seeds(model_seeds),
        "pretrain_learning_rate": pretrain_learning_rate,
        "fine_tune_learning_rate": fine_tune_learning_rate,
        "state_margin": state_margin,
        "flip_state_margin": flip_state_margin,
        "pair_margin": pair_margin,
        "flip_state_loss_weight": flip_state_loss_weight,
        "pair_loss_weight": pair_loss_weight,
        "minimum_overall_train_agreement": minimum_overall_train_agreement,
        "minimum_family_flip_pair_exact": minimum_family_flip_pair_exact,
        "maximum_residual_saturation_rate": maximum_residual_saturation_rate,
    }
    _validate_protocol(protocol)
    if (
        isinstance(saturation_loss_weight, bool)
        or not isinstance(saturation_loss_weight, (int, float))
        or not math.isfinite(saturation_loss_weight)
        or saturation_loss_weight < 0
    ):
        raise ValueError("saturation_loss_weight must be non-negative and finite")
    if (
        isinstance(saturation_penalty_allowance, bool)
        or not isinstance(saturation_penalty_allowance, (int, float))
        or not math.isfinite(saturation_penalty_allowance)
        or not 0 <= saturation_penalty_allowance < 1
    ):
        raise ValueError("saturation_penalty_allowance must be within [0, 1)")
    resolved_model_seeds = cast(tuple[int, ...], protocol["model_seeds"])
    destination = Path(output_dir)
    summary_path = destination / "flip_curriculum_train_diagnostic.json"
    report_path = destination / "flip_curriculum_train_diagnostic.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite flip curriculum train diagnostic: "
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
    checkpoint_epochs = _checkpoint_epochs(
        fine_tune_epochs=fine_tune_epochs,
        checkpoint_interval=checkpoint_interval,
    )

    methods: dict[str, dict[str, object]] = {}
    for method in TRAINABLE_METHODS:
        per_seed: list[dict[str, object]] = []
        for model_seed in resolved_model_seeds:
            model = _build_model(method, seed=model_seed)
            initial_agreement = _model_agreement(model, train_states)
            training_result = _train_curriculum(
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
                minimum_overall_train_agreement=minimum_overall_train_agreement,
                minimum_family_flip_pair_exact=minimum_family_flip_pair_exact,
                maximum_residual_saturation_rate=(
                    maximum_residual_saturation_rate
                ),
            )
            training_result["model_seed"] = model_seed
            training_result["initial_train_agreement"] = initial_agreement
            per_seed.append(training_result)
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

    parameter_counts = {
        cast(int, row["parameter_count"]) for row in methods.values()
    }
    parameter_budget_equal = len(parameter_counts) == 1
    candidate_training_conditions_met = parameter_budget_equal and all(
        cast(bool, row["all_seed_training_conditions_met"])
        for row in methods.values()
    )
    formal_protocol_run = (
        protocol == FORMAL_PROTOCOL
        and saturation_loss_weight == 0
        and saturation_penalty_allowance == 0
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
        "status": "flip_curriculum_train_diagnostic",
        "ticket": 36,
        "source_ticket": 35,
        "protocol_class": "exploratory_train_only_curriculum",
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
            "pretrain_epochs": pretrain_epochs,
            "fine_tune_epochs": fine_tune_epochs,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_fine_tune_epochs": list(checkpoint_epochs),
            "pretrain_learning_rate": pretrain_learning_rate,
            "fine_tune_learning_rate": fine_tune_learning_rate,
            "state_objective": "bounded_multi_margin",
            "state_margin": state_margin,
            "flip_state_objective": "family_balanced_bounded_multi_margin",
            "flip_state_margin": flip_state_margin,
            "flip_state_loss_weight": flip_state_loss_weight,
            "pair_objective": "flip_pair_cross_target_margin",
            "pair_margin": pair_margin,
            "pair_loss_weight": pair_loss_weight,
            **(
                {
                    "raw_residual_saturation_penalty": {
                        "boundary": "atanh(0.95)",
                        "eligible_actions": "hard_feasible_transport",
                        "objective": (
                            "mean_excess_outside_saturation_budget"
                            if saturation_penalty_allowance > 0
                            else "mean_excess_absolute_raw_residual"
                        ),
                        "weight": saturation_loss_weight,
                        **(
                            {
                                "allowed_saturation_rate": (
                                    saturation_penalty_allowance
                                )
                            }
                            if saturation_penalty_allowance > 0
                            else {}
                        ),
                    }
                }
                if saturation_loss_weight > 0
                else {}
            ),
            "pairs_per_family_per_batch": pairs_per_family_per_batch,
            "batches_per_epoch": len(balanced_batches),
            "batch_order": "fixed_family_then_candidate_order",
            "family_pairs_per_batch": {
                family: pairs_per_family_per_batch
                for family in RELATIONAL_FAMILIES
            },
            "checkpoint_selection_order": [
                "overall_and_saturation_conditions_met",
                "family_flip_pair_exact_pass_count",
                "minimum_family_flip_pair_exact",
                "overall_train_agreement",
                "earlier_fine_tune_epoch",
            ],
        },
        "thresholds": {
            "minimum_overall_train_agreement": minimum_overall_train_agreement,
            "minimum_family_flip_pair_exact": minimum_family_flip_pair_exact,
            "maximum_residual_saturation_rate": (
                maximum_residual_saturation_rate
            ),
        },
        "selection_controls": {
            "label_used_for_candidate_pair_selection": False,
            "oracle_flip_used_for_candidate_pair_selection": False,
            "margin_used_for_candidate_pair_selection": False,
            "model_output_used_for_candidate_pair_selection": False,
            "candidate_pair_selection_order": "first_generated_pairs_per_family",
            "training_labels_used_after_pair_selection": True,
            "training_metrics_used_for_checkpoint_selection": True,
        },
        "methods": methods,
        "candidate_training_conditions_met": candidate_training_conditions_met,
        "all_training_conditions_met": all_training_conditions_met,
        "decision": decision,
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
            "Freeze a separate held-out relational evaluation protocol."
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
    return FlipCurriculumTrainDiagnosticResult(summary_path, report_path)


def _train_curriculum(
    model: MDEnhancedSchedulerNetwork,
    batches: Sequence[_PreparedBatch],
    train_twins: Sequence[RelationalTwin],
    train_states: Sequence[RelationalState],
    *,
    pretrain_epochs: int,
    fine_tune_epochs: int,
    checkpoint_interval: int,
    pretrain_learning_rate: float,
    fine_tune_learning_rate: float,
    state_margin: float,
    flip_state_margin: float,
    pair_margin: float,
    flip_state_loss_weight: float,
    pair_loss_weight: float,
    saturation_loss_weight: float,
    saturation_penalty_allowance: float,
    minimum_overall_train_agreement: float,
    minimum_family_flip_pair_exact: float,
    maximum_residual_saturation_rate: float,
) -> dict[str, object]:
    pretrain_optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=pretrain_learning_rate,
    )
    pretrain_losses = [
        _train_epoch(
            model,
            batches,
            pretrain_optimizer,
            state_margin=state_margin,
            flip_state_margin=flip_state_margin,
            pair_margin=pair_margin,
            flip_state_loss_weight=0.0,
            pair_loss_weight=0.0,
            saturation_loss_weight=saturation_loss_weight,
            saturation_penalty_allowance=saturation_penalty_allowance,
        )
        for _epoch in range(pretrain_epochs)
    ]
    checkpoints = [
        _capture_checkpoint(
            model,
            train_twins,
            train_states,
            fine_tune_epoch=0,
            minimum_family_flip_pair_exact=minimum_family_flip_pair_exact,
        )
    ]

    fine_tune_optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=fine_tune_learning_rate,
    )
    fine_tune_losses: list[dict[str, float]] = []
    checkpoint_epochs = set(
        _checkpoint_epochs(
            fine_tune_epochs=fine_tune_epochs,
            checkpoint_interval=checkpoint_interval,
        )
    )
    for epoch in range(1, fine_tune_epochs + 1):
        fine_tune_losses.append(
            _train_epoch(
                model,
                batches,
                fine_tune_optimizer,
                state_margin=state_margin,
                flip_state_margin=flip_state_margin,
                pair_margin=pair_margin,
                flip_state_loss_weight=flip_state_loss_weight,
                pair_loss_weight=pair_loss_weight,
                saturation_loss_weight=saturation_loss_weight,
                saturation_penalty_allowance=saturation_penalty_allowance,
            )
        )
        if epoch in checkpoint_epochs:
            checkpoints.append(
                _capture_checkpoint(
                    model,
                    train_twins,
                    train_states,
                    fine_tune_epoch=epoch,
                    minimum_family_flip_pair_exact=(
                        minimum_family_flip_pair_exact
                    ),
                )
            )

    selected = max(
        checkpoints,
        key=lambda checkpoint: _checkpoint_score(
            checkpoint,
            minimum_overall_train_agreement=minimum_overall_train_agreement,
            minimum_family_flip_pair_exact=minimum_family_flip_pair_exact,
            maximum_residual_saturation_rate=maximum_residual_saturation_rate,
        ),
    )
    model.load_state_dict(selected.state_dict)
    family_passed = all(
        cast(bool, row["criterion_met"])
        for row in selected.family_exact.values()
    )
    overall_passed = (
        selected.overall_train_agreement >= minimum_overall_train_agreement
    )
    saturation_passed = (
        selected.residual_saturation_rate
        <= maximum_residual_saturation_rate
    )
    return {
        "selected_fine_tune_epoch": selected.fine_tune_epoch,
        "overall_train_agreement": selected.overall_train_agreement,
        "residual_saturation_rate": selected.residual_saturation_rate,
        "oracle_flip_pair_exact_by_family": selected.family_exact,
        "checkpoint_metrics": [
            _checkpoint_record(checkpoint) for checkpoint in checkpoints
        ],
        "loss_diagnostics": {
            "initial_pretrain_total_loss": pretrain_losses[0]["total_loss"],
            "final_pretrain_total_loss": pretrain_losses[-1]["total_loss"],
            "initial_fine_tune_total_loss": fine_tune_losses[0]["total_loss"],
            "final_fine_tune_total_loss": fine_tune_losses[-1]["total_loss"],
            "final_fine_tune_state_loss": fine_tune_losses[-1]["state_loss"],
            "final_fine_tune_flip_state_loss": fine_tune_losses[-1][
                "flip_state_loss"
            ],
            "final_fine_tune_pair_loss": fine_tune_losses[-1]["pair_loss"],
        },
        "criteria": {
            "overall_train_agreement_met": overall_passed,
            "all_family_flip_pair_exact_met": family_passed,
            "residual_saturation_rate_met": saturation_passed,
            "all_training_conditions_met": (
                overall_passed and family_passed and saturation_passed
            ),
        },
    }


def _train_epoch(
    model: MDEnhancedSchedulerNetwork,
    batches: Sequence[_PreparedBatch],
    optimizer: torch.optim.Optimizer,
    *,
    state_margin: float,
    flip_state_margin: float,
    pair_margin: float,
    flip_state_loss_weight: float,
    pair_loss_weight: float,
    saturation_loss_weight: float,
    saturation_penalty_allowance: float,
) -> dict[str, float]:
    model.train()
    total_losses: list[float] = []
    state_losses: list[float] = []
    flip_state_losses: list[float] = []
    pair_losses: list[float] = []
    for batch in batches:
        diagnostics = model.forward_with_diagnostics(
            batch.robot_features,
            batch.task_features,
            batch.task_adjacency,
            md_inputs=cast(MDPolicyInputs, batch.md_inputs),
        )
        logits = _flatten_transport_scores(diagnostics.scores)
        state_loss = F.multi_margin_loss(
            logits, batch.targets, margin=state_margin
        )
        flip_state_loss = _family_balanced_flip_state_loss(
            logits,
            batch.targets,
            batch.twins,
            margin=flip_state_margin,
        )
        pair_loss = _flip_pair_cross_target_loss(
            logits,
            batch.targets,
            margin=pair_margin,
        )
        total_loss = (
            state_loss
            + flip_state_loss_weight * flip_state_loss
            + pair_loss_weight * pair_loss
        )
        if saturation_loss_weight > 0:
            total_loss = total_loss + saturation_loss_weight * (
                _raw_residual_saturation_loss(
                    diagnostics,
                    cast(MDPolicyInputs, batch.md_inputs),
                    allowed_saturation_rate=saturation_penalty_allowance,
                )
            )
        optimizer.zero_grad()
        total_loss.backward()
        if not all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            raise RuntimeError("flip curriculum produced non-finite gradients")
        optimizer.step()
        total_losses.append(float(total_loss.detach()))
        state_losses.append(float(state_loss.detach()))
        flip_state_losses.append(float(flip_state_loss.detach()))
        pair_losses.append(float(pair_loss.detach()))
    return {
        "total_loss": _mean(total_losses),
        "state_loss": _mean(state_losses),
        "flip_state_loss": _mean(flip_state_losses),
        "pair_loss": _mean(pair_losses),
    }


def _family_balanced_flip_state_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    twins: Sequence[RelationalTwin],
    *,
    margin: float,
) -> torch.Tensor:
    family_losses: list[torch.Tensor] = []
    for family in RELATIONAL_FAMILIES:
        state_indices: list[int] = []
        for pair_index, twin in enumerate(twins):
            before_index = 2 * pair_index
            after_index = before_index + 1
            if (
                twin.family == family
                and targets[before_index] != targets[after_index]
            ):
                state_indices.extend((before_index, after_index))
        if state_indices:
            family_losses.append(
                F.multi_margin_loss(
                    logits[state_indices],
                    targets[state_indices],
                    margin=margin,
                )
            )
    if not family_losses:
        return logits.sum() * 0.0
    return torch.stack(family_losses).mean()


def _raw_residual_saturation_loss(
    diagnostics: MDPolicyDiagnostics,
    md_inputs: MDPolicyInputs,
    *,
    allowed_saturation_rate: float,
) -> torch.Tensor:
    device = diagnostics.raw_residual.device
    eligible = md_inputs.hard_feasibility_mask.to(
        device=device
    ) & md_inputs.task_is_transport.to(device=device).unsqueeze(1)
    absolute_raw = diagnostics.raw_residual.abs()[eligible]
    allowed_count = math.floor(
        allowed_saturation_rate * absolute_raw.numel()
    )
    if allowed_count:
        absolute_raw = torch.sort(absolute_raw).values[:-allowed_count]
    return F.relu(absolute_raw - math.atanh(0.95)).mean()


def _capture_checkpoint(
    model: MDEnhancedSchedulerNetwork,
    train_twins: Sequence[RelationalTwin],
    train_states: Sequence[RelationalState],
    *,
    fine_tune_epoch: int,
    minimum_family_flip_pair_exact: float,
) -> _TrainingCheckpoint:
    logits, diagnostics = _model_logits(model, train_states)
    predictions = cast(list[int], torch.argmax(logits, dim=-1).tolist())
    return _TrainingCheckpoint(
        fine_tune_epoch=fine_tune_epoch,
        state_dict={
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        overall_train_agreement=_agreement(predictions, train_states),
        residual_saturation_rate=float(
            diagnostics.residual_saturation_rate.mean()
        ),
        family_exact=_flip_pair_exact_by_family(
            train_twins,
            train_states,
            predictions,
            minimum_family_flip_pair_exact,
        ),
    )


def _checkpoint_score(
    checkpoint: _TrainingCheckpoint,
    *,
    minimum_overall_train_agreement: float,
    minimum_family_flip_pair_exact: float,
    maximum_residual_saturation_rate: float,
) -> tuple[bool, int, float, float, int]:
    exact_values = [
        cast(float, row["exact_pair_accuracy"])
        for row in checkpoint.family_exact.values()
        if row["exact_pair_accuracy"] is not None
    ]
    family_pass_count = sum(
        exact >= minimum_family_flip_pair_exact for exact in exact_values
    )
    return (
        checkpoint.overall_train_agreement >= minimum_overall_train_agreement
        and checkpoint.residual_saturation_rate
        <= maximum_residual_saturation_rate,
        family_pass_count,
        min(exact_values, default=-1.0),
        checkpoint.overall_train_agreement,
        -checkpoint.fine_tune_epoch,
    )


def _checkpoint_record(checkpoint: _TrainingCheckpoint) -> dict[str, object]:
    return {
        "fine_tune_epoch": checkpoint.fine_tune_epoch,
        "overall_train_agreement": checkpoint.overall_train_agreement,
        "residual_saturation_rate": checkpoint.residual_saturation_rate,
        "oracle_flip_pair_exact_by_family": checkpoint.family_exact,
    }


def _checkpoint_epochs(
    *, fine_tune_epochs: int, checkpoint_interval: int
) -> tuple[int, ...]:
    epochs = list(range(0, fine_tune_epochs + 1, checkpoint_interval))
    if epochs[-1] != fine_tune_epochs:
        epochs.append(fine_tune_epochs)
    return tuple(epochs)


def _validate_protocol(protocol: dict[str, object]) -> None:
    for name in (
        "candidate_pool_per_family",
        "train_pairs_per_family",
        "pairs_per_family_per_batch",
        "pretrain_epochs",
        "fine_tune_epochs",
        "checkpoint_interval",
    ):
        value = protocol[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    candidate_count = cast(int, protocol["candidate_pool_per_family"])
    train_count = cast(int, protocol["train_pairs_per_family"])
    batch_count = cast(int, protocol["pairs_per_family_per_batch"])
    if candidate_count < train_count:
        raise ValueError(
            "candidate_pool_per_family must be >= train_pairs_per_family"
        )
    if train_count % batch_count:
        raise ValueError(
            "train_pairs_per_family must be divisible by pairs_per_family_per_batch"
        )
    data_seed = protocol["data_seed"]
    if isinstance(data_seed, bool) or not isinstance(data_seed, int) or data_seed < 0:
        raise ValueError("data_seed must be a non-negative integer")
    for name in (
        "pretrain_learning_rate",
        "fine_tune_learning_rate",
        "state_margin",
        "flip_state_margin",
        "pair_margin",
        "flip_state_loss_weight",
        "pair_loss_weight",
    ):
        value = protocol[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
    for name in (
        "minimum_overall_train_agreement",
        "minimum_family_flip_pair_exact",
        "maximum_residual_saturation_rate",
    ):
        value = protocol[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be finite and within [0, 1]")


def _render_report(summary: dict[str, object]) -> str:
    methods = cast(dict[str, dict[str, object]], summary["methods"])
    lines = [
        "# Ticket 36 Flip-Balanced Curriculum Train Diagnostic",
        "",
        f"Decision: **{summary['decision']}**.",
        "",
        "Two-stage family-balanced pretraining and relational fine-tuning.",
        "",
        "## Selected Checkpoints",
        "",
        "| Method | Seed | Fine-tune epoch | Agreement | Saturation | All conditions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method_name in TRAINABLE_METHODS:
        per_seed = cast(list[dict[str, object]], methods[method_name]["per_seed"])
        for row in per_seed:
            criteria = cast(dict[str, bool], row["criteria"])
            lines.append(
                f"| {method_name} | {row['model_seed']} | "
                f"{row['selected_fine_tune_epoch']} | "
                f"{cast(float, row['overall_train_agreement']):.4f} | "
                f"{cast(float, row['residual_saturation_rate']):.4f} | "
                f"{str(criteria['all_training_conditions_met']).lower()} |"
            )
    lines.extend(
        [
            "",
            "## Family Flip-Pair Exact",
            "",
            "| Method | Seed | Family | Flip pairs | Exact accuracy | Pass |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for method_name in TRAINABLE_METHODS:
        per_seed = cast(list[dict[str, object]], methods[method_name]["per_seed"])
        for row in per_seed:
            family_rows = cast(
                dict[str, dict[str, object]],
                row["oracle_flip_pair_exact_by_family"],
            )
            for family in RELATIONAL_FAMILIES:
                family_row = family_rows[family]
                accuracy = family_row["exact_pair_accuracy"]
                accuracy_text = (
                    "NA" if accuracy is None else f"{cast(float, accuracy):.4f}"
                )
                lines.append(
                    f"| {method_name} | {row['model_seed']} | {family} | "
                    f"{family_row['oracle_flip_pair_count']} | {accuracy_text} | "
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
            "This train-only diagnostic contains no held-out model result, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ticket 36 flip-balanced train-only curriculum"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_flip_curriculum_train_diagnostic(args.output_dir)
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
