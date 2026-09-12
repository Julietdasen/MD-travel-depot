"""Controlled multi-transport relational gate for pair-aware scoring."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence, cast

import torch
import torch.nn.functional as F

from experiments.md_policy_signal_gate import (
    _MatchedParameterMLPNetwork,
    _bootstrap_mean_ci,
    _wilson_interval,
)
from models.md_enhanced_policy import (
    MDEnhancedPolicyConfig,
    MDEnhancedSchedulerNetwork,
    MDPolicyDiagnostics,
)
from models.md_policy import MD_OPPORTUNITY_FEATURE_COUNT, MDPolicyInputs


ROBOT_COUNT = 3
CANDIDATE_TASK_COUNT = 4
TASK_COUNT = 2 * CANDIDATE_TASK_COUNT
RELATIONAL_FAMILIES = (
    "competitor_robot_position",
    "competitor_robot_speed",
    "alternative_task_pickup",
    "alternative_downstream_priority",
)
METHOD_NAMES = (
    "physics_only",
    "local_eta_priority",
    "matched_parameter_mlp",
    "cross_attention_full",
    "pair_aware_attention",
)
TRAINABLE_METHODS = METHOD_NAMES[2:]
DEFAULT_MODEL_SEEDS = (3101, 3102, 3103)
MARGIN_UTILITY_TOLERANCE = 0.05
MARGIN_STRATA = (
    ("near_tie_lt_0.01", 0.0, 0.01),
    ("less_ambiguous_ge_0.01", 0.01, 0.05),
    ("clear_ge_0.05", 0.05, math.inf),
)
Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class RelationalState:
    state_id: str
    pair_id: str
    family: str
    robot_positions: tuple[Point, ...]
    robot_loaded_speeds: tuple[float, ...]
    task_pickups: tuple[Point, ...]
    task_deliveries: tuple[Point, ...]
    downstream_priorities: tuple[float, ...]
    eta: tuple[tuple[float, ...], ...]
    action_values: tuple[float, ...]
    oracle_action: int
    hard_mask: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True, slots=True)
class RelationalTwin:
    family: str
    pair_id: str
    changed_entity_index: int
    before: RelationalState
    after: RelationalState


@dataclass(frozen=True, slots=True)
class RelationalGateResult:
    summary_path: Path
    report_path: Path


def build_relational_twins(
    *, pairs_per_family: int = 25, seed: int = 2027
) -> tuple[RelationalTwin, ...]:
    """Build non-local counterfactuals over small multi-transport states."""

    if (
        isinstance(pairs_per_family, bool)
        or not isinstance(pairs_per_family, int)
        or pairs_per_family <= 0
    ):
        raise ValueError("pairs_per_family must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    generator = random.Random(seed)
    twins: list[RelationalTwin] = []
    for family in RELATIONAL_FAMILIES:
        for index in range(pairs_per_family):
            pair_id = f"{family}-{index:04d}"
            for _attempt in range(4000):
                base = _random_state_spec(generator)
                changed_index, changed = _relational_intervention(
                    family, base, generator
                )
                before = _state_from_spec(
                    state_id=f"{pair_id}-before",
                    pair_id=pair_id,
                    family=family,
                    spec=base,
                )
                after = _state_from_spec(
                    state_id=f"{pair_id}-after",
                    pair_id=pair_id,
                    family=family,
                    spec=changed,
                )
                if before.oracle_action == after.oracle_action:
                    continue
                if not _change_is_nonlocal(
                    family,
                    changed_index,
                    before.oracle_action,
                    after.oracle_action,
                ):
                    continue
                twins.append(
                    RelationalTwin(
                        family,
                        pair_id,
                        changed_index,
                        before,
                        after,
                    )
                )
                break
            else:
                raise RuntimeError(f"could not build relational twin {pair_id}")

    signatures = {_template_signature(twin) for twin in twins}
    if len(signatures) != len(twins):
        raise RuntimeError("relational generator produced duplicate templates")
    return tuple(twins)


def select_margin_balanced_twins(
    twins: Sequence[RelationalTwin],
    *,
    pairs_per_family: int,
    min_margin: float = 0.02,
) -> tuple[RelationalTwin, ...]:
    """Select a deterministic, balanced subset with clear oracle choices."""
    if isinstance(pairs_per_family, bool) or not isinstance(pairs_per_family, int) or pairs_per_family <= 0:
        raise ValueError("pairs_per_family must be a positive integer")
    if isinstance(min_margin, bool) or not isinstance(min_margin, (int, float)) or not math.isfinite(min_margin) or min_margin < 0:
        raise ValueError("min_margin must be finite and non-negative")
    selected: list[RelationalTwin] = []
    for family in RELATIONAL_FAMILIES:
        candidates = [
            twin
            for twin in twins
            if twin.family == family
            and _state_completion_margin(twin.before) >= min_margin
            and _state_completion_margin(twin.after) >= min_margin
        ]
        if len(candidates) < pairs_per_family:
            raise ValueError(
                f"not enough margin-qualified pairs for {family}: "
                f"need {pairs_per_family}, found {len(candidates)}"
            )
        selected.extend(candidates[:pairs_per_family])
    return tuple(selected)



def run_md_policy_relational_gate(
    output_dir: str | Path,
    *,
    pairs_per_family: int = 25,
    selection_pool_per_family: int | None = None,
    min_margin: float | None = None,
    epochs: int = 200,
    seed: int = 2027,
    model_seeds: Sequence[int] = DEFAULT_MODEL_SEEDS,
    training_objective: str = "bounded_margin",
    margin: float = 0.1,
) -> RelationalGateResult:
    """Train equal-budget multi-seed probes on held-out relational twins."""

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if training_objective not in ("bounded_margin", "cross_entropy"):
        raise ValueError("unsupported relational training objective")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(margin)
        or margin <= 0
    ):
        raise ValueError("margin must be positive and finite")
    resolved_model_seeds = _validated_model_seeds(model_seeds)
    started_at = time.perf_counter()
    if selection_pool_per_family is not None:
        if (
            isinstance(selection_pool_per_family, bool)
            or not isinstance(selection_pool_per_family, int)
            or selection_pool_per_family < pairs_per_family
        ):
            raise ValueError(
                "selection_pool_per_family must be an integer >= pairs_per_family"
            )
    if min_margin is not None and (
        isinstance(min_margin, bool)
        or not isinstance(min_margin, (int, float))
        or not math.isfinite(min_margin)
        or min_margin < 0
    ):
        raise ValueError("min_margin must be finite and non-negative")
    if min_margin is not None and selection_pool_per_family is None:
        raise ValueError("min_margin requires selection_pool_per_family")
    if selection_pool_per_family is not None and min_margin is None:
        raise ValueError("selection_pool_per_family requires min_margin")
    generation_pairs_per_family = (
        selection_pool_per_family
        if selection_pool_per_family is not None
        else pairs_per_family
    )
    twins = build_relational_twins(
        pairs_per_family=generation_pairs_per_family,
        seed=seed,
    )
    if min_margin is not None:
        twins = select_margin_balanced_twins(
            twins,
            pairs_per_family=pairs_per_family,
            min_margin=float(min_margin),
        )
    train_twins, evaluation_twins = _split_twins(twins, pairs_per_family)
    train_templates = {_template_signature(twin) for twin in train_twins}
    evaluation_templates = {
        _template_signature(twin) for twin in evaluation_twins
    }
    template_overlap = train_templates & evaluation_templates
    if template_overlap:
        raise RuntimeError("train/evaluation relational templates overlap")

    train_states = tuple(
        state for twin in train_twins for state in (twin.before, twin.after)
    )
    evaluation_states = tuple(
        state for twin in evaluation_twins for state in (twin.before, twin.after)
    )

    metrics: dict[str, dict[str, object]] = {}
    for method in METHOD_NAMES[:2]:
        static_metrics = _evaluate_method(
            method,
            None,
            evaluation_twins,
            evaluation_states,
            learning_curve=(),
            initial_agreement=None,
            train_agreement=None,
            seed=seed,
        )
        static_metrics["per_seed"] = []
        metrics[method] = static_metrics

    for method in TRAINABLE_METHODS:
        per_seed = []
        for model_seed in resolved_model_seeds:
            model = _build_model(method, seed=model_seed)
            initial_agreement = _model_agreement(model, evaluation_states)
            curve = _train_model(
                model,
                train_states,
                epochs=epochs,
                training_objective=training_objective,
                margin=float(margin),
            )
            train_agreement = _model_agreement(model, train_states)
            seed_metrics = _evaluate_method(
                method,
                model,
                evaluation_twins,
                evaluation_states,
                learning_curve=curve,
                initial_agreement=initial_agreement,
                train_agreement=train_agreement,
                seed=seed + model_seed,
            )
            seed_metrics["model_seed"] = model_seed
            per_seed.append(seed_metrics)
        metrics[method] = _aggregate_seed_metrics(
            per_seed,
            bootstrap_seed=seed,
        )

    matched = metrics["matched_parameter_mlp"]
    pair_aware = metrics["pair_aware_attention"]
    matched_agreement = cast(float, matched["first_action_agreement"])
    matched_flip = cast(float, matched["relational_flip_accuracy"])
    pair_aware_agreement = cast(float, pair_aware["first_action_agreement"])
    pair_aware_flip = cast(float, pair_aware["relational_flip_accuracy"])
    family_flip = cast(dict[str, float], pair_aware["family_flip_accuracy"])
    train_fit_values = [
        cast(float, seed_metrics["train_first_action_agreement"])
        for method in ("matched_parameter_mlp", "pair_aware_attention")
        for seed_metrics in cast(list[dict[str, object]], metrics[method]["per_seed"])
    ]
    comparison_valid = min(train_fit_values) >= 0.8
    pair_aware_pair_agreements = cast(
        list[float], pair_aware["pair_agreement_values"]
    )
    matched_pair_agreements = cast(
        list[float], matched["pair_agreement_values"]
    )
    agreement_differences = [
        pair_value - matched_value
        for pair_value, matched_value in zip(
            pair_aware_pair_agreements,
            matched_pair_agreements,
            strict=True,
        )
    ]
    agreement_difference_ci95 = list(
        _bootstrap_mean_ci(agreement_differences, seed=seed + 97)
    )
    parameter_counts = {
        method: cast(int, metrics[method]["parameter_count"])
        for method in TRAINABLE_METHODS
    }
    parameter_budget_equal = len(set(parameter_counts.values())) == 1
    pair_aware_supported = bool(
        min_margin is None
        and comparison_valid
        and parameter_budget_equal
        and pair_aware_agreement >= matched_agreement + 0.05
        and pair_aware_flip >= matched_flip + 0.05
        and min(family_flip.values()) >= 0.6
        and agreement_difference_ci95[0] > 0.0
        and cast(float, pair_aware["residual_saturation_rate"]) <= 0.25
    )
    claim = (
        "diagnostic_only"
        if min_margin is not None
        else (
            "inconclusive"
            if not comparison_valid
            else ("go" if pair_aware_supported else "no_go")
        )
    )

    report_phase = (
        "margin_balanced_diagnostic"
        if min_margin is not None
        else (
            "initial_cross_entropy_gate"
            if training_objective == "cross_entropy"
            else "bounded_margin_follow_up_held_out_gate"
        )
    )

    summary = {
        "schema_version": "2.0.0",
        "status": "pair_aware_relational_gate",
        "ticket": 29,
        "report_phase": report_phase,
        "train_only_diagnosis": {
            "is_this_report": False,
            "description": "Train-only optimization diagnosis is a separate run; this report contains held-out gate results.",
        },
        "robot_count": ROBOT_COUNT,
        "transport_robot_count": ROBOT_COUNT,
        "candidate_transport_task_count": CANDIDATE_TASK_COUNT,
        "downstream_context_task_count": CANDIDATE_TASK_COUNT,
        "pairs_per_family": pairs_per_family,
        "sampling": {
            "margin_balanced": min_margin is not None,
            "selection_pool_per_family": selection_pool_per_family,
            "minimum_state_margin": min_margin,
        },
        "pair_count": len(twins),
        "state_count": 2 * len(twins),
        "train_pair_count": len(train_twins),
        "evaluation_pair_count": len(evaluation_twins),
        "epochs": epochs,
        "training": {
            "objective": training_objective,
            "margin": float(margin),
            "learning_rate": 0.01,
        },
        "data_seed": seed,
        "model_seeds": list(resolved_model_seeds),
        "wall_time_seconds": time.perf_counter() - started_at,
        "relational_families": list(RELATIONAL_FAMILIES),
        "oracle": {
            "kind": "exact_assignment_completion_enumeration",
            "assignment_shape": "3 robots x 4 transport tasks",
            "scope": (
                "first-action completion proxy only; not a scheduling benchmark"
            ),
        },
        "controls": {
            "same_hard_mask_within_twins": True,
            "changed_entity_excluded_from_both_twin_oracle_actions": True,
            "signed_opportunity_prior_enabled": False,
            "shared_flattened_masked_argmax_decoder": True,
            "unique_pair_template_count": len(
                {_template_signature(twin) for twin in twins}
            ),
            "train_evaluation_template_overlap": len(template_overlap),
            "production_dataset_written": False,
            "confidence_interval_unit": "relational_pair",
            "utility_aware_tolerance": MARGIN_UTILITY_TOLERANCE,
            "margin_strata": [
                {"name": name, "lower": lower, "upper": None if math.isinf(upper) else upper}
                for name, lower, upper in MARGIN_STRATA
            ],
            "parameter_budget_equal": parameter_budget_equal,
            "trainable_parameter_counts": parameter_counts,
        },
        "pre_registered_go_criterion": {
            "minimum_train_agreement_for_architecture_comparison": 0.8,
            "minimum_agreement_margin_over_matched_mlp": 0.05,
            "minimum_flip_margin_over_matched_mlp": 0.05,
            "minimum_per_family_flip_accuracy": 0.6,
            "agreement_difference_ci95_lower_bound_strictly_above": 0.0,
            "maximum_mean_residual_saturation_rate": 0.25,
        },
        "pre_registered_comparison_validity_check": {
            "minimum_train_agreement": 0.8,
            "applies_to": [
                "matched_parameter_mlp",
                "pair_aware_attention",
            ],
            "per_seed_train_agreements": train_fit_values,
            "prerequisite_met": comparison_valid,
            "status": (
                "valid" if comparison_valid else "insufficient_train_fit"
            ),
        },
        "decision": {
            "pair_aware_relational_claim": claim,
            "probe_learnability_prerequisite_met": comparison_valid,
            "pair_aware_incremental_claim_supported": pair_aware_supported,
            "pair_aware_minus_matched_agreement": (
                pair_aware_agreement - matched_agreement
            ),
            "pair_aware_minus_matched_agreement_ci95": (
                agreement_difference_ci95
            ),
            "ticket_17_formal_data_authorized": pair_aware_supported,
            "ticket_20_authorized": False,
        },
        "diagnosis": {
            "minimum_pair_aware_or_matched_train_agreement": min(
                train_fit_values
            ),
            "interpretation": (
                "Margin-balanced subset is diagnostic only; formal Ticket 29 gate is unchanged."
                if min_margin is not None
                else (
                    "Train fit is insufficient for architecture comparison."
                    if not comparison_valid
                    else (
                        "Pair-aware attention satisfies every pre-registered gate."
                        if pair_aware_supported
                        else "The comparison is valid but at least one pair-aware "
                        "go criterion is not met."
                    )
                )
            ),
            "architecture_observation": (
                "Pair-aware axial attention consumes competitor pair metadata "
                "along task and robot axes while excluding focal ETA from its "
                "own learned residual."
            ),
        },
        "methods": metrics,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "relational_gate_summary.json"
    report_path = destination / "relational_gate_report.md"
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return RelationalGateResult(summary_path, report_path)


@dataclass(frozen=True, slots=True)
class _StateSpec:
    robot_positions: tuple[Point, ...]
    robot_loaded_speeds: tuple[float, ...]
    task_pickups: tuple[Point, ...]
    task_deliveries: tuple[Point, ...]
    downstream_priorities: tuple[float, ...]


def _random_state_spec(generator: random.Random) -> _StateSpec:
    return _StateSpec(
        robot_positions=tuple(_point(generator) for _ in range(ROBOT_COUNT)),
        robot_loaded_speeds=tuple(
            generator.uniform(0.65, 1.6) for _ in range(ROBOT_COUNT)
        ),
        task_pickups=tuple(
            _point(generator) for _ in range(CANDIDATE_TASK_COUNT)
        ),
        task_deliveries=tuple(
            _point(generator) for _ in range(CANDIDATE_TASK_COUNT)
        ),
        downstream_priorities=tuple(
            generator.uniform(0.25, 1.0)
            for _ in range(CANDIDATE_TASK_COUNT)
        ),
    )


def _point(generator: random.Random) -> Point:
    return generator.uniform(0.0, 2.0), generator.uniform(0.0, 2.0)


def _relational_intervention(
    family: str,
    spec: _StateSpec,
    generator: random.Random,
) -> tuple[int, _StateSpec]:
    if family == "competitor_robot_position":
        index = generator.randrange(ROBOT_COUNT)
        positions = list(spec.robot_positions)
        positions[index] = _point(generator)
        return index, replace(spec, robot_positions=tuple(positions))
    if family == "competitor_robot_speed":
        index = generator.randrange(ROBOT_COUNT)
        speeds = list(spec.robot_loaded_speeds)
        speeds[index] = generator.uniform(0.45, 2.0)
        return index, replace(spec, robot_loaded_speeds=tuple(speeds))
    if family == "alternative_task_pickup":
        index = generator.randrange(CANDIDATE_TASK_COUNT)
        pickups = list(spec.task_pickups)
        pickups[index] = _point(generator)
        return index, replace(spec, task_pickups=tuple(pickups))
    if family == "alternative_downstream_priority":
        index = generator.randrange(CANDIDATE_TASK_COUNT)
        priorities = list(spec.downstream_priorities)
        priorities[index] = generator.uniform(0.05, 1.25)
        return index, replace(spec, downstream_priorities=tuple(priorities))
    raise ValueError(f"unknown relational family: {family}")


def _state_from_spec(
    *,
    state_id: str,
    pair_id: str,
    family: str,
    spec: _StateSpec,
) -> RelationalState:
    eta = tuple(
        tuple(
            _transport_eta(
                spec.robot_positions[robot],
                spec.robot_loaded_speeds[robot],
                spec.task_pickups[task],
                spec.task_deliveries[task],
            )
            for task in range(CANDIDATE_TASK_COUNT)
        )
        for robot in range(ROBOT_COUNT)
    )
    action_values = _exact_action_values(
        eta,
        spec.downstream_priorities,
    )
    oracle_action = max(
        range(len(action_values)),
        key=lambda action: (action_values[action], -action),
    )
    hard_mask = tuple(
        (True,) * CANDIDATE_TASK_COUNT
        + (False,) * CANDIDATE_TASK_COUNT
        for _ in range(ROBOT_COUNT)
    )
    return RelationalState(
        state_id=state_id,
        pair_id=pair_id,
        family=family,
        robot_positions=spec.robot_positions,
        robot_loaded_speeds=spec.robot_loaded_speeds,
        task_pickups=spec.task_pickups,
        task_deliveries=spec.task_deliveries,
        downstream_priorities=spec.downstream_priorities,
        eta=eta,
        action_values=action_values,
        oracle_action=oracle_action,
        hard_mask=hard_mask,
    )


def _transport_eta(
    robot_position: Point,
    loaded_speed: float,
    pickup: Point,
    delivery: Point,
) -> float:
    unloaded_leg = math.dist(robot_position, pickup) / 2.0
    loaded_leg = math.dist(pickup, delivery) / loaded_speed
    return unloaded_leg + loaded_leg


def _exact_action_values(
    eta: Sequence[Sequence[float]],
    priorities: Sequence[float],
) -> tuple[float, ...]:
    edge_utility = tuple(
        tuple(
            1.25 * priorities[task] - eta[robot][task]
            for task in range(CANDIDATE_TASK_COUNT)
        )
        for robot in range(ROBOT_COUNT)
    )
    assignments = tuple(
        (
            tasks,
            sum(edge_utility[robot][task] for robot, task in enumerate(tasks)),
        )
        for tasks in itertools.permutations(
            range(CANDIDATE_TASK_COUNT), ROBOT_COUNT
        )
    )
    values = []
    for robot in range(ROBOT_COUNT):
        for task in range(CANDIDATE_TASK_COUNT):
            completion = max(
                total
                for assigned_tasks, total in assignments
                if assigned_tasks[robot] == task
            )
            values.append(completion + 0.05 * edge_utility[robot][task])
    return tuple(values)


def _change_is_nonlocal(
    family: str,
    changed_index: int,
    before_action: int,
    after_action: int,
) -> bool:
    if family.startswith("competitor_robot"):
        return all(
            action // CANDIDATE_TASK_COUNT != changed_index
            for action in (before_action, after_action)
        )
    return all(
        action % CANDIDATE_TASK_COUNT != changed_index
        for action in (before_action, after_action)
    )


def _template_signature(twin: RelationalTwin) -> str:
    return json.dumps(
        {
            "changed": twin.changed_entity_index,
            "before": {
                "robot_positions": twin.before.robot_positions,
                "robot_speeds": twin.before.robot_loaded_speeds,
                "task_pickups": twin.before.task_pickups,
                "deliveries": twin.before.task_deliveries,
                "priorities": twin.before.downstream_priorities,
            },
            "after": {
                "robot_positions": twin.after.robot_positions,
                "robot_speeds": twin.after.robot_loaded_speeds,
                "task_pickups": twin.after.task_pickups,
                "deliveries": twin.after.task_deliveries,
                "priorities": twin.after.downstream_priorities,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _split_twins(
    twins: Sequence[RelationalTwin],
    pairs_per_family: int,
) -> tuple[tuple[RelationalTwin, ...], tuple[RelationalTwin, ...]]:
    train_count = max(1, int(0.8 * pairs_per_family))
    if train_count >= pairs_per_family:
        train_count = pairs_per_family - 1 if pairs_per_family > 1 else 1
    counts = {family: 0 for family in RELATIONAL_FAMILIES}
    train: list[RelationalTwin] = []
    evaluation: list[RelationalTwin] = []
    for twin in twins:
        index = counts[twin.family]
        counts[twin.family] += 1
        (train if index < train_count else evaluation).append(twin)
    if not evaluation:
        evaluation = list(train)
    return tuple(train), tuple(evaluation)


def _build_model(
    method: str,
    *,
    seed: int,
) -> MDEnhancedSchedulerNetwork:
    torch.manual_seed(seed)
    config = MDEnhancedPolicyConfig(
        hidden_dim=16,
        residual_bound=0.75,
        legacy_transport_bound=0.15,
        physics_scale=0.5,
        eta_scale=1.0,
        use_cross_attention=method == "cross_attention_full",
        use_pair_aware_attention=method == "pair_aware_attention",
        use_signed_opportunity_prior=False,
    )
    model_class = (
        _MatchedParameterMLPNetwork
        if method == "matched_parameter_mlp"
        else MDEnhancedSchedulerNetwork
    )
    return model_class(
        robot_input_dimensions=7,
        task_input_dimension=9,
        embed_dim=16,
        ff_dim=32,
        n_transformer_heads=4,
        n_transformer_layers=1,
        n_gatn_heads=4,
        n_gatn_layers=1,
        dropout=0.0,
        use_idle=False,
        md_config=config,
    ).cpu()


def _batch(
    states: Sequence[RelationalState],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MDPolicyInputs]:
    batch_size = len(states)
    robot_features = torch.zeros(batch_size, ROBOT_COUNT, 7)
    task_features = torch.zeros(batch_size, TASK_COUNT, 9)
    robot_metadata = torch.zeros(batch_size, ROBOT_COUNT, 4)
    task_metadata = torch.zeros(batch_size, TASK_COUNT, 8)
    pair_metadata = torch.zeros(batch_size, ROBOT_COUNT, TASK_COUNT, 5)
    opportunity_context = torch.zeros(
        batch_size,
        ROBOT_COUNT,
        TASK_COUNT,
        MD_OPPORTUNITY_FEATURE_COUNT,
    )
    typed_adjacency = torch.zeros(batch_size, 2, TASK_COUNT, TASK_COUNT)
    for batch_index, state in enumerate(states):
        for robot in range(ROBOT_COUNT):
            robot_features[batch_index, robot, :2] = torch.tensor(
                state.robot_positions[robot]
            )
            robot_metadata[batch_index, robot] = torch.tensor(
                (1.0, 0.0, 2.0, state.robot_loaded_speeds[robot])
            )
        for task in range(CANDIDATE_TASK_COUNT):
            downstream = task + CANDIDATE_TASK_COUNT
            priority = state.downstream_priorities[task]
            task_features[batch_index, task, :2] = torch.tensor(
                state.task_pickups[task]
            )
            task_features[batch_index, downstream, :2] = torch.tensor(
                state.task_deliveries[task]
            )
            task_metadata[batch_index, task] = torch.tensor(
                (1.0, 1.0, 0.0, 0.0, 1.0, 0.0, priority, 1.0)
            )
            task_metadata[batch_index, downstream] = torch.tensor(
                (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, priority, 0.0)
            )
            typed_adjacency[batch_index, 1, task, downstream] = 1.0
            loaded_leg = math.dist(
                state.task_pickups[task],
                state.task_deliveries[task],
            )
            for robot in range(ROBOT_COUNT):
                pair_metadata[batch_index, robot, task] = torch.tensor(
                    (
                        state.eta[robot][task],
                        1.0,
                        1.0,
                        1.0,
                        loaded_leg,
                    )
                )
    task_adjacency = typed_adjacency[:, 1]
    md_inputs = MDPolicyInputs(
        robot_metadata=robot_metadata,
        task_metadata=task_metadata,
        pair_metadata=pair_metadata,
        task_is_transport=torch.tensor(
            [
                [True] * CANDIDATE_TASK_COUNT
                + [False] * CANDIDATE_TASK_COUNT
                for _state in states
            ]
        ),
        typed_adjacency=typed_adjacency,
        downstream_task_index=torch.tensor(
            [
                list(range(CANDIDATE_TASK_COUNT, TASK_COUNT))
                + [-1] * CANDIDATE_TASK_COUNT
                for _state in states
            ],
            dtype=torch.long,
        ),
        hard_feasibility_mask=torch.tensor(
            [state.hard_mask for state in states],
            dtype=torch.bool,
        ),
        opportunity_context=opportunity_context,
    )
    return robot_features, task_features, task_adjacency, md_inputs


def _flatten_transport_scores(scores: torch.Tensor) -> torch.Tensor:
    return scores[:, :, :CANDIDATE_TASK_COUNT].reshape(scores.shape[0], -1)


def _train_model(
    model: MDEnhancedSchedulerNetwork,
    states: Sequence[RelationalState],
    *,
    epochs: int,
    training_objective: str,
    margin: float,
) -> tuple[float, ...]:
    robot_features, task_features, task_adjacency, md_inputs = _batch(states)
    targets = torch.tensor(
        [state.oracle_action for state in states],
        dtype=torch.long,
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    losses = []
    model.train()
    for _epoch in range(epochs):
        diagnostics = model.forward_with_diagnostics(
            robot_features,
            task_features,
            task_adjacency,
            md_inputs=md_inputs,
        )
        logits = _flatten_transport_scores(diagnostics.scores)
        if training_objective == "bounded_margin":
            loss = F.multi_margin_loss(logits, targets, margin=margin)
        elif training_objective == "cross_entropy":
            loss = F.cross_entropy(logits, targets)
        else:
            raise ValueError("unsupported relational training objective")
        optimizer.zero_grad()
        loss.backward()
        if not all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            raise RuntimeError("relational probe produced non-finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return tuple(losses)


def _model_agreement(
    model: MDEnhancedSchedulerNetwork,
    states: Sequence[RelationalState],
) -> float:
    logits, _diagnostics = _model_logits(model, states)
    predictions = torch.argmax(logits, dim=-1).tolist()
    return sum(
        int(prediction == state.oracle_action)
        for prediction, state in zip(predictions, states, strict=True)
    ) / len(states)


def _model_logits(
    model: MDEnhancedSchedulerNetwork,
    states: Sequence[RelationalState],
) -> tuple[torch.Tensor, MDPolicyDiagnostics]:
    robot_features, task_features, task_adjacency, md_inputs = _batch(states)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        diagnostics = model.forward_with_diagnostics(
            robot_features.to(device),
            task_features.to(device),
            task_adjacency.to(device),
            md_inputs=md_inputs.to(device),
        )
    cpu_diagnostics = diagnostics.to("cpu")
    return _flatten_transport_scores(cpu_diagnostics.scores), cpu_diagnostics


def _evaluate_method(
    method: str,
    model: MDEnhancedSchedulerNetwork | None,
    twins: Sequence[RelationalTwin],
    states: Sequence[RelationalState],
    *,
    learning_curve: tuple[float, ...],
    initial_agreement: float | None,
    train_agreement: float | None,
    seed: int,
) -> dict[str, object]:
    diagnostics = None
    if model is not None:
        logits, diagnostics = _model_logits(model, states)
        residual_magnitude = float(
            diagnostics.bounded_residual[
                :, :, :CANDIDATE_TASK_COUNT
            ].abs().mean()
        )
        saturation = float(diagnostics.residual_saturation_rate.mean())
        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        context_gate_magnitudes = {
            "robot_to_task": abs(
                float(torch.tanh(model.robot_to_task_gate).detach())
            ),
            "task_to_robot": abs(
                float(torch.tanh(model.task_to_robot_gate).detach())
            ),
        }
    else:
        eta = torch.tensor([state.eta for state in states])
        priorities = torch.tensor(
            [state.downstream_priorities for state in states]
        ).unsqueeze(1)
        if method == "physics_only":
            logits = -eta / (1.0 + eta)
        elif method == "local_eta_priority":
            logits = 1.25 * priorities - eta
        else:
            raise ValueError(f"unknown relational method: {method}")
        logits = logits.reshape(len(states), -1)
        residual_magnitude = 0.0
        saturation = 0.0
        parameters = 0
        context_gate_magnitudes = {
            "robot_to_task": 0.0,
            "task_to_robot": 0.0,
        }

    predictions = torch.argmax(logits, dim=-1).tolist()
    agreements = [
        int(prediction == state.oracle_action)
        for prediction, state in zip(predictions, states, strict=True)
    ]
    regrets = [
        max(state.action_values) - state.action_values[prediction]
        for prediction, state in zip(predictions, states, strict=True)
    ]
    state_index = {state.state_id: index for index, state in enumerate(states)}
    flip_successes: list[int] = []
    pair_agreements: list[float] = []
    pair_regrets: list[float] = []
    family_successes: dict[str, list[int]] = {
        family: [] for family in RELATIONAL_FAMILIES
    }
    for twin in twins:
        before_index = state_index[twin.before.state_id]
        after_index = state_index[twin.after.state_id]
        success = int(
            (
                predictions[before_index],
                predictions[after_index],
            )
            == (
                twin.before.oracle_action,
                twin.after.oracle_action,
            )
        )
        flip_successes.append(success)
        family_successes[twin.family].append(success)
        pair_agreements.append(
            (agreements[before_index] + agreements[after_index]) / 2
        )
        pair_regrets.append((regrets[before_index] + regrets[after_index]) / 2)

    final_agreement = sum(agreements) / len(agreements)
    margin_strata = _margin_stratified_metrics(states, predictions, regrets)
    return {
        "initial_evaluation_agreement": (
            final_agreement if initial_agreement is None else initial_agreement
        ),
        "train_first_action_agreement": train_agreement,
        "first_action_agreement": final_agreement,
        "agreement_ci95": list(
            _bootstrap_mean_ci(pair_agreements, seed=seed + 11)
        ),
        "relational_flip_accuracy": sum(flip_successes) / len(flip_successes),
        "flip_ci95": list(
            _wilson_interval(sum(flip_successes), len(flip_successes))
        ),
        "family_flip_accuracy": {
            family: sum(values) / len(values)
            for family, values in family_successes.items()
        },
        "mean_completion_proxy_regret": sum(regrets) / len(regrets),
        "utility_aware_accuracy_at_0.05": sum(
            regret <= MARGIN_UTILITY_TOLERANCE for regret in regrets
        )
        / len(regrets),
        "margin_strata": margin_strata,
        "regret_ci95": list(
            _bootstrap_mean_ci(pair_regrets, seed=seed + 12)
        ),
        "mean_residual_magnitude": residual_magnitude,
        "residual_saturation_rate": saturation,
        "parameter_count": parameters,
        "context_gate_magnitudes": context_gate_magnitudes,
        "initial_train_loss": learning_curve[0] if learning_curve else 0.0,
        "final_train_loss": learning_curve[-1] if learning_curve else 0.0,
        "pair_agreement_values": pair_agreements,
        "pair_flip_values": flip_successes,
        "pair_regret_values": pair_regrets,
    }


def _state_completion_margin(state: RelationalState) -> float:
    values = sorted(state.action_values, reverse=True)
    return values[0] - values[1]


def _validated_model_seeds(model_seeds: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(model_seeds)
    if len(resolved) != 3:
        raise ValueError("model_seeds must contain exactly three seeds")
    if len(set(resolved)) != len(resolved):
        raise ValueError("model_seeds must be unique")
    if any(
        isinstance(model_seed, bool)
        or not isinstance(model_seed, int)
        or model_seed < 0
        for model_seed in resolved
    ):
        raise ValueError("model_seeds must contain non-negative integers")
    return resolved


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty metric sequence")
    return sum(values) / len(values)


def _population_std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(_mean([(value - mean) ** 2 for value in values]))


def _mean_pair_values(
    per_seed: Sequence[dict[str, object]],
    key: str,
) -> list[float]:
    rows = [cast(list[float], metrics[key]) for metrics in per_seed]
    pair_count = len(rows[0])
    if any(len(row) != pair_count for row in rows):
        raise RuntimeError(f"inconsistent pair metric length for {key}")
    return [
        _mean([row[index] for row in rows])
        for index in range(pair_count)
    ]


def _aggregate_margin_strata(
    per_seed: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    def mean_optional(rows: Sequence[dict[str, object]], key: str) -> float | None:
        values = [row[key] for row in rows]
        if any(value is None for value in values):
            return None
        return _mean([float(cast(float, value)) for value in values])

    for name, _lower, _upper in MARGIN_STRATA:
        rows = [
            cast(dict[str, object], cast(dict[str, dict[str, object]], metrics["margin_strata"])[name])
            for metrics in per_seed
        ]
        result[name] = {
            "state_count": cast(int, rows[0]["state_count"]),
            "mean_margin": mean_optional(rows, "mean_margin"),
            "exact_agreement": mean_optional(rows, "exact_agreement"),
            "mean_completion_proxy_regret": mean_optional(rows, "mean_completion_proxy_regret"),
            "utility_aware_accuracy_at_0.05": mean_optional(rows, "utility_aware_accuracy_at_0.05"),
        }
    return result


def _aggregate_seed_metrics(
    per_seed: Sequence[dict[str, object]],
    *,
    bootstrap_seed: int,
) -> dict[str, object]:
    if not per_seed:
        raise ValueError("per_seed metrics must not be empty")
    parameter_counts = {
        cast(int, metrics["parameter_count"]) for metrics in per_seed
    }
    if len(parameter_counts) != 1:
        raise RuntimeError("parameter count changed across model seeds")

    pair_agreements = _mean_pair_values(per_seed, "pair_agreement_values")
    pair_flips = _mean_pair_values(per_seed, "pair_flip_values")
    pair_regrets = _mean_pair_values(per_seed, "pair_regret_values")
    family_flip = {
        family: _mean(
            [
                cast(dict[str, float], metrics["family_flip_accuracy"])[family]
                for metrics in per_seed
            ]
        )
        for family in RELATIONAL_FAMILIES
    }
    scalar_keys = (
        "initial_evaluation_agreement",
        "train_first_action_agreement",
        "first_action_agreement",
        "relational_flip_accuracy",
        "mean_completion_proxy_regret",
        "utility_aware_accuracy_at_0.05",
        "mean_residual_magnitude",
        "residual_saturation_rate",
        "initial_train_loss",
        "final_train_loss",
    )
    scalar_values = {
        key: [cast(float, metrics[key]) for metrics in per_seed]
        for key in scalar_keys
    }
    margin_strata = _aggregate_margin_strata(per_seed)
    gates = {
        name: _mean(
            [
                cast(dict[str, float], metrics["context_gate_magnitudes"])[name]
                for metrics in per_seed
            ]
        )
        for name in ("robot_to_task", "task_to_robot")
    }
    aggregated: dict[str, object] = {
        key: _mean(values) for key, values in scalar_values.items()
    }
    aggregated.update(
        {
            "agreement_ci95": list(
                _bootstrap_mean_ci(pair_agreements, seed=bootstrap_seed + 11)
            ),
            "flip_ci95": list(
                _bootstrap_mean_ci(pair_flips, seed=bootstrap_seed + 12)
            ),
            "family_flip_accuracy": family_flip,
            "regret_ci95": list(
                _bootstrap_mean_ci(pair_regrets, seed=bootstrap_seed + 13)
            ),
            "margin_strata": margin_strata,
            "parameter_count": parameter_counts.pop(),
            "context_gate_magnitudes": gates,
            "seed_std": {
                key: _population_std(values)
                for key, values in scalar_values.items()
            },
            "pair_agreement_values": pair_agreements,
            "pair_flip_values": pair_flips,
            "pair_regret_values": pair_regrets,
            "per_seed": list(per_seed),
        }
    )
    return aggregated


def _render_report(summary: dict[str, object]) -> str:
    decision = cast(dict[str, object], summary["decision"])
    methods = cast(dict[str, dict[str, object]], summary["methods"])
    report_phase = cast(str, summary["report_phase"])
    difference_ci = cast(
        list[float], decision["pair_aware_minus_matched_agreement_ci95"]
    )
    lines = [
        "# Ticket 29 Pair-Aware Relational Gate",
        "",
        (
            "This controlled 3-transport-robot probe is not Ticket 20 and does "
            "not write a production expert dataset."
        ),
        "",
        f"**Report phase:** `{report_phase}`.",
        (
            "This is the initial cross-entropy gate."
            if report_phase == "initial_cross_entropy_gate"
            else (
                "This is a margin-balanced diagnostic subset; it does not alter the formal gate."
                if report_phase == "margin_balanced_diagnostic"
                else "This is the bounded-margin follow-up held-out gate."
            )
        ),
        "",
        "The train-only optimization diagnosis is a separate run; this report contains held-out gate results.",
        (
            "Pair-aware relational claim: "
            f"**{decision['pair_aware_relational_claim']}**"
        ),
        "",
        (
            "| Method | Train | Agreement [95% CI] | Flip [95% CI] | "
            "Regret | Utility@0.05 | Saturation | Context gates |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_NAMES:
        metrics = methods[method]
        agreement_ci = cast(list[float], metrics["agreement_ci95"])
        flip_ci = cast(list[float], metrics["flip_ci95"])
        gates = cast(dict[str, float], metrics["context_gate_magnitudes"])
        lines.append(
            f"| {method}"
            + (
                " | n/a"
                if metrics["train_first_action_agreement"] is None
                else f" | {metrics['train_first_action_agreement']:.3f}"
            )
            + f" | {metrics['first_action_agreement']:.3f} "
            + f"[{agreement_ci[0]:.3f}, {agreement_ci[1]:.3f}]"
            + f" | {metrics['relational_flip_accuracy']:.3f} "
            + f"[{flip_ci[0]:.3f}, {flip_ci[1]:.3f}]"
            + f" | {metrics['mean_completion_proxy_regret']:.4f}"
            + f" | {metrics['utility_aware_accuracy_at_0.05']:.3f}"
            + f" | {metrics['residual_saturation_rate']:.3f}"
            + f" | {gates['robot_to_task']:.3f}/"
            + f"{gates['task_to_robot']:.3f} |"
        )
    lines.extend([
        "",
        "Margin-stratified diagnostics (evaluation states):",
        "| Method | Stratum | States | Exact | Regret | Utility@0.05 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for method in TRAINABLE_METHODS:
        strata = cast(dict[str, dict[str, object]], methods[method]["margin_strata"])
        for name, _lower, _upper in MARGIN_STRATA:
            row = strata[name]
            values = [row["exact_agreement"], row["mean_completion_proxy_regret"], row["utility_aware_accuracy_at_0.05"]]
            formatted = ["n/a" if value is None else f"{cast(float, value):.3f}" for value in values]
            lines.append(f"| {method} | {name} | {row['state_count']} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")

    lines.extend(
        [
            "",
            (
                "Every intervention changes a competitor excluded from both "
                "oracle first actions; labels come from exact 3x4 assignment "
                "completion values."
            ),
            (
                "The 0.80 train-agreement prerequisite was pre-registered for "
                "every pair-aware and matched-MLP model seed."
            ),
            (
                "Pair-aware minus matched-MLP agreement: "
                f"{decision['pair_aware_minus_matched_agreement']:.3f} "
                f"[95% CI {difference_ci[0]:.3f}, {difference_ci[1]:.3f}]."
            ),
            (
                "Pair-aware axial attention consumes competitor pair metadata "
                "while excluding focal ETA from its own learned residual."
            ),
            (
                "The claim is limited to held-out controlled relational "
                "templates and does not establish end-to-end schedule quality."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _margin_stratified_metrics(
    states: Sequence[RelationalState],
    predictions: Sequence[int],
    regrets: Sequence[float],
) -> dict[str, dict[str, object]]:
    """Report diagnostic metrics for clear and near-tie oracle decisions."""
    rows = [
        (_state_completion_margin(state), int(prediction == state.oracle_action), regret, int(regret <= MARGIN_UTILITY_TOLERANCE))
        for state, prediction, regret in zip(states, predictions, regrets, strict=True)
    ]
    result: dict[str, dict[str, object]] = {}
    for name, lower, upper in MARGIN_STRATA:
        selected = [row for row in rows if lower <= row[0] < upper]
        result[name] = {
            "state_count": len(selected),
            "mean_margin": _mean([row[0] for row in selected]) if selected else None,
            "exact_agreement": _mean([float(row[1]) for row in selected]) if selected else None,
            "mean_completion_proxy_regret": _mean([row[2] for row in selected]) if selected else None,
            "utility_aware_accuracy_at_0.05": _mean([float(row[3]) for row in selected]) if selected else None,
        }
    return result

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Ticket 29 pair-aware relational gate"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-family", type=int, default=25)
    parser.add_argument(
        "--selection-pool-per-family",
        type=int,
        default=None,
        help="Generate a larger pool before deterministic margin-balanced selection.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=None,
        help="Enable diagnostic margin-balanced selection at this state margin.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--training-objective",
        choices=("bounded_margin", "cross_entropy"),
        default="bounded_margin",
    )
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs=3,
        default=DEFAULT_MODEL_SEEDS,
        metavar=("SEED_A", "SEED_B", "SEED_C"),
    )
    args = parser.parse_args()
    result = run_md_policy_relational_gate(
        args.output_dir,
        pairs_per_family=args.pairs_per_family,
        selection_pool_per_family=args.selection_pool_per_family,
        min_margin=args.min_margin,
        epochs=args.epochs,
        seed=args.seed,
        model_seeds=args.model_seeds,
        training_objective=args.training_objective,
        margin=args.margin,
    )
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
