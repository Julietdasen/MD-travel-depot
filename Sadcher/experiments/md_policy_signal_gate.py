"""Deterministic Ticket 19 semantic-twin gate for MD policy signal."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence, cast

import pulp  # type: ignore[import-untyped]
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.md_enhanced_policy import (
    MDEnhancedPolicyConfig,
    MDEnhancedSchedulerNetwork,
    MDPolicyDiagnostics,
)
from models.md_policy import (
    MD_OPPORTUNITY_FEATURE_COUNT,
    MDOpportunityContext,
    MDOpportunityFeature,
    MDOpportunityValues,
    MDPolicyInputs,
)


SEMANTIC_FAMILIES = (
    "eta_vs_criticality",
    "eta_vs_coalition_availability",
    "last_blocker",
    "capacity_scarcity",
)
METHOD_NAMES = (
    "physics_only",
    "residual_only",
    "no_downstream_no_coalition",
    "matched_parameter_mlp",
    "cross_attention_full",
    "eta_unlock_heuristic",
)


@dataclass(frozen=True, slots=True)
class SemanticState:
    state_id: str
    pair_id: str
    intervention_family: str
    eta: tuple[float, float]
    opportunity: tuple[MDOpportunityValues, MDOpportunityValues]
    oracle_utility: tuple[float, float]
    oracle_action: int
    hard_mask: tuple[tuple[bool, ...], ...] = (
        (True, True, False, False),
    )


@dataclass(frozen=True, slots=True)
class SemanticTwin:
    family: str
    pair_id: str
    before: SemanticState
    after: SemanticState


@dataclass(frozen=True, slots=True)
class MDPolicySignalGateResult:
    summary_path: Path
    report_path: Path


class _MatchedParameterMLPNetwork(MDEnhancedSchedulerNetwork):
    """Exchange pooled context with the same trainable budget as two MHA blocks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for attention in (
            self.robot_to_task_attention,
            self.task_to_robot_attention,
        ):
            for parameter in attention.parameters():
                parameter.requires_grad_(False)
        hidden = self.enhanced_config.hidden_dim
        self.robot_context_mlp = self._context_mlp(hidden)
        self.task_context_mlp = self._context_mlp(hidden)
        # Two hidden vectors exactly close the MHA/MLP bias-count difference.
        self.matched_context_bias = nn.Parameter(torch.zeros(2, hidden))

    @staticmethod
    def _context_mlp(hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def _exchange_context(
        self, robot_context: torch.Tensor, task_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task_summary = task_context.mean(dim=1, keepdim=True).expand(
            -1, robot_context.shape[1], -1
        )
        robot_summary = robot_context.mean(dim=1, keepdim=True).expand(
            -1, task_context.shape[1], -1
        )
        robot_update = self.robot_context_mlp(
            torch.cat((robot_context, task_summary), dim=-1)
        )
        task_update = self.task_context_mlp(
            torch.cat((task_context, robot_summary), dim=-1)
        )
        robot_context = (
            robot_context
            + torch.tanh(self.robot_to_task_gate)
            * (robot_update + self.matched_context_bias[0])
        )
        task_context = (
            task_context
            + torch.tanh(self.task_to_robot_gate)
            * (task_update + self.matched_context_bias[1])
        )
        return robot_context, task_context


def build_semantic_twins(
    *, pairs_per_family: int = 50, seed: int = 2026
) -> tuple[SemanticTwin, ...]:
    """Build targeted counterfactuals without generating scheduling instances."""

    if (
        isinstance(pairs_per_family, bool)
        or not isinstance(pairs_per_family, int)
        or pairs_per_family <= 0
    ):
        raise ValueError("pairs_per_family must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    generator = random.Random(seed)
    twins = []
    for family in SEMANTIC_FAMILIES:
        for index in range(pairs_per_family):
            pair_id = f"{family}-{index:04d}"
            for _attempt in range(100):
                eta = (
                    generator.uniform(0.15, 0.45),
                    generator.uniform(0.15, 0.45),
                )
                before_context, after_context = _intervention(family, generator)
                before = _semantic_state(
                    state_id=f"{pair_id}-before",
                    pair_id=pair_id,
                    family=family,
                    eta=eta,
                    opportunity=(
                        before_context[0].values,
                        before_context[1].values,
                    ),
                )
                after = _semantic_state(
                    state_id=f"{pair_id}-after",
                    pair_id=pair_id,
                    family=family,
                    eta=eta,
                    opportunity=(
                        after_context[0].values,
                        after_context[1].values,
                    ),
                )
                if before.oracle_action != after.oracle_action:
                    break
            else:
                raise RuntimeError(f"could not build valid semantic twin {pair_id}")
            twins.append(SemanticTwin(family, pair_id, before, after))
    signatures = {_template_signature(twin) for twin in twins}
    if len(signatures) != len(twins):
        raise RuntimeError("semantic twin generator produced duplicate templates")
    return tuple(twins)


def run_md_policy_signal_gate(
    output_dir: str | Path,
    *,
    pairs_per_family: int = 50,
    epochs: int = 30,
    seed: int = 2026,
) -> MDPolicySignalGateResult:
    """Train/evaluate small probes and emit the pre-registered go/no-go decision."""

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    twins = build_semantic_twins(pairs_per_family=pairs_per_family, seed=seed)
    all_states = tuple(
        state for twin in twins for state in (twin.before, twin.after)
    )
    mip_actions = _solve_exact_proxy_mip(all_states)
    expected_actions = tuple(state.oracle_action for state in all_states)
    if mip_actions != expected_actions:
        raise RuntimeError("closed-form semantic labels disagree with exact proxy MIP")

    train_twins, evaluation_twins = _split_twins(twins, pairs_per_family)
    train_templates = {_template_signature(twin) for twin in train_twins}
    evaluation_templates = {
        _template_signature(twin) for twin in evaluation_twins
    }
    template_overlap = train_templates & evaluation_templates
    if template_overlap:
        raise RuntimeError("train/evaluation semantic templates overlap")
    train_states = tuple(
        state for twin in train_twins for state in (twin.before, twin.after)
    )
    evaluation_states = tuple(
        state for twin in evaluation_twins for state in (twin.before, twin.after)
    )
    models: dict[str, MDEnhancedSchedulerNetwork] = {}
    learning_curves: dict[str, tuple[float, ...]] = {}
    initial_agreements: dict[str, float] = {}
    for offset, method in enumerate(
        (
            "residual_only",
            "no_downstream_no_coalition",
            "matched_parameter_mlp",
            "cross_attention_full",
        )
    ):
        model = _build_model(method, seed=seed + offset)
        initial_agreements[method] = _model_agreement(
            model, method, evaluation_states
        )
        curve = _train_probe(
            model,
            method,
            train_states,
            epochs=epochs,
            seed=seed + offset,
        )
        models[method] = model
        learning_curves[method] = curve

    method_metrics: dict[str, dict[str, object]] = {}
    for method in METHOD_NAMES:
        evaluation_model = models.get(method)
        method_metrics[method] = _evaluate_method(
            method,
            evaluation_model,
            evaluation_twins,
            evaluation_states,
            seed=seed,
            learning_curve=learning_curves.get(method, ()),
            initial_evaluation_agreement=initial_agreements.get(method),
        )

    full = method_metrics["cross_attention_full"]
    physics = method_metrics["physics_only"]
    heuristic = method_metrics["eta_unlock_heuristic"]
    matched = method_metrics["matched_parameter_mlp"]
    full_agreement = cast(float, full["first_action_agreement"])
    full_initial_agreement = cast(float, full["initial_evaluation_agreement"])
    full_flip = cast(float, full["semantic_action_flip_accuracy"])
    full_family_flip = cast(dict[str, float], full["family_flip_accuracy"])
    strongest_baseline_agreement = max(
        cast(float, physics["first_action_agreement"]),
        cast(float, heuristic["first_action_agreement"]),
    )
    strongest_baseline_flip = max(
        cast(float, physics["semantic_action_flip_accuracy"]),
        cast(float, heuristic["semantic_action_flip_accuracy"]),
    )
    signal_supported = bool(
        full_agreement >= full_initial_agreement + 0.05
        and full_agreement >= strongest_baseline_agreement + 0.05
        and full_flip >= strongest_baseline_flip + 0.05
        and min(full_family_flip.values()) >= 0.6
    )
    cross_attention_supported = bool(
        full_agreement
        >= cast(float, matched["first_action_agreement"]) + 0.02
    )
    decision = "go" if signal_supported else "no_go"
    kill_triggered = not signal_supported

    summary = {
        "schema_version": "1.0.0",
        "status": "semantic_signal_gate",
        "ticket": 19,
        "pairs_per_family": pairs_per_family,
        "semantic_families": list(SEMANTIC_FAMILIES),
        "state_count": len(all_states),
        "train_pair_count": len(train_twins),
        "evaluation_pair_count": len(evaluation_twins),
        "seed": seed,
        "epochs": epochs,
        "oracle": {
            "kind": "exact_proxy_mip",
            "solver": "PuLP CBC",
            "scope": (
                "one-step semantic utility only; this is not a full scheduling oracle"
            ),
            "status": "optimal",
        },
        "controls": {
            "hard_mask_digest": _mask_digest(evaluation_states),
            "decoder": "shared_masked_singleton_argmax",
            "future_schedule_features": False,
            "production_data_generated": False,
            "signed_opportunity_prior_enabled": False,
            "unique_pair_template_count": len(
                {_template_signature(twin) for twin in twins}
            ),
            "train_evaluation_template_overlap": len(template_overlap),
            "confidence_interval_unit": "semantic_pair",
        },
        "pre_registered_kill_criterion": {
            "minimum_agreement_gain_over_initial": 0.05,
            "minimum_agreement_margin_over_physics_or_eta_unlock": 0.05,
            "minimum_flip_margin_over_physics_or_eta_unlock": 0.05,
            "minimum_per_family_flip_accuracy": 0.6,
        },
        "decision": {
            "semantic_residual_signal": decision,
            "kill_triggered": kill_triggered,
            "cross_attention_incremental_claim_supported": cross_attention_supported,
            "paper_claim": (
                "downstream-conditioned residual has controlled semantic signal"
                if signal_supported
                else "downstream-conditioned residual failed the semantic signal gate"
            ),
        },
        "methods": method_metrics,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "signal_gate_summary.json"
    report_path = destination / "signal_gate_report.md"
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return MDPolicySignalGateResult(summary_path, report_path)


def _intervention(
    family: str,
    generator: random.Random,
) -> tuple[
    tuple[MDOpportunityContext, MDOpportunityContext],
    tuple[MDOpportunityContext, MDOpportunityContext],
]:
    left = _random_context(generator)
    right = _random_context(generator)
    high = generator.uniform(0.75, 1.0)
    low = generator.uniform(0.0, 0.2)
    if family == "eta_vs_criticality":
        before = (
            replace(left, critical_path_proxy=high),
            replace(right, critical_path_proxy=low),
        )
        after = (
            replace(left, critical_path_proxy=low),
            replace(right, critical_path_proxy=high),
        )
    elif family == "eta_vs_coalition_availability":
        before = (
            replace(left, coalition_availability=high, coalition_scarcity=low),
            replace(right, coalition_availability=low, coalition_scarcity=high),
        )
        after = (
            replace(left, coalition_availability=low, coalition_scarcity=high),
            replace(right, coalition_availability=high, coalition_scarcity=low),
        )
    elif family == "last_blocker":
        before = (
            replace(left, last_material_blocker=1.0),
            replace(right, last_material_blocker=0.0),
        )
        after = (
            replace(left, last_material_blocker=0.0),
            replace(right, last_material_blocker=1.0),
        )
    elif family == "capacity_scarcity":
        before = (
            replace(left, capacity_scarcity=high),
            replace(right, capacity_scarcity=low),
        )
        after = (
            replace(left, capacity_scarcity=low),
            replace(right, capacity_scarcity=high),
        )
    else:
        raise ValueError(f"unknown semantic family: {family}")
    return before, after


def _random_context(generator: random.Random) -> MDOpportunityContext:
    return MDOpportunityContext(
        critical_path_proxy=generator.uniform(0.0, 0.45),
        last_material_blocker=float(generator.random() < 0.25),
        coalition_availability=generator.uniform(0.15, 0.85),
        coalition_scarcity=generator.uniform(0.15, 0.85),
        capacity_scarcity=generator.uniform(0.05, 0.65),
    )


def _template_signature(twin: SemanticTwin) -> str:
    return json.dumps(
        {
            "eta": twin.before.eta,
            "before": twin.before.opportunity,
            "after": twin.after.opportunity,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_state(
    *,
    state_id: str,
    pair_id: str,
    family: str,
    eta: tuple[float, float],
    opportunity: tuple[MDOpportunityValues, MDOpportunityValues],
) -> SemanticState:
    utilities = (
        _proxy_utility(eta[0], opportunity[0]),
        _proxy_utility(eta[1], opportunity[1]),
    )
    action = 0 if utilities[0] >= utilities[1] else 1
    return SemanticState(
        state_id=state_id,
        pair_id=pair_id,
        intervention_family=family,
        eta=eta,
        opportunity=opportunity,
        oracle_utility=utilities,
        oracle_action=action,
    )


def _proxy_utility(
    eta: float, opportunity: MDOpportunityValues
) -> float:
    criticality, last_blocker, availability, scarcity, capacity_scarcity = (
        opportunity
    )
    return (
        -eta
        + 1.2 * criticality
        + 1.4 * last_blocker
        + 0.8 * availability
        - 0.8 * scarcity
        + 0.9 * capacity_scarcity
    )


def _solve_exact_proxy_mip(states: Sequence[SemanticState]) -> tuple[int, ...]:
    problem = pulp.LpProblem("ticket_19_semantic_first_actions", pulp.LpMaximize)
    choices: dict[int, tuple[Any, ...]] = {}
    objective: list[Any] = []
    for state_index, state in enumerate(states):
        row = tuple(
            problem.add_variable(
                f"state_{state_index}_action_{action}",
                lowBound=0,
                upBound=1,
                cat=pulp.LpBinary,
            )
            for action in range(2)
        )
        choices[state_index] = row
        problem += pulp.lpSum(row) == 1
        objective.extend(
            state.oracle_utility[action] * row[action] for action in range(2)
        )
    problem += pulp.lpSum(objective)
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False, threads=1))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            f"semantic exact proxy MIP was not optimal: {pulp.LpStatus[status]}"
        )
    return tuple(
        max(range(2), key=lambda action: float(pulp.value(choices[index][action])))
        for index in range(len(states))
    )


def _split_twins(
    twins: Sequence[SemanticTwin], pairs_per_family: int
) -> tuple[tuple[SemanticTwin, ...], tuple[SemanticTwin, ...]]:
    train_count = max(1, int(0.8 * pairs_per_family))
    if train_count >= pairs_per_family:
        train_count = pairs_per_family - 1 if pairs_per_family > 1 else 1
    train: list[SemanticTwin] = []
    evaluation: list[SemanticTwin] = []
    family_counts = {family: 0 for family in SEMANTIC_FAMILIES}
    for twin in twins:
        index = family_counts[twin.family]
        family_counts[twin.family] += 1
        (train if index < train_count else evaluation).append(twin)
    if not evaluation:
        evaluation = list(train)
    return tuple(train), tuple(evaluation)


def _build_model(method: str, *, seed: int) -> MDEnhancedSchedulerNetwork:
    torch.manual_seed(seed)
    config = MDEnhancedPolicyConfig(
        hidden_dim=16,
        residual_bound=0.5,
        legacy_transport_bound=0.25,
        use_cross_attention=method
        not in ("matched_parameter_mlp", "no_downstream_no_coalition"),
        use_downstream_encoding=method != "no_downstream_no_coalition",
        use_critical_path_proxy=method != "no_downstream_no_coalition",
        use_last_material_blocker=method != "no_downstream_no_coalition",
        use_coalition_context=method != "no_downstream_no_coalition",
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


def _train_probe(
    model: MDEnhancedSchedulerNetwork,
    method: str,
    states: Sequence[SemanticState],
    *,
    epochs: int,
    seed: int,
) -> tuple[float, ...]:
    torch.manual_seed(seed)
    robot_features, task_features, task_adjacency, md_inputs = _batch(states)
    targets = torch.tensor(
        [state.oracle_action for state in states], dtype=torch.long
    )
    learning_rate = 0.02 if method == "residual_only" else 0.05
    optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=learning_rate,
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
        logits = _diagnostic_logits(method, diagnostics)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        if not all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            raise RuntimeError(f"{method} produced non-finite semantic gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return tuple(losses)


def _batch(
    states: Sequence[SemanticState],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MDPolicyInputs]:
    batch_size = len(states)
    robot_features = torch.zeros(batch_size, 1, 7)
    task_features = torch.zeros(batch_size, 4, 9)
    robot_metadata = torch.tensor(
        [[[1.0, 0.0, 2.0, 1.0]] for _state in states],
        dtype=torch.float32,
    )
    task_metadata = torch.zeros(batch_size, 4, 8)
    pair_metadata = torch.zeros(batch_size, 1, 4, 5)
    opportunity_context = torch.zeros(
        batch_size, 1, 4, MD_OPPORTUNITY_FEATURE_COUNT
    )
    typed_adjacency = torch.zeros(batch_size, 2, 4, 4)
    for batch_index, state in enumerate(states):
        for candidate in range(2):
            task_metadata[batch_index, candidate, 0] = 1.0
            task_metadata[batch_index, candidate, 1] = 1.0
            task_metadata[batch_index, candidate, 7] = 1.0
            process_index = candidate + 2
            task_metadata[batch_index, process_index, 5] = 1.0
            task_metadata[batch_index, process_index, 6] = state.opportunity[
                candidate
            ][0]
            task_metadata[batch_index, process_index, 7] = 1.0
            pair_metadata[batch_index, 0, candidate] = torch.tensor(
                (state.eta[candidate], 1.0, 1.0, 1.0, state.eta[candidate])
            )
            opportunity_context[batch_index, 0, candidate] = torch.tensor(
                state.opportunity[candidate]
            )
            typed_adjacency[batch_index, 1, candidate, process_index] = 1.0
    task_adjacency = typed_adjacency[:, 1]
    md_inputs = MDPolicyInputs(
        robot_metadata=robot_metadata,
        task_metadata=task_metadata,
        pair_metadata=pair_metadata,
        task_is_transport=torch.tensor(
            [[True, True, False, False] for _state in states]
        ),
        typed_adjacency=typed_adjacency,
        downstream_task_index=torch.tensor(
            [[2, 3, -1, -1] for _state in states], dtype=torch.long
        ),
        hard_feasibility_mask=torch.tensor(
            [state.hard_mask for state in states], dtype=torch.bool
        ),
        opportunity_context=opportunity_context,
    )
    return robot_features, task_features, task_adjacency, md_inputs


def _diagnostic_logits(
    method: str, diagnostics: MDPolicyDiagnostics
) -> torch.Tensor:
    if method == "residual_only":
        return diagnostics.bounded_residual[:, 0, :2]
    return diagnostics.scores[:, 0, :2]


def _model_agreement(
    model: MDEnhancedSchedulerNetwork,
    method: str,
    states: Sequence[SemanticState],
) -> float:
    robot_features, task_features, task_adjacency, md_inputs = _batch(states)
    model.eval()
    with torch.no_grad():
        diagnostics = model.forward_with_diagnostics(
            robot_features,
            task_features,
            task_adjacency,
            md_inputs=md_inputs,
        )
    predictions = torch.argmax(
        _diagnostic_logits(method, diagnostics), dim=-1
    ).tolist()
    return sum(
        int(prediction == state.oracle_action)
        for prediction, state in zip(predictions, states, strict=True)
    ) / len(states)


def _evaluate_method(
    method: str,
    model: MDEnhancedSchedulerNetwork | None,
    twins: Sequence[SemanticTwin],
    states: Sequence[SemanticState],
    *,
    seed: int,
    learning_curve: tuple[float, ...],
    initial_evaluation_agreement: float | None,
) -> dict[str, object]:
    robot_features, task_features, task_adjacency, md_inputs = _batch(states)
    diagnostics = None
    if model is not None:
        with torch.no_grad():
            diagnostics = model.forward_with_diagnostics(
                robot_features,
                task_features,
                task_adjacency,
                md_inputs=md_inputs,
            )
        logits = _diagnostic_logits(method, diagnostics)
        residual_magnitude = float(
            diagnostics.bounded_residual[:, 0, :2].abs().mean()
        )
        component_scales = [
            float(value)
            for value in diagnostics.component_scales.mean(dim=0)
        ]
        saturation_rate = float(
            diagnostics.residual_saturation_rate.mean()
        )
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    elif method == "physics_only":
        eta = md_inputs.pair_metadata[:, 0, :2, 0]
        logits = -eta / (1.0 + eta)
        residual_magnitude = 0.0
        component_scales = [0.0, float(logits.abs().mean()), 0.0]
        saturation_rate = 0.0
        parameter_count = 0
    elif method == "eta_unlock_heuristic":
        eta = md_inputs.pair_metadata[:, 0, :2, 0]
        assert md_inputs.opportunity_context is not None
        last_blocker = md_inputs.opportunity_context[
            :, 0, :2, MDOpportunityFeature.LAST_MATERIAL_BLOCKER
        ]
        logits = -eta / (1.0 + eta) + 0.6 * last_blocker
        residual_magnitude = 0.0
        component_scales = [0.0, float(logits.abs().mean()), 0.0]
        saturation_rate = 0.0
        parameter_count = 0
    else:
        raise ValueError(f"unknown method: {method}")

    predictions = torch.argmax(logits, dim=-1).tolist()
    targets = [state.oracle_action for state in states]
    agreements = [
        int(prediction == target)
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    regrets = [
        max(state.oracle_utility)
        - state.oracle_utility[prediction]
        for state, prediction in zip(states, predictions, strict=True)
    ]
    state_index = {state.state_id: index for index, state in enumerate(states)}
    flip_successes = []
    pair_agreements = []
    pair_regrets = []
    family_successes: dict[str, list[int]] = {
        family: [] for family in SEMANTIC_FAMILIES
    }
    for twin in twins:
        before_index = state_index[twin.before.state_id]
        after_index = state_index[twin.after.state_id]
        before_prediction = predictions[before_index]
        after_prediction = predictions[after_index]
        success = int(
            (before_prediction, after_prediction)
            == (twin.before.oracle_action, twin.after.oracle_action)
        )
        flip_successes.append(success)
        family_successes[twin.family].append(success)
        pair_agreements.append(
            (agreements[before_index] + agreements[after_index]) / 2
        )
        pair_regrets.append((regrets[before_index] + regrets[after_index]) / 2)

    return {
        "first_action_agreement": sum(agreements) / len(agreements),
        "agreement_ci95": list(
            _bootstrap_mean_ci(pair_agreements, seed=seed + 1)
        ),
        "semantic_action_flip_accuracy": (
            sum(flip_successes) / len(flip_successes)
        ),
        "flip_ci95": list(
            _wilson_interval(sum(flip_successes), len(flip_successes))
        ),
        "family_flip_accuracy": {
            family: sum(values) / len(values)
            for family, values in family_successes.items()
        },
        "mean_one_step_oracle_regret": sum(regrets) / len(regrets),
        "regret_ci95": list(_bootstrap_mean_ci(pair_regrets, seed=seed + 2)),
        "mean_residual_magnitude": residual_magnitude,
        "component_scale_mean_abs": {
            "legacy": component_scales[0],
            "physics": component_scales[1],
            "residual": component_scales[2],
        },
        "residual_saturation_rate": saturation_rate,
        "parameter_count": parameter_count,
        "initial_evaluation_agreement": (
            sum(agreements) / len(agreements)
            if initial_evaluation_agreement is None
            else initial_evaluation_agreement
        ),
        "initial_train_loss": (
            learning_curve[0] if learning_curve else 0.0
        ),
        "final_train_loss": (
            learning_curve[-1] if learning_curve else 0.0
        ),
    }


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    centre = (proportion + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, replicates: int = 500
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires values")
    generator = random.Random(seed)
    means = sorted(
        sum(generator.choice(values) for _value in values) / len(values)
        for _replicate in range(replicates)
    )
    return means[int(0.025 * replicates)], means[int(0.975 * replicates) - 1]


def _mask_digest(states: Sequence[SemanticState]) -> str:
    import hashlib

    payload = json.dumps(
        [state.hard_mask for state in states], separators=(",", ":")
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _render_report(summary: dict[str, object]) -> str:
    decision = summary["decision"]
    assert isinstance(decision, dict)
    lines = [
        "# Ticket 19 Semantic Signal Gate",
        "",
        (
            "This controlled semantic probe is not a Ticket 20 benchmark and "
            "does not generate production expert data."
        ),
        "",
        f"Decision: **{decision['semantic_residual_signal']}**",
        "",
        (
            "| Method | Initial agreement | Agreement [95% CI] | "
            "Flip [95% CI] | Regret [95% CI] | L/P/R scale | "
            "Residual | Saturation |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    methods = summary["methods"]
    assert isinstance(methods, dict)
    for method in METHOD_NAMES:
        metrics = methods[method]
        assert isinstance(metrics, dict)
        agreement_ci = cast(list[float], metrics["agreement_ci95"])
        flip_ci = cast(list[float], metrics["flip_ci95"])
        regret_ci = cast(list[float], metrics["regret_ci95"])
        scales = cast(dict[str, float], metrics["component_scale_mean_abs"])
        lines.append(
            "| "
            + method
            + f" | {metrics['initial_evaluation_agreement']:.3f}"
            + f" | {metrics['first_action_agreement']:.3f} "
            + f"[{agreement_ci[0]:.3f}, {agreement_ci[1]:.3f}]"
            + f" | {metrics['semantic_action_flip_accuracy']:.3f} "
            + f"[{flip_ci[0]:.3f}, {flip_ci[1]:.3f}]"
            + f" | {metrics['mean_one_step_oracle_regret']:.4f} "
            + f"[{regret_ci[0]:.4f}, {regret_ci[1]:.4f}]"
            + f" | {scales['legacy']:.3f}/{scales['physics']:.3f}/"
            + f"{scales['residual']:.3f}"
            + f" | {metrics['mean_residual_magnitude']:.4f}"
            + f" | {metrics['residual_saturation_rate']:.3f} |"
        )
    controls = summary["controls"]
    assert isinstance(controls, dict)
    lines.extend(
        [
            "",
            (
                f"Unique pair templates: {controls['unique_pair_template_count']}; "
                "train/evaluation overlap: "
                f"{controls['train_evaluation_template_overlap']}; "
                "confidence-interval unit: semantic pair."
            ),
            (
                "The signed opportunity prior was disabled, so improvement over "
                "initial evaluation comes from fitted parameters."
            ),
            "",
            "The exact proxy MIP optimizes only the pre-registered one-step "
            "semantic utility. Full scheduling quality, scaling, and latency "
            "remain outside this gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Ticket 19 MD semantic signal gate"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-family", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = run_md_policy_signal_gate(
        args.output_dir,
        pairs_per_family=args.pairs_per_family,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
