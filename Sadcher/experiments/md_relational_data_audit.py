"""Deterministic data audit for the Ticket 29 relational twins."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence, cast

from experiments.md_policy_relational_gate import (
    CANDIDATE_TASK_COUNT,
    RELATIONAL_FAMILIES,
    ROBOT_COUNT,
    RelationalState,
    RelationalTwin,
    _StateSpec,
    _split_twins,
    _state_completion_margin,
    _state_from_spec,
    _template_signature,
    build_relational_twins,
)


AUDIT_MARGIN_STRATA = (
    ("near_tie_lt_0.01", 0.0, 0.01),
    ("small_ge_0.01_lt_0.05", 0.01, 0.05),
    ("moderate_ge_0.05_lt_0.10", 0.05, 0.10),
    ("clear_ge_0.10", 0.10, math.inf),
)
PERTURBATION_MAGNITUDES = {
    "robot_position": 0.01,
    "robot_speed": 0.01,
    "task_pickup": 0.01,
    "downstream_priority": 0.01,
}
BASELINE_NAMES = (
    "local_eta",
    "eta_plus_priority",
    "competitor_aware_relational",
)
Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class RelationalDataAuditResult:
    summary_path: Path
    report_path: Path


def run_relational_data_audit(
    output_dir: str | Path,
    *,
    pairs_per_family: int = 25,
    seed: int = 2027,
    perturbation_seed: int = 29001,
) -> RelationalDataAuditResult:
    """Audit the natural Ticket 29 twin distribution without training models."""
    if (
        isinstance(pairs_per_family, bool)
        or not isinstance(pairs_per_family, int)
        or pairs_per_family < 2
    ):
        raise ValueError("pairs_per_family must be an integer of at least two")
    for name, value in (("seed", seed), ("perturbation_seed", perturbation_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    destination = Path(output_dir)
    summary_path = destination / "relational_data_audit.json"
    report_path = destination / "relational_data_audit.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite audit report: {names}")

    twins = build_relational_twins(
        pairs_per_family=pairs_per_family,
        seed=seed,
    )
    train_twins, evaluation_twins = _split_twins(twins, pairs_per_family)
    train_templates = {_template_signature(twin) for twin in train_twins}
    evaluation_templates = {
        _template_signature(twin) for twin in evaluation_twins
    }
    overlap = train_templates & evaluation_templates
    if overlap:
        raise RuntimeError("train/evaluation relational templates overlap")

    train_states = _states(train_twins)
    evaluation_states = _states(evaluation_twins)
    all_states = _states(twins)
    evaluation_strata = margin_strata(evaluation_states)
    stratum_ids = [
        state_id
        for row in evaluation_strata.values()
        for state_id in cast(list[str], row["state_ids"])
    ]
    strata_cover_evaluation = set(stratum_ids) == {
        state.state_id for state in evaluation_states
    }
    strata_are_mutually_exclusive = len(stratum_ids) == len(set(stratum_ids))
    if not strata_cover_evaluation or not strata_are_mutually_exclusive:
        raise RuntimeError("evaluation margin strata must partition all states")

    family_statistics: dict[str, dict[str, object]] = {
        family: _family_statistics(twins, family)
        for family in RELATIONAL_FAMILIES
    }
    baselines: dict[str, dict[str, object]] = {
        name: _baseline_metrics(name, evaluation_twins)
        for name in BASELINE_NAMES
    }
    perturbations = perturbation_statistics(
        train_states,
        evaluation_states,
        seed=perturbation_seed,
    )
    feature_comparison = {
        name: _compare_distributions(
            _feature_values(train_states, extractor),
            _feature_values(evaluation_states, extractor),
        )
        for name, extractor in _feature_extractors().items()
    }
    best_local_pair_accuracy = max(
        cast(float, baselines[name]["oracle_pair_accuracy"])
        for name in ("local_eta", "eta_plus_priority")
    )
    relational_pair_accuracy = cast(
        float,
        baselines["competitor_aware_relational"]["oracle_pair_accuracy"],
    )
    all_families_flip = all(
        cast(float, row["oracle_action_flip_rate"]) == 1.0
        for row in family_statistics.values()
    )

    pair_records = []
    train_pair_ids = {twin.pair_id for twin in train_twins}
    for twin in twins:
        pair_records.append(
            {
                "pair_id": twin.pair_id,
                "family": twin.family,
                "split": "train" if twin.pair_id in train_pair_ids else "evaluation",
                "changed_entity_index": twin.changed_entity_index,
                "before_oracle_action": twin.before.oracle_action,
                "after_oracle_action": twin.after.oracle_action,
                "oracle_action_flipped": (
                    twin.before.oracle_action != twin.after.oracle_action
                ),
                "before_oracle_completion_margin": _state_completion_margin(
                    twin.before
                ),
                "after_oracle_completion_margin": _state_completion_margin(
                    twin.after
                ),
            }
        )

    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "ticket_29_relational_data_audit",
        "ticket": 29,
        "data_seed": seed,
        "perturbation_seed": perturbation_seed,
        "pairs_per_family": pairs_per_family,
        "pair_count": len(twins),
        "state_count": len(all_states),
        "train_pair_count": len(train_twins),
        "evaluation_pair_count": len(evaluation_twins),
        "family_statistics": family_statistics,
        "pair_records": pair_records,
        "train_evaluation_distribution_comparison": feature_comparison,
        "evaluation_margin_strata": evaluation_strata,
        "baselines": baselines,
        "perturbations": perturbations,
        "controls": {
            "train_evaluation_template_overlap": len(overlap),
            "margin_strata_are_mutually_exclusive": strata_are_mutually_exclusive,
            "margin_strata_cover_all_evaluation_states": strata_cover_evaluation,
            "production_expert_dataset_written": False,
            "report_files_are_independent_from_relational_gate": True,
        },
        "interpretation": {
            "engineered_relational_dependence_supported": all_families_flip,
            "competitor_aware_rule_advantage_supported": (
                relational_pair_accuracy > best_local_pair_accuracy
            ),
            "data_contains_relational_learning_signal": (
                all_families_flip
                and relational_pair_accuracy > best_local_pair_accuracy
            ),
            "supports": [
                "Each accepted twin changes the oracle action after a non-local intervention.",
                "The explicit completion-aware rule tests whether that dependence is recoverable from competitor context.",
                "The held-out split has zero exact template overlap with training.",
            ],
            "does_not_support": [
                "Twin acceptance is conditioned on an oracle flip, so the family flip rate is not a natural-frequency estimate.",
                "A rule-baseline advantage does not establish that the learned pair-aware scorer generalizes.",
                "This audit does not evaluate end-to-end scheduling quality or authorize Ticket 20 data generation.",
            ],
        },
    }

    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return RelationalDataAuditResult(summary_path, report_path)


def margin_stratum_name(margin: float) -> str:
    """Return the unique audit stratum for a finite non-negative margin."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    for name, lower, upper in AUDIT_MARGIN_STRATA:
        if lower <= margin < upper:
            return name
    raise RuntimeError("margin strata do not cover the supplied value")


def margin_strata(
    states: Sequence[RelationalState],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[tuple[str, float]]] = {
        name: [] for name, _lower, _upper in AUDIT_MARGIN_STRATA
    }
    for state in states:
        margin = _state_completion_margin(state)
        grouped[margin_stratum_name(margin)].append((state.state_id, margin))
    return {
        name: {
            "state_count": len(rows),
            "state_ids": [state_id for state_id, _margin in rows],
            "margin_distribution": _distribution(
                [margin for _state_id, margin in rows]
            ),
        }
        for name, rows in grouped.items()
    }


def perturbation_statistics(
    train_states: Sequence[RelationalState],
    evaluation_states: Sequence[RelationalState],
    *,
    seed: int,
) -> dict[str, object]:
    generator = random.Random(seed)
    by_feature: dict[str, dict[str, int | float]] = {
        name: {"comparison_count": 0, "label_flip_count": 0}
        for name in PERTURBATION_MAGNITUDES
    }
    by_split: dict[str, dict[str, dict[str, int | float]]] = {
        split: {
            name: {"comparison_count": 0, "label_flip_count": 0}
            for name in PERTURBATION_MAGNITUDES
        }
        for split in ("train", "evaluation")
    }
    records: list[dict[str, object]] = []
    for split, states in (("train", train_states), ("evaluation", evaluation_states)):
        for state in states:
            for feature, magnitude in PERTURBATION_MAGNITUDES.items():
                perturbed = _perturb_state(
                    state,
                    feature=feature,
                    magnitude=magnitude,
                    generator=generator,
                )
                flipped = perturbed.oracle_action != state.oracle_action
                _increment_flip_row(by_feature[feature], flipped)
                _increment_flip_row(by_split[split][feature], flipped)
                records.append(
                    {
                        "state_id": state.state_id,
                        "split": split,
                        "feature": feature,
                        "oracle_action_before": state.oracle_action,
                        "oracle_action_after": perturbed.oracle_action,
                        "label_flipped": flipped,
                    }
                )
    for row in by_feature.values():
        _finish_flip_row(row)
    for split_rows in by_split.values():
        for row in split_rows.values():
            _finish_flip_row(row)
    total_comparisons = sum(
        int(row["comparison_count"]) for row in by_feature.values()
    )
    total_flips = sum(int(row["label_flip_count"]) for row in by_feature.values())
    return {
        "magnitudes": dict(PERTURBATION_MAGNITUDES),
        "comparison_count": total_comparisons,
        "label_flip_count": total_flips,
        "label_flip_rate": total_flips / total_comparisons,
        "by_feature": by_feature,
        "by_split": by_split,
        "records": records,
    }


def _states(twins: Sequence[RelationalTwin]) -> tuple[RelationalState, ...]:
    return tuple(state for twin in twins for state in (twin.before, twin.after))


def _family_statistics(
    twins: Sequence[RelationalTwin], family: str
) -> dict[str, object]:
    selected = [twin for twin in twins if twin.family == family]
    flips = sum(twin.before.oracle_action != twin.after.oracle_action for twin in selected)
    return {
        "pair_count": len(selected),
        "oracle_action_flip_count": flips,
        "oracle_action_flip_rate": flips / len(selected),
        "before_margin_distribution": _distribution(
            [_state_completion_margin(twin.before) for twin in selected]
        ),
        "after_margin_distribution": _distribution(
            [_state_completion_margin(twin.after) for twin in selected]
        ),
    }


def _feature_extractors() -> dict[str, Callable[[RelationalState], list[float]]]:
    return {
        "robot_to_pickup_distance": lambda state: [
            math.dist(robot, pickup)
            for robot in state.robot_positions
            for pickup in state.task_pickups
        ],
        "loaded_leg_distance": lambda state: [
            math.dist(pickup, delivery)
            for pickup, delivery in zip(
                state.task_pickups, state.task_deliveries, strict=True
            )
        ],
        "robot_speed": lambda state: list(state.robot_loaded_speeds),
        "downstream_priority": lambda state: list(state.downstream_priorities),
        "eta": lambda state: [value for row in state.eta for value in row],
        "oracle_completion_margin": lambda state: [
            _state_completion_margin(state)
        ],
    }


def _feature_values(
    states: Sequence[RelationalState],
    extractor: Callable[[RelationalState], list[float]],
) -> list[float]:
    return [value for state in states for value in extractor(state)]


def _compare_distributions(
    train_values: Sequence[float], evaluation_values: Sequence[float]
) -> dict[str, object]:
    train = _distribution(train_values)
    evaluation = _distribution(evaluation_values)
    if train is None or evaluation is None:
        raise ValueError("train/evaluation distributions must not be empty")
    return {
        "train": train,
        "evaluation": evaluation,
        "evaluation_minus_train_mean": evaluation["mean"] - train["mean"],
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mean = sum(ordered) / len(ordered)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "q25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "q75": _quantile(ordered, 0.75),
        "maximum": ordered[-1],
        "mean": mean,
        "population_std": math.sqrt(
            sum((value - mean) ** 2 for value in ordered) / len(ordered)
        ),
    }


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _baseline_prediction(name: str, state: RelationalState) -> int:
    if name == "local_eta":
        scores = [-eta for row in state.eta for eta in row]
    elif name == "eta_plus_priority":
        scores = [
            1.25 * state.downstream_priorities[task] - state.eta[robot][task]
            for robot in range(ROBOT_COUNT)
            for task in range(CANDIDATE_TASK_COUNT)
        ]
    elif name == "competitor_aware_relational":
        edge_utility = tuple(
            tuple(
                1.25 * state.downstream_priorities[task]
                - state.eta[robot][task]
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
        scores = [
            max(
                total
                for assigned_tasks, total in assignments
                if assigned_tasks[robot] == task
            )
            for robot in range(ROBOT_COUNT)
            for task in range(CANDIDATE_TASK_COUNT)
        ]
    else:
        raise ValueError(f"unknown baseline: {name}")
    return max(range(len(scores)), key=lambda action: (scores[action], -action))


def _baseline_metrics(
    name: str, twins: Sequence[RelationalTwin]
) -> dict[str, object]:
    states = _states(twins)
    predictions = {
        state.state_id: _baseline_prediction(name, state) for state in states
    }
    agreements = [
        predictions[state.state_id] == state.oracle_action for state in states
    ]
    pair_rows = []
    family_rows: dict[str, list[tuple[bool, bool]]] = {
        family: [] for family in RELATIONAL_FAMILIES
    }
    for twin in twins:
        before = predictions[twin.before.state_id]
        after = predictions[twin.after.state_id]
        exact_pair = (
            before == twin.before.oracle_action and after == twin.after.oracle_action
        )
        predicted_flip = before != after
        pair_rows.append((exact_pair, predicted_flip))
        family_rows[twin.family].append((exact_pair, predicted_flip))
    return {
        "evaluation_state_count": len(states),
        "first_action_agreement": sum(agreements) / len(agreements),
        "oracle_pair_accuracy": sum(row[0] for row in pair_rows) / len(pair_rows),
        "prediction_flip_rate": sum(row[1] for row in pair_rows) / len(pair_rows),
        "by_family": {
            family: {
                "pair_count": len(rows),
                "oracle_pair_accuracy": sum(row[0] for row in rows) / len(rows),
                "prediction_flip_rate": sum(row[1] for row in rows) / len(rows),
            }
            for family, rows in family_rows.items()
        },
    }


def _perturb_state(
    state: RelationalState,
    *,
    feature: str,
    magnitude: float,
    generator: random.Random,
) -> RelationalState:
    spec = _StateSpec(
        robot_positions=state.robot_positions,
        robot_loaded_speeds=state.robot_loaded_speeds,
        task_pickups=state.task_pickups,
        task_deliveries=state.task_deliveries,
        downstream_priorities=state.downstream_priorities,
    )
    if feature == "robot_position":
        spec = replace(
            spec,
            robot_positions=tuple(
                _perturb_point(point, magnitude, generator)
                for point in spec.robot_positions
            ),
        )
    elif feature == "robot_speed":
        spec = replace(
            spec,
            robot_loaded_speeds=tuple(
                max(0.05, value + generator.uniform(-magnitude, magnitude))
                for value in spec.robot_loaded_speeds
            ),
        )
    elif feature == "task_pickup":
        spec = replace(
            spec,
            task_pickups=tuple(
                _perturb_point(point, magnitude, generator)
                for point in spec.task_pickups
            ),
        )
    elif feature == "downstream_priority":
        spec = replace(
            spec,
            downstream_priorities=tuple(
                max(0.0, value + generator.uniform(-magnitude, magnitude))
                for value in spec.downstream_priorities
            ),
        )
    else:
        raise ValueError(f"unknown perturbation feature: {feature}")
    return _state_from_spec(
        state_id=f"{state.state_id}-perturbed-{feature}",
        pair_id=state.pair_id,
        family=state.family,
        spec=spec,
    )


def _perturb_point(
    point: Point, magnitude: float, generator: random.Random
) -> Point:
    return (
        min(2.0, max(0.0, point[0] + generator.uniform(-magnitude, magnitude))),
        min(2.0, max(0.0, point[1] + generator.uniform(-magnitude, magnitude))),
    )


def _increment_flip_row(row: dict[str, int | float], flipped: bool) -> None:
    row["comparison_count"] = int(row["comparison_count"]) + 1
    row["label_flip_count"] = int(row["label_flip_count"]) + int(flipped)


def _finish_flip_row(row: dict[str, int | float]) -> None:
    comparisons = int(row["comparison_count"])
    row["label_flip_rate"] = int(row["label_flip_count"]) / comparisons


def _render_report(summary: dict[str, object]) -> str:
    families = summary["family_statistics"]
    baselines = summary["baselines"]
    comparison = summary["train_evaluation_distribution_comparison"]
    perturbations = summary["perturbations"]
    strata = summary["evaluation_margin_strata"]
    interpretation = summary["interpretation"]
    assert isinstance(families, dict)
    assert isinstance(baselines, dict)
    assert isinstance(comparison, dict)
    assert isinstance(perturbations, dict)
    assert isinstance(strata, dict)
    assert isinstance(interpretation, dict)

    lines = [
        "# Ticket 29 Relational Data Audit",
        "",
        "This report audits the controlled relational twins. It is independent from the gate reports and does not create a Ticket 20 production expert dataset.",
        "",
        "## Relational Families",
        "",
        "| Family | Pairs | Oracle flips | Before margin mean | After margin mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in RELATIONAL_FAMILIES:
        row = families[family]
        assert isinstance(row, dict)
        before = row["before_margin_distribution"]
        after = row["after_margin_distribution"]
        assert isinstance(before, dict) and isinstance(after, dict)
        lines.append(
            f"| {family} | {row['pair_count']} | "
            f"{float(row['oracle_action_flip_rate']):.3f} | "
            f"{float(before['mean']):.6f} | {float(after['mean']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Train/Evaluation Distributions",
            "",
            "Distance is reported separately for robot-to-pickup and pickup-to-delivery legs.",
            "",
            "| Feature | Train mean | Evaluation mean | Eval - train |",
            "|---|---:|---:|---:|",
        ]
    )
    for feature, raw_row in comparison.items():
        assert isinstance(raw_row, dict)
        train = raw_row["train"]
        evaluation = raw_row["evaluation"]
        assert isinstance(train, dict) and isinstance(evaluation, dict)
        lines.append(
            f"| {feature} | {float(train['mean']):.6f} | "
            f"{float(evaluation['mean']):.6f} | "
            f"{float(raw_row['evaluation_minus_train_mean']):+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Rule Baselines",
            "",
            "| Baseline | State agreement | Oracle pair accuracy | Prediction flip rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in BASELINE_NAMES:
        row = baselines[name]
        assert isinstance(row, dict)
        lines.append(
            f"| {name} | {float(row['first_action_agreement']):.3f} | "
            f"{float(row['oracle_pair_accuracy']):.3f} | "
            f"{float(row['prediction_flip_rate']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Perturbation Stability",
            "",
            f"Overall label flip rate: **{float(perturbations['label_flip_rate']):.3f}** ({perturbations['label_flip_count']}/{perturbations['comparison_count']}).",
            "",
            "| Perturbed feature | Magnitude | Comparisons | Label flip rate |",
            "|---|---:|---:|---:|",
        ]
    )
    by_feature = perturbations["by_feature"]
    magnitudes = perturbations["magnitudes"]
    assert isinstance(by_feature, dict) and isinstance(magnitudes, dict)
    for feature in PERTURBATION_MAGNITUDES:
        row = by_feature[feature]
        assert isinstance(row, dict)
        lines.append(
            f"| {feature} | {float(magnitudes[feature]):.3f} | "
            f"{row['comparison_count']} | {float(row['label_flip_rate']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Evaluation Margin Strata",
            "",
            "| Stratum | States | Mean margin |",
            "|---|---:|---:|",
        ]
    )
    for name, _lower, _upper in AUDIT_MARGIN_STRATA:
        row = strata[name]
        assert isinstance(row, dict)
        distribution = row["margin_distribution"]
        mean = "n/a"
        if isinstance(distribution, dict):
            mean = f"{float(distribution['mean']):.6f}"
        lines.append(f"| {name} | {row['state_count']} | {mean} |")

    supports = interpretation["supports"]
    does_not_support = interpretation["does_not_support"]
    assert isinstance(supports, list) and isinstance(does_not_support, list)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Data contains a relational learning signal: **{str(interpretation['data_contains_relational_learning_signal']).lower()}**.",
            "",
            "Supported by this audit:",
            *[f"- {item}" for item in supports],
            "",
            "Not supported by this audit:",
            *[f"- {item}" for item in does_not_support],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Ticket 29 relational data audit"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-family", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--perturbation-seed", type=int, default=29001)
    args = parser.parse_args()
    result = run_relational_data_audit(
        args.output_dir,
        pairs_per_family=args.pairs_per_family,
        seed=args.seed,
        perturbation_seed=args.perturbation_seed,
    )
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
