"""Freeze an independent, model-blind relational held-out package."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast

from experiments.md_policy_relational_gate import (
    RELATIONAL_FAMILIES,
    RelationalTwin,
    _state_completion_margin,
    _template_signature,
)
from experiments.md_train_only_optimization_diagnostic import (
    _select_first_pairs_per_family,
)
from experiments.md_unconditioned_relational_audit import (
    build_unconditioned_relational_candidates,
)


HELD_OUT_MARGIN_STRATA: Final = (
    ("near_tie_lt_0.01", 0.0, 0.01),
    ("small_ge_0.01_lt_0.03", 0.01, 0.03),
    ("medium_ge_0.03_lt_0.05", 0.03, 0.05),
    ("high_ge_0.05", 0.05, math.inf),
)
FORMAL_PROTOCOL: Final = {
    "candidate_pool_per_family": 5000,
    "train_pairs_per_family": 500,
    "quota_per_family_stratum": 25,
    "train_seed": 3030,
    "evaluation_seed": 3939,
    "perturbation_seed": 3940,
    "pair_margin_strata": HELD_OUT_MARGIN_STRATA,
}


@dataclass(frozen=True, slots=True)
class IndependentHeldOutPackageResult:
    summary_path: Path
    report_path: Path
    package_path: Path | None


def run_independent_held_out_package(
    output_dir: str | Path,
    *,
    candidate_pool_per_family: int = 5000,
    train_pairs_per_family: int = 500,
    quota_per_family_stratum: int = 25,
    train_seed: int = 3030,
    evaluation_seed: int = 3939,
    perturbation_seed: int = 3940,
) -> IndependentHeldOutPackageResult:
    """Freeze held-out pairs before any model output is observed."""
    _validate_protocol(
        candidate_pool_per_family=candidate_pool_per_family,
        train_pairs_per_family=train_pairs_per_family,
        quota_per_family_stratum=quota_per_family_stratum,
        train_seed=train_seed,
        evaluation_seed=evaluation_seed,
        perturbation_seed=perturbation_seed,
    )
    protocol = {
        "candidate_pool_per_family": candidate_pool_per_family,
        "train_pairs_per_family": train_pairs_per_family,
        "quota_per_family_stratum": quota_per_family_stratum,
        "train_seed": train_seed,
        "evaluation_seed": evaluation_seed,
        "perturbation_seed": perturbation_seed,
        "pair_margin_strata": HELD_OUT_MARGIN_STRATA,
    }
    formal_protocol_run = protocol == FORMAL_PROTOCOL
    destination = Path(output_dir)
    summary_path = destination / "independent_held_out_summary.json"
    report_path = destination / "independent_held_out_report.md"
    package_path = destination / "independent_held_out_package.json"
    existing = [
        path for path in (summary_path, report_path, package_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite independent held-out package: "
            + ", ".join(str(path) for path in existing)
        )

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
    selected, cells, insufficient_cells = _select_evaluation_pairs(
        evaluation_candidates,
        quota_per_family_stratum=quota_per_family_stratum,
    )

    train_templates = {_template_signature(twin) for twin in train_twins}
    evaluation_templates = {
        _template_signature(twin) for _index, twin in selected
    }
    template_overlap = train_templates & evaluation_templates
    feasible = not insufficient_cells and not template_overlap
    protocol_status = (
        ("frozen" if formal_protocol_run else "diagnostic_only")
        if feasible
        else "infeasible"
    )
    records = [
        _evaluation_record(candidate_index, twin)
        for candidate_index, twin in selected
    ]
    flip_count = sum(
        cast(bool, record["oracle_action_flipped"]) for record in records
    )
    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "independent_held_out_relational_package",
        "ticket": 39,
        "source_ticket": 38,
        "formal_protocol_run": formal_protocol_run,
        "protocol_status": protocol_status,
        "frozen_protocol": _protocol_record(protocol),
        "cells": cells,
        "insufficient_cells": insufficient_cells,
        "train_pair_count": len(train_twins),
        "evaluation_pair_count": len(records) if feasible else 0,
        "evaluation_state_count": 2 * len(records) if feasible else 0,
        "selected_oracle_flip_pair_count": flip_count if feasible else 0,
        "selected_oracle_no_flip_pair_count": (
            len(records) - flip_count if feasible else 0
        ),
        "selected_natural_oracle_flip_rate": (
            flip_count / len(records) if feasible else None
        ),
        "train_template_count": len(train_templates),
        "evaluation_template_count": len(evaluation_templates) if feasible else 0,
        "train_evaluation_template_overlap": len(template_overlap),
        "evaluation_package_written": feasible,
        "model_training_or_evaluation_run": False,
        "ticket_40_authorized": formal_protocol_run and feasible,
        "controls": {
            "production_expert_dataset_written": False,
            "ticket_17_unfrozen": False,
            "ticket_20_unfrozen": False,
            "ticket_32_unblocked": False,
        },
        "next_step": (
            "Ticket 40 is authorized only against this committed package."
            if formal_protocol_run and feasible
            else (
                "This reduced package does not authorize model evaluation."
                if feasible
                else "Model evaluation remains unauthorized."
            )
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if feasible:
        package: dict[str, object] = {
            "schema_version": "1.0.0",
            "status": "frozen_independent_held_out_relational_package",
            "ticket": 39,
            "formal_protocol_run": formal_protocol_run,
            "train_provenance": {
                "candidate_pool_per_family": candidate_pool_per_family,
                "pairs_per_family": train_pairs_per_family,
                "seed": train_seed,
                "selection_order": "first_generated_pairs_per_family",
            },
            "evaluation_provenance": {
                "candidate_pool_per_family": candidate_pool_per_family,
                "quota_per_family_stratum": quota_per_family_stratum,
                "seed": evaluation_seed,
                "selection_order": (
                    "family_then_margin_stratum_then_candidate_generation_order"
                ),
            },
            "perturbation_seed": perturbation_seed,
            "pair_margin_definition": (
                "min(before_oracle_completion_margin, "
                "after_oracle_completion_margin)"
            ),
            "pair_margin_strata": {
                name: {
                    "lower": lower,
                    "upper": None if math.isinf(upper) else upper,
                }
                for name, lower, upper in HELD_OUT_MARGIN_STRATA
            },
            "evaluation_pair_count": len(records),
            "evaluation_state_count": 2 * len(records),
            "evaluation_records": records,
            "train_template_count": len(train_templates),
            "evaluation_template_count": len(evaluation_templates),
            "train_evaluation_template_overlap": len(template_overlap),
            "selection_controls": {
                "family_used_for_selection": True,
                "pair_margin_used_for_selection": True,
                "oracle_action_used_for_selection": False,
                "oracle_flip_used_for_selection": False,
                "model_output_used_for_selection": False,
                "oracle_labels_observed_only_after_selection": True,
            },
            "controls": {
                "production_expert_dataset_written": False,
                "model_training_or_evaluation_run": False,
            },
        }
        package_path.write_text(
            json.dumps(package, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        resolved_package_path: Path | None = package_path
    else:
        resolved_package_path = None
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return IndependentHeldOutPackageResult(
        summary_path,
        report_path,
        resolved_package_path,
    )


def _select_evaluation_pairs(
    candidates: Sequence[RelationalTwin],
    *,
    quota_per_family_stratum: int,
) -> tuple[
    tuple[tuple[int, RelationalTwin], ...],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    selected: list[tuple[int, RelationalTwin]] = []
    cells: list[dict[str, object]] = []
    insufficient: list[dict[str, object]] = []
    for family in RELATIONAL_FAMILIES:
        family_candidates = [twin for twin in candidates if twin.family == family]
        for stratum, lower, upper in HELD_OUT_MARGIN_STRATA:
            available = [
                (index, twin)
                for index, twin in enumerate(family_candidates)
                if lower <= _pair_margin(twin) < upper
            ]
            chosen = available[:quota_per_family_stratum]
            flip_count = sum(
                twin.before.oracle_action != twin.after.oracle_action
                for _index, twin in chosen
            )
            cell: dict[str, object] = {
                "family": family,
                "pair_margin_stratum": stratum,
                "available_candidate_count": len(available),
                "required_quota": quota_per_family_stratum,
                "selected_pair_count": len(chosen),
                "selected_oracle_flip_pair_count": flip_count,
                "selected_oracle_no_flip_pair_count": len(chosen) - flip_count,
                "sufficient": len(chosen) == quota_per_family_stratum,
            }
            cells.append(cell)
            if len(chosen) != quota_per_family_stratum:
                insufficient.append(dict(cell))
            selected.extend(chosen)
    return tuple(selected), cells, insufficient


def _evaluation_record(
    candidate_index: int,
    twin: RelationalTwin,
) -> dict[str, object]:
    before_margin = _state_completion_margin(twin.before)
    after_margin = _state_completion_margin(twin.after)
    pair_margin = min(before_margin, after_margin)
    return {
        "family": twin.family,
        "candidate_index_within_family": candidate_index,
        "pair_id": twin.pair_id,
        "changed_entity_index": twin.changed_entity_index,
        "before_oracle_action": twin.before.oracle_action,
        "after_oracle_action": twin.after.oracle_action,
        "oracle_action_flipped": (
            twin.before.oracle_action != twin.after.oracle_action
        ),
        "before_oracle_completion_margin": before_margin,
        "after_oracle_completion_margin": after_margin,
        "pair_margin": pair_margin,
        "pair_margin_stratum": _pair_margin_stratum(pair_margin),
    }


def _pair_margin(twin: RelationalTwin) -> float:
    return min(
        _state_completion_margin(twin.before),
        _state_completion_margin(twin.after),
    )


def _pair_margin_stratum(margin: float) -> str:
    for name, lower, upper in HELD_OUT_MARGIN_STRATA:
        if lower <= margin < upper:
            return name
    raise RuntimeError("held-out margin strata do not cover pair margin")


def _protocol_record(protocol: dict[str, object]) -> dict[str, object]:
    record = dict(protocol)
    record["pair_margin_definition"] = "min(before_margin, after_margin)"
    record["pair_margin_strata"] = [
        {
            "name": name,
            "lower": lower,
            "upper": None if math.isinf(upper) else upper,
        }
        for name, lower, upper in HELD_OUT_MARGIN_STRATA
    ]
    record["selection_order"] = (
        "family_then_margin_stratum_then_candidate_generation_order"
    )
    return record


def _validate_protocol(
    *,
    candidate_pool_per_family: int,
    train_pairs_per_family: int,
    quota_per_family_stratum: int,
    train_seed: int,
    evaluation_seed: int,
    perturbation_seed: int,
) -> None:
    for name, value in (
        ("candidate_pool_per_family", candidate_pool_per_family),
        ("train_pairs_per_family", train_pairs_per_family),
        ("quota_per_family_stratum", quota_per_family_stratum),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if train_pairs_per_family > candidate_pool_per_family:
        raise ValueError(
            "train_pairs_per_family must not exceed candidate_pool_per_family"
        )
    for name, value in (
        ("train_seed", train_seed),
        ("evaluation_seed", evaluation_seed),
        ("perturbation_seed", perturbation_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if len({train_seed, evaluation_seed, perturbation_seed}) != 3:
        raise ValueError("train, evaluation, and perturbation seeds must be unique")


def _render_report(summary: dict[str, object]) -> str:
    cells = cast(list[dict[str, object]], summary["cells"])
    lines = [
        "# Ticket 39 Independent Held-Out Relational Package",
        "",
        f"Protocol status: **{summary['protocol_status']}**.",
        "",
        "## Stratum Supply And Selection",
        "",
        "| Family | Pair-margin stratum | Available | Selected | Flip after selection |",
        "|---|---|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['family']} | {row['pair_margin_stratum']} | "
            f"{row['available_candidate_count']} | {row['selected_pair_count']} | "
            f"{row['selected_oracle_flip_pair_count']} |"
        )
    lines.extend(
        [
            "",
            "## Package",
            "",
            f"Evaluation pairs: **{summary['evaluation_pair_count']}**.",
            f"Evaluation states: **{summary['evaluation_state_count']}**.",
            f"Train/evaluation template overlap: **{summary['train_evaluation_template_overlap']}**.",
            f"Natural flip rate after selection: **{summary['selected_natural_oracle_flip_rate']}**.",
            "",
            "## Next Step",
            "",
            cast(str, summary["next_step"]),
            "",
            "## Limitations",
            "",
            "This ticket freezes data only, contains no model result, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze Ticket 39 independent held-out relational package"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_independent_held_out_package(args.output_dir)
    print(result.summary_path)
    print(result.report_path)
    if result.package_path is not None:
        print(result.package_path)


if __name__ == "__main__":
    main()
