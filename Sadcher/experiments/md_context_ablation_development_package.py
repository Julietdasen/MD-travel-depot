"""Freeze the model-blind Ticket 42 context-ablation development package."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast

from experiments.md_context_ablation_relational_data import (
    RELATIONAL_FAMILIES,
    RelationalTwin,
    build_unconditioned_relational_candidates,
    state_completion_margin as _state_completion_margin,
    template_signature as _template_signature,
)


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
HELD_OUT_MARGIN_STRATA: Final = (
    ("near_tie_lt_0.01", 0.0, 0.01),
    ("small_ge_0.01_lt_0.03", 0.01, 0.03),
    ("medium_ge_0.03_lt_0.05", 0.03, 0.05),
    ("high_ge_0.05", 0.05, math.inf),
)
TICKET_39_PROTOCOL: Final = {
    "candidate_pool_per_family": 5000,
    "evaluation_seed": 3939,
    "perturbation_seed": 3940,
    "quota_per_family_stratum": 25,
}
CORRECTED_HELD_OUT_PACKAGE: Final = (
    REPOSITORY_ROOT
    / "reports"
    / "md_independent_held_out_package_protocol_corrected_2026-08-28"
    / "independent_held_out_package.json"
)
FORMAL_PROTOCOL: Final = {
    "training_candidate_pool_per_family": 5000,
    "training_pairs_per_family": 500,
    "training_seed": 3030,
    "development_candidates_per_family": 10000,
    "development_base_state_seed": 4242,
    "development_perturbation_seed": 4243,
    "quota_per_family_stratum": 50,
    "pair_margin_strata": HELD_OUT_MARGIN_STRATA,
}


@dataclass(frozen=True, slots=True)
class ContextAblationDevelopmentPackageResult:
    summary_path: Path
    package_path: Path | None
    markdown_path: Path


def run_context_ablation_development_package(
    output_dir: str | Path,
    *,
    training_candidate_pool_per_family: int = 5000,
    training_pairs_per_family: int = 500,
    training_seed: int = 3030,
    development_candidates_per_family: int = 10000,
    development_base_state_seed: int = 4242,
    development_perturbation_seed: int = 4243,
    quota_per_family_stratum: int = 50,
    corrected_held_out_package_path: str | Path = CORRECTED_HELD_OUT_PACKAGE,
) -> ContextAblationDevelopmentPackageResult:
    """Generate, validate, and freeze Ticket 42 without invoking a model."""
    _validate_protocol(
        training_candidate_pool_per_family=training_candidate_pool_per_family,
        training_pairs_per_family=training_pairs_per_family,
        training_seed=training_seed,
        development_candidates_per_family=development_candidates_per_family,
        development_base_state_seed=development_base_state_seed,
        development_perturbation_seed=development_perturbation_seed,
        quota_per_family_stratum=quota_per_family_stratum,
    )
    destination = Path(output_dir)
    summary_path = destination / "context_ablation_development_summary.json"
    package_path = destination / "context_ablation_development_package.json"
    markdown_path = destination / "context_ablation_development_package.md"
    existing = [
        path
        for path in (summary_path, package_path, markdown_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite context-ablation development package: "
            + ", ".join(str(path) for path in existing)
        )

    protocol = {
        "training_candidate_pool_per_family": (
            training_candidate_pool_per_family
        ),
        "training_pairs_per_family": training_pairs_per_family,
        "training_seed": training_seed,
        "development_candidates_per_family": (
            development_candidates_per_family
        ),
        "development_base_state_seed": development_base_state_seed,
        "development_perturbation_seed": development_perturbation_seed,
        "quota_per_family_stratum": quota_per_family_stratum,
        "pair_margin_strata": HELD_OUT_MARGIN_STRATA,
    }
    formal_protocol_run = protocol == FORMAL_PROTOCOL

    training_candidates = build_unconditioned_relational_candidates(
        candidates_per_family=training_candidate_pool_per_family,
        seed=training_seed,
    )
    training_twins = _select_first_pairs_per_family(
        training_candidates,
        pairs_per_family=training_pairs_per_family,
    )
    development_candidates = build_unconditioned_relational_candidates(
        candidates_per_family=development_candidates_per_family,
        seed=development_base_state_seed,
        perturbation_seed=development_perturbation_seed,
    )
    selected, cells, insufficient_cells = _select_development_pairs(
        development_candidates,
        quota_per_family_stratum=quota_per_family_stratum,
    )
    records = [
        _development_record(candidate_index, twin)
        for candidate_index, twin in selected
    ]
    cells = _with_post_selection_flip_counts(cells, records)

    replay_selected, replay_mismatch = _replay_development_records(
        records,
        development_candidates_per_family=development_candidates_per_family,
        development_base_state_seed=development_base_state_seed,
        development_perturbation_seed=development_perturbation_seed,
        quota_per_family_stratum=quota_per_family_stratum,
    )
    held_out_twins, held_out_replay_mismatch = _replay_corrected_held_out(
        Path(corrected_held_out_package_path)
    )
    training_templates = {_template_signature(twin) for twin in training_twins}
    development_templates = {
        _template_signature(twin) for twin in replay_selected
    }
    held_out_templates = {_template_signature(twin) for twin in held_out_twins}
    training_overlap = training_templates & development_templates
    held_out_overlap = held_out_templates & development_templates
    integrity = _package_integrity(records)
    feasible = (
        not insufficient_cells
        and replay_mismatch == 0
        and held_out_replay_mismatch == 0
        and not training_overlap
        and not held_out_overlap
        and cast(bool, integrity["valid"])
    )
    frozen = formal_protocol_run and feasible
    status = (
        "development_package_frozen"
        if frozen
        else "development_package_infeasible"
    )
    protocol_status = (
        "frozen"
        if frozen
        else ("diagnostic_only" if feasible else "infeasible")
    )
    flip_count = sum(
        cast(bool, record["oracle_action_flipped"]) for record in records
    )
    selected_pair_count = len(records) if feasible else 0

    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": status,
        "ticket": 42,
        "source_tickets": [38, 39, 41],
        "formal_protocol_run": formal_protocol_run,
        "protocol_status": protocol_status,
        "frozen_protocol": _protocol_record(protocol),
        "cells": cells,
        "insufficient_cells": insufficient_cells,
        "training_pair_count": len(training_twins),
        "development_pair_count": selected_pair_count,
        "development_state_count": 2 * selected_pair_count,
        "unique_development_pair_count": (
            cast(int, integrity["unique_pair_count"]) if feasible else 0
        ),
        "pairs_with_exactly_before_after_states": (
            cast(int, integrity["complete_pair_count"]) if feasible else 0
        ),
        "selected_oracle_flip_pair_count": flip_count if feasible else 0,
        "selected_oracle_no_flip_pair_count": (
            len(records) - flip_count if feasible else 0
        ),
        "selected_natural_oracle_flip_rate": (
            flip_count / len(records) if feasible and records else None
        ),
        "package_replay_mismatch_count": replay_mismatch,
        "corrected_held_out_replay_mismatch_count": held_out_replay_mismatch,
        "training_template_count": len(training_templates),
        "development_template_count": (
            len(development_templates) if feasible else 0
        ),
        "prior_held_out_template_count": len(held_out_templates),
        "training_development_template_overlap": len(training_overlap),
        "prior_held_out_development_template_overlap": len(held_out_overlap),
        "development_package_written": feasible,
        "model_training_or_evaluation_run": False,
        "architecture_candidate_selected": False,
        "architecture_gate_reopened": False,
        "ticket_43_authorized": frozen,
        "controls": {
            "production_expert_dataset_written": False,
            "ticket_17_unfrozen": False,
            "ticket_20_unfrozen": False,
            "ticket_32_unblocked": False,
            "ticket_39_artifact_modified": False,
            "ticket_40_artifact_read_or_modified": False,
            "ticket_41_artifact_modified": False,
        },
        "next_step": (
            "Ticket 43 is authorized to consume only this committed formal "
            "development package."
            if frozen
            else (
                "This reduced package cannot authorize Ticket 43."
                if feasible
                else "Ticket 43 remains unauthorized because the package is infeasible."
            )
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved_package_path: Path | None = None
    if feasible:
        package = _package_payload(
            status=status,
            formal_protocol_run=formal_protocol_run,
            training_candidate_pool_per_family=(
                training_candidate_pool_per_family
            ),
            training_pairs_per_family=training_pairs_per_family,
            training_seed=training_seed,
            development_candidates_per_family=(
                development_candidates_per_family
            ),
            development_base_state_seed=development_base_state_seed,
            development_perturbation_seed=development_perturbation_seed,
            quota_per_family_stratum=quota_per_family_stratum,
            corrected_held_out_package_path=Path(
                corrected_held_out_package_path
            ),
            records=records,
            training_template_count=len(training_templates),
            development_template_count=len(development_templates),
            held_out_template_count=len(held_out_templates),
            replay_mismatch=replay_mismatch,
            held_out_replay_mismatch=held_out_replay_mismatch,
            training_overlap=len(training_overlap),
            held_out_overlap=len(held_out_overlap),
            ticket_43_authorized=frozen,
        )
        package_path.write_text(
            json.dumps(package, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        resolved_package_path = package_path
    markdown_path.write_text(
        render_context_ablation_development_report(summary),
        encoding="utf-8",
    )
    return ContextAblationDevelopmentPackageResult(
        summary_path=summary_path,
        package_path=resolved_package_path,
        markdown_path=markdown_path,
    )


def _select_development_pairs(
    candidates: Sequence[RelationalTwin],
    *,
    quota_per_family_stratum: int,
) -> tuple[
    tuple[tuple[int, RelationalTwin], ...],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Select only by family, pair margin, and generation order."""
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
            cell: dict[str, object] = {
                "family": family,
                "pair_margin_stratum": stratum,
                "available_candidate_count": len(available),
                "required_quota": quota_per_family_stratum,
                "selected_pair_count": len(chosen),
                "sufficient": len(chosen) == quota_per_family_stratum,
            }
            cells.append(cell)
            if not cast(bool, cell["sufficient"]):
                insufficient.append(dict(cell))
            selected.extend(chosen)
    return tuple(selected), cells, insufficient


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


def _with_post_selection_flip_counts(
    cells: Sequence[dict[str, object]],
    records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    completed: list[dict[str, object]] = []
    for cell in cells:
        rows = [
            row
            for row in records
            if row["family"] == cell["family"]
            and row["pair_margin_stratum"] == cell["pair_margin_stratum"]
        ]
        flip_count = sum(cast(bool, row["oracle_action_flipped"]) for row in rows)
        completed.append(
            {
                **cell,
                "selected_oracle_flip_pair_count": flip_count,
                "selected_oracle_no_flip_pair_count": len(rows) - flip_count,
            }
        )
    return completed


def _development_record(
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
        "states": [
            {"state_role": "before", "state_id": twin.before.state_id},
            {"state_role": "after", "state_id": twin.after.state_id},
        ],
        "before_oracle_action": twin.before.oracle_action,
        "after_oracle_action": twin.after.oracle_action,
        "before_oracle_completion_margin": before_margin,
        "after_oracle_completion_margin": after_margin,
        "pair_margin": pair_margin,
        "pair_margin_stratum": _pair_margin_stratum(pair_margin),
        "oracle_action_flipped": (
            twin.before.oracle_action != twin.after.oracle_action
        ),
    }


def _held_out_record(
    candidate_index: int,
    twin: RelationalTwin,
) -> dict[str, object]:
    record = _development_record(candidate_index, twin)
    del record["states"]
    return record


def _pair_margin(twin: RelationalTwin) -> float:
    return min(
        _state_completion_margin(twin.before),
        _state_completion_margin(twin.after),
    )


def _pair_margin_stratum(margin: float) -> str:
    for name, lower, upper in HELD_OUT_MARGIN_STRATA:
        if lower <= margin < upper:
            return name
    raise RuntimeError("development margin strata do not cover pair margin")


def _replay_development_records(
    records: Sequence[dict[str, object]],
    *,
    development_candidates_per_family: int,
    development_base_state_seed: int,
    development_perturbation_seed: int,
    quota_per_family_stratum: int,
) -> tuple[tuple[RelationalTwin, ...], int]:
    candidates = build_unconditioned_relational_candidates(
        candidates_per_family=development_candidates_per_family,
        seed=development_base_state_seed,
        perturbation_seed=development_perturbation_seed,
    )
    selected, _cells, _insufficient = _select_development_pairs(
        candidates,
        quota_per_family_stratum=quota_per_family_stratum,
    )
    expected = [
        _development_record(candidate_index, twin)
        for candidate_index, twin in selected
    ]
    mismatch = abs(len(records) - len(expected)) + sum(
        actual != replayed
        for actual, replayed in zip(records, expected)
    )
    return tuple(twin for _index, twin in selected), mismatch


def _replay_corrected_held_out(
    package_path: Path,
) -> tuple[tuple[RelationalTwin, ...], int]:
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("corrected Ticket 39 package must be a JSON object")
    package = cast(dict[str, object], payload)
    candidates = build_unconditioned_relational_candidates(
        candidates_per_family=cast(
            int, TICKET_39_PROTOCOL["candidate_pool_per_family"]
        ),
        seed=cast(int, TICKET_39_PROTOCOL["evaluation_seed"]),
        perturbation_seed=cast(int, TICKET_39_PROTOCOL["perturbation_seed"]),
    )
    selected, _cells, insufficient = _select_development_pairs(
        candidates,
        quota_per_family_stratum=cast(
            int, TICKET_39_PROTOCOL["quota_per_family_stratum"]
        ),
    )
    if insufficient:
        raise RuntimeError("corrected Ticket 39 package is infeasible on replay")
    expected = [
        _held_out_record(candidate_index, twin)
        for candidate_index, twin in selected
    ]
    actual = cast(list[dict[str, object]], package.get("evaluation_records", []))
    mismatch = abs(len(actual) - len(expected)) + sum(
        record != replayed for record, replayed in zip(actual, expected)
    )
    return tuple(twin for _index, twin in selected), mismatch


def _package_integrity(records: Sequence[dict[str, object]]) -> dict[str, object]:
    pair_ids = [cast(str, record["pair_id"]) for record in records]
    complete_count = 0
    for record in records:
        states = cast(list[dict[str, object]], record["states"])
        if [state["state_role"] for state in states] == ["before", "after"]:
            complete_count += 1
    unique_pair_count = len(set(pair_ids))
    return {
        "unique_pair_count": unique_pair_count,
        "complete_pair_count": complete_count,
        "valid": (
            unique_pair_count == len(records)
            and complete_count == len(records)
        ),
    }


def _package_payload(
    *,
    status: str,
    formal_protocol_run: bool,
    training_candidate_pool_per_family: int,
    training_pairs_per_family: int,
    training_seed: int,
    development_candidates_per_family: int,
    development_base_state_seed: int,
    development_perturbation_seed: int,
    quota_per_family_stratum: int,
    corrected_held_out_package_path: Path,
    records: list[dict[str, object]],
    training_template_count: int,
    development_template_count: int,
    held_out_template_count: int,
    replay_mismatch: int,
    held_out_replay_mismatch: int,
    training_overlap: int,
    held_out_overlap: int,
    ticket_43_authorized: bool,
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "status": status,
        "ticket": 42,
        "formal_protocol_run": formal_protocol_run,
        "training_provenance": {
            "source_ticket": 38,
            "candidate_pool_per_family": training_candidate_pool_per_family,
            "pairs_per_family": training_pairs_per_family,
            "data_seed": training_seed,
            "selection_order": "first_generated_pairs_per_family",
        },
        "development_provenance": {
            "candidate_pool_per_family": development_candidates_per_family,
            "quota_per_family_stratum": quota_per_family_stratum,
            "base_state_seed": development_base_state_seed,
            "perturbation_seed": development_perturbation_seed,
            "selection_order": (
                "family_then_margin_stratum_then_candidate_generation_order"
            ),
        },
        "prior_held_out_provenance": {
            "source_ticket": 39,
            "protocol_corrected": True,
            "package_path": _repository_relative_path(
                corrected_held_out_package_path
            ),
            "base_state_seed": TICKET_39_PROTOCOL["evaluation_seed"],
            "perturbation_seed": TICKET_39_PROTOCOL["perturbation_seed"],
        },
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
        "development_pair_count": len(records),
        "development_state_count": 2 * len(records),
        "development_records": records,
        "package_replay_mismatch_count": replay_mismatch,
        "corrected_held_out_replay_mismatch_count": held_out_replay_mismatch,
        "training_template_count": training_template_count,
        "development_template_count": development_template_count,
        "prior_held_out_template_count": held_out_template_count,
        "training_development_template_overlap": training_overlap,
        "prior_held_out_development_template_overlap": held_out_overlap,
        "selection_controls": {
            "selection_fields": ["family", "pair_margin"],
            "family_used_for_selection": True,
            "pair_margin_used_for_selection": True,
            "oracle_action_used_for_selection": False,
            "oracle_flip_used_for_selection": False,
            "model_output_used_for_selection": False,
            "ticket_41_classification_used_for_selection": False,
            "oracle_flip_statistics_computed_after_selection": True,
        },
        "controls": {
            "model_training_or_evaluation_run": False,
            "architecture_candidate_selected": False,
            "architecture_gate_reopened": False,
            "production_expert_dataset_written": False,
        },
        "ticket_43_authorized": ticket_43_authorized,
    }


def _repository_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


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
    training_candidate_pool_per_family: int,
    training_pairs_per_family: int,
    training_seed: int,
    development_candidates_per_family: int,
    development_base_state_seed: int,
    development_perturbation_seed: int,
    quota_per_family_stratum: int,
) -> None:
    for name, value in (
        (
            "training_candidate_pool_per_family",
            training_candidate_pool_per_family,
        ),
        ("training_pairs_per_family", training_pairs_per_family),
        (
            "development_candidates_per_family",
            development_candidates_per_family,
        ),
        ("quota_per_family_stratum", quota_per_family_stratum),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if training_candidate_pool_per_family < 2:
        raise ValueError("training_candidate_pool_per_family must be at least two")
    if development_candidates_per_family < 2:
        raise ValueError("development_candidates_per_family must be at least two")
    if training_pairs_per_family > training_candidate_pool_per_family:
        raise ValueError(
            "training_pairs_per_family must not exceed the training candidate pool"
        )
    for name, value in (
        ("training_seed", training_seed),
        ("development_base_state_seed", development_base_state_seed),
        ("development_perturbation_seed", development_perturbation_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if len(
        {
            training_seed,
            development_base_state_seed,
            development_perturbation_seed,
        }
    ) != 3:
        raise ValueError("training, base-state, and perturbation seeds must be unique")


def render_context_ablation_development_report(
    summary: dict[str, object],
) -> str:
    cells = cast(list[dict[str, object]], summary["cells"])
    lines = [
        "# Ticket 42 Context Ablation Development Package",
        "",
        f"Status: **{summary['status']}**.",
        f"Protocol status: **{summary['protocol_status']}**.",
        "",
        "## Stratum Supply And Selection",
        "",
        "| Family | Pair-margin stratum | Available | Selected | Flip after selection | No flip after selection |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['family']} | {row['pair_margin_stratum']} | "
            f"{row['available_candidate_count']} | {row['selected_pair_count']} | "
            f"{row['selected_oracle_flip_pair_count']} | "
            f"{row['selected_oracle_no_flip_pair_count']} |"
        )
    lines.extend(
        [
            "",
            "## Package Checks",
            "",
            f"Development pairs: **{summary['development_pair_count']}**.",
            f"Development states: **{summary['development_state_count']}**.",
            f"Package replay mismatch: **{summary['package_replay_mismatch_count']}**.",
            f"Training/development template overlap: **{summary['training_development_template_overlap']}**.",
            f"Prior held-out/development template overlap: **{summary['prior_held_out_development_template_overlap']}**.",
            "",
            "## Authorization",
            "",
            f"Ticket 43 authorized: **{summary['ticket_43_authorized']}**.",
            cast(str, summary["next_step"]),
            "",
            "## Scope",
            "",
            "This ticket generated development data only. It ran no model, selected no architecture, generated no production expert dataset, and did not reopen an architecture gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze Ticket 42 context-ablation development package"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_context_ablation_development_package(args.output_dir)
    print(result.summary_path)
    if result.package_path is not None:
        print(result.package_path)
    print(result.markdown_path)


if __name__ == "__main__":
    main()
