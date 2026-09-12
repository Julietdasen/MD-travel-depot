"""Frozen margin-stratified relational evaluation protocol for Ticket 31."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from experiments.md_unconditioned_relational_audit import PAIR_MARGIN_STRATA
from experiments.md_policy_relational_gate import RELATIONAL_FAMILIES


@dataclass(frozen=True, slots=True)
class FrozenRelationalEvaluationResult:
    summary_path: Path
    report_path: Path
    package_path: Path | None


def run_frozen_margin_stratified_relational_evaluation(
    output_dir: str | Path,
    *,
    candidate_audit_path: str | Path,
    quota_per_family_stratum: int = 1,
    train_seed: int = 3131,
    evaluation_seed: int = 3132,
    perturbation_seed: int = 3133,
) -> FrozenRelationalEvaluationResult:
    """Freeze the protocol or report that its pre-registered cells are infeasible."""
    if (
        isinstance(quota_per_family_stratum, bool)
        or not isinstance(quota_per_family_stratum, int)
        or quota_per_family_stratum <= 0
    ):
        raise ValueError("quota_per_family_stratum must be a positive integer")
    for name, value in (
        ("train_seed", train_seed),
        ("evaluation_seed", evaluation_seed),
        ("perturbation_seed", perturbation_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if len({train_seed, evaluation_seed, perturbation_seed}) != 3:
        raise ValueError("train, evaluation, and perturbation seeds must be unique")

    destination = Path(output_dir)
    summary_path = destination / "frozen_relational_evaluation_summary.json"
    report_path = destination / "frozen_relational_evaluation_report.md"
    package_path = destination / "frozen_relational_evaluation_package.json"
    existing = [
        path for path in (summary_path, report_path, package_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite frozen relational evaluation: "
            + ", ".join(str(path) for path in existing)
        )

    audit = _load_candidate_audit(Path(candidate_audit_path))
    family_statistics = cast(
        dict[str, dict[str, object]], audit["family_statistics"]
    )
    cells: list[dict[str, object]] = []
    insufficient_cells: list[dict[str, object]] = []
    for family in RELATIONAL_FAMILIES:
        strata = cast(
            dict[str, dict[str, int | float]],
            family_statistics[family]["margin_strata"],
        )
        for stratum, _lower, _upper in PAIR_MARGIN_STRATA:
            available = cast(int, strata[stratum]["pair_count"])
            row: dict[str, object] = {
                "family": family,
                "pair_margin_stratum": stratum,
                "available_candidate_count": available,
                "required_evaluation_quota": quota_per_family_stratum,
                "sufficient": available >= quota_per_family_stratum,
            }
            cells.append(row)
            if available < quota_per_family_stratum:
                insufficient_cells.append(dict(row))

    if not insufficient_cells:
        raise RuntimeError(
            "candidate supply is feasible; freeze implementation must be extended "
            "before selecting a package"
        )

    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "ticket_31_frozen_margin_stratified_relational_evaluation",
        "ticket": 31,
        "protocol_status": "infeasible",
        "infeasibility_reason": (
            "At least one pre-registered family x pair-margin stratum has fewer "
            "unconditioned candidates than its fixed evaluation quota."
        ),
        "candidate_audit_provenance": {
            "schema_version": audit["schema_version"],
            "ticket": audit["ticket"],
            "status": audit["status"],
            "data_seed": audit["data_seed"],
            "candidates_per_family": audit["candidates_per_family"],
            "candidate_pair_count": audit["candidate_pair_count"],
        },
        "frozen_protocol": {
            "pair_margin_definition": "min(before_margin, after_margin)",
            "pair_margin_strata": [
                {
                    "name": name,
                    "lower": lower,
                    "upper": None if upper == float("inf") else upper,
                }
                for name, lower, upper in PAIR_MARGIN_STRATA
            ],
            "quota_per_family_stratum": quota_per_family_stratum,
            "train_seed": train_seed,
            "evaluation_seed": evaluation_seed,
            "perturbation_seed": perturbation_seed,
            "insufficient_cell_action": "report_infeasible_and_stop",
            "selection_order": (
                "RELATIONAL_FAMILIES order, then PAIR_MARGIN_STRATA order, "
                "then ascending deterministic candidate index"
            ),
        },
        "cells": cells,
        "insufficient_cells": insufficient_cells,
        "selection_controls": {
            "oracle_action_used": False,
            "oracle_flip_used": False,
            "model_output_used": False,
            "quota_changed_after_supply_check": False,
            "margin_boundary_changed_after_supply_check": False,
        },
        "evaluation_package_written": False,
        "model_training_or_evaluation_run": False,
        "controls": {
            "production_expert_dataset_written": False,
            "ticket_20_authorized": False,
            "ticket_32_unblocked": False,
        },
        "next_decision": (
            "Do not run Ticket 32. Revisit the completion proxy or pre-register a "
            "new margin protocol in a separate ticket; do not lower this protocol's "
            "frozen >=0.10 boundary in place."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return FrozenRelationalEvaluationResult(summary_path, report_path, None)


def _load_candidate_audit(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate audit must contain a JSON object")
    required = {
        "schema_version",
        "status",
        "ticket",
        "data_seed",
        "candidates_per_family",
        "candidate_pair_count",
        "family_statistics",
        "selection_controls",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"candidate audit is missing fields: {missing}")
    if payload["ticket"] != 30 or payload["status"] != (
        "ticket_30_unconditioned_relational_candidate_audit"
    ):
        raise ValueError("candidate audit is not a Ticket 30 audit")
    controls = payload["selection_controls"]
    if not isinstance(controls, dict):
        raise ValueError("candidate audit selection controls must be an object")
    if any(
        controls.get(name) is not False
        for name in (
            "label_used_for_acceptance",
            "oracle_flip_used_for_acceptance",
            "margin_used_for_acceptance",
            "model_output_used_for_acceptance",
        )
    ):
        raise ValueError("candidate audit used a forbidden selection signal")
    return cast(dict[str, object], payload)


def _render_report(summary: dict[str, object]) -> str:
    protocol = cast(dict[str, object], summary["frozen_protocol"])
    insufficient = cast(list[dict[str, object]], summary["insufficient_cells"])
    lines = [
        "# Ticket 31 Frozen Margin-Stratified Relational Evaluation",
        "",
        "## Protocol infeasible",
        "",
        cast(str, summary["infeasibility_reason"]),
        "",
        f"Frozen quota per family x stratum: **{protocol['quota_per_family_stratum']}**.",
        "No evaluation package was written. No model was trained or evaluated, and no Ticket 20 production expert dataset was generated.",
        "",
        "| Family | Pair-margin stratum | Available | Required |",
        "|---|---|---:|---:|",
    ]
    for row in insufficient:
        lines.append(
            f"| {row['family']} | {row['pair_margin_stratum']} | "
            f"{row['available_candidate_count']} | "
            f"{row['required_evaluation_quota']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            cast(str, summary["next_decision"]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or reject the Ticket 31 relational evaluation protocol"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-audit", required=True)
    parser.add_argument("--quota-per-family-stratum", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=3131)
    parser.add_argument("--evaluation-seed", type=int, default=3132)
    parser.add_argument("--perturbation-seed", type=int, default=3133)
    args = parser.parse_args()
    result = run_frozen_margin_stratified_relational_evaluation(
        args.output_dir,
        candidate_audit_path=args.candidate_audit,
        quota_per_family_stratum=args.quota_per_family_stratum,
        train_seed=args.train_seed,
        evaluation_seed=args.evaluation_seed,
        perturbation_seed=args.perturbation_seed,
    )
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
