"""Unconditioned relational candidate audit for Ticket 30."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

from experiments.md_relational_data_audit import (
    _compare_distributions,
    _feature_extractors,
    _feature_values,
)
from experiments.md_policy_relational_gate import (
    RELATIONAL_FAMILIES,
    RelationalTwin,
    _random_state_spec,
    _StateSpec,
    _relational_intervention,
    _split_twins,
    _state_completion_margin,
    _state_from_spec,
    _template_signature,
)


PAIR_MARGIN_STRATA = (
    ("near_tie_lt_0.01", 0.0, 0.01),
    ("small_ge_0.01_lt_0.05", 0.01, 0.05),
    ("moderate_ge_0.05_lt_0.10", 0.05, 0.10),
    ("clear_ge_0.10", 0.10, math.inf),
)


@dataclass(frozen=True, slots=True)
class UnconditionedCandidateAuditResult:
    summary_path: Path
    report_path: Path


def build_unconditioned_relational_candidates(
    *,
    candidates_per_family: int,
    seed: int,
    perturbation_seed: int | None = None,
) -> tuple[RelationalTwin, ...]:
    """Generate structurally valid pairs before observing oracle outcomes."""
    twins, _attempt_count = _generate_candidates(
        candidates_per_family=candidates_per_family,
        seed=seed,
        perturbation_seed=perturbation_seed,
    )
    return twins


def run_unconditioned_relational_candidate_audit(
    output_dir: str | Path,
    *,
    candidates_per_family: int = 5000,
    seed: int = 3030,
) -> UnconditionedCandidateAuditResult:
    """Write the deterministic Ticket 30 candidate supply audit."""
    _validate_generation_arguments(candidates_per_family, seed)
    destination = Path(output_dir)
    summary_path = destination / "unconditioned_candidate_audit.json"
    report_path = destination / "unconditioned_candidate_audit.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite candidate audit: "
            + ", ".join(str(path) for path in existing)
        )

    twins, attempt_count = _generate_candidates(
        candidates_per_family=candidates_per_family,
        seed=seed,
    )
    train_twins, evaluation_twins = _split_twins(twins, candidates_per_family)
    train_templates = {_template_signature(twin) for twin in train_twins}
    evaluation_templates = {
        _template_signature(twin) for twin in evaluation_twins
    }
    overlap = train_templates & evaluation_templates
    if overlap:
        raise RuntimeError("train/evaluation candidate templates overlap")

    train_states = tuple(
        state for twin in train_twins for state in (twin.before, twin.after)
    )
    evaluation_states = tuple(
        state
        for twin in evaluation_twins
        for state in (twin.before, twin.after)
    )
    family_statistics = {
        family: _family_statistics(twins, family)
        for family in RELATIONAL_FAMILIES
    }
    distribution_comparison = {
        name: _compare_distributions(
            _feature_values(train_states, extractor),
            _feature_values(evaluation_states, extractor),
        )
        for name, extractor in _feature_extractors().items()
    }
    candidate_records = [_candidate_record(twin) for twin in twins]
    no_flip_count = sum(
        not cast(bool, record["oracle_action_flipped"])
        for record in candidate_records
    )
    clear_counts = []
    for row in family_statistics.values():
        strata = cast(dict[str, dict[str, int | float]], row["margin_strata"])
        clear_counts.append(cast(int, strata["clear_ge_0.10"]["pair_count"]))
    uniform_clear_quota = min(clear_counts)

    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "ticket_30_unconditioned_relational_candidate_audit",
        "ticket": 30,
        "data_seed": seed,
        "candidates_per_family": candidates_per_family,
        "candidate_pair_count": len(twins),
        "oracle_flip_pair_count": len(twins) - no_flip_count,
        "oracle_no_flip_pair_count": no_flip_count,
        "natural_oracle_flip_rate": (len(twins) - no_flip_count) / len(twins),
        "generation": {
            "attempt_count": attempt_count,
            "accepted_candidate_count": len(twins),
            "structural_acceptance_rate": len(twins) / attempt_count,
        },
        "train_pair_count": len(train_twins),
        "evaluation_pair_count": len(evaluation_twins),
        "pair_margin_definition": "min(before_margin, after_margin)",
        "pair_margin_strata": [
            {
                "name": name,
                "lower": lower,
                "upper": None if math.isinf(upper) else upper,
            }
            for name, lower, upper in PAIR_MARGIN_STRATA
        ],
        "family_statistics": family_statistics,
        "candidate_records": candidate_records,
        "train_evaluation_distribution_comparison": distribution_comparison,
        "selection_controls": {
            "label_used_for_acceptance": False,
            "oracle_flip_used_for_acceptance": False,
            "margin_used_for_acceptance": False,
            "model_output_used_for_acceptance": False,
            "allowed_acceptance_inputs": [
                "structural_validity",
                "intervention_field",
                "template_uniqueness",
            ],
        },
        "ticket_31_supply": {
            "clear_ge_0.10_count_by_family": dict(
                zip(RELATIONAL_FAMILIES, clear_counts, strict=True)
            ),
            "maximum_uniform_clear_quota": uniform_clear_quota,
            "four_strata_protocol_feasible": uniform_clear_quota > 0,
        },
        "controls": {
            "unique_template_count": len(
                {_template_signature(twin) for twin in twins}
            ),
            "train_evaluation_template_overlap": len(overlap),
            "production_expert_dataset_written": False,
            "historical_ticket_29_generator_modified": False,
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return UnconditionedCandidateAuditResult(summary_path, report_path)


def pair_margin_stratum(margin: float) -> str:
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("pair margin must be finite and non-negative")
    for name, lower, upper in PAIR_MARGIN_STRATA:
        if lower <= margin < upper:
            return name
    raise RuntimeError("pair margin strata do not cover the supplied value")


def _generate_candidates(
    *,
    candidates_per_family: int,
    seed: int,
    perturbation_seed: int | None = None,
) -> tuple[tuple[RelationalTwin, ...], int]:
    _validate_generation_arguments(
        candidates_per_family, seed, perturbation_seed
    )
    state_generator = random.Random(seed)
    intervention_generator = (
        state_generator
        if perturbation_seed is None
        else random.Random(perturbation_seed)
    )
    twins: list[RelationalTwin] = []
    signatures: set[str] = set()
    attempt_count = 0
    for family in RELATIONAL_FAMILIES:
        for index in range(candidates_per_family):
            pair_id = f"unconditioned-{family}-{index:06d}"
            for _attempt in range(100):
                attempt_count += 1
                before_spec = _random_state_spec(state_generator)
                changed_index, after_spec = _relational_intervention(
                    family, before_spec, intervention_generator
                )
                signature = _structural_template_signature(
                    family,
                    changed_index,
                    before_spec,
                    after_spec,
                )
                if signature in signatures:
                    continue
                twin = RelationalTwin(
                    family=family,
                    pair_id=pair_id,
                    changed_entity_index=changed_index,
                    before=_state_from_spec(
                        state_id=f"{pair_id}-before",
                        pair_id=pair_id,
                        family=family,
                        spec=before_spec,
                    ),
                    after=_state_from_spec(
                        state_id=f"{pair_id}-after",
                        pair_id=pair_id,
                        family=family,
                        spec=after_spec,
                    ),
                )
                twins.append(twin)
                signatures.add(signature)
                break
            else:
                raise RuntimeError(f"could not build unique candidate {pair_id}")
    return tuple(twins), attempt_count



def _structural_template_signature(
    family: str,
    changed_index: int,
    before: _StateSpec,
    after: _StateSpec,
) -> str:
    def payload(spec: _StateSpec) -> dict[str, object]:
        return {
            "robot_positions": spec.robot_positions,
            "robot_loaded_speeds": spec.robot_loaded_speeds,
            "task_pickups": spec.task_pickups,
            "task_deliveries": spec.task_deliveries,
            "downstream_priorities": spec.downstream_priorities,
        }

    return json.dumps(
        {
            "family": family,
            "changed_entity_index": changed_index,
            "before": payload(before),
            "after": payload(after),
        },
        sort_keys=True,
        separators=(",", ":"),
    )

def _validate_generation_arguments(
    candidates_per_family: int,
    seed: int,
    perturbation_seed: int | None = None,
) -> None:
    if (
        isinstance(candidates_per_family, bool)
        or not isinstance(candidates_per_family, int)
        or candidates_per_family < 2
    ):
        raise ValueError("candidates_per_family must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if perturbation_seed is not None and (
        isinstance(perturbation_seed, bool)
        or not isinstance(perturbation_seed, int)
        or perturbation_seed < 0
    ):
        raise ValueError("perturbation_seed must be a non-negative integer")


def _pair_margin(twin: RelationalTwin) -> float:
    return min(
        _state_completion_margin(twin.before),
        _state_completion_margin(twin.after),
    )


def _candidate_record(twin: RelationalTwin) -> dict[str, object]:
    before_margin = _state_completion_margin(twin.before)
    after_margin = _state_completion_margin(twin.after)
    pair_margin = min(before_margin, after_margin)
    return {
        "pair_id": twin.pair_id,
        "family": twin.family,
        "changed_entity_index": twin.changed_entity_index,
        "before_oracle_action": twin.before.oracle_action,
        "after_oracle_action": twin.after.oracle_action,
        "oracle_action_flipped": twin.before.oracle_action != twin.after.oracle_action,
        "before_oracle_completion_margin": before_margin,
        "after_oracle_completion_margin": after_margin,
        "pair_margin": pair_margin,
        "pair_margin_stratum": pair_margin_stratum(pair_margin),
    }


def _family_statistics(
    twins: Sequence[RelationalTwin], family: str
) -> dict[str, object]:
    selected = [twin for twin in twins if twin.family == family]
    strata: dict[str, dict[str, int | float]] = {}
    for name, _lower, _upper in PAIR_MARGIN_STRATA:
        rows = [twin for twin in selected if pair_margin_stratum(_pair_margin(twin)) == name]
        flips = sum(twin.before.oracle_action != twin.after.oracle_action for twin in rows)
        strata[name] = {
            "pair_count": len(rows),
            "pair_fraction": len(rows) / len(selected),
            "flip_pair_count": flips,
            "no_flip_pair_count": len(rows) - flips,
            "natural_flip_rate": flips / len(rows) if rows else 0.0,
        }
    total_flips = sum(
        twin.before.oracle_action != twin.after.oracle_action for twin in selected
    )
    return {
        "candidate_pair_count": len(selected),
        "oracle_flip_pair_count": total_flips,
        "oracle_no_flip_pair_count": len(selected) - total_flips,
        "natural_oracle_flip_rate": total_flips / len(selected),
        "margin_strata": strata,
    }


def _render_report(summary: dict[str, object]) -> str:
    family_statistics = cast(
        dict[str, dict[str, object]], summary["family_statistics"]
    )
    supply = cast(dict[str, object], summary["ticket_31_supply"])
    clear_counts = cast(dict[str, int], supply["clear_ge_0.10_count_by_family"])
    lines = [
        "# Ticket 30 Unconditioned Relational Candidate Audit",
        "",
        "Candidates are retained before oracle labels, flips, margins, or model outputs are observed. This audit does not create a Ticket 20 production expert dataset.",
        "",
        f"Candidate pairs: **{summary['candidate_pair_count']}**.",
        f"Natural oracle flip rate: **{cast(float, summary['natural_oracle_flip_rate']):.4f}**.",
        "",
        "| Family | Candidates | Natural flip rate | <0.01 | [0.01,0.05) | [0.05,0.10) | >=0.10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    stratum_names = [name for name, _lower, _upper in PAIR_MARGIN_STRATA]
    for family in RELATIONAL_FAMILIES:
        row = family_statistics[family]
        strata = cast(dict[str, dict[str, int | float]], row["margin_strata"])
        counts = [cast(int, strata[name]["pair_count"]) for name in stratum_names]
        lines.append(
            f"| {family} | {row['candidate_pair_count']} | "
            f"{cast(float, row['natural_oracle_flip_rate']):.4f} | "
            + " | ".join(str(count) for count in counts)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Ticket 31 Supply Decision",
            "",
            "Clear (`pair_margin >=0.10`) counts by family: "
            + ", ".join(
                f"{family}={clear_counts[family]}" for family in RELATIONAL_FAMILIES
            )
            + ".",
            f"Maximum uniform clear-stratum quota: **{supply['maximum_uniform_clear_quota']}**.",
            f"Four-strata protocol feasible: **{str(supply['four_strata_protocol_feasible']).lower()}**.",
            "",
            "Ticket 31 must report infeasible rather than lower the frozen >=0.10 boundary when the uniform quota is zero.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Ticket 30 unconditioned relational candidate audit"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidates-per-family", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3030)
    args = parser.parse_args()
    result = run_unconditioned_relational_candidate_audit(
        args.output_dir,
        candidates_per_family=args.candidates_per_family,
        seed=args.seed,
    )
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
