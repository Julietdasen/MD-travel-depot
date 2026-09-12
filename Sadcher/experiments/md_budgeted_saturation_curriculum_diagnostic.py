"""Budget-aware residual saturation train-only diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast

from experiments.md_policy_relational_gate import (
    DEFAULT_MODEL_SEEDS,
    _validated_model_seeds,
)
from experiments.md_saturation_controlled_curriculum_diagnostic import (
    FORMAL_PROTOCOL as BASE_FORMAL_PROTOCOL,
)
from experiments.md_saturation_controlled_curriculum_diagnostic import (
    _render_report as _render_saturation_report,
)
from experiments.md_saturation_controlled_curriculum_diagnostic import (
    run_saturation_controlled_curriculum_diagnostic,
)


FORMAL_PROTOCOL: Final = {
    **BASE_FORMAL_PROTOCOL,
    "saturation_penalty_allowance": 0.25,
}


@dataclass(frozen=True, slots=True)
class BudgetedSaturationCurriculumDiagnosticResult:
    summary_path: Path
    report_path: Path


def run_budgeted_saturation_curriculum_diagnostic(
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
    saturation_loss_weight: float = 0.1,
    saturation_penalty_allowance: float = 0.25,
    minimum_overall_train_agreement: float = 0.70,
    minimum_family_flip_pair_exact: float = 0.60,
    maximum_residual_saturation_rate: float = 0.25,
) -> BudgetedSaturationCurriculumDiagnosticResult:
    """Run Ticket 38 without constructing or reading held-out evaluation data."""
    resolved_model_seeds = _validated_model_seeds(model_seeds)
    if (
        isinstance(saturation_penalty_allowance, bool)
        or not isinstance(saturation_penalty_allowance, (int, float))
        or not math.isfinite(saturation_penalty_allowance)
        or not 0 < saturation_penalty_allowance < 1
    ):
        raise ValueError("saturation_penalty_allowance must be within (0, 1)")
    protocol = {
        "candidate_pool_per_family": candidate_pool_per_family,
        "train_pairs_per_family": train_pairs_per_family,
        "pairs_per_family_per_batch": pairs_per_family_per_batch,
        "pretrain_epochs": pretrain_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "checkpoint_interval": checkpoint_interval,
        "data_seed": data_seed,
        "model_seeds": resolved_model_seeds,
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
    destination = Path(output_dir)
    summary_path = destination / "budgeted_saturation_curriculum_diagnostic.json"
    report_path = destination / "budgeted_saturation_curriculum_diagnostic.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite budgeted saturation curriculum diagnostic: "
            + ", ".join(str(path) for path in existing)
        )

    with tempfile.TemporaryDirectory() as temporary:
        base_result = run_saturation_controlled_curriculum_diagnostic(
            Path(temporary) / "budgeted-curriculum",
            candidate_pool_per_family=candidate_pool_per_family,
            train_pairs_per_family=train_pairs_per_family,
            pairs_per_family_per_batch=pairs_per_family_per_batch,
            pretrain_epochs=pretrain_epochs,
            fine_tune_epochs=fine_tune_epochs,
            checkpoint_interval=checkpoint_interval,
            data_seed=data_seed,
            model_seeds=resolved_model_seeds,
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
            maximum_residual_saturation_rate=maximum_residual_saturation_rate,
        )
        summary = cast(
            dict[str, object], json.loads(base_result.summary_path.read_text())
        )

    formal_protocol_run = protocol == FORMAL_PROTOCOL
    candidate_training_conditions_met = cast(
        bool, summary["candidate_training_conditions_met"]
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
    controls = cast(dict[str, object], summary["controls"])
    controls["held_out_stage_authorized"] = all_training_conditions_met
    summary.update(
        {
            "status": "budgeted_saturation_curriculum_diagnostic",
            "ticket": 38,
            "source_ticket": 37,
            "protocol_class": "exploratory_train_only_budgeted_saturation",
            "formal_protocol_run": formal_protocol_run,
            "all_training_conditions_met": all_training_conditions_met,
            "decision": decision,
            "controls": controls,
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
    )
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return BudgetedSaturationCurriculumDiagnosticResult(
        summary_path,
        report_path,
    )


def _render_report(summary: dict[str, object]) -> str:
    report = _render_saturation_report(summary)
    return report.replace(
        "# Ticket 37 Saturation-Controlled Curriculum Diagnostic",
        "# Ticket 38 Budgeted Saturation Curriculum Diagnostic",
        1,
    ).replace(
        "Two-stage relational curriculum with raw-residual saturation control.",
        "Two-stage relational curriculum with a 25% residual saturation budget.",
        1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ticket 38 budgeted saturation train-only curriculum"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_budgeted_saturation_curriculum_diagnostic(args.output_dir)
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
