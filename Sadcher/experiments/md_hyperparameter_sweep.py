"""Controlled single-device hyperparameter sweep over the frozen Ticket 42 package.

This runner deliberately evaluates only the frozen development package. It does
not construct, read, or authorize any held-out evaluation artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast

import torch

from experiments.md_context_ablation_development_package import (
    _replay_development_records,
    _template_signature,
)
from experiments.md_context_ablation_relational_data import (
    RelationalState as DevelopmentRelationalState,
    RelationalTwin as DevelopmentRelationalTwin,
)
from experiments.md_flip_curriculum_train_diagnostic import (
    _capture_checkpoint,


    _prepare_batch,

    _train_epoch,
)
from experiments.md_policy_relational_gate import (

    RelationalState,
    RelationalTwin,
)
from experiments.md_task_process_context_ablation import (
    DEVELOPMENT_PACKAGE,
    _development_prediction_records,
    _grouped_metrics,
    _load_development_package,
)
from experiments.md_task_process_context_models import build_context_ablation_model
from experiments.md_train_only_optimization_diagnostic import (
    _select_first_pairs_per_family,
)
from experiments.md_unconditioned_relational_audit import (
    build_unconditioned_relational_candidates,
)
from models.md_enhanced_policy import MDEnhancedSchedulerNetwork


SWEEP_CONFIGS: Final[dict[str, dict[str, object]]] = {
    "C0": {"pretrain_learning_rate": 0.01, "fine_tune_learning_rate": 0.002, "weight_decay": 0.0, "pairs_per_family_per_batch": 100, "pretrain_epochs": 200, "fine_tune_epochs": 100, "scheduler": "none"},
    "C1": {"pretrain_learning_rate": 0.005, "fine_tune_learning_rate": 0.001, "weight_decay": 0.0, "pairs_per_family_per_batch": 100, "pretrain_epochs": 200, "fine_tune_epochs": 100, "scheduler": "none"},
    "C2": {"pretrain_learning_rate": 0.02, "fine_tune_learning_rate": 0.004, "weight_decay": 0.0, "pairs_per_family_per_batch": 100, "pretrain_epochs": 200, "fine_tune_epochs": 100, "scheduler": "none"},
    "C3": {"pretrain_learning_rate": 0.01, "fine_tune_learning_rate": 0.002, "weight_decay": 1e-4, "pairs_per_family_per_batch": 100, "pretrain_epochs": 200, "fine_tune_epochs": 100, "scheduler": "none"},
    "C4": {"pretrain_learning_rate": 0.01, "fine_tune_learning_rate": 0.002, "weight_decay": 1e-3, "pairs_per_family_per_batch": 100, "pretrain_epochs": 200, "fine_tune_epochs": 100, "scheduler": "none"},
    "C5": {"pretrain_learning_rate": 0.01, "fine_tune_learning_rate": 0.002, "weight_decay": 0.0, "pairs_per_family_per_batch": 50, "pretrain_epochs": 200, "fine_tune_epochs": 100, "scheduler": "none"},
    "C6": {"pretrain_learning_rate": 0.01, "fine_tune_learning_rate": 0.002, "weight_decay": 0.0, "pairs_per_family_per_batch": 250, "pretrain_epochs": 200, "fine_tune_epochs": 100, "scheduler": "none"},
    "C7": {"pretrain_learning_rate": 0.01, "fine_tune_learning_rate": 0.002, "weight_decay": 1e-4, "pairs_per_family_per_batch": 100, "pretrain_epochs": 300, "fine_tune_epochs": 150, "scheduler": "cosine"},
}
DEFAULT_PACKAGE = DEVELOPMENT_PACKAGE
TRAIN_PAIRS_PER_FAMILY: Final = 500
DATA_SEED: Final = 3030
CHECKPOINT_INTERVAL: Final = 10
MINIMUM_OVERALL_TRAIN_AGREEMENT: Final = 0.70
MINIMUM_FAMILY_FLIP_PAIR_EXACT: Final = 0.60
MAXIMUM_RESIDUAL_SATURATION_RATE: Final = 0.25
SATURATION_LOSS_WEIGHT: Final = 0.1
SATURATION_PENALTY_ALLOWANCE: Final = 0.25


@dataclass(frozen=True, slots=True)
class SweepResult:
    summary_path: Path
    report_path: Path


def _resolve_device(device: str | torch.device) -> tuple[str, torch.device]:
    requested = str(device).strip().lower()
    if "," in requested or any(char.isspace() for char in requested):
        raise ValueError("device must identify exactly one device")
    parsed = torch.device(requested)
    if parsed.type != "cuda":
        raise ValueError("sweep runs require a single CUDA device")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    index = 0 if parsed.index is None else parsed.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable")
    return requested, torch.device("cuda", index)


def _runtime(requested: str, device: torch.device) -> dict[str, object]:
    return {
        "requested_device": requested,
        "resolved_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "single_device_execution": True,
        "ddp": False,
        "data_parallel": False,
    }


def _validate_config(name: str, config: dict[str, object]) -> None:
    if name not in SWEEP_CONFIGS:
        raise ValueError(f"unknown sweep config: {name}")
    for key in ("pretrain_learning_rate", "fine_tune_learning_rate", "weight_decay"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    for key in ("pairs_per_family_per_batch", "pretrain_epochs", "fine_tune_epochs"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if TRAIN_PAIRS_PER_FAMILY % cast(int, config["pairs_per_family_per_batch"]):
        raise ValueError("pairs_per_family_per_batch must divide 500")
    if config["scheduler"] not in ("none", "cosine"):
        raise ValueError("scheduler must be none or cosine")


def _optimizer(model: MDEnhancedSchedulerNetwork, *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )



def _scheduler(optimizer: torch.optim.Optimizer, *, name: str, epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError("scheduler must be none or cosine")


def _train(
    model: MDEnhancedSchedulerNetwork,
    batches: Sequence[object],
    train_twins: Sequence[RelationalState],
    train_states: Sequence[RelationalState],
    *,
    config: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch_log: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    pretrain_epochs = cast(int, config["pretrain_epochs"])
    fine_tune_epochs = cast(int, config["fine_tune_epochs"])
    scheduler_name = cast(str, config["scheduler"])
    weight_decay = cast(float, config["weight_decay"])

    def run_stage(stage: str, epochs: int, lr: float, *, pair_weight: float, flip_weight: float) -> None:
        optimizer = _optimizer(model, lr=lr, weight_decay=weight_decay)
        scheduler = _scheduler(optimizer, name=scheduler_name, epochs=epochs)
        for epoch in range(1, epochs + 1):
            losses = _train_epoch(
                model, batches, optimizer,
                state_margin=0.1,
                flip_state_margin=0.1,
                pair_margin=0.1,
                flip_state_loss_weight=flip_weight,
                pair_loss_weight=pair_weight,
                saturation_loss_weight=SATURATION_LOSS_WEIGHT,
                saturation_penalty_allowance=SATURATION_PENALTY_ALLOWANCE,
            )
            if scheduler is not None:
                scheduler.step()
            epoch_log.append({"stage": stage, "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], **losses})
            if stage == "fine_tune" and (epoch % CHECKPOINT_INTERVAL == 0 or epoch == fine_tune_epochs):
                snapshot = _capture_checkpoint(
                    model,
                    cast(Sequence[RelationalState], train_twins),
                    train_states,
                    fine_tune_epoch=epoch,
                    minimum_family_flip_pair_exact=MINIMUM_FAMILY_FLIP_PAIR_EXACT,
                )
                checkpoints.append({"epoch": epoch, "state_dict": snapshot.state_dict, "overall_train_agreement": snapshot.overall_train_agreement, "residual_saturation_rate": snapshot.residual_saturation_rate, "oracle_flip_pair_exact_by_family": snapshot.family_exact})

    run_stage("pretrain", pretrain_epochs, cast(float, config["pretrain_learning_rate"]), pair_weight=0.0, flip_weight=0.0)
    initial = _capture_checkpoint(
        model,
        cast(Sequence[RelationalState], train_twins),
        train_states,
        fine_tune_epoch=0,
        minimum_family_flip_pair_exact=MINIMUM_FAMILY_FLIP_PAIR_EXACT,
    )
    checkpoints.append({"epoch": 0, "state_dict": initial.state_dict, "overall_train_agreement": initial.overall_train_agreement, "residual_saturation_rate": initial.residual_saturation_rate, "oracle_flip_pair_exact_by_family": initial.family_exact})
    run_stage("fine_tune", fine_tune_epochs, cast(float, config["fine_tune_learning_rate"]), pair_weight=0.25, flip_weight=0.5)

    def score(row: dict[str, object]) -> tuple[bool, int, float, float, int]:
        exact = [cast(float, value["exact_pair_accuracy"]) for value in cast(dict[str, dict[str, object]], row["oracle_flip_pair_exact_by_family"]).values() if value["exact_pair_accuracy"] is not None]
        return (
            cast(float, row["overall_train_agreement"]) >= MINIMUM_OVERALL_TRAIN_AGREEMENT and cast(float, row["residual_saturation_rate"]) <= MAXIMUM_RESIDUAL_SATURATION_RATE,
            sum(value >= MINIMUM_FAMILY_FLIP_PAIR_EXACT for value in exact),
            min(exact, default=-1.0),
            cast(float, row["overall_train_agreement"]),
            -cast(int, row["epoch"]),
        )

    selected = max(checkpoints, key=score)
    model.load_state_dict(cast(dict[str, torch.Tensor], selected["state_dict"]))
    best_path = checkpoint_dir / "best_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "selected_checkpoint": {key: value for key, value in selected.items() if key != "state_dict"}, "config": config}, best_path)
    reloaded = torch.load(best_path, map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(cast(dict[str, torch.Tensor], reloaded["model_state_dict"]))
    (output_dir / "epoch_log.json").write_text(json.dumps(epoch_log, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    criteria = {
        "overall_train_agreement_met": cast(float, selected["overall_train_agreement"]) >= MINIMUM_OVERALL_TRAIN_AGREEMENT,
        "all_family_flip_pair_exact_met": all(cast(float, value["exact_pair_accuracy"]) >= MINIMUM_FAMILY_FLIP_PAIR_EXACT for value in cast(dict[str, dict[str, object]], selected["oracle_flip_pair_exact_by_family"]).values() if value["exact_pair_accuracy"] is not None),
        "residual_saturation_rate_met": cast(float, selected["residual_saturation_rate"]) <= MAXIMUM_RESIDUAL_SATURATION_RATE,
    }
    criteria["all_training_conditions_met"] = all(criteria.values())
    return {
        "selected_fine_tune_epoch": selected["epoch"],
        "overall_train_agreement": selected["overall_train_agreement"],
        "residual_saturation_rate": selected["residual_saturation_rate"],
        "oracle_flip_pair_exact_by_family": selected["oracle_flip_pair_exact_by_family"],
        "criteria": criteria,
        "checkpoint_metrics": [{key: value for key, value in row.items() if key != "state_dict"} for row in checkpoints],
        "epoch_count": len(epoch_log),
        "best_checkpoint": str(best_path.relative_to(output_dir)),
        "checkpoint_reload_verified": True,
    }


def _development_metrics(model: MDEnhancedSchedulerNetwork, twins: Sequence[DevelopmentRelationalTwin]) -> dict[str, object]:
    states = cast(Sequence[DevelopmentRelationalState], tuple(state for twin in twins for state in (twin.before, twin.after)))
    records = _development_prediction_records(model, twins, states)
    grouped = _grouped_metrics(records)
    overall = cast(dict[str, object], grouped["overall"])
    pickup = cast(dict[str, dict[str, object]], grouped["by_family"])["alternative_task_pickup"]
    return {
        "overall_exact_pair_accuracy": overall["exact_pair_accuracy"],
        "overall_state_agreement": overall["state_agreement"],
        "pickup_exact_pair_accuracy": pickup["exact_pair_accuracy"],
        "pickup_state_agreement": pickup["state_agreement"],
        "residual_saturation_rate": overall["mean_residual_saturation_rate"],
        "grouped": grouped,
        "records": records,
    }


def run_single(
    output_dir: str | Path,
    *,
    config_name: str,
    model_seed: int,
    device: str | torch.device,
    package_path: str | Path = DEFAULT_PACKAGE,
    smoke: bool = False,
) -> SweepResult:
    config = dict(SWEEP_CONFIGS[config_name])
    if smoke:
        config.update({"pairs_per_family_per_batch": 500, "pretrain_epochs": 1, "fine_tune_epochs": 1})
    _validate_config(config_name, config)
    requested, runtime_device = _resolve_device(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    package = _load_development_package(Path(package_path))
    provenance = cast(dict[str, object], package["development_provenance"])
    records = cast(list[dict[str, object]], package["development_records"])
    dev_twins, replay_mismatch = _replay_development_records(
        records,
        development_candidates_per_family=cast(int, provenance["candidate_pool_per_family"]),
        development_base_state_seed=cast(int, provenance["base_state_seed"]),
        development_perturbation_seed=cast(int, provenance["perturbation_seed"]),
        quota_per_family_stratum=cast(int, provenance["quota_per_family_stratum"]),
    )
    train_candidates = build_unconditioned_relational_candidates(candidates_per_family=5000, seed=DATA_SEED)
    train_twins = _select_first_pairs_per_family(train_candidates, pairs_per_family=TRAIN_PAIRS_PER_FAMILY)
    training_templates = {_template_signature(twin) for twin in train_twins}
    development_templates = {_template_signature(twin) for twin in dev_twins}
    if replay_mismatch or training_templates & development_templates:
        raise RuntimeError("Ticket 42 replay or training/development overlap check failed")
    train_states = tuple(state for twin in train_twins for state in (twin.before, twin.after))
    from experiments.md_pair_structured_train_diagnostic import _family_balanced_batches
    batches = tuple(_prepare_batch(batch, device=runtime_device) for batch in _family_balanced_batches(train_twins, pairs_per_family_per_batch=cast(int, config["pairs_per_family_per_batch"])))
    model = cast(MDEnhancedSchedulerNetwork, build_context_ablation_model("current_pair_aware", seed=model_seed, device=runtime_device))
    training = _train(model, batches, cast(Sequence[RelationalState], train_twins), train_states, config=config, output_dir=destination)
    metrics = _development_metrics(model, dev_twins)
    summary = {
        "schema_version": "1.0.0",
        "status": "completed",
        "stage": "single_run",
        "config_name": config_name,
        "config": config,
        "model_seed": model_seed,
        "data_seed": DATA_SEED,
        "train_pairs_per_family": TRAIN_PAIRS_PER_FAMILY,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "ticket_42_package_path": str(Path(package_path).resolve()),
        "training_development_template_overlap": len(training_templates & development_templates),
        "package_replay_mismatch_count": replay_mismatch,
        "runtime": _runtime(requested, runtime_device),
        "training": training,
        "development": metrics,
        "controls": {"heldout_data_read": False, "ticket_44_executed": False, "architecture_modified": False, "preprocessing_modified": False, "loss_definition_modified": False, "split_modified": False, "single_device_execution": True},
    }
    summary_path = destination / "run.json"
    report_path = destination / "run.md"
    summary_path.write_text(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_render_run_report(summary), encoding="utf-8")
    (destination / "config.json").write_text(json.dumps({"config": config, "model_seed": model_seed, "data_seed": DATA_SEED}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SweepResult(summary_path, report_path)


def _render_run_report(summary: dict[str, object]) -> str:
    config = cast(dict[str, object], summary["config"])
    development = cast(dict[str, object], summary["development"])
    training = cast(dict[str, object], summary["training"])
    return "\n".join([
        f"# Controlled Hyperparameter Run {summary['config_name']}", "",
        f"Status: **{summary['status']}**.",
        f"Device: **{summary['runtime']['resolved_device']}** (CUDA_VISIBLE_DEVICES={summary['runtime']['cuda_visible_devices']}).",
        f"Model seed: **{summary['model_seed']}**.", "",
        "| Parameter | Value |", "|---|---:|",
        *[f"| {key} | {value} |" for key, value in config.items()], "",
        f"Best checkpoint: **{training['best_checkpoint']}**.",
        f"Training valid: **{str(training['criteria']['all_training_conditions_met']).lower()}**.",
        f"Training/development template overlap: **{summary['training_development_template_overlap']}**.", "",
        "## Development Metrics", "",
        f"Pickup exact pair accuracy: **{development['pickup_exact_pair_accuracy']:.6f}**.",
        f"Overall exact pair accuracy: **{development['overall_exact_pair_accuracy']:.6f}**.",
        f"Residual saturation: **{development['residual_saturation_rate']:.6f}**.", "",
        "Held-out data was not read; Ticket 44 was not executed.", "",
    ])


def _load_runs(root: Path) -> list[dict[str, object]]:
    return [cast(dict[str, object], json.loads(path.read_text(encoding="utf-8"))) for path in sorted(root.glob("*/run.json"))]


def aggregate_stage1(root: str | Path, output_path: str | Path) -> Path:
    runs = _load_runs(Path(root))
    baseline = next(row for row in runs if row["config_name"] == "C0")
    baseline_overall = cast(float, baseline["development"]["overall_exact_pair_accuracy"])
    eligible = [row for row in runs if row["config_name"] != "C0" and cast(bool, row["training"]["criteria"]["all_training_conditions_met"]) and cast(float, row["training"]["residual_saturation_rate"]) <= MAXIMUM_RESIDUAL_SATURATION_RATE and baseline_overall - cast(float, row["development"]["overall_exact_pair_accuracy"]) <= 0.005]
    ranking = sorted(eligible, key=lambda row: (cast(float, row["development"]["pickup_exact_pair_accuracy"]), cast(float, row["development"]["overall_exact_pair_accuracy"])), reverse=True)
    selected = [row["config_name"] for row in ranking[:2]]
    payload = {"stage": "stage1", "run_count": len(runs), "baseline": baseline["config_name"], "ranking": [{"config_name": row["config_name"], "pickup_exact_pair_accuracy": row["development"]["pickup_exact_pair_accuracy"], "overall_exact_pair_accuracy": row["development"]["overall_exact_pair_accuracy"], "training_valid": row["training"]["criteria"]["all_training_conditions_met"], "residual_saturation_rate": row["development"]["residual_saturation_rate"]} for row in ranking], "selected_candidates": selected, "failed_runs": [row["config_name"] for row in runs if row not in ranking and row["config_name"] != "C0"]}
    destination = Path(output_path)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def aggregate_stage2(root: str | Path, stage1_path: str | Path, output_path: str | Path) -> Path:
    runs = _load_runs(Path(root))
    stage1 = cast(dict[str, object], json.loads(Path(stage1_path).read_text(encoding="utf-8")))
    names = ["C0", *cast(list[str], stage1["selected_candidates"])]
    grouped = {name: [row for row in runs if row["config_name"] == name] for name in names}
    baseline = grouped["C0"]
    comparisons = []
    for name in names[1:]:
        pairs = []
        for base, candidate in zip(sorted(baseline, key=lambda row: row["model_seed"]), sorted(grouped[name], key=lambda row: row["model_seed"]), strict=True):
            pairs.append({"seed": base["model_seed"], "pickup_exact_gain": candidate["development"]["pickup_exact_pair_accuracy"] - base["development"]["pickup_exact_pair_accuracy"], "overall_exact_decline": base["development"]["overall_exact_pair_accuracy"] - candidate["development"]["overall_exact_pair_accuracy"], "candidate_residual_saturation": candidate["development"]["residual_saturation_rate"], "training_valid": candidate["training"]["criteria"]["all_training_conditions_met"]})
        mean_pickup = sum(float(row["pickup_exact_gain"]) for row in pairs) / len(pairs)
        mean_decline = sum(float(row["overall_exact_decline"]) for row in pairs) / len(pairs)
        accepted = mean_pickup >= 0.02 and sum(float(row["pickup_exact_gain"]) > 0 for row in pairs) >= 2 and mean_decline <= 0.005 and all(float(row["candidate_residual_saturation"]) <= 0.25 and row["training_valid"] for row in pairs)
        comparisons.append({"candidate": name, "seed_comparisons": pairs, "mean_pickup_exact_gain": mean_pickup, "mean_overall_exact_decline": mean_decline, "accepted": accepted})
    payload = {"stage": "stage2", "run_count": len(runs), "expected_run_count": 9, "comparisons": comparisons, "final_acceptance": any(row["accepted"] for row in comparisons), "ticket_44_executed": False}
    destination = Path(output_path)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = destination.with_suffix(".md")
    lines = ["# Controlled Hyperparameter Sweep", "", f"Stage 2 acceptance: **{str(payload['final_acceptance']).lower()}**.", "", "| Candidate | Mean pickup gain | Mean overall decline | Accepted |", "|---|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['candidate']} | {row['mean_pickup_exact_gain']:.6f} | {row['mean_overall_exact_decline']:.6f} | {str(row['accepted']).lower()} |")
    lines.extend(["", f"Stage 2 runs: **{len(runs)}**; Ticket 44 executed: **false**.", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one controlled Sadcher hyperparameter sweep job")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", choices=sorted(SWEEP_CONFIGS), required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run_single(args.output_dir, config_name=args.config, model_seed=args.model_seed, device=args.device, package_path=args.package, smoke=args.smoke)
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
