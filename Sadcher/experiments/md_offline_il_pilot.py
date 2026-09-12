"""Reproducible Ticket 23 offline-IL pilot and lightweight visual report."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data_generation.md_expert_dataset import MDExpertDatasetLoader
from experiments.protocol import DatasetSplit
from imitation_learning.md_train import (
    MDILTrainingConfig,
    load_md_policy_checkpoint,
    train_md_policy,
)
from models.md_legacy_features import legacy_policy_inputs_from_samples
from models.md_policy_features import md_policy_inputs_from_samples


PILOT_SCHEMA_VERSION = "1.0.0"
PILOT_TOPIC = "md_offline_il_pilot"


def run_md_offline_il_pilot(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    config: MDILTrainingConfig,
    run_date: str | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Path]:
    """Train from stored expert samples and write an auditable pilot package."""

    observed_date = date.today().isoformat() if run_date is None else run_date
    date.fromisoformat(observed_date)
    destination = Path(output_dir)
    training_dir = destination / "training"
    destination.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    result = train_md_policy(
        dataset_root,
        training_dir,
        config=config,
        device=device,
    )
    elapsed_seconds = time.perf_counter() - started
    training_summary = _load_mapping(result.summary_path)
    if training_summary.get("solver_calls_during_training") != 0:
        raise RuntimeError("offline IL pilot unexpectedly recorded solver calls")

    loader = MDExpertDatasetLoader(dataset_root)
    records = loader.load_records()
    split_records = {
        split.value: tuple(record for record in records if record.split is split)
        for split in DatasetSplit
    }
    record_counts = {
        split: len(split_values) for split, split_values in split_records.items()
    }
    sample_counts = {
        split: sum(len(record.samples) for record in split_values)
        for split, split_values in split_records.items()
    }
    validation = _evaluate_validation_checkpoint(
        result.checkpoint_path,
        loader,
        device=device,
    )
    all_losses = (*result.train_losses, *result.validation_losses)
    if not all(math.isfinite(value) for value in all_losses):
        raise RuntimeError("offline IL pilot recorded non-finite losses")

    paths = {
        "summary": destination / "pilot_summary.json",
        "report": destination / "RESULTS.md",
        "training_loss_plot": destination / "training_loss.png",
        "dataset_split_plot": destination / "dataset_split.png",
        "checkpoint": result.checkpoint_path,
        "training_summary": result.summary_path,
    }
    summary = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "pilot",
        "ticket": 23,
        "topic": PILOT_TOPIC,
        "run_date": observed_date,
        "dataset": {
            "source": str(dataset_root),
            "record_counts": record_counts,
            "sample_counts": sample_counts,
            "quality_counts": dict(
                sorted(Counter(record.quality.value for record in records).items())
            ),
        },
        "training": {
            "epochs": config.epochs,
            "train_losses": list(result.train_losses),
            "validation_losses": list(result.validation_losses),
            "best_validation_loss": min(result.validation_losses),
            "final_train_loss": result.train_losses[-1],
            "final_validation_loss": result.validation_losses[-1],
            "all_losses_finite": True,
            "solver_calls": 0,
            "elapsed_seconds": elapsed_seconds,
            "device": str(device),
        },
        "validation": validation,
        "artifacts": {
            key: str(path.relative_to(destination))
            for key, path in paths.items()
            if key != "summary"
        },
        "limitations": {
            "formal_benchmark_pending": True,
            "note": (
                "This pilot checks offline-IL trainability only; Tickets 20, 21, "
                "and 28 provide formal scheduling quality and latency evidence."
            ),
        },
    }
    _plot_losses(
        result.train_losses, result.validation_losses, paths["training_loss_plot"]
    )
    _plot_dataset_split(record_counts, sample_counts, paths["dataset_split_plot"])
    paths["report"].write_text(_markdown_report(summary), encoding="utf-8")
    _write_json(paths["summary"], summary)
    return paths


def _evaluate_validation_checkpoint(
    checkpoint_path: Path,
    loader: MDExpertDatasetLoader,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    model, _ = load_md_policy_checkpoint(checkpoint_path, device=device)
    samples = loader.load_samples(split=DatasetSplit.VALIDATION)
    if not samples:
        raise ValueError("offline IL pilot requires validation samples")

    top1_hits = 0
    score_count = 0
    with torch.no_grad():
        for sample in samples:
            batch = (sample,)
            legacy = legacy_policy_inputs_from_samples(batch)
            md_inputs = md_policy_inputs_from_samples(batch)
            scores = model(
                legacy.robot_features.to(device),
                legacy.task_features.to(device),
                legacy.task_adjacency.to(device),
                md_inputs=md_inputs,
            ).cpu()
            if not torch.isfinite(scores).all():
                raise RuntimeError("offline IL checkpoint produced non-finite scores")
            feasible = md_inputs.hard_feasibility_mask.cpu()
            if not torch.any(feasible):
                raise ValueError("validation sample has no feasible expert actions")
            masked = scores.masked_fill(~feasible, float("-inf"))
            selected = int(masked[0].flatten().argmax())
            expert = torch.tensor(sample.expert_assignment, dtype=torch.bool)
            top1_hits += int(expert.flatten()[selected])
            score_count += scores.numel()

    return {
        "evaluated_samples": len(samples),
        "finite_score_count": score_count,
        "all_scores_finite": True,
        "masked_top1_expert_pair_hits": top1_hits,
        "masked_top1_expert_pair_accuracy": top1_hits / len(samples),
        "metric_note": (
            "A hit means the highest-scoring legal robot-task pair belongs to the "
            "expert assignment; it is not decoder-level schedule accuracy."
        ),
    }


def _markdown_report(summary: Mapping[str, Any]) -> str:
    dataset = summary["dataset"]
    training = summary["training"]
    validation = summary["validation"]
    return "\n".join(
        (
            "# Ticket 23 Offline IL Pilot",
            "",
            f"Run date: {summary['run_date']}",
            "",
            (
                "This is a trainability pilot, not a formal benchmark. Formal "
                "scheduling quality and latency evaluation remains in Tickets 20, 21, and 28."
            ),
            "",
            "| Check | Result |",
            "|---|---:|",
            f"| Train records / samples | {dataset['record_counts']['train']} / {dataset['sample_counts']['train']} |",
            f"| Validation records / samples | {dataset['record_counts']['validation']} / {dataset['sample_counts']['validation']} |",
            f"| Epochs | {training['epochs']} |",
            f"| Final train loss | {training['final_train_loss']:.6f} |",
            f"| Best validation loss | {training['best_validation_loss']:.6f} |",
            f"| Masked top-1 expert-pair accuracy | {validation['masked_top1_expert_pair_accuracy']:.2%} |",
            f"| Solver calls during training | {training['solver_calls']} |",
            f"| Training elapsed (s) | {training['elapsed_seconds']:.3f} |",
            "",
            (
                "All recorded losses and validation scores were finite. Training "
                "completed without a non-finite-gradient failure."
            ),
            "",
        )
    )


def _plot_losses(
    train_losses: Sequence[float],
    validation_losses: Sequence[float],
    path: Path,
) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, train_losses, marker="o", color="#2563EB", label="Train")
    axis.plot(
        epochs,
        validation_losses,
        marker="s",
        color="#D97706",
        label="Validation",
    )
    axis.set_title("Ticket 23 Offline IL Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Binary cross-entropy")
    axis.set_xticks(epochs)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_dataset_split(
    record_counts: Mapping[str, int],
    sample_counts: Mapping[str, int],
    path: Path,
) -> None:
    splits = [split.value for split in DatasetSplit]
    labels = [split.title() for split in splits]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].bar(labels, [record_counts[name] for name in splits], color="#287271")
    axes[1].bar(labels, [sample_counts[name] for name in splits], color="#D9822B")
    axes[0].set_title("Expert records")
    axes[1].set_title("Decision samples")
    for axis in axes:
        axis.set_ylabel("Count")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Ticket 17 Dataset Split Used by Ticket 23")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Ticket 23 offline-IL pilot")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-date", default=date.today().isoformat())
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or f"runs/{PILOT_TOPIC}_{args.run_date}"
    paths = run_md_offline_il_pilot(
        args.dataset_dir,
        output_dir,
        config=MDILTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
        ),
        run_date=args.run_date,
        device=args.device,
    )
    print(paths["summary"])


if __name__ == "__main__":
    main()
