"""Path B: capacity-upgrade sweep for the C0 architecture.

Path A (same-capacity fine-tune of the hidden_dim=16 C0 checkpoint on
process_scarce + dependency_deep scaled shards) failed: 60-task
process_scarce makespan went from 398.75 (baseline C0) to 405.25 (Path A),
i.e. the fine-tune slightly regressed on process_scarce even though
validation loss decreased. Diagnosis: at hidden_dim=16 the network cannot
simultaneously represent both structural signatures, so val loss and
downstream makespan decouple.

Path B keeps the C0 backbone (pair-aware attention, no cross-attention,
same feature stack, same loss) and only *scales the capacity*: wider hidden
dim, wider embed dim, more transformer / GAT layers, some dropout, cosine
LR schedule with warmup. Training is from scratch on the combined Path A
dataset (300 new scaled shards + 250 legacy balanced/scale shards).

This single script is used for both:

- B1 (hidden_dim = embed_dim = 48, transformer_layers = gat_layers = 2)
- B2 (hidden_dim = embed_dim = 64, transformer_layers = gat_layers = 2)

so the two experiments only differ in CLI arguments. Output goes under a
run-specific directory the caller passes via --output.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from data_generation.md_expert_dataset import MDExpertDatasetLoader
from experiments.md_c0_milp_supervised_pilot import _c0_training_config
from experiments.protocol import DatasetSplit
from imitation_learning.md_train import (
    MD_IL_CHECKPOINT_VERSION,
    MDILTrainingConfig,
    _batch_loss,
    _gradients_finite,
    _mean,
    _sample_batches,
    load_md_policy_checkpoint,
)
from models.md_enhanced_policy import MDEnhancedSchedulerNetwork


PATH_B_SCHEMA = "md-c0-pathB-capacity-upgrade-1.0"

DEFAULT_DATASET = Path(
    "reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset"
)


def _build_training_config(
    *,
    hidden_dim: int,
    embed_dim: int,
    transformer_layers: int,
    gat_layers: int,
    epochs: int,
    seed: int,
    batch_size: int,
    peak_lr: float,
) -> MDILTrainingConfig:
    """Start from the pinned C0 flags and override capacity / LR knobs."""

    base = _c0_training_config(epochs=epochs, seed=seed)
    kwargs = asdict(base)
    kwargs.update(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": peak_lr,
            "seed": seed,
            "embed_dim": embed_dim,
            # ff_dim scales with embed_dim by the same 2x ratio C0 pilot used.
            "ff_dim": embed_dim * 2,
            "hidden_dim": hidden_dim,
            "transformer_layers": transformer_layers,
            "gat_layers": gat_layers,
        }
    )
    return MDILTrainingConfig(**kwargs)


def _model_spec_with_dropout(config: MDILTrainingConfig, dropout: float) -> dict[str, Any]:
    return {
        "robot_input_dimensions": 7,
        "task_input_dimension": 9,
        "embed_dim": config.embed_dim,
        "ff_dim": config.ff_dim,
        "n_transformer_heads": config.transformer_heads,
        "n_transformer_layers": config.transformer_layers,
        "n_gatn_heads": config.gat_heads,
        "n_gatn_layers": config.gat_layers,
        "dropout": dropout,
        "use_idle": False,
    }


def _build_model(config: MDILTrainingConfig, dropout: float) -> MDEnhancedSchedulerNetwork:
    return MDEnhancedSchedulerNetwork(
        md_config=config.md_policy_config(),
        **_model_spec_with_dropout(config, dropout),
    )


def _partition_records(dataset_dir: Path) -> dict[str, tuple]:
    loader = MDExpertDatasetLoader(dataset_dir)
    train = loader.load_records(split=DatasetSplit.TRAIN)
    validation = loader.load_records(split=DatasetSplit.VALIDATION)
    test = loader.load_records(split=DatasetSplit.TEST)
    return {"train": train, "validation": validation, "test": test}


def _cosine_lr(
    epoch: int, *, warmup_epochs: int, total_epochs: int, peak_lr: float, min_lr: float
) -> float:
    """Warmup for `warmup_epochs` (linear 0 -> peak_lr), then cosine decay to min_lr.

    `epoch` is 1-based (i.e. first training epoch is epoch=1).
    """
    if epoch <= warmup_epochs:
        # Linear warmup that ends at exactly peak_lr at the last warmup epoch.
        return peak_lr * float(epoch) / max(1, warmup_epochs)
    # Cosine decay across the remaining epochs.
    remaining_total = max(1, total_epochs - warmup_epochs)
    step = min(epoch - warmup_epochs, remaining_total)
    cos = 0.5 * (1.0 + math.cos(math.pi * step / remaining_total))
    return min_lr + (peak_lr - min_lr) * cos


def _train(
    dataset_dir: Path,
    output_dir: Path,
    *,
    config: MDILTrainingConfig,
    dropout: float,
    peak_lr: float,
    min_lr: float,
    warmup_epochs: int,
    epochs: int,
    early_stopping_patience: int,
    seed: int,
    device: torch.device,
    from_scratch: bool,
    pretrained_checkpoint: Path | None,
) -> dict:
    partitions = _partition_records(dataset_dir)
    train_records = partitions["train"]
    validation_records = partitions["validation"]
    if not train_records or not validation_records:
        raise RuntimeError("dataset missing train or validation records")
    train_samples = tuple(s for r in train_records for s in r.samples)
    validation_samples = tuple(s for r in validation_records for s in r.samples)
    if not train_samples or not validation_samples:
        raise RuntimeError("no decision samples available after partitioning")

    torch.manual_seed(seed)
    model = _build_model(config, dropout).to(device)

    if not from_scratch and pretrained_checkpoint is not None:
        pre_model, _ = load_md_policy_checkpoint(pretrained_checkpoint, device=device)
        incompatible = model.load_state_dict(pre_model.state_dict(), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint state_dict does not match capacity-upgraded model: "
                f"missing={list(incompatible.missing_keys)} "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr)

    def _apply_lr(epoch: int) -> float:
        lr = _cosine_lr(
            epoch,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            peak_lr=peak_lr,
            min_lr=min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    train_losses: list[float] = []
    validation_losses: list[float] = []
    lr_schedule: list[float] = []
    best_validation = math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        current_lr = _apply_lr(epoch)
        lr_schedule.append(current_lr)
        epoch_started = time.perf_counter()
        model.train()
        batch_losses = []
        for batch in _sample_batches(
            train_samples,
            batch_size=config.batch_size,
            seed=seed + epoch,
            shuffle=True,
        ):
            loss = _batch_loss(model, batch, device, enhancement_config=None)
            optimizer.zero_grad()
            loss.backward()
            if not _gradients_finite(model):
                raise RuntimeError("path B produced non-finite gradients")
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_loss = _mean(batch_losses, f"training-epoch-{epoch}")
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_losses = [
                float(
                    _batch_loss(model, batch, device, enhancement_config=None).cpu()
                )
                for batch in _sample_batches(
                    validation_samples,
                    batch_size=config.batch_size,
                    seed=seed,
                    shuffle=False,
                )
            ]
        validation_loss = _mean(val_losses, f"validation-epoch-{epoch}")
        validation_losses.append(validation_loss)
        epoch_elapsed = time.perf_counter() - epoch_started

        improved = validation_loss < best_validation
        print(
            f"[{time.strftime('%H:%M:%S')}] epoch {epoch:03d} lr={current_lr:.2e} "
            f"train={train_loss:.6f} val={validation_loss:.6f} "
            f"best={best_validation:.6f} "
            f"{'(new best)' if improved else f'(patience {patience_counter + 1}/{early_stopping_patience})'} "
            f"elapsed={epoch_elapsed:.1f}s",
            flush=True,
        )

        if improved:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(
                    f"[{time.strftime('%H:%M:%S')}] early stop after "
                    f"{patience_counter} epochs w/o improvement",
                    flush=True,
                )
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    summary_path = output_dir / "training_summary.json"
    train_ids = tuple(r.instance_id for r in train_records)
    validation_ids = tuple(r.instance_id for r in validation_records)
    checkpoint = {
        "schema_version": MD_IL_CHECKPOINT_VERSION,
        "model_kind": "md_enhanced",
        "state_dict": best_state,
        "model_spec": _model_spec_with_dropout(config, dropout),
        "md_policy_config": asdict(config.md_policy_config()),
        "training_config": asdict(config),
        "training_enhancements": asdict(config.training_enhancement_config()),
        "value_head_state_dict": None,
        "dataset": {
            "train_instance_ids": list(train_ids),
            "validation_instance_ids": list(validation_ids),
            "train_sample_count": len(train_samples),
            "validation_sample_count": len(validation_samples),
        },
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "best_validation_loss": best_validation,
        "solver_calls_during_training": 0,
        "path_b": {
            "dropout": dropout,
            "peak_lr": peak_lr,
            "min_lr": min_lr,
            "warmup_epochs": warmup_epochs,
            "epochs_requested": epochs,
            "epochs_run": len(train_losses),
            "early_stopping_patience": early_stopping_patience,
            "best_epoch": best_epoch,
            "from_scratch": from_scratch,
            "pretrained_checkpoint": (
                None if pretrained_checkpoint is None else str(pretrained_checkpoint)
            ),
            "lr_schedule": lr_schedule,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {k: v for k, v in checkpoint.items() if k not in ("state_dict", "value_head_state_dict")}
    summary["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
    }
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "summary_path": str(summary_path),
        "best_validation_loss": best_validation,
        "best_epoch": best_epoch,
        "epochs_run": len(train_losses),
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
    }


def run(
    output_root: Path,
    *,
    dataset_dir: Path,
    hidden_dim: int,
    embed_dim: int,
    transformer_layers: int,
    gat_layers: int,
    dropout: float,
    peak_lr: float,
    warmup_epochs: int,
    min_lr: float,
    epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    seed: int,
    device: str,
    from_scratch: bool,
    pretrained_checkpoint: Path | None,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    training_dir = output_root / "training"

    config = _build_training_config(
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        transformer_layers=transformer_layers,
        gat_layers=gat_layers,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        peak_lr=peak_lr,
    )

    device_obj = torch.device(device)
    training_result = _train(
        dataset_dir,
        training_dir,
        config=config,
        dropout=dropout,
        peak_lr=peak_lr,
        min_lr=min_lr,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
        device=device_obj,
        from_scratch=from_scratch,
        pretrained_checkpoint=pretrained_checkpoint,
    )

    metadata = {
        "schema": PATH_B_SCHEMA,
        "output_root": str(output_root),
        "dataset_dir": str(dataset_dir),
        "capacity": {
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
            "transformer_layers": transformer_layers,
            "gat_layers": gat_layers,
            "dropout": dropout,
        },
        "lr_schedule": {
            "peak_lr": peak_lr,
            "warmup_epochs": warmup_epochs,
            "min_lr": min_lr,
        },
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "batch_size": batch_size,
        "seed": seed,
        "from_scratch": from_scratch,
        "pretrained_checkpoint": (
            None if pretrained_checkpoint is None else str(pretrained_checkpoint)
        ),
        "training": training_result,
    }
    (output_root / "pilot_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--peak-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--from-scratch", action="store_true", default=True)
    parser.add_argument(
        "--load-pretrained",
        dest="from_scratch",
        action="store_false",
        help="Load `--pretrained-checkpoint` before training (overrides default from-scratch).",
    )
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output,
                dataset_dir=args.dataset_dir,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                transformer_layers=args.transformer_layers,
                gat_layers=args.gat_layers,
                dropout=args.dropout,
                peak_lr=args.peak_lr,
                warmup_epochs=args.warmup_epochs,
                min_lr=args.min_lr,
                epochs=args.epochs,
                early_stopping_patience=args.early_stopping_patience,
                batch_size=args.batch_size,
                seed=args.seed,
                device=args.device,
                from_scratch=args.from_scratch,
                pretrained_checkpoint=args.pretrained_checkpoint,
            ),
            indent=2,
        )
    )
