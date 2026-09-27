"""Path C v1: replace pointwise CE loss with margin ranking loss.

Motivation
----------
Path A / B1 / B2 all showed the same failure mode: lower val loss -> worse
downstream makespan on 60-task process_scarce. Diagnosis: pointwise BCE
forces the model to reproduce MILP tie-breaking (when multiple first actions
tie for optimal, CE labels one and treats the rest as errors).

This script fine-tunes the C0 (hidden_dim=16) checkpoint from
md_c0_milp_supervised_scale_pilot_2026-09-13 on the Path A combined dataset
using a *margin ranking* loss:

    loss_state = mean_k max(0, margin - score(pref) + score(neg_k))

where preferred pair == expert (r*, t*) and negatives are sampled uniformly
from legal_actions(s) \\ {(r*, t*)}. If a state has <2 legal actions, its loss
is 0. If it has < K+1 legal actions we sample all K' < K available.

Backbone, encoder, features, hard-mask semantics are unchanged; only the
training-time loss function differs.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from data_generation.md_expert_dataset import MDExpertDatasetLoader
from experiments.md_c0_milp_supervised_pilot import _c0_training_config
from experiments.protocol import DatasetSplit
from imitation_learning.md_train import (
    MD_IL_CHECKPOINT_VERSION,
    _build_model,
    _gradients_finite,
    _mean,
    _model_spec,
    _sample_batches,
    load_md_policy_checkpoint,
)
from models.md_legacy_features import legacy_policy_inputs_from_samples
from models.md_policy_features import md_policy_inputs_from_samples


PATH_C_SCHEMA = "md-c0-pathC-ranking-v1-1.0"
DEFAULT_PRETRAINED = Path(
    "reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt"
)
DEFAULT_DATASET = Path("reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset")


def _partition_records(dataset_dir: Path) -> dict[str, tuple]:
    loader = MDExpertDatasetLoader(dataset_dir)
    return {
        "train": loader.load_records(split=DatasetSplit.TRAIN),
        "validation": loader.load_records(split=DatasetSplit.VALIDATION),
    }


def _forward_scores(model, samples, device):
    """Return scores tensor shape (B, R, T) plus hard_mask (B, R, T) on device."""
    legacy = legacy_policy_inputs_from_samples(samples)
    md_inputs = md_policy_inputs_from_samples(samples)
    robot_features = legacy.robot_features.to(device)
    task_features = legacy.task_features.to(device)
    task_adjacency = legacy.task_adjacency.to(device)
    scores = model(
        robot_features, task_features, task_adjacency, md_inputs=md_inputs
    )
    hard_mask = md_inputs.hard_feasibility_mask.to(device)
    return scores, hard_mask


def _ranking_loss(
    scores: torch.Tensor,
    hard_mask: torch.Tensor,
    samples,
    *,
    num_negatives: int,
    margin: float,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Margin ranking loss.

    For every state in the batch:
      * Positives: legal AND expert_assignment == True.
      * Negatives: legal AND expert_assignment == False.
    For each positive pair (r*, t*) we sample K negatives (r_k, t_k) uniformly
    without replacement from the negative pool of the SAME state; excluding
    ALL preferred pairs from the negative pool (not just (r*, t*)) so that
    equally-optimal ties are not treated as negatives -- this deviates from
    the literal spec ("legal_actions \\ {(r*, t*)}") but matches the stated
    intent of the experiment (do not punish equivalent first actions).

    Loss is averaged across positive-negative pairs contributing to this batch.
    Returns (loss, stats) where stats contains counts for logging.
    """
    batch_size, robot_count, task_count = scores.shape
    losses: list[torch.Tensor] = []
    positive_count = 0
    negative_pool_lt_k = 0
    skipped_states = 0

    for b_idx, sample in enumerate(samples):
        legal = hard_mask[b_idx]  # (R, T) bool
        expert = torch.tensor(
            sample.expert_assignment, dtype=torch.bool, device=device
        )
        # sanity: expert must be within legal
        pos_mask = legal & expert
        neg_mask = legal & (~expert)
        pos_indices = pos_mask.nonzero(as_tuple=False)
        neg_indices = neg_mask.nonzero(as_tuple=False)
        if pos_indices.numel() == 0 or neg_indices.numel() == 0:
            skipped_states += 1
            continue

        n_neg_avail = neg_indices.size(0)
        k_eff = min(num_negatives, n_neg_avail)
        if k_eff < num_negatives:
            negative_pool_lt_k += 1

        # Score tensor slice for this state.
        state_scores = scores[b_idx]  # (R, T)

        for p_row in pos_indices:
            r_star = int(p_row[0].item())
            t_star = int(p_row[1].item())
            s_pos = state_scores[r_star, t_star]

            # Sample k_eff negatives without replacement.
            pool_size = n_neg_avail
            if pool_size <= k_eff:
                sampled = list(range(pool_size))
            else:
                sampled = rng.sample(range(pool_size), k_eff)
            neg_rows = neg_indices[sampled]
            neg_r = neg_rows[:, 0]
            neg_t = neg_rows[:, 1]
            s_neg = state_scores[neg_r, neg_t]
            # hinge: max(0, margin - s_pos + s_neg), average across K
            hinge = torch.clamp(margin - s_pos + s_neg, min=0.0)
            losses.append(hinge.mean())
            positive_count += 1

    if not losses:
        # Return zero loss with grad path; keeps trainer happy.
        loss = scores.sum() * 0.0
    else:
        loss = torch.stack(losses).mean()
    stats = {
        "positive_pairs": positive_count,
        "skipped_states": skipped_states,
        "batch_states": batch_size,
        "neg_pool_lt_k": negative_pool_lt_k,
    }
    return loss, stats


def _load_pretrained_state(pretrained_path: Path, device: torch.device):
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint missing: {pretrained_path}")
    model, _ = load_md_policy_checkpoint(pretrained_path, device=device)
    return copy.deepcopy(model.state_dict())


def _save_checkpoint(
    path: Path,
    *,
    state,
    config,
    train_losses,
    validation_losses,
    train_ids,
    validation_ids,
    train_sample_count,
    validation_sample_count,
    epoch,
    pretrained_checkpoint,
    learning_rate,
    num_negatives,
    margin,
    seed,
) -> None:
    checkpoint = {
        "schema_version": MD_IL_CHECKPOINT_VERSION,
        "model_kind": "md_enhanced",
        "state_dict": state,
        "model_spec": _model_spec(config),
        "md_policy_config": asdict(config.md_policy_config()),
        "training_config": asdict(config),
        "training_enhancements": asdict(config.training_enhancement_config()),
        "value_head_state_dict": None,
        "dataset": {
            "train_instance_ids": list(train_ids),
            "validation_instance_ids": list(validation_ids),
            "train_sample_count": train_sample_count,
            "validation_sample_count": validation_sample_count,
        },
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "best_validation_loss": min(validation_losses) if validation_losses else None,
        "solver_calls_during_training": 0,
        "path_c_ranking_v1": {
            "pretrained_checkpoint": str(pretrained_checkpoint),
            "epoch": epoch,
            "learning_rate": learning_rate,
            "num_negatives": num_negatives,
            "margin": margin,
            "seed": seed,
        },
    }
    torch.save(checkpoint, path)


def _run_training(
    dataset_dir: Path,
    output_dir: Path,
    pretrained_path: Path,
    *,
    epochs: int,
    learning_rate: float,
    num_negatives: int,
    margin: float,
    seed: int,
    device: torch.device,
    checkpoint_epochs: tuple[int, ...],
    log_lines: list[str],
) -> dict:
    config = _c0_training_config(epochs=epochs, seed=seed)
    partitions = _partition_records(dataset_dir)
    train_records = partitions["train"]
    validation_records = partitions["validation"]
    if not train_records or not validation_records:
        raise RuntimeError("empty train or validation partition")
    train_samples = tuple(s for r in train_records for s in r.samples)
    validation_samples = tuple(s for r in validation_records for s in r.samples)
    if not train_samples or not validation_samples:
        raise RuntimeError("no decision samples after partitioning")

    torch.manual_seed(seed)
    rng_train = random.Random(seed + 7919)
    rng_val = random.Random(seed + 3121)
    model = _build_model(config).to(device)
    pre_state = _load_pretrained_state(pretrained_path, device)
    model.load_state_dict(pre_state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_ids = tuple(r.instance_id for r in train_records)
    validation_ids = tuple(r.instance_id for r in validation_records)

    def _eval_validation() -> tuple[float, dict]:
        model.eval()
        losses = []
        totals = {"positive_pairs": 0, "skipped_states": 0, "batch_states": 0, "neg_pool_lt_k": 0}
        with torch.no_grad():
            for batch in _sample_batches(
                validation_samples,
                batch_size=config.batch_size,
                seed=seed,
                shuffle=False,
            ):
                scores, hard_mask = _forward_scores(model, batch, device)
                loss, stats = _ranking_loss(
                    scores,
                    hard_mask,
                    batch,
                    num_negatives=num_negatives,
                    margin=margin,
                    rng=rng_val,
                    device=device,
                )
                losses.append(float(loss.detach().cpu()))
                for k, v in stats.items():
                    totals[k] += v
        return _mean(losses, "validation"), totals

    output_dir.mkdir(parents=True, exist_ok=True)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    saved_paths: dict[int, str] = {}

    # Epoch-0 validation loss (pretrained baseline under ranking objective).
    epoch0_val, epoch0_stats = _eval_validation()
    validation_losses.append(epoch0_val)
    msg = (
        f"[{time.strftime('%H:%M:%S')}] epoch 000 (pretrained) "
        f"val={epoch0_val:.6f} pos_pairs={epoch0_stats['positive_pairs']} "
        f"skipped_states={epoch0_stats['skipped_states']}"
    )
    print(msg, flush=True)
    log_lines.append(msg)

    prev_train_loss: float | None = None
    flat_epoch_count = 0

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        batch_losses: list[float] = []
        agg = {"positive_pairs": 0, "skipped_states": 0, "batch_states": 0, "neg_pool_lt_k": 0}
        for batch in _sample_batches(
            train_samples,
            batch_size=config.batch_size,
            seed=seed + epoch,
            shuffle=True,
        ):
            scores, hard_mask = _forward_scores(model, batch, device)
            loss, stats = _ranking_loss(
                scores,
                hard_mask,
                batch,
                num_negatives=num_negatives,
                margin=margin,
                rng=rng_train,
                device=device,
            )
            optimizer.zero_grad()
            loss.backward()
            if not _gradients_finite(model):
                raise RuntimeError("Path C training produced non-finite gradients")
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            for k, v in stats.items():
                agg[k] += v
        train_loss = _mean(batch_losses, f"training-epoch-{epoch}")
        train_losses.append(train_loss)

        validation_loss, val_stats = _eval_validation()
        validation_losses.append(validation_loss)
        elapsed = time.perf_counter() - epoch_started
        msg = (
            f"[{time.strftime('%H:%M:%S')}] epoch {epoch:03d} "
            f"train={train_loss:.6f} val={validation_loss:.6f} "
            f"pos_pairs(train)={agg['positive_pairs']} skipped={agg['skipped_states']} "
            f"elapsed={elapsed:.1f}s"
        )
        print(msg, flush=True)
        log_lines.append(msg)

        if epoch in checkpoint_epochs:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            _save_checkpoint(
                ckpt_path,
                state=copy.deepcopy(model.state_dict()),
                config=config,
                train_losses=train_losses,
                validation_losses=validation_losses,
                train_ids=train_ids,
                validation_ids=validation_ids,
                train_sample_count=len(train_samples),
                validation_sample_count=len(validation_samples),
                epoch=epoch,
                pretrained_checkpoint=pretrained_path,
                learning_rate=learning_rate,
                num_negatives=num_negatives,
                margin=margin,
                seed=seed,
            )
            saved_paths[epoch] = str(ckpt_path)

        # Flat-loss watchdog (stop if 2 consecutive epochs perfectly flat).
        if prev_train_loss is not None and abs(prev_train_loss - train_loss) < 1e-8:
            flat_epoch_count += 1
            if flat_epoch_count >= 2:
                warn = f"[WATCHDOG] train loss flat for 2 consecutive epochs; stopping."
                print(warn, flush=True)
                log_lines.append(warn)
                break
        else:
            flat_epoch_count = 0
        prev_train_loss = train_loss

    # Save "best_checkpoint.pt" alias to the last epoch checkpoint.
    final_epoch = epochs if not train_losses else min(epochs, len(train_losses))
    final_state = copy.deepcopy(model.state_dict())
    best_path = output_dir / "best_checkpoint.pt"
    _save_checkpoint(
        best_path,
        state=final_state,
        config=config,
        train_losses=train_losses,
        validation_losses=validation_losses,
        train_ids=train_ids,
        validation_ids=validation_ids,
        train_sample_count=len(train_samples),
        validation_sample_count=len(validation_samples),
        epoch=final_epoch,
        pretrained_checkpoint=pretrained_path,
        learning_rate=learning_rate,
        num_negatives=num_negatives,
        margin=margin,
        seed=seed,
    )

    summary = {
        "schema": PATH_C_SCHEMA,
        "pretrained_checkpoint": str(pretrained_path),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "epochs_requested": epochs,
        "epochs_run": len(train_losses),
        "learning_rate": learning_rate,
        "num_negatives": num_negatives,
        "margin": margin,
        "seed": seed,
        "device": str(device),
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "train_instance_count": len(train_ids),
        "validation_instance_count": len(validation_ids),
        "epoch0_validation_loss": epoch0_val,
        "train_losses": train_losses,
        "validation_losses": validation_losses,  # includes epoch 0
        "checkpoint_paths": {str(k): v for k, v in {**saved_paths, "final": str(best_path)}.items()},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run(
    output_root: Path,
    *,
    pretrained_checkpoint: Path,
    dataset_dir: Path,
    epochs: int,
    learning_rate: float,
    num_negatives: int,
    margin: float,
    seed: int,
    device: str,
    checkpoint_epochs: tuple[int, ...],
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    training_dir = output_root / "training"
    log_lines: list[str] = []
    device_obj = torch.device(device)
    summary = _run_training(
        dataset_dir,
        training_dir,
        pretrained_checkpoint,
        epochs=epochs,
        learning_rate=learning_rate,
        num_negatives=num_negatives,
        margin=margin,
        seed=seed,
        device=device_obj,
        checkpoint_epochs=checkpoint_epochs,
        log_lines=log_lines,
    )
    (output_root / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (output_root / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/md_c0_pathC_ranking_v1_2026-09-17"),
    )
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint-epochs",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
    )
    args = parser.parse_args()
    result = run(
        args.output,
        pretrained_checkpoint=args.pretrained_checkpoint,
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_negatives=args.negatives,
        margin=args.margin,
        seed=args.seed,
        device=args.device,
        checkpoint_epochs=tuple(args.checkpoint_epochs),
    )
    print(json.dumps(result, indent=2))
