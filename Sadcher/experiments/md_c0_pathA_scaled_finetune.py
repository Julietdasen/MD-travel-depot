"""Path A: extend MILP-IL v2 C0 training with process_scarce and dependency_deep
scaled shards, and fine-tune the existing checkpoint on the combined dataset.

Motivation
----------
The MILP-IL v2 C0 checkpoint from md_c0_milp_supervised_scale_pilot_2026-09-13
loses its advantage over `greedy_unlock` on 60-task `process_scarce` (relative
gap goes from -15% at tc=12 to +0.8% at tc=60). Beam-search follow-up showed
this is a *policy quality* ceiling, not a search issue: on every 60-task
instance the policy top-1 is already the best branch, so more search does not
help.

Diagnosis: the pilot only saw `balanced` (tc=12), `scale_medium` (tc=18) and a
custom `scale42` shard (tc=42, but with balanced-like transport_ratio /
precedence_density). It never saw the scarce-process or dense-precedence
structural signatures at 24 / 42 / 60 tasks.

This script:

1. Generates two new profile-aligned scaled shards (`process_scarce_scaled`,
   `dependency_deep_scaled`) at task_count ∈ {24, 42, 60}, 50 instances per
   (profile, task_count) tuple.
2. Symlinks the existing pilot dataset shards next to the new ones so the
   MDExpertDatasetLoader treats them as one dataset.
3. Fine-tunes the C0 checkpoint from
   reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt
   on the combined dataset with a low LR (2e-5) and early stopping on
   validation loss.

The original checkpoint, training script, and pilot report are not modified.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch

from data_generation.md_expert_dataset import MDExpertDatasetLoader
from data_generation.md_expert_dataset_generation import (
    MDExpertGenerationConfig,
    generate_md_expert_dataset,
)
from data_generation.md_instance_profiles import INSTANCE_PROFILES
from experiments.md_c0_milp_supervised_pilot import _c0_training_config
from experiments.protocol import DatasetSplit
from imitation_learning.md_train import (
    MD_IL_CHECKPOINT_VERSION,
    _batch_loss,
    _build_model,
    _gradients_finite,
    _mean,
    _model_spec,
    _sample_batches,
    load_md_policy_checkpoint,
)


PATH_A_SCHEMA = "md-c0-pathA-scaled-finetune-1.0"

# The original checkpoint from the multi-scale MILP-IL pilot.
DEFAULT_PRETRAINED = Path(
    "reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt"
)
# Original dataset directory — its shards get symlinked next to the new ones so
# the loader treats old + new as one dataset without copying JSONs.
DEFAULT_LEGACY_DATASET = Path(
    "reports/md_c0_milp_supervised_scale_pilot_2026-09-13/dataset"
)


def _scaled_profile_shard_config(
    profile_name: str,
    task_count: int,
    *,
    seed_start: int,
    count: int,
    time_limit_seconds: float,
) -> MDExpertGenerationConfig:
    """Build a generation config that mirrors experiments/md_c0_scaled_target_profiles_eval.py:_scaled_profile_config.

    Structural parameters (transport_ratio, precedence_density, capacity_slack,
    speed_ratio, skill_count) are kept as the base profile. task_count, robot
    counts and critical_path_length scale by `task_count / base_task_count`.
    """
    base = dict(INSTANCE_PROFILES[profile_name].parameters)
    base_task_count = base["task_count"]
    scale = task_count / base_task_count
    process_tasks = int(task_count * (1 - base["transport_ratio"]))
    critical_path = min(
        max(base["critical_path_length"], round(base["critical_path_length"] * scale)),
        process_tasks,
    )
    process_robots = max(1, round(base["process_robot_count"] * scale))
    transport_robots = max(
        base.get("transport_robot_count", 1),
        round(base["transport_robot_count"] * scale),
    )
    prefix = f"md-{profile_name}-scaled-tc{task_count:02d}-milp-il"
    return MDExpertGenerationConfig(
        count=count,
        seed_start=seed_start,
        task_group_prefix=prefix,
        instance_prefix=prefix,
        task_count=task_count,
        transport_ratio=base["transport_ratio"],
        precedence_density=base["precedence_density"],
        critical_path_length=critical_path,
        capacity_slack=base["capacity_slack"],
        speed_ratio=base["speed_ratio"],
        process_robot_count=process_robots,
        transport_robot_count=transport_robots,
        skill_count=base["skill_count"],
        scarce_skill_count=base.get("scarce_skill_count", 0),
        material_downstream_stratified=base.get("material_downstream_stratified", False),
        process_duration_range=base.get("process_duration_range", None),
        time_limit_seconds=time_limit_seconds,
        threads=4,
    )


# Seed ranges are chosen to avoid every reserved band:
#   confirmation split          75800-75949
#   regression benchmark        76000-76149
#   scaled_target_profiles_eval 301-304
#   md_c0_milp_supervised_pilot 80000-...    (balanced)
#   md_c0_milp_supervised_scale 82000-...    (scale_medium)
#                               84000-...    (scale42)
# The bands below leave 200-seed gaps between shards so that adding more tasks
# later cannot accidentally overlap.
_SHARD_SEED_STARTS = {
    ("process_scarce", 24): 86000,
    ("process_scarce", 42): 86200,
    ("process_scarce", 60): 86400,
    ("dependency_deep", 24): 87000,
    ("dependency_deep", 42): 87200,
    ("dependency_deep", 60): 87400,
}


def _symlink_legacy_dataset(
    legacy_root: Path, target_root: Path
) -> tuple[int, int]:
    """Mirror the legacy shard JSONs (and its _generation manifests) into
    target_root using symlinks. Returns (record_link_count, manifest_link_count).
    """
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "_generation").mkdir(parents=True, exist_ok=True)
    record_links = 0
    manifest_links = 0
    for path in sorted(legacy_root.glob("*.json")):
        dest = target_root / path.name
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(path.resolve())
        record_links += 1
    for path in sorted((legacy_root / "_generation").glob("*.json")):
        dest = target_root / "_generation" / path.name
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(path.resolve())
        manifest_links += 1
    return record_links, manifest_links


def _generate_shards(
    dataset_dir: Path,
    profiles: tuple[str, ...],
    task_counts: tuple[int, ...],
    count_per_tier: int,
    time_limit_seconds: float,
    log_lines: list[str],
) -> list[dict]:
    """Generate the new profile-aligned scaled shards. One shard per
    (profile, task_count) so seed ranges are disjoint and prefixes carry the
    task-count so task_level_split hashing spreads instances stably.
    """
    all_summaries: list[dict] = []
    for profile in profiles:
        for tc in task_counts:
            seed_start = _SHARD_SEED_STARTS[(profile, tc)]
            cfg = _scaled_profile_shard_config(
                profile,
                tc,
                seed_start=seed_start,
                count=count_per_tier,
                time_limit_seconds=time_limit_seconds,
            )
            shard_started = time.perf_counter()
            print(
                f"[{time.strftime('%H:%M:%S')}] SHARD start "
                f"profile={profile} tc={tc} count={count_per_tier} "
                f"seed_start={seed_start} time_limit={time_limit_seconds:.0f}s "
                f"process_robots={cfg.process_robot_count} "
                f"transport_robots={cfg.transport_robot_count} "
                f"critical_path={cfg.critical_path_length}",
                flush=True,
            )
            log_lines.append(
                f"SHARD_START profile={profile} tc={tc} count={count_per_tier} "
                f"seed_start={seed_start} pr={cfg.process_robot_count} "
                f"tr={cfg.transport_robot_count} cp={cfg.critical_path_length}"
            )
            summary = generate_md_expert_dataset(
                dataset_dir, config=cfg, overwrite=False
            )
            shard_elapsed = time.perf_counter() - shard_started
            row = {
                "profile": profile,
                "task_count": tc,
                "seed_start": seed_start,
                "count": count_per_tier,
                "generated": summary.generated_count,
                "skipped": summary.skipped_count,
                "elapsed_seconds": shard_elapsed,
                "config": {
                    "transport_ratio": cfg.transport_ratio,
                    "precedence_density": cfg.precedence_density,
                    "critical_path_length": cfg.critical_path_length,
                    "capacity_slack": cfg.capacity_slack,
                    "speed_ratio": cfg.speed_ratio,
                    "process_robot_count": cfg.process_robot_count,
                    "transport_robot_count": cfg.transport_robot_count,
                    "skill_count": cfg.skill_count,
                    "time_limit_seconds": cfg.time_limit_seconds,
                    "threads": cfg.threads,
                },
                "skipped_records": [
                    {
                        "instance_id": skip.instance_id,
                        "seed": skip.seed,
                        "status": skip.status,
                        "message": skip.message,
                    }
                    for skip in summary.skipped
                ],
            }
            all_summaries.append(row)
            print(
                f"[{time.strftime('%H:%M:%S')}] SHARD done "
                f"profile={profile} tc={tc} generated={summary.generated_count}/"
                f"{count_per_tier} skipped={summary.skipped_count} "
                f"elapsed={shard_elapsed:.1f}s",
                flush=True,
            )
            log_lines.append(
                f"SHARD_END profile={profile} tc={tc} "
                f"generated={summary.generated_count} "
                f"skipped={summary.skipped_count} elapsed={shard_elapsed:.1f}s"
            )
    return all_summaries


def _partition_records(dataset_dir: Path) -> dict[str, tuple]:
    """Load MD expert records and split them via the deterministic task-level
    hash. Same protocol as train_md_policy so we honour the same held-out set.
    """
    loader = MDExpertDatasetLoader(dataset_dir)
    train = loader.load_records(split=DatasetSplit.TRAIN)
    validation = loader.load_records(split=DatasetSplit.VALIDATION)
    test = loader.load_records(split=DatasetSplit.TEST)
    return {"train": train, "validation": validation, "test": test}


def _load_pretrained_state(
    pretrained_path: Path, device: torch.device
) -> dict[str, torch.Tensor]:
    """Load the state_dict from the pilot checkpoint."""
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint missing: {pretrained_path}")
    model, _ = load_md_policy_checkpoint(pretrained_path, device=device)
    return copy.deepcopy(model.state_dict())


def _finetune(
    dataset_dir: Path,
    output_dir: Path,
    pretrained_path: Path,
    *,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Fine-tune the C0 model on the combined dataset.

    - Same architecture as _c0_training_config (hidden_dim=16, ...).
    - LR is lower than pilot (default 2e-5, was 1e-3) so we do not blow away
      the balanced signal.
    - Best checkpoint == lowest validation loss.
    - Early stop after `early_stopping_patience` epochs of no improvement.
    """
    config = _c0_training_config(epochs=epochs, seed=seed)
    partitions = _partition_records(dataset_dir)
    train_records = partitions["train"]
    validation_records = partitions["validation"]
    if not train_records or not validation_records:
        raise RuntimeError(
            "combined dataset has no train or validation records; "
            "check that _partition_records saw the symlinked and new shards"
        )
    train_samples = tuple(s for r in train_records for s in r.samples)
    validation_samples = tuple(s for r in validation_records for s in r.samples)
    if not train_samples or not validation_samples:
        raise RuntimeError("no decision samples available after partitioning")

    torch.manual_seed(seed)
    model = _build_model(config).to(device)
    pre_state = _load_pretrained_state(pretrained_path, device)
    incompatible = model.load_state_dict(pre_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint state_dict does not match new model: "
            f"missing={list(incompatible.missing_keys)} "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Evaluate pretrained (epoch 0) validation loss so we never save a
    # regression relative to the source checkpoint.
    model.eval()
    with torch.no_grad():
        pre_val_losses = [
            float(
                _batch_loss(model, batch, device, enhancement_config=None).cpu()
            )
            for batch in _sample_batches(
                validation_samples, batch_size=config.batch_size, seed=seed, shuffle=False
            )
        ]
    pretrained_validation_loss = _mean(pre_val_losses, "pretrained_validation")

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation = pretrained_validation_loss
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0  # 0 == pretrained checkpoint (i.e. no fine-tune improvement)
    patience_counter = 0

    for epoch in range(1, epochs + 1):
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
                raise RuntimeError("fine-tune produced non-finite gradients")
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
            f"[{time.strftime('%H:%M:%S')}] epoch {epoch:03d} "
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
                    f"[{time.strftime('%H:%M:%S')}] early stop after {patience_counter} epochs w/o improvement",
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
        "model_spec": _model_spec(config),
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
        "fine_tune": {
            "pretrained_checkpoint": str(pretrained_path),
            "pretrained_validation_loss": pretrained_validation_loss,
            "best_epoch": best_epoch,
            "learning_rate": learning_rate,
            "epochs_requested": epochs,
            "epochs_run": len(train_losses),
            "early_stopping_patience": early_stopping_patience,
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
        "pretrained_validation_loss": pretrained_validation_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(train_losses),
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
    }


def run(
    output_root: Path,
    *,
    pretrained_checkpoint: Path,
    legacy_dataset: Path,
    profiles: tuple[str, ...],
    task_counts: tuple[int, ...],
    count_per_tier: int,
    milp_time_limit_seconds: float,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    seed: int,
    device: str,
    skip_generation: bool,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_root / "dataset"
    training_dir = output_root / "training"

    log_lines: list[str] = []
    manifest_started = time.time()

    record_links, manifest_links = _symlink_legacy_dataset(legacy_dataset, dataset_dir)
    print(
        f"[{time.strftime('%H:%M:%S')}] legacy symlinks: records={record_links} "
        f"manifests={manifest_links} legacy_root={legacy_dataset}",
        flush=True,
    )
    log_lines.append(
        f"LEGACY_SYMLINKS records={record_links} manifests={manifest_links}"
    )

    if skip_generation:
        shards_summary: list[dict] = []
        print("[SKIP] --skip-generation, expecting shard files present", flush=True)
    else:
        shards_summary = _generate_shards(
            dataset_dir,
            profiles,
            task_counts,
            count_per_tier,
            milp_time_limit_seconds,
            log_lines,
        )

    generation_manifest = {
        "schema": PATH_A_SCHEMA,
        "started_at": manifest_started,
        "finished_at": time.time(),
        "legacy_dataset": str(legacy_dataset),
        "legacy_symlink_counts": {"records": record_links, "manifests": manifest_links},
        "profiles": list(profiles),
        "task_counts": list(task_counts),
        "count_per_tier": count_per_tier,
        "milp_time_limit_seconds": milp_time_limit_seconds,
        "shards": shards_summary,
    }
    (output_root / "dataset_generation_summary.json").write_text(
        json.dumps(generation_manifest, indent=2) + "\n"
    )

    device_obj = torch.device(device)
    training_result = _finetune(
        dataset_dir,
        training_dir,
        pretrained_checkpoint,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
        device=device_obj,
    )

    result = {
        "schema": PATH_A_SCHEMA,
        "output_root": str(output_root),
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "dataset_generation": generation_manifest,
        "training": training_result,
    }
    (output_root / "pilot_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/md_c0_pathA_scaled_finetune_2026-09-17"),
    )
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--legacy-dataset", type=Path, default=DEFAULT_LEGACY_DATASET)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["process_scarce", "dependency_deep"],
    )
    parser.add_argument("--task-counts", nargs="+", type=int, default=[24, 42, 60])
    parser.add_argument("--count-per-tier", type=int, default=50)
    parser.add_argument("--milp-time-limit", type=float, default=300.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output,
                pretrained_checkpoint=args.pretrained_checkpoint,
                legacy_dataset=args.legacy_dataset,
                profiles=tuple(args.profiles),
                task_counts=tuple(args.task_counts),
                count_per_tier=args.count_per_tier,
                milp_time_limit_seconds=args.milp_time_limit,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                early_stopping_patience=args.early_stopping_patience,
                seed=args.seed,
                device=args.device,
                skip_generation=args.skip_generation,
            ),
            indent=2,
        )
    )
