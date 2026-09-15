"""MILP-supervised IL training pilot for a C0-architecture policy.

Generates a MILP-labeled dataset from balanced-profile instances and trains
a policy with the C0 architecture (hidden_dim=16, pair-aware attention, no
cross-attention). Reuses:

- data_generation.md_expert_dataset_generation.generate_md_expert_dataset
  for the (MILP solve -> step-by-step replay -> MDDecisionSample) pipeline.
- experiments.md_offline_il_pilot.run_md_offline_il_pilot for the training
  run, artifact packaging, and validation checkpoint sanity check.

All output lives under a fresh dated report root; no existing C0 data,
scripts, or checkpoints are touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_generation.md_expert_dataset_generation import (
    MDExpertGenerationConfig,
    generate_md_expert_dataset,
)
from data_generation.md_instance_profiles import INSTANCE_PROFILES
from experiments.md_offline_il_pilot import run_md_offline_il_pilot
from imitation_learning.md_train import MDILTrainingConfig


# balanced profile parameters (from data_generation/md_instance_profiles.py).
BALANCED = dict(INSTANCE_PROFILES["balanced"].parameters)
SEED_START = 80_000  # outside HANDOFF-reserved 75000-76149 boundaries
DATASET_COUNT = 150
MILP_TIME_LIMIT_SECONDS = 60.0
MILP_THREADS = 4


def _c0_training_config(*, epochs: int, seed: int) -> MDILTrainingConfig:
    """MDILTrainingConfig with the C0 architecture pinned."""
    return MDILTrainingConfig(
        epochs=epochs,
        batch_size=32,
        learning_rate=1e-3,
        seed=seed,
        embed_dim=16,
        ff_dim=32,
        transformer_heads=4,
        transformer_layers=1,
        gat_heads=4,
        gat_layers=1,
        hidden_dim=16,
        use_cross_attention=False,
        use_pair_aware_attention=True,
        cross_attention_heads=4,
        residual_bound=0.75,
        legacy_transport_bound=0.15,
        physics_scale=0.5,
        eta_scale=1.0,
        use_typed_edges=True,
        use_downstream_encoding=True,
        use_transport_eta=True,
        use_capacity_features=True,
        use_critical_path_proxy=True,
        use_last_material_blocker=True,
        use_coalition_context=True,
        use_capacity_scarcity=True,
    )


def run(output_root: Path, *, epochs: int, seed: int, device: str, count: int) -> dict:
    dataset_dir = output_root / "dataset"
    training_dir = output_root
    output_root.mkdir(parents=True, exist_ok=True)

    config = MDExpertGenerationConfig(
        count=count,
        seed_start=SEED_START,
        task_group_prefix="md-balanced-milp-il",
        instance_prefix="md-balanced-milp-il",
        task_count=BALANCED["task_count"],
        transport_ratio=BALANCED["transport_ratio"],
        precedence_density=BALANCED["precedence_density"],
        critical_path_length=BALANCED["critical_path_length"],
        capacity_slack=BALANCED["capacity_slack"],
        speed_ratio=BALANCED["speed_ratio"],
        process_robot_count=BALANCED["process_robot_count"],
        transport_robot_count=BALANCED["transport_robot_count"],
        skill_count=BALANCED["skill_count"],
        time_limit_seconds=MILP_TIME_LIMIT_SECONDS,
        threads=MILP_THREADS,
    )
    generation_summary = generate_md_expert_dataset(
        dataset_dir, config=config, overwrite=False
    )
    generation_report = {
        "requested_count": generation_summary.requested_count,
        "generated_count": generation_summary.generated_count,
        "skipped_count": generation_summary.skipped_count,
        "skipped": [
            {
                "instance_id": skip.instance_id,
                "seed": skip.seed,
                "status": skip.status,
                "message": skip.message,
            }
            for skip in generation_summary.skipped
        ],
    }
    (output_root / "dataset_generation_summary.json").write_text(
        json.dumps(generation_report, indent=2) + "\n"
    )

    device_obj = torch.device(device)
    il_config = _c0_training_config(epochs=epochs, seed=seed)
    paths = run_md_offline_il_pilot(
        dataset_dir,
        training_dir,
        config=il_config,
        device=device_obj,
    )
    return {
        "output_root": str(output_root),
        "dataset_generation": generation_report,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/md_c0_milp_supervised_pilot_2026-09-13"),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=DATASET_COUNT)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
                count=args.count,
            ),
            indent=2,
        )
    )
