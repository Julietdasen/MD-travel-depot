"""Multi-scale MILP-supervised IL training pilot.

Extends md_c0_milp_supervised_pilot.py by mixing three instance sizes in
the training set (task_count 12 / 24 / 42), so the resulting model has
seen the same scaled-profile geometry that the scale-ladder evaluation
covers. Uses the same reused infrastructure — MD MILP dispatch (OR-Tools
CP-SAT by default, Gurobi opt-in via ``MRTA_MILP_SOLVER=gurobi``) +
generate_md_expert_record_from_oracle + train_md_policy — no new label
schema or dataloader shims.

Instances at 42 tasks sometimes time out at 300s (see the pure-MILP
baseline pilot). generate_md_expert_dataset skips non-{OPTIMAL,FEASIBLE}
results; feasible-but-timed-out schedules are still admitted because
those are still a *better than random* supervision signal (they came from
the solver's best incumbent within the time budget).
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
from experiments.md_c0_milp_supervised_pilot import _c0_training_config

BALANCED = dict(INSTANCE_PROFILES["balanced"].parameters)
SCALE_MEDIUM = dict(INSTANCE_PROFILES["scale_medium"].parameters)


def _scaled_42_config(seed_start: int, count: int) -> MDExpertGenerationConfig:
    """task_count=42 mirror of the scale ladder used for stress-testing.

    Robot counts scale from the balanced profile's 3/2 ratio, keeping the
    same ratios used in md_c0_scale_stress_pilot.py (scale = tc / 18).
    """
    scale = 42 / 18
    return MDExpertGenerationConfig(
        count=count,
        seed_start=seed_start,
        task_group_prefix="md-scale42-milp-il",
        instance_prefix="md-scale42-milp-il",
        task_count=42,
        transport_ratio=1.0 / 3.0,
        precedence_density=0.30,
        critical_path_length=6,
        capacity_slack=0.10,
        speed_ratio=0.70,
        process_robot_count=max(1, round(4 * scale)),
        transport_robot_count=max(1, round(3 * scale)),
        skill_count=3,
        time_limit_seconds=120.0,  # let 42-task MILP work harder
        threads=4,
    )


def _profile_config(
    profile_name: str, seed_start: int, count: int, time_limit_seconds: float = 60.0
) -> MDExpertGenerationConfig:
    params = dict(INSTANCE_PROFILES[profile_name].parameters)
    return MDExpertGenerationConfig(
        count=count,
        seed_start=seed_start,
        task_group_prefix=f"md-{profile_name}-milp-il",
        instance_prefix=f"md-{profile_name}-milp-il",
        task_count=params["task_count"],
        transport_ratio=params["transport_ratio"],
        precedence_density=params["precedence_density"],
        critical_path_length=params["critical_path_length"],
        capacity_slack=params["capacity_slack"],
        speed_ratio=params["speed_ratio"],
        process_robot_count=params["process_robot_count"],
        transport_robot_count=params["transport_robot_count"],
        skill_count=params["skill_count"],
        time_limit_seconds=time_limit_seconds,
        threads=4,
    )


def run(
    output_root: Path,
    *,
    balanced_count: int,
    scale_medium_count: int,
    scaled42_count: int,
    epochs: int,
    seed: int,
    device: str,
) -> dict:
    dataset_dir = output_root / "dataset"
    output_root.mkdir(parents=True, exist_ok=True)

    # Three shards, all pointed at the same dataset_dir. instance_prefix
    # differs so no path collisions; seed ranges are disjoint.
    shards = []
    if balanced_count > 0:
        shards.append(("balanced", _profile_config("balanced", 80200, balanced_count)))
    if scale_medium_count > 0:
        shards.append(("scale_medium", _profile_config("scale_medium", 82000, scale_medium_count)))
    if scaled42_count > 0:
        shards.append(("scale42", _scaled_42_config(84000, scaled42_count)))

    combined_summary = {"shards": []}
    for shard_name, cfg in shards:
        summary = generate_md_expert_dataset(dataset_dir, config=cfg, overwrite=False)
        combined_summary["shards"].append(
            {
                "shard": shard_name,
                "requested_count": summary.requested_count,
                "generated_count": summary.generated_count,
                "skipped_count": summary.skipped_count,
                "skipped": [
                    {
                        "instance_id": skip.instance_id,
                        "seed": skip.seed,
                        "status": skip.status,
                        "message": skip.message,
                    }
                    for skip in summary.skipped
                ],
            }
        )
    (output_root / "dataset_generation_summary.json").write_text(
        json.dumps(combined_summary, indent=2) + "\n"
    )

    device_obj = torch.device(device)
    il_config = _c0_training_config(epochs=epochs, seed=seed)
    paths = run_md_offline_il_pilot(
        dataset_dir,
        output_root,
        config=il_config,
        device=device_obj,
    )
    return {
        "output_root": str(output_root),
        "dataset_generation": combined_summary,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/md_c0_milp_supervised_scale_pilot_2026-09-13"),
    )
    parser.add_argument("--balanced-count", type=int, default=100)
    parser.add_argument("--scale-medium-count", type=int, default=100)
    parser.add_argument("--scaled42-count", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output,
                balanced_count=args.balanced_count,
                scale_medium_count=args.scale_medium_count,
                scaled42_count=args.scaled42_count,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
            ),
            indent=2,
        )
    )
