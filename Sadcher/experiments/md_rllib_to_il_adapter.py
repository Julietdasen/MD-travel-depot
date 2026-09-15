"""Convert an RLlib PPO checkpoint (from md_ray_ppo) into IL checkpoint
format so it can be loaded by imitation_learning.md_train.load_md_policy_checkpoint
and evaluated by the same scripts used for MILP-IL v2 (six-profile eval and
scale-ladder eval).

RLlib stores model weights under `weights` (numpy state_dict with `net.*`
prefix). The IL checkpoint format expects:
  - schema_version, model_kind, state_dict, model_spec, md_policy_config

We take the RLlib weights, strip the `net.` prefix, cast to torch tensors,
and copy the model_spec / md_policy_config from the IL warm-start checkpoint.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

MD_IL_CHECKPOINT_VERSION = "1.0.0"


def convert(rllib_checkpoint_dir: Path, il_warm_start_checkpoint: Path, output_path: Path) -> None:
    policy_state_path = rllib_checkpoint_dir / "policies" / "default_policy" / "policy_state.pkl"
    with open(policy_state_path, "rb") as f:
        rllib_state = pickle.load(f)

    numpy_weights = rllib_state["weights"]
    torch_state_dict = {}
    for key, value in numpy_weights.items():
        # Strip 'net.' prefix that MDRLlibModel wraps around the underlying network
        # only keep the trained actor net.*, drop reference_net.* (frozen IL copy) and value_head.*
        if not key.startswith("net."):
            continue
        clean_key = key[len("net."):]
        torch_state_dict[clean_key] = torch.from_numpy(value)  # kept

    # Copy config metadata from the IL warm-start checkpoint (same architecture)
    warm = torch.load(il_warm_start_checkpoint, map_location="cpu", weights_only=False)
    payload = {
        "schema_version": MD_IL_CHECKPOINT_VERSION,
        "model_kind": "md_enhanced",
        "state_dict": torch_state_dict,
        "model_spec": warm["model_spec"],
        "md_policy_config": warm["md_policy_config"],
        "provenance": {
            "converted_from_rllib": str(rllib_checkpoint_dir),
            "il_warm_start": str(il_warm_start_checkpoint),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"wrote IL-format checkpoint to {output_path} ({len(torch_state_dict)} tensors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rllib-checkpoint", type=Path, required=True,
                        help="Path to RLlib checkpoint directory (e.g. runs/.../best_iter_0020)")
    parser.add_argument("--il-warm-start", type=Path,
                        default=Path("reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt"),
                        help="IL checkpoint the RL warm-started from — used to copy model_spec + md_policy_config")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.rllib_checkpoint, args.il_warm_start, args.output)
