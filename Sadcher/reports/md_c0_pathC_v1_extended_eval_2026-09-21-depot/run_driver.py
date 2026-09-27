"""Wrapper that monkey-patches SEEDS to 8-seed range and calls the existing
scaled_target_profiles_eval `run()` function.  We do not modify the eval script
itself; instead we override the module-level SEEDS constant before invoking run().
"""
from __future__ import annotations

import json
from pathlib import Path

import experiments.md_c0_scaled_target_profiles_eval as mod

mod.SEEDS = (301, 302, 303, 304, 305, 306, 307, 308)

CHECKPOINT = Path("reports/md_c0_pathC_v1_2026-09-21-depot/training/checkpoint_epoch_20.pt")
OUTPUT = Path("reports/md_c0_pathC_v1_extended_eval_2026-09-21-depot")

summary = mod.run(
    checkpoint=CHECKPOINT,
    output=OUTPUT,
    device_name="cpu",
    profiles=("process_scarce", "dependency_deep"),
    task_counts=(24, 42, 60),
    milp_time_limit_seconds=60.0,
)

print(json.dumps({"schema": summary["schema"], "seeds": summary["seeds"]}, indent=2))
print("DONE")
