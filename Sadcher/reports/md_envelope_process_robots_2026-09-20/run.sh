#!/usr/bin/env bash
set -euo pipefail
CKPT=reports/md_c0_pathC_v1_stability_2026-09-18/seed3101/training/checkpoint_epoch_5.pt
OUT_BASE=reports/md_envelope_process_robots_2026-09-20
export MRTA_MILP_SOLVER=ortools
for pr in 2 3 4 6 8; do
  echo "=== process_robot_count=$pr ==="
  /root/miniconda3/envs/md/bin/python -m experiments.md_c0_scaled_target_profiles_eval \
    --checkpoint "$CKPT" \
    --profiles process_scarce \
    --tc-list 24 \
    --override-process-robot-count "$pr" \
    --override-transport-robot-count 4 \
    --output "${OUT_BASE}/pr${pr}" \
    --device cuda:0
done
echo "SWEEP3 DONE"
