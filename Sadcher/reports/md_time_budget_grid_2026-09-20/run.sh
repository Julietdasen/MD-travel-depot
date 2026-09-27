#!/usr/bin/env bash
set -euo pipefail
CKPT=reports/md_c0_pathC_v1_stability_2026-09-18/seed3101/training/checkpoint_epoch_5.pt
OUT_BASE=reports/md_time_budget_grid_2026-09-20
export MRTA_MILP_SOLVER=ortools
for tc in 60 80; do
  for budget in 5 15 60 300; do
    tag="tc${tc}_b${budget}"
    echo "=== ${tag} ==="
    /root/miniconda3/envs/md/bin/python -m experiments.md_c0_scaled_target_profiles_eval \
      --checkpoint "$CKPT" \
      --profiles process_scarce \
      --tc-list ${tc} \
      --milp-time-limit ${budget} \
      --skip-greedies \
      --output "${OUT_BASE}/${tag}" \
      --device cuda:0
  done
done
echo "TIME_BUDGET DONE"
