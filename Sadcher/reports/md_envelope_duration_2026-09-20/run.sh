#!/usr/bin/env bash
set -euo pipefail
CKPT=reports/md_c0_pathC_v1_stability_2026-09-18/seed3101/training/checkpoint_epoch_5.pt
OUT_BASE=reports/md_envelope_duration_2026-09-20
export MRTA_MILP_SOLVER=ortools
declare -A RANGES
RANGES[narrow]="8,12"
RANGES[medium]="5,20"
RANGES[wide]="3,40"
for name in narrow medium wide; do
  r=${RANGES[$name]}
  lo=${r%%,*}; hi=${r##*,}
  echo "=== duration=${name} (${lo},${hi}) ==="
  /root/miniconda3/envs/md/bin/python -m experiments.md_c0_scaled_target_profiles_eval \
    --checkpoint "$CKPT" \
    --profiles process_scarce \
    --tc-list 24 \
    --override-process-robot-count 4 \
    --override-transport-robot-count 4 \
    --override-process-duration-min "$lo" \
    --override-process-duration-max "$hi" \
    --output "${OUT_BASE}/${name}" \
    --device cuda:0
done
echo "SWEEP2 DONE"
