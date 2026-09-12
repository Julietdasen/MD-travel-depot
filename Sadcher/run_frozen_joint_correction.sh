#!/usr/bin/env bash
set -euo pipefail
PY=/data/ZJZ/miniconda3/envs/mrta-sadcher/bin/python
ROOT=/data/ZJZ/3D-Foundation-Model/MRTA/Sadcher
REPORT=$ROOT/reports/md_frozen_joint_correction_2026-09-06
cd "$ROOT"
mkdir -p "$REPORT"
for S in 3101 3102 3103; do
  GPU=$((S - 3101))
  "$PY" -u -m imitation_learning.md_frozen_joint_correction_train \
    --residual-dataset "$ROOT/reports/md_residual_scale10_2026-09-04/dataset_seed${S}" \
    --residual-checkpoint "$ROOT/reports/md_residual_scale10_2026-09-04/training/seed${S}/C0_residual_tail_seed${S}/best_checkpoint.pt" \
    --output-dir "$REPORT/seed${S}" --seed "$S" --epochs 100 --device "cuda:$GPU" \
    > "$REPORT/seed${S}.log" 2>&1 &
done
wait
