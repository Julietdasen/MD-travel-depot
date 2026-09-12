#!/usr/bin/env bash
set -euo pipefail
PY=/data/ZJZ/miniconda3/envs/mrta-sadcher/bin/python
ROOT=/data/ZJZ/3D-Foundation-Model/MRTA/Sadcher
C0=$ROOT/reports/md_c0_end_to_end_diagnostic_pilot_2026-09-01/training
RES=$ROOT/reports/md_residual_scale10_2026-09-04/training
JOINT=$ROOT/reports/md_frozen_joint_correction_2026-09-06
cd "$ROOT"
exec "$PY" -u -m experiments.md_frozen_joint_evaluation \
  --output-dir reports/md_frozen_joint_correction_2026-09-06/evaluation \
  --c0-checkpoint 3101="$C0/C0_seed3101/checkpoints/best_checkpoint.pt" \
  --c0-checkpoint 3102="$C0/C0_seed3102/checkpoints/best_checkpoint.pt" \
  --c0-checkpoint 3103="$C0/C0_seed3103/checkpoints/best_checkpoint.pt" \
  --residual-checkpoint 3101="$RES/seed3101/C0_residual_tail_seed3101/best_checkpoint.pt" \
  --residual-checkpoint 3102="$RES/seed3102/C0_residual_tail_seed3102/best_checkpoint.pt" \
  --residual-checkpoint 3103="$RES/seed3103/C0_residual_tail_seed3103/best_checkpoint.pt" \
  --joint-checkpoint 3101="$JOINT/seed3101/best_checkpoint.pt" \
  --joint-checkpoint 3102="$JOINT/seed3102/best_checkpoint.pt" \
  --joint-checkpoint 3103="$JOINT/seed3103/best_checkpoint.pt" \
  --device cpu
