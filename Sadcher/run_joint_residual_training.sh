#!/usr/bin/env bash
set -euo pipefail
PY=/data/ZJZ/miniconda3/envs/mrta-sadcher/bin/python
ROOT=/data/ZJZ/3D-Foundation-Model/MRTA/Sadcher
REPORT=$ROOT/reports/md_joint_residual_2026-09-05
cd "$ROOT"
for S in 3101 3102 3103; do
  C0=$ROOT/reports/md_c0_end_to_end_diagnostic_pilot_2026-09-01/training/C0_seed${S}/checkpoints/best_checkpoint.pt
  DATA=$ROOT/reports/md_residual_scale10_2026-09-04/dataset_seed${S}
  if [[ ! -f "$REPORT/c0_init_seed${S}/training_summary.json" ]]; then
    "$PY" -u -m imitation_learning.md_joint_residual_train --c0-dataset-root datasets/md_expert_v1_small --residual-dataset "$DATA" --legacy-checkpoint "$C0" --output-dir "$REPORT/c0_init_seed${S}" --seed "$S" --epochs 100 --batch-size 32 --device cuda:0
  fi
  RES=$ROOT/reports/md_residual_scale10_2026-09-04/training/seed${S}/C0_residual_tail_seed${S}/best_checkpoint.pt
  if [[ ! -f "$REPORT/residual_init_seed${S}/training_summary.json" ]]; then
    "$PY" -u -m imitation_learning.md_joint_residual_train --c0-dataset-root datasets/md_expert_v1_small --residual-dataset "$DATA" --legacy-checkpoint "$C0" --initialization-checkpoint "$RES" --output-dir "$REPORT/residual_init_seed${S}" --seed "$S" --epochs 100 --batch-size 32 --device cuda:0
  fi
done
