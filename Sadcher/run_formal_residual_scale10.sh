#!/usr/bin/env bash
set -euo pipefail

PY=/data/ZJZ/miniconda3/envs/mrta-sadcher/bin/python
ROOT=/data/ZJZ/3D-Foundation-Model/MRTA/Sadcher
REPORT=/data/ZJZ/3D-Foundation-Model/MRTA/Sadcher/reports/md_residual_scale10_2026-09-04
cd "$ROOT"
mkdir -p "$REPORT/logs"

for S in 3101 3102 3103; do
  OUT="$REPORT/dataset_seed${S}"
  C0="$ROOT/reports/md_c0_end_to_end_diagnostic_pilot_2026-09-01/training/C0_seed${S}/checkpoints/best_checkpoint.pt"
  "$PY" -u -m data_generation.md_residual_pipeline \
    --output-dir "$OUT" --model-seed "$S" --c0-checkpoint "$S=$C0" \
    --splits train development --workers 4 --time-limit 10 --resume
  "$PY" -u -m imitation_learning.md_residual_train \
    --c0-dataset-root datasets/md_expert_v1_small --residual-dataset "$OUT" \
    --legacy-checkpoint "$C0" --output-dir "$REPORT/training/seed${S}" \
    --seed "$S" --epochs 100 --batch-size 32
done
