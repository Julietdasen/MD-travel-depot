#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
LOG=c0_milp_supervised_scale.log
OUT=reports/md_c0_milp_supervised_scale_pilot_2026-09-13
SCALE_EVAL=reports/md_c0_milp_supervised_scale_eval_2026-09-13
PROFILE_EVAL=reports/md_c0_milp_supervised_scale_profile_eval_2026-09-13

echo "=== [$(date)] START multi-scale MILP-IL pilot ===" | tee -a "$LOG"

echo "=== [$(date)] STAGE data+train ===" | tee -a "$LOG"
python -m experiments.md_c0_milp_supervised_scale_pilot \
  --output "$OUT" \
  --balanced-count 100 \
  --scale-medium-count 100 \
  --scaled42-count 50 \
  --epochs 200 \
  --seed 3101 \
  --device cuda:0 >> "$LOG" 2>&1

CKPT="$OUT/training/best_checkpoint.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found at $CKPT" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] STAGE scale eval ===" | tee -a "$LOG"
python -m experiments.md_c0_milp_supervised_scale_eval \
  --checkpoint "$CKPT" \
  --output "$SCALE_EVAL" \
  --device cuda:0 >> "$LOG" 2>&1

echo "=== [$(date)] STAGE profile eval ===" | tee -a "$LOG"
python -m experiments.md_c0_milp_supervised_profile_eval \
  --checkpoint "$CKPT" \
  --output "$PROFILE_EVAL" \
  --device cuda:0 >> "$LOG" 2>&1

echo "=== [$(date)] DONE ===" | tee -a "$LOG"
