#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
LOG=c0_scaled_target_profiles_eval.log

echo "=== [$(date)] START scaled target-profiles eval ===" | tee -a "$LOG"
python -u -m experiments.md_c0_scaled_target_profiles_eval \
  --checkpoint reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt \
  --output reports/md_c0_scaled_target_profiles_eval_2026-09-14 \
  --device cuda:0 >> "$LOG" 2>&1
echo "=== [$(date)] DONE ===" | tee -a "$LOG"
