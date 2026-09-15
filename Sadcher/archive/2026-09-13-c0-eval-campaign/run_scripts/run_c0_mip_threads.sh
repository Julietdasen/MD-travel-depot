#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
LOG=c0_mip_threads.log

echo "=== [$(date)] START MIP threads sweep ===" | tee -a "$LOG"
python -m experiments.md_c0_mip_threads_pilot \
  --checkpoint reports/md_c0_gpu_retrain_pilot_2026-09-13/training/C0_seed3101/checkpoints/best_checkpoint.pt \
  --model-seed 3101 \
  --output reports/md_c0_mip_threads_pilot_2026-09-13 \
  --device cuda:0 >> "$LOG" 2>&1
echo "=== [$(date)] DONE ===" | tee -a "$LOG"
