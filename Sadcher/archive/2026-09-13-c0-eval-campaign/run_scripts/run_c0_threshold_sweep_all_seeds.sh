#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
LOG=c0_threshold_sweep_all_seeds.log

echo "=== [$(date)] START threshold sweep for all 3 seeds ===" | tee -a "$LOG"
for seed in 3102 3103; do
  echo "--- seed $seed ---" | tee -a "$LOG"
  python -m experiments.md_c0_confidence_threshold_sweep \
    --checkpoint reports/md_c0_gpu_retrain_pilot_2026-09-13/training/C0_seed${seed}/checkpoints/best_checkpoint.pt \
    --model-seed ${seed} \
    --output reports/md_c0_confidence_threshold_sweep_2026-09-13_seed${seed} \
    --device cuda:0 >> "$LOG" 2>&1
done
echo "=== [$(date)] DONE ===" | tee -a "$LOG"
