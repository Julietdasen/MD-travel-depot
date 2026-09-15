#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
export CUDA_VISIBLE_DEVICES=0
LOG=/root/autodl-tmp/MD/Sadcher/c0_threshold_sweep.log
echo "=== [$(date)] START THRESHOLD SWEEP ===" > $LOG
timeout 3600 python experiments/md_c0_confidence_threshold_sweep.py \
  --checkpoint reports/md_c0_gpu_retrain_pilot_2026-09-13/training/C0_seed3101/checkpoints/best_checkpoint.pt \
  --model-seed 3101 \
  --output reports/md_c0_confidence_threshold_sweep_2026-09-13 \
  --device cuda:0 >> $LOG 2>&1 || echo "[threshold_sweep failed/timeout]" >> $LOG
echo "=== [$(date)] DONE ===" >> $LOG
