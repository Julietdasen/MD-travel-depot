#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
export CUDA_VISIBLE_DEVICES=0
LOG=/root/autodl-tmp/MD/Sadcher/c0_profile_eval.log
echo "=== [$(date)] START C0 PROFILE EVAL ===" > $LOG
timeout 3600 python experiments/md_c0_profile_eval_pilot.py \
  --checkpoint-root reports/md_c0_gpu_retrain_pilot_2026-09-13/training \
  --output reports/md_c0_profile_eval_pilot_2026-09-13 \
  --device cuda:0 >> $LOG 2>&1 || echo "[c0_profile_eval failed/timeout]" >> $LOG
echo "=== [$(date)] DONE ===" >> $LOG
