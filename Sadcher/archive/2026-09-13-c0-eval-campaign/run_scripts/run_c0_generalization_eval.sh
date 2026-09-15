#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
export CUDA_VISIBLE_DEVICES=0
LOG=/root/autodl-tmp/MD/Sadcher/c0_generalization_eval.log
echo "=== [$(date)] START HELD-OUT GENERALIZATION EVAL ===" > $LOG
timeout 3600 python experiments/md_c0_profile_eval_pilot.py \
  --checkpoint-root reports/md_c0_gpu_retrain_pilot_2026-09-13/training \
  --output reports/md_c0_generalization_eval_2026-09-13 \
  --device cuda:0 \
  --seeds 9101 9102 9103 9104 9105 9106 9107 9108 9109 9110 >> $LOG 2>&1 || echo "[generalization_eval failed/timeout]" >> $LOG
echo "=== [$(date)] DONE ===" >> $LOG
