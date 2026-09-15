#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
export CUDA_VISIBLE_DEVICES=0
ROOT=reports/md_c0_gpu_retrain_pilot_2026-09-13
LOG=/root/autodl-tmp/MD/Sadcher/c0_gpu_train.log
echo "=== [$(date)] FREEZE ===" > $LOG
python -m experiments.md_c0_end_to_end_diagnostic_pilot freeze --root $ROOT >> $LOG 2>&1
echo "=== [$(date)] SMOKE ===" >> $LOG
python -m experiments.md_c0_end_to_end_diagnostic_pilot smoke --root $ROOT --device cuda:0 >> $LOG 2>&1
for seed in 3101 3102 3103; do
  echo "=== [$(date)] TRAIN SEED $seed ===" >> $LOG
  python -m experiments.md_c0_end_to_end_diagnostic_pilot train-c0 --root $ROOT --device cuda:0 --seed $seed >> $LOG 2>&1
done
echo "=== [$(date)] RUN FORMAL PILOT ===" >> $LOG
python -m experiments.md_c0_end_to_end_diagnostic_pilot run --root $ROOT --device cuda:0 >> $LOG 2>&1
echo "=== [$(date)] C0 GPU TRAIN ALL DONE ===" >> $LOG
