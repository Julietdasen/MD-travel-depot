#!/bin/bash
# Path C B2 regret-weighted driver: seed=3101, 20 epochs, per-epoch checkpoints,
# then evaluate every checkpoint on the 32-instance scaled-target grid.
set -eu -o pipefail

cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export MRTA_MILP_SOLVER=ortools

PY=/root/miniconda3/envs/md/bin/python
ROOT=reports/md_c0_pathC_b2_regret_weighted_2026-09-20
DATASET=reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset
PRETRAINED=reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt
SEED=3101
OUT=${ROOT}/seed${SEED}
mkdir -p ${OUT}

echo "[$(date -Is)] driver start pid=$$" | tee ${ROOT}/driver.log

if [ -f "${OUT}/training/training_summary.json" ]; then
  echo "[$(date -Is)] seed=${SEED} training already done, skip" | tee -a ${ROOT}/driver.log
else
  echo "[$(date -Is)] seed=${SEED} train start" | tee -a ${ROOT}/driver.log
  ${PY} -m experiments.md_c0_pathC_b2_regret_weighted \
    --output ${OUT} \
    --pretrained-checkpoint ${PRETRAINED} \
    --dataset-dir ${DATASET} \
    --epochs 20 \
    --learning-rate 5e-6 \
    --negatives 5 \
    --margin 1.0 \
    --seed ${SEED} \
    --device cuda:0 \
    --checkpoint-epochs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    > ${OUT}/run.log 2>&1
  echo "[$(date -Is)] seed=${SEED} train done" | tee -a ${ROOT}/driver.log
fi

if [ -f "${OUT}/eval/final_report.md" ]; then
  echo "[$(date -Is)] seed=${SEED} eval already done, skip" | tee -a ${ROOT}/driver.log
else
  echo "[$(date -Is)] seed=${SEED} eval start" | tee -a ${ROOT}/driver.log
  ${PY} -m experiments.md_c0_pathC_ranking_v1_eval \
    --training-dir ${OUT}/training \
    --training-summary ${OUT}/training/training_summary.json \
    --output ${OUT}/eval \
    --epochs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    --device cuda:0 \
    > ${OUT}/eval.log 2>&1
  echo "[$(date -Is)] seed=${SEED} eval done" | tee -a ${ROOT}/driver.log
fi

echo "[$(date -Is)] driver finished ok" | tee -a ${ROOT}/driver.log
