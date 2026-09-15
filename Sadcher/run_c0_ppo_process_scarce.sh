#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
export MD_RAY_TEMP_DIR=/root/autodl-tmp/MD/Sadcher/tmp_ray
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
mkdir -p tmp_ray runs

ITERATIONS=${ITERATIONS:-60}
TASK_COUNT=${TASK_COUNT:-42}
KL_WEIGHT=${KL_WEIGHT:-0.005}
OUT=${OUT:-runs/md_ray_ppo_process_scarce_tc${TASK_COUNT}_2026-09-14}
LOG=${LOG:-rl_process_scarce_tc${TASK_COUNT}.log}

echo "=== [$(date)] START RL on scaled process_scarce (tc=$TASK_COUNT, $ITERATIONS iter, kl=$KL_WEIGHT) ===" | tee -a "$LOG"
python -u -m experiments.md_c0_ppo_process_scarce_pilot \
  --il-checkpoint reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt \
  --task-count "$TASK_COUNT" \
  --seed 0 \
  --iterations "$ITERATIONS" \
  --kl-weight "$KL_WEIGHT" \
  --num-env-runners 0 \
  --num-gpus 1.0 \
  --output-dir "$OUT" >> "$LOG" 2>&1
echo "=== [$(date)] DONE ===" | tee -a "$LOG"
