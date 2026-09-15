#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
export MD_RAY_TEMP_DIR=/root/autodl-tmp/MD/Sadcher/tmp_ray
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
mkdir -p tmp_ray runs
LOG=rl_train.log
ITERATIONS=${ITERATIONS:-30}
OUT=runs/md_ray_ppo_train_2026-09-14

echo "=== [$(date)] START RL training ($ITERATIONS iterations, warm-start from MILP-IL v2) ===" | tee -a "$LOG"
python -u -m reinforcement_learning.md_ray_ppo \
  --il-checkpoint reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt \
  --seed 0 \
  --iterations "$ITERATIONS" \
  --num-env-runners 0 \
  --num-gpus 1.0 \
  --output-dir "$OUT" >> "$LOG" 2>&1
echo "=== [$(date)] DONE ===" | tee -a "$LOG"
