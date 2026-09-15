#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
LOG=pure_milp_baseline.log

echo "=== [$(date)] START pure MILP baseline ===" | tee -a "$LOG"
python -m experiments.md_pure_milp_baseline_pilot \
  --output reports/md_pure_milp_baseline_pilot_2026-09-13 \
  --time-limit 300 \
  --threads 4 >> "$LOG" 2>&1
echo "=== [$(date)] DONE ===" | tee -a "$LOG"
