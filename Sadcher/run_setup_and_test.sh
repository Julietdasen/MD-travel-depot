#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate md
cd /root/autodl-tmp/MD/Sadcher

LOG=/root/autodl-tmp/MD/Sadcher/setup_and_test.log
echo "=== [$(date)] START INSTALL ===" > $LOG

pip install --no-input \
  "ipywidgets>=8.1.5" "matplotlib>=3.7.5" "pandas>=2.0.3" "pulp>=2.9.0" \
  "torch==2.4.0" "torch-geometric>=2.6.1" "tqdm>=4.67.1" "pyyaml>=6.0.2" \
  "icecream>=2.1.4" "optuna>=4.2.1" "opencv-python>=4.11.0.86" \
  "gymnasium>=1.1.1" "tensorboard>=2.19.0" "skrl[torch]>=1.4.3" \
  "ray[rllib]>=2.49,<2.50" pytest >> $LOG 2>&1

echo "=== [$(date)] INSTALL DONE ===" >> $LOG

echo "=== [$(date)] SMOKE TESTS ===" >> $LOG
python -m pytest tests/test_md_instance_profiles.py tests/test_md_policy_smoke.py -v >> $LOG 2>&1 || echo "[pytest failed]" >> $LOG

echo "=== [$(date)] PROFILE PILOT ===" >> $LOG
timeout 1800 python experiments/md_profile_pilot.py >> $LOG 2>&1 || echo "[profile_pilot failed/timeout]" >> $LOG

echo "=== [$(date)] PREFIX PILOT ===" >> $LOG

echo "=== [$(date)] ALL DONE ===" >> $LOG
