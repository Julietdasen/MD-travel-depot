#!/usr/bin/env bash
set -euo pipefail
PY=/data/ZJZ/miniconda3/envs/mrta-sadcher/bin/python
ROOT=/data/ZJZ/3D-Foundation-Model/MRTA/Sadcher
REPORT=$ROOT/reports/md_joint_residual_2026-09-05
cd "$ROOT"

run_one() {
  local seed=$1 init=$2 gpu=$3
  local c0=$ROOT/reports/md_c0_end_to_end_diagnostic_pilot_2026-09-01/training/C0_seed${seed}/checkpoints/best_checkpoint.pt
  local data=$ROOT/reports/md_residual_scale10_2026-09-04/dataset_seed${seed}
  local out=$REPORT/${init}_init_v2_seed${seed}
  local args=()
  if [[ $init == residual ]]; then
    args+=(--initialization-checkpoint "$ROOT/reports/md_residual_scale10_2026-09-04/training/seed${seed}/C0_residual_tail_seed${seed}/best_checkpoint.pt")
  fi
  "$PY" -u -m imitation_learning.md_joint_residual_train_v2 \
    --c0-dataset-root datasets/md_expert_v1_small --residual-dataset "$data" \
    --legacy-checkpoint "$c0" --output-dir "$out" --seed "$seed" \
    --epochs 100 --batch-size 32 --device "cuda:$gpu" "${args[@]}" \
    > "$REPORT/${init}_init_v2_seed${seed}.log" 2>&1
}

run_one 3101 c0 0 &
run_one 3101 residual 1 &
run_one 3102 c0 2 &
run_one 3102 residual 3 &
run_one 3103 c0 4 &
run_one 3103 residual 5 &
wait
