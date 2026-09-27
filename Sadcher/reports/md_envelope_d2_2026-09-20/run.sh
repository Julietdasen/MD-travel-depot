#!/usr/bin/env bash
set -euo pipefail
CKPT=reports/md_c0_pathC_v1_stability_2026-09-18/seed3101/training/checkpoint_epoch_5.pt
OUT_BASE=reports/md_envelope_d2_2026-09-20
export MRTA_MILP_SOLVER=ortools

echo "=========== D2a: scarce_skill sweep (tc=24, pr=4, tr=4, skill=3) ==========="
for scarce in 0 1 2 3; do
  echo "--- scarce_skill=$scarce ---"
  /root/miniconda3/envs/md/bin/python -m experiments.md_c0_scaled_target_profiles_eval \
    --checkpoint "$CKPT" \
    --profiles process_scarce \
    --tc-list 24 \
    --override-process-robot-count 4 \
    --override-transport-robot-count 4 \
    --override-scarce-skill-count $scarce \
    --output "${OUT_BASE}/d2a_scarce${scarce}" \
    --device cuda:0
done

echo "=========== D2b: scale-preserved sweep (pr/process_task=0.2, skill=3, scarce=1) ==========="
# tc=24 → pr=4 tr=4; tc=42 → pr=7 tr=7; tc=60 → pr=10 tr=10
for combo in "24,4,4" "42,7,7" "60,10,10"; do
  IFS=',' read -r tc pr tr <<< "$combo"
  echo "--- tc=$tc pr=$pr tr=$tr scarce=1 ---"
  /root/miniconda3/envs/md/bin/python -m experiments.md_c0_scaled_target_profiles_eval \
    --checkpoint "$CKPT" \
    --profiles process_scarce \
    --tc-list $tc \
    --override-process-robot-count $pr \
    --override-transport-robot-count $tr \
    --override-scarce-skill-count 1 \
    --output "${OUT_BASE}/d2b_tc${tc}_pr${pr}" \
    --device cuda:0
done
echo "D2 DONE"
