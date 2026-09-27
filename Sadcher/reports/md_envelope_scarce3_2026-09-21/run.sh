#!/bin/bash
# scarce_skill_count=3 sweep: the "true coalition" regime given skill_count=3.
# Every skill is scarce (owned by exactly one robot), so coalition composition
# choice is unavoidable for 40-50% of process tasks (see coalition_diagnostic).
#
# Runs at scale-preserved pr scaling (pr = round(2 * tc/12)):
#   tc=24, pr=4, tr=4
#   tc=42, pr=7, tr=7
#   tc=60, pr=10, tr=10
#
# Uses same 4 seeds as envelope reports for direct comparability.
set -e
CKPT=reports/md_c0_pathC_v1_stability_2026-09-18/seed3101/training/checkpoint_epoch_5.pt
PY=/root/miniconda3/envs/md/bin/python
export MRTA_MILP_SOLVER=ortools
OUT_ROOT=reports/md_envelope_scarce3_2026-09-21
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"

echo "=== scarce=3 sweep start ===" | tee "$LOG"

for cfg in "24 4 4" "42 7 7" "60 10 10"; do
  read tc pr tr <<< "$cfg"
  OUT="$OUT_ROOT/tc${tc}_pr${pr}"
  echo "--- tc=$tc pr=$pr tr=$tr scarce=3 ---" | tee -a "$LOG"
  $PY -m experiments.md_c0_scaled_target_profiles_eval \
    --checkpoint "$CKPT" \
    --profiles process_scarce \
    --tc-list "$tc" \
    --override-process-robot-count "$pr" \
    --override-transport-robot-count "$tr" \
    --override-scarce-skill-count 3 \
    --output "$OUT" \
    --device cuda:0 2>&1 | tee -a "$LOG"
done

echo "=== SCARCE3 DONE ===" | tee -a "$LOG"
