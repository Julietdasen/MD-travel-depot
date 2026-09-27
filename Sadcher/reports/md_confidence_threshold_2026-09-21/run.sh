#!/bin/bash
# Threshold ablation at tc=24 process_scarce (envelope sweet spot).
# Baseline (threshold=0.0): C0 with fallback only when confidence < 0.
# Higher thresholds -> more MILP fallbacks -> stronger but slower.
# Confidence distribution p10=-1.25, p25=-0.07, p50=0.0, p75=1.25 (from
# reports/md_confidence_dist_2026-09-21/tc24_pr4_scarce0/summary.json).
set -e
CKPT=reports/md_c0_pathC_v1_stability_2026-09-18/seed3101/training/checkpoint_epoch_5.pt
PY=/root/miniconda3/envs/md/bin/python
export MRTA_MILP_SOLVER=ortools
OUT_ROOT=reports/md_confidence_threshold_2026-09-21
LOG="$OUT_ROOT/run.log"
mkdir -p "$OUT_ROOT"

echo "=== threshold ablation start ===" | tee "$LOG"

for th in 0.0 0.1 0.5 1.5; do
  OUT="$OUT_ROOT/th${th}"
  echo "--- threshold=$th ---" | tee -a "$LOG"
  $PY -m experiments.md_c0_scaled_target_profiles_eval \
    --checkpoint "$CKPT" \
    --profiles process_scarce \
    --tc-list 24 \
    --override-scarce-skill-count 1 \
    --confidence-threshold "$th" \
    --output "$OUT" \
    --skip-greedies \
    --device cuda:0 2>&1 | tee -a "$LOG"
done
echo "=== THRESHOLD DONE ===" | tee -a "$LOG"
