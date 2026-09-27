#!/bin/bash
# 3-seed stability verification.
# Verify the two headline data points from RESULTS_SUMMARY:
#   - tc=60 pr=10 scarce=3 : C0 vs greedy = -6.9pp (headline: scarce救大规模)
#   - tc=24 pr=4  scarce=1 : C0 vs greedy = -11.6pp (envelope sweet spot)
# on seed3102 and seed3103 ep5, to confirm the shape isn't a single-seed artifact.
set -e
PY=/root/miniconda3/envs/md/bin/python
export MRTA_MILP_SOLVER=ortools
OUT_ROOT=reports/md_3seed_stability_verification_2026-09-21
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"

echo "=== 3-seed verification start ===" | tee "$LOG"

for CKPT_SEED in 3102 3103; do
  CKPT="reports/md_c0_pathC_v1_stability_2026-09-18/seed${CKPT_SEED}/training/checkpoint_epoch_5.pt"

  # Headline: tc60/pr10/scarce=3
  OUT="$OUT_ROOT/seed${CKPT_SEED}_tc60_pr10_scarce3"
  echo "--- ckpt=seed${CKPT_SEED} tc=60 pr=10 scarce=3 ---" | tee -a "$LOG"
  $PY -m experiments.md_c0_scaled_target_profiles_eval \
    --checkpoint "$CKPT" \
    --profiles process_scarce \
    --tc-list 60 \
    --override-process-robot-count 10 \
    --override-transport-robot-count 10 \
    --override-scarce-skill-count 3 \
    --output "$OUT" \
    --device cuda:0 2>&1 | tee -a "$LOG"

  # Envelope sweet spot: tc24/pr4/scarce=1
  OUT="$OUT_ROOT/seed${CKPT_SEED}_tc24_pr4_scarce1"
  echo "--- ckpt=seed${CKPT_SEED} tc=24 pr=4 scarce=1 ---" | tee -a "$LOG"
  $PY -m experiments.md_c0_scaled_target_profiles_eval \
    --checkpoint "$CKPT" \
    --profiles process_scarce \
    --tc-list 24 \
    --override-process-robot-count 4 \
    --override-transport-robot-count 4 \
    --override-scarce-skill-count 1 \
    --output "$OUT" \
    --device cuda:0 2>&1 | tee -a "$LOG"
done

echo "=== 3-SEED VERIFICATION DONE ===" | tee -a "$LOG"
