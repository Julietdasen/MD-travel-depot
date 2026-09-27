# Experiment E — Path B2: capacity=64 from-scratch train

## Hypothesis

Path A (same-capacity fine-tune, hidden=16) failed to close the 60-task
process_scarce gap and slightly regressed vs baseline C0. Diagnosis: hidden=16
is too small — val loss keeps decreasing while downstream makespan degrades,
so loss and makespan decouple. B2 tests whether the more aggressive capacity
step (hidden=64, 4× baseline) trained from scratch on the augmented Path A
dataset eliminates the process_scarce vs dependency_deep trade-off.

## Setup

- Script: `experiments/md_c0_pathB_capacity_upgrade.py` (shared with B1).
- Architecture: `hidden_dim=embed_dim=64`, `ff_dim=128`, `transformer_layers=2`,
  `gat_layers=2`, `dropout=0.1`, C0 backbone flags preserved.
- Optimisation: Adam, cosine LR (peak 1e-3, warmup 5, min 1e-5), batch 32,
  seed 3101, from scratch, 100 epochs cap, early stop patience 15.
- Dataset: `reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset`
  (300 scaled-profile shards + 250 legacy symlinks; 9838 train / 1291 val samples).
- Ran 40 epochs (early stop). Best epoch 25, val loss 0.339690.

## Result

Downstream eval `md_c0_scaled_target_profiles_eval` on 2 profiles × 4 task
counts × 4 seeds.

- **60-task process_scarce**: B2 413.5 vs greedy_unlock 395.75 → **+4.49%**
  (worse than greedy, worse than baseline C0 at 398.75). Primary criterion
  FAIL.
- **process_scarce tc=42**: 346.0 vs baseline 332.25 → +13.75 makespan
  regression.
- **dependency_deep tc=12**: 265.0 vs baseline 245.5 → +19.5 regression.
- **dependency_deep tc=60**: 492.25 vs baseline 500.5 → -8.25 improvement,
  and gap vs greedy_unlock closed from +1.62% to -0.05%.
- **process_scarce tc=12**: -15.82% vs greedy (best of all runs).

## Verdict

**FAIL.** B2 does not close the 60-task process_scarce gap; it is worse than
baseline C0. The trade-off is *not* eliminated. B2 regresses on
dependency_deep tc=12/24 while barely improving tc=60. Capacity monotonicity
(B2 vs B1 hidden=48) is inconclusive — B2 is not systematically better than
B1.

## Interpretation

Combined with Path A's identical failure signature (val loss down, makespan
sideways), this points to the augmented dataset itself as the ceiling: the
two profile gradients pull in conflicting directions and 300 additional
shards + baseline pilot data is not enough for a fresh backbone to resolve
them, regardless of capacity. Recommended next tracks: profile-aware loss
reweighting, per-profile curriculum, or larger 60-task shard counts to
saturate the loss on the hardest tier.

## Artifacts

- Training: `reports/md_c0_pathB2_capacity64_2026-09-17/`
- Eval: `reports/md_c0_pathB2_capacity64_eval_2026-09-17/`
- Report: `reports/md_c0_pathB2_capacity64_eval_2026-09-17/final_report.md`
- Sibling: `EXPERIMENT_D_PATH_B1_CAPACITY48.md`
