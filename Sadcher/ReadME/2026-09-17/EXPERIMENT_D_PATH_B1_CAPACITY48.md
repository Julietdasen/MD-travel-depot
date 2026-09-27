# Experiment D — Path B1: gentle capacity upgrade (hidden_dim = 48)

## Motivation
Path A (hidden_dim = 16 fine-tune on the process_scarce + dependency_deep
scaled shards) failed: 60-task process_scarce makespan went from 398.75
(baseline C0) to 405.25 (Path A), and val loss ↔ downstream makespan
decoupled. Hypothesis: the C0 hidden_dim = 16 model does not have enough
capacity to jointly represent both structural signatures. Path B1 tests a
gentle capacity bump; B2 (hidden_dim = 64) is a parallel run.

## Setup
- Script: `experiments/md_c0_pathB_capacity_upgrade.py` (shared with B2).
- Dataset: `reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset/` — 300
  new process_scarce + dependency_deep scaled shards plus 250 legacy
  balanced / scale_medium / scale42 shards (task-level split).
- Capacity vs C0: `hidden_dim` 16 → 48, `embed_dim` 16 → 48, `ff_dim` 32 →
  96, `transformer_layers` 1 → 2, `gat_layers` 1 → 2, `dropout` 0 → 0.1.
  All other flags (pair-aware attention, no cross-attention, feature
  toggles) unchanged.
- Training: from scratch, Adam + cosine schedule (peak 1e-3, 5-epoch linear
  warmup, min 1e-5), batch_size 32, seed 3101, cuda:0, 100 epochs, early
  stop patience 15.

## Training outcome
Early-stopped at epoch 38 (best epoch 23). Best val loss = 0.3407, well
below Path A's 0.4066. Train loss kept dropping to 0.278 by epoch 38 while
val loss plateaued near 0.34 — overfit signal even with dropout.

## Downstream eval

| profile | tc | baseline C0 | Path A | **B1** | greedy | B1 vs baseline | B1 vs greedy |
|---|---:|---:|---:|---:|---:|---:|---:|
| dependency_deep | 12 | 245.50 | 245.50 | **249.25** | 245.75 | +1.5% | +1.4% |
| dependency_deep | 24 | 284.00 | 282.25 | **314.25** | 281.50 | +10.7% | +11.6% |
| dependency_deep | 42 | 383.25 | 374.75 | **384.75** | 392.25 | +0.4% | -1.9% |
| dependency_deep | 60 | 500.50 | 496.50 | **499.75** | 492.50 | -0.1% | +1.5% |
| process_scarce  | 12 | 175.75 | 175.25 | **174.50** | 207.00 | -0.7% | -15.7% |
| process_scarce  | 24 | 263.50 | 269.25 | **268.50** | 304.00 | +1.9% | -11.7% |
| process_scarce  | 42 | 332.25 | 338.50 | **339.25** | 346.50 | +2.1% | -2.1% |
| process_scarce  | 60 | 398.75 | 405.25 | **415.00** | 395.75 | +4.1% | +4.9% |

## Verdict — FAIL
- **Primary criterion** (process_scarce tc=60 vs greedy_unlock ≥ 0): FAIL.
  B1 = +4.86 % worse than greedy, vs +0.76 % for baseline C0.
- **Secondary** (no regression vs baseline C0): FAIL on 6 / 8 tiers.
  dependency_deep tc=24 gets much worse (284.0 → 314.3, +10.7 %).
- **Trade-off eliminated?** No. Both profiles are worse than baseline at
  tc=60, and dependency_deep tc=24 blows up.
- **Loss ↔ makespan decoupling**: the same pathology Path A showed at
  hidden=16 reappears at hidden=48 — B1's best val loss is much lower than
  Path A's, but downstream makespan is worse. Doubling depth and 3×
  widening hidden_dim on a ~10 k-sample training set overfits.

## Artifacts
- Checkpoint: `reports/md_c0_pathB1_capacity48_2026-09-17/training/best_checkpoint.pt`
- Training summary: `reports/md_c0_pathB1_capacity48_2026-09-17/training/training_summary.json`
- Training log: `reports/md_c0_pathB1_capacity48_2026-09-17/run.log`
- Eval summary: `reports/md_c0_pathB1_capacity48_eval_2026-09-17/summary.json`
- Eval final report: `reports/md_c0_pathB1_capacity48_eval_2026-09-17/final_report.md`
- Full B1 report: `reports/md_c0_pathB1_capacity48_2026-09-17/final_report.md`

## Follow-up
- B2 (hidden_dim = 64) is a data point on whether more capacity would
  amplify or attenuate the overfit; the B1 signal makes a monotone-worse
  outcome likely without additional data or regularization.
- If capacity direction is retired, next lever is training signal
  (structured / margin loss, value head, MCTS-labeled data) on the C0
  hidden = 16 backbone.
