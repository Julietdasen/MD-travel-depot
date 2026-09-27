# Path B1 — capacity-upgrade training (hidden_dim = 48) — final report

## Setup
- Script: `experiments/md_c0_pathB_capacity_upgrade.py` (shared with B2).
- Dataset: `reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset/` (300 new
  scaled process_scarce + dependency_deep shards + 250 legacy balanced /
  scale_medium / scale42 shards, task-level split via `experiments.protocol`).
- Architecture (delta vs C0):
  - `hidden_dim` 16 → 48
  - `embed_dim` 16 → 48 (`ff_dim` auto-scaled 32 → 96)
  - `transformer_layers` 1 → 2
  - `gat_layers` 1 → 2
  - `dropout` 0.0 → 0.1
  - All other flags kept identical to `_c0_training_config` (pair-aware
    attention, no cross-attention, same feature toggles).
- Training: from scratch, Adam, cosine LR (peak 1e-3, 5-epoch linear warmup,
  min 1e-5), batch_size = 32, seed = 3101, device cuda:0, 100 epochs with
  early stopping on validation loss (patience 15).

## Training outcome
- Ran 38 / 100 epochs (early-stopped after 15 no-improvement epochs).
- Best validation loss = **0.3407** at epoch 23 (vs Path A's best 0.4066 at
  hidden=16).
- Train/val gap widened from epoch 23 onward (train kept dropping to ~0.28
  while val plateaued near 0.34) — classic small-dataset overfit signal even
  with 10 % dropout.

## Downstream evaluation (`md_c0_scaled_target_profiles_eval`)

| profile | tc | MILP | C0 (baseline hidden=16) | Path A (hidden=16 FT) | **B1 (hidden=48)** | best greedy | C0 gap | A gap | B1 gap | B1 vs greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dependency_deep | 12 | 226.5 | 245.50 | 245.50 | **249.25** | 245.75 (unlock) | +8.4% | +8.4% | +10.0% | +1.4% |
| dependency_deep | 24 | 262.2 | 284.00 | 282.25 | **314.25** | 281.50 (unlock) | +8.3% | +7.6% | +19.8% | +11.6% |
| dependency_deep | 42 | 342.8 | 383.25 | 374.75 | **384.75** | 392.25 (unlock) | +11.8% | +9.3% | +12.3% | -1.9% |
| dependency_deep | 60 | 444.5 | 500.50 | 496.50 | **499.75** | 492.50 (unlock) | +12.6% | +11.7% | +12.4% | +1.5% |
| process_scarce  | 12 | 164.2 | 175.75 | 175.25 | **174.50** | 207.00 (distance) | +7.0% | +6.7% | +6.2% | -15.7% |
| process_scarce  | 24 | 250.0 | 263.50 | 269.25 | **268.50** | 304.00 (unlock) | +5.4% | +7.7% | +7.4% | -11.7% |
| process_scarce  | 42 | 304.0 | 332.25 | 338.50 | **339.25** | 346.50 (unlock) | +9.3% | +11.4% | +11.6% | -2.1% |
| process_scarce  | 60 | 356.0 | 398.75 | 405.25 | **415.00** | 395.75 (unlock) | +12.0% | +13.8% | +16.6% | +4.9% |

## Judgment vs success criteria
- **Primary (process_scarce tc=60 vs greedy_unlock ≥ 0)**: **FAIL**. B1 =
  415.00 vs 395.75, i.e. **+4.86 %** *worse* than greedy_unlock. Baseline C0
  was already at +0.76 % on this tier, so B1 regressed further, not closer.
- **Secondary (no regression on other tiers)**: **FAIL**. B1 regresses vs
  baseline C0 on 6 of 8 tiers. Worst: dependency_deep tc=24 goes from
  284.00 → 314.25 (+30.25 makespan, +10.7 %). process_scarce tc=60 goes from
  398.75 → 415.00 (+16.25 makespan, +4.1 %).
- **Trade-off eliminated?**: **NO**. Bigger capacity did not fix Path A's
  process_scarce ↔ dependency_deep tension; both profiles are worse than
  baseline at 60 tasks, and dependency_deep tc=24 became much worse.

## Diagnosis
Best-val-loss B1 (0.341) is well below Path A's (0.407) yet downstream
makespan is worse — the loss ↔ makespan decoupling that Path A exhibited
at hidden=16 also holds at hidden=48. Doubling the transformer / GAT depth
plus 3× wider hidden dim on a 9838-sample training set drove `train_loss`
from 0.398 to 0.278 in 38 epochs while `val_loss` plateaued at ~0.34 by
epoch 23. Combined with the downstream regression, the signal is:
this size class overfits the ~10 k decision samples of the combined
dataset. To scale capacity further B2 (hidden=64) would need substantially
more data, stronger regularization, or a different training signal
(structured loss / margin loss / value head). Result of B1 alone: **capacity
scaling in this regime is net-negative**.
