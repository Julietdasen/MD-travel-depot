# Path C v1 (margin ranking loss) evaluation, 2026-09-17

Fine-tuned from `reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt` on `reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset` (9838 train / 1291 val samples), lr=5e-06 K=5 margin=1.0 seed=3101 epochs=20.

## Primary judgement (60-task process_scarce)

Target: mean makespan across the 4 seeds <= 395.75 (baseline C0 = 398.75, greedy_unlock = 395.75). Improvement threshold >= 3 timesteps.

| epoch | 60-task process_scarce makespan | vs baseline C0 | vs greedy_unlock | pass? |
|---:|---:|---:|---:|:---:|
| 5 | 393.50 | -5.25 | -2.25 | YES |
| 10 | 401.00 | +2.25 | +5.25 | no |
| 15 | 401.75 | +3.00 | +6.00 | no |
| 20 | 402.25 | +3.50 | +6.50 | no |

**Primary verdict: PASSED.**
Checkpoints beating threshold: [5].

### Comparison vs prior attempts (60-task process_scarce)

| approach | 60-task process_scarce makespan |
|---|---:|
| baseline C0 (pre-Path A) | 398.75 |
| greedy_unlock | 395.75 |
| Path A (h=16 fine-tune) | 405.2 |
| Path B1 (h=48 scratch) | 415.0 |
| Path B2 (h=64 scratch) | 413.5 |
| Path C v1 epoch 5 | 393.5 |
| Path C v1 epoch 10 | 401.0 |
| Path C v1 epoch 15 | 401.75 |
| Path C v1 epoch 20 | 402.25 |

## Full per-tier table (all epochs)

For each (profile, task_count) the row shows baseline C0 (pre-Path A) then Path C v1 makespan per epoch.

| profile | tc | baseline C0 | best greedy | MILP | ep5 | ep10 | ep15 | ep20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dependency_deep | 12 | 245.5 | 245.75 (greedy_unlock) | 226.5 | 244.50 | 245.25 | 245.25 | 245.00 |
| dependency_deep | 24 | 284.0 | 281.5 (greedy_unlock) | 262.25 | 288.50 | 283.00 | 288.25 | 283.00 |
| dependency_deep | 42 | 383.25 | 392.25 (greedy_unlock) | 342.75 | 372.25 | 378.25 | 370.75 | 379.25 |
| dependency_deep | 60 | 500.5 | 492.5 (greedy_unlock) | 444.5 | 487.00 | 478.00 | 480.75 | 482.75 |
| process_scarce | 12 | 175.75 | 207.0 (greedy_distance) | 164.25 | 175.75 | 175.50 | 175.50 | 175.75 |
| process_scarce | 24 | 263.5 | 304.0 (greedy_unlock) | 250.0 | 270.25 | 275.75 | 267.75 | 275.00 |
| process_scarce | 42 | 332.25 | 346.5 (greedy_unlock) | 304.0 | 338.75 | 333.25 | 336.25 | 332.50 |
| process_scarce | 60 | 398.75 | 395.75 (greedy_unlock) | 356.0 | 393.50 | 401.00 | 401.75 | 402.25 |

## Secondary trends

Validation loss per epoch (index 0 = pretrained, ranking objective): [0.6738, 0.6674, 0.6472, 0.6339, 0.6282, 0.6244, 0.6303, 0.6198, 0.6108, 0.6161, 0.6001, 0.5948, 0.593, 0.5824, 0.5875, 0.5849, 0.582, 0.5717, 0.5786, 0.573, 0.5674]

Train loss per epoch: [0.6325, 0.6125, 0.6039, 0.5932, 0.586, 0.5753, 0.5656, 0.5623, 0.56, 0.5548, 0.5519, 0.5454, 0.542, 0.5415, 0.539, 0.5313, 0.533, 0.5293, 0.5321, 0.5263]

### Val loss vs 60-task process_scarce makespan direction

* epoch 5: val_loss = 0.6244, 60-task process_scarce makespan = 393.5
* epoch 10: val_loss = 0.6001, 60-task process_scarce makespan = 401.0
* epoch 15: val_loss = 0.5849, 60-task process_scarce makespan = 401.75
* epoch 20: val_loss = 0.5674, 60-task process_scarce makespan = 402.25

### Tier non-regression count vs baseline C0 (out of 8)

* epoch 5: 5/8 tiers not regressed vs baseline C0
* epoch 10: 5/8 tiers not regressed vs baseline C0
* epoch 15: 4/8 tiers not regressed vs baseline C0
* epoch 20: 5/8 tiers not regressed vs baseline C0

## Interpretation guidance

* If **any** checkpoint has 60-task process_scarce makespan <= 395.75 (i.e. beats greedy_unlock and improves >=3 timesteps over baseline C0 398.75), the ranking signal is directionally correct. Advance to Path C v2 (add continuation-makespan-based labels + listwise loss).
* If no checkpoint clears the primary, but *all* four checkpoints stay at least as good as Path A (405.2) / B1 (415.0) / B2 (413.5), the ranking loss is not enough on its own; continuation-value supervision is needed.
* If any checkpoint is worse than Path A/B1/B2 on 60-task process_scarce (makespan > 415), ranking loss is a strictly worse direction; abandon and pivot to candidate beta (add MILP-provided score margins) or candidate gamma.

