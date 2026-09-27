# Path C v1 (margin ranking loss) evaluation, 2026-09-17

Fine-tuned from `reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt` on `reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset` (9838 train / 1291 val samples), lr=5e-06 K=5 margin=1.0 seed=3101 epochs=20.

## Primary judgement (60-task process_scarce)

Target: mean makespan across the 4 seeds <= 395.75 (baseline C0 = 398.75, greedy_unlock = 395.75). Improvement threshold >= 3 timesteps.

| epoch | 60-task process_scarce makespan | vs baseline C0 | vs greedy_unlock | pass? |
|---:|---:|---:|---:|:---:|
| 1 | 397.75 | -1.00 | +2.00 | no |
| 2 | 396.25 | -2.50 | +0.50 | no |
| 3 | 407.50 | +8.75 | +11.75 | no |
| 4 | 404.25 | +5.50 | +8.50 | no |
| 5 | 393.50 | -5.25 | -2.25 | YES |
| 6 | 396.25 | -2.50 | +0.50 | no |
| 7 | 395.00 | -3.75 | -0.75 | YES |
| 8 | 394.00 | -4.75 | -1.75 | YES |
| 9 | 411.50 | +12.75 | +15.75 | no |
| 10 | 401.00 | +2.25 | +5.25 | no |
| 11 | 398.25 | -0.50 | +2.50 | no |
| 12 | 402.50 | +3.75 | +6.75 | no |
| 13 | 405.75 | +7.00 | +10.00 | no |
| 14 | 401.75 | +3.00 | +6.00 | no |
| 15 | 401.75 | +3.00 | +6.00 | no |
| 16 | 397.25 | -1.50 | +1.50 | no |
| 17 | 400.50 | +1.75 | +4.75 | no |
| 18 | 406.25 | +7.50 | +10.50 | no |
| 19 | 404.25 | +5.50 | +8.50 | no |
| 20 | 402.25 | +3.50 | +6.50 | no |

**Primary verdict: PASSED.**
Checkpoints beating threshold: [5, 7, 8].

### Comparison vs prior attempts (60-task process_scarce)

| approach | 60-task process_scarce makespan |
|---|---:|
| baseline C0 (pre-Path A) | 398.75 |
| greedy_unlock | 395.75 |
| Path A (h=16 fine-tune) | 405.2 |
| Path B1 (h=48 scratch) | 415.0 |
| Path B2 (h=64 scratch) | 413.5 |
| Path C v1 epoch 1 | 397.75 |
| Path C v1 epoch 2 | 396.25 |
| Path C v1 epoch 3 | 407.5 |
| Path C v1 epoch 4 | 404.25 |
| Path C v1 epoch 5 | 393.5 |
| Path C v1 epoch 6 | 396.25 |
| Path C v1 epoch 7 | 395.0 |
| Path C v1 epoch 8 | 394.0 |
| Path C v1 epoch 9 | 411.5 |
| Path C v1 epoch 10 | 401.0 |
| Path C v1 epoch 11 | 398.25 |
| Path C v1 epoch 12 | 402.5 |
| Path C v1 epoch 13 | 405.75 |
| Path C v1 epoch 14 | 401.75 |
| Path C v1 epoch 15 | 401.75 |
| Path C v1 epoch 16 | 397.25 |
| Path C v1 epoch 17 | 400.5 |
| Path C v1 epoch 18 | 406.25 |
| Path C v1 epoch 19 | 404.25 |
| Path C v1 epoch 20 | 402.25 |

## Full per-tier table (all epochs)

For each (profile, task_count) the row shows baseline C0 (pre-Path A) then Path C v1 makespan per epoch.

| profile | tc | baseline C0 | best greedy | MILP | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 | ep10 | ep11 | ep12 | ep13 | ep14 | ep15 | ep16 | ep17 | ep18 | ep19 | ep20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dependency_deep | 12 | 245.5 | 245.75 (greedy_unlock) | 226.5 | 244.25 | 244.25 | 244.25 | 244.50 | 244.50 | 244.00 | 244.00 | 244.50 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.00 | 245.00 |
| dependency_deep | 24 | 284.0 | 281.5 (greedy_unlock) | 262.25 | 285.00 | 285.00 | 285.00 | 284.75 | 288.50 | 288.50 | 291.50 | 288.25 | 283.00 | 283.00 | 282.25 | 282.25 | 282.25 | 288.25 | 288.25 | 280.75 | 283.00 | 283.00 | 283.00 | 283.00 |
| dependency_deep | 42 | 383.25 | 392.25 (greedy_unlock) | 342.75 | 382.75 | 381.25 | 373.75 | 373.50 | 372.25 | 372.50 | 372.50 | 381.00 | 377.75 | 378.25 | 369.00 | 369.00 | 369.25 | 370.75 | 370.75 | 377.75 | 370.25 | 370.25 | 380.50 | 379.25 |
| dependency_deep | 60 | 500.5 | 492.5 (greedy_unlock) | 444.5 | 492.25 | 490.50 | 493.00 | 493.50 | 487.00 | 479.75 | 479.75 | 481.75 | 477.00 | 478.00 | 476.75 | 479.50 | 483.00 | 480.75 | 480.75 | 482.25 | 488.25 | 488.25 | 482.00 | 482.75 |
| process_scarce | 12 | 175.75 | 207.0 (greedy_distance) | 164.25 | 175.75 | 175.75 | 175.75 | 175.75 | 175.75 | 175.75 | 175.75 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.75 | 175.75 | 175.75 |
| process_scarce | 24 | 263.5 | 304.0 (greedy_unlock) | 250.0 | 263.25 | 263.25 | 270.25 | 270.25 | 270.25 | 275.75 | 275.75 | 275.75 | 275.75 | 275.75 | 275.75 | 275.75 | 268.00 | 267.75 | 267.75 | 267.75 | 267.75 | 275.00 | 275.00 | 275.00 |
| process_scarce | 42 | 332.25 | 346.5 (greedy_unlock) | 304.0 | 337.50 | 339.75 | 340.00 | 339.25 | 338.75 | 336.50 | 336.25 | 338.50 | 338.50 | 333.25 | 338.00 | 338.00 | 338.00 | 338.00 | 336.25 | 336.25 | 336.25 | 336.00 | 336.00 | 332.50 |
| process_scarce | 60 | 398.75 | 395.75 (greedy_unlock) | 356.0 | 397.75 | 396.25 | 407.50 | 404.25 | 393.50 | 396.25 | 395.00 | 394.00 | 411.50 | 401.00 | 398.25 | 402.50 | 405.75 | 401.75 | 401.75 | 397.25 | 400.50 | 406.25 | 404.25 | 402.25 |

## Secondary trends

Validation loss per epoch (index 0 = pretrained, ranking objective): [0.6738, 0.6674, 0.6472, 0.6339, 0.6282, 0.6244, 0.6303, 0.6198, 0.6108, 0.6161, 0.6001, 0.5948, 0.593, 0.5824, 0.5875, 0.5849, 0.582, 0.5717, 0.5786, 0.573, 0.5674]

Train loss per epoch: [0.6325, 0.6125, 0.6039, 0.5932, 0.586, 0.5753, 0.5656, 0.5623, 0.56, 0.5548, 0.5519, 0.5454, 0.542, 0.5415, 0.539, 0.5313, 0.533, 0.5293, 0.5321, 0.5263]

### Val loss vs 60-task process_scarce makespan direction

* epoch 1: val_loss = 0.6674, 60-task process_scarce makespan = 397.75
* epoch 2: val_loss = 0.6472, 60-task process_scarce makespan = 396.25
* epoch 3: val_loss = 0.6339, 60-task process_scarce makespan = 407.5
* epoch 4: val_loss = 0.6282, 60-task process_scarce makespan = 404.25
* epoch 5: val_loss = 0.6244, 60-task process_scarce makespan = 393.5
* epoch 6: val_loss = 0.6303, 60-task process_scarce makespan = 396.25
* epoch 7: val_loss = 0.6198, 60-task process_scarce makespan = 395.0
* epoch 8: val_loss = 0.6108, 60-task process_scarce makespan = 394.0
* epoch 9: val_loss = 0.6161, 60-task process_scarce makespan = 411.5
* epoch 10: val_loss = 0.6001, 60-task process_scarce makespan = 401.0
* epoch 11: val_loss = 0.5948, 60-task process_scarce makespan = 398.25
* epoch 12: val_loss = 0.593, 60-task process_scarce makespan = 402.5
* epoch 13: val_loss = 0.5824, 60-task process_scarce makespan = 405.75
* epoch 14: val_loss = 0.5875, 60-task process_scarce makespan = 401.75
* epoch 15: val_loss = 0.5849, 60-task process_scarce makespan = 401.75
* epoch 16: val_loss = 0.582, 60-task process_scarce makespan = 397.25
* epoch 17: val_loss = 0.5717, 60-task process_scarce makespan = 400.5
* epoch 18: val_loss = 0.5786, 60-task process_scarce makespan = 406.25
* epoch 19: val_loss = 0.573, 60-task process_scarce makespan = 404.25
* epoch 20: val_loss = 0.5674, 60-task process_scarce makespan = 402.25

### Tier non-regression count vs baseline C0 (out of 8)

* epoch 1: 6/8 tiers not regressed vs baseline C0
* epoch 2: 6/8 tiers not regressed vs baseline C0
* epoch 3: 4/8 tiers not regressed vs baseline C0
* epoch 4: 4/8 tiers not regressed vs baseline C0
* epoch 5: 5/8 tiers not regressed vs baseline C0
* epoch 6: 5/8 tiers not regressed vs baseline C0
* epoch 7: 5/8 tiers not regressed vs baseline C0
* epoch 8: 5/8 tiers not regressed vs baseline C0
* epoch 9: 5/8 tiers not regressed vs baseline C0
* epoch 10: 5/8 tiers not regressed vs baseline C0
* epoch 11: 6/8 tiers not regressed vs baseline C0
* epoch 12: 5/8 tiers not regressed vs baseline C0
* epoch 13: 5/8 tiers not regressed vs baseline C0
* epoch 14: 4/8 tiers not regressed vs baseline C0
* epoch 15: 4/8 tiers not regressed vs baseline C0
* epoch 16: 6/8 tiers not regressed vs baseline C0
* epoch 17: 5/8 tiers not regressed vs baseline C0
* epoch 18: 5/8 tiers not regressed vs baseline C0
* epoch 19: 5/8 tiers not regressed vs baseline C0
* epoch 20: 5/8 tiers not regressed vs baseline C0

## Interpretation guidance

* If **any** checkpoint has 60-task process_scarce makespan <= 395.75 (i.e. beats greedy_unlock and improves >=3 timesteps over baseline C0 398.75), the ranking signal is directionally correct. Advance to Path C v2 (add continuation-makespan-based labels + listwise loss).
* If no checkpoint clears the primary, but *all* four checkpoints stay at least as good as Path A (405.2) / B1 (415.0) / B2 (413.5), the ranking loss is not enough on its own; continuation-value supervision is needed.
* If any checkpoint is worse than Path A/B1/B2 on 60-task process_scarce (makespan > 415), ranking loss is a strictly worse direction; abandon and pivot to candidate beta (add MILP-provided score margins) or candidate gamma.

