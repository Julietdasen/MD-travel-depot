# Path C v1 (margin ranking loss) evaluation, 2026-09-17

Fine-tuned from `reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt` on `reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset` (9838 train / 1291 val samples), lr=5e-06 K=5 margin=1.0 seed=3102 epochs=20.

## Primary judgement (60-task process_scarce)

Target: mean makespan across the 4 seeds <= 395.75 (baseline C0 = 398.75, greedy_unlock = 395.75). Improvement threshold >= 3 timesteps.

| epoch | 60-task process_scarce makespan | vs baseline C0 | vs greedy_unlock | pass? |
|---:|---:|---:|---:|:---:|
| 1 | 397.75 | -1.00 | +2.00 | no |
| 2 | 396.25 | -2.50 | +0.50 | no |
| 3 | 404.00 | +5.25 | +8.25 | no |
| 4 | 405.75 | +7.00 | +10.00 | no |
| 5 | 393.00 | -5.75 | -2.75 | YES |
| 6 | 396.25 | -2.50 | +0.50 | no |
| 7 | 397.00 | -1.75 | +1.25 | no |
| 8 | 398.25 | -0.50 | +2.50 | no |
| 9 | 411.50 | +12.75 | +15.75 | no |
| 10 | 412.50 | +13.75 | +16.75 | no |
| 11 | 398.75 | +0.00 | +3.00 | no |
| 12 | 398.25 | -0.50 | +2.50 | no |
| 13 | 410.50 | +11.75 | +14.75 | no |
| 14 | 401.75 | +3.00 | +6.00 | no |
| 15 | 401.75 | +3.00 | +6.00 | no |
| 16 | 401.75 | +3.00 | +6.00 | no |
| 17 | 397.50 | -1.25 | +1.75 | no |
| 18 | 404.00 | +5.25 | +8.25 | no |
| 19 | 404.25 | +5.50 | +8.50 | no |
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
| Path C v1 epoch 1 | 397.75 |
| Path C v1 epoch 2 | 396.25 |
| Path C v1 epoch 3 | 404.0 |
| Path C v1 epoch 4 | 405.75 |
| Path C v1 epoch 5 | 393.0 |
| Path C v1 epoch 6 | 396.25 |
| Path C v1 epoch 7 | 397.0 |
| Path C v1 epoch 8 | 398.25 |
| Path C v1 epoch 9 | 411.5 |
| Path C v1 epoch 10 | 412.5 |
| Path C v1 epoch 11 | 398.75 |
| Path C v1 epoch 12 | 398.25 |
| Path C v1 epoch 13 | 410.5 |
| Path C v1 epoch 14 | 401.75 |
| Path C v1 epoch 15 | 401.75 |
| Path C v1 epoch 16 | 401.75 |
| Path C v1 epoch 17 | 397.5 |
| Path C v1 epoch 18 | 404.0 |
| Path C v1 epoch 19 | 404.25 |
| Path C v1 epoch 20 | 402.25 |

## Full per-tier table (all epochs)

For each (profile, task_count) the row shows baseline C0 (pre-Path A) then Path C v1 makespan per epoch.

| profile | tc | baseline C0 | best greedy | MILP | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 | ep10 | ep11 | ep12 | ep13 | ep14 | ep15 | ep16 | ep17 | ep18 | ep19 | ep20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dependency_deep | 12 | 245.5 | 245.75 (greedy_unlock) | 226.5 | 244.25 | 244.25 | 244.25 | 244.50 | 244.50 | 244.00 | 244.00 | 244.50 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.25 | 245.00 | 245.00 |
| dependency_deep | 24 | 284.0 | 281.5 (greedy_unlock) | 262.25 | 285.00 | 285.00 | 285.00 | 284.75 | 288.50 | 288.50 | 291.50 | 288.25 | 283.00 | 283.00 | 282.25 | 288.25 | 288.25 | 288.25 | 283.00 | 280.75 | 283.00 | 283.00 | 283.00 | 283.00 |
| dependency_deep | 42 | 383.25 | 392.25 (greedy_unlock) | 342.75 | 382.75 | 373.75 | 374.00 | 373.50 | 372.25 | 372.50 | 381.00 | 381.00 | 381.75 | 377.75 | 369.00 | 369.00 | 362.75 | 370.75 | 370.25 | 377.00 | 370.25 | 381.00 | 380.50 | 379.25 |
| dependency_deep | 60 | 500.5 | 492.5 (greedy_unlock) | 444.5 | 492.25 | 490.25 | 490.25 | 490.50 | 487.00 | 479.75 | 480.25 | 481.75 | 477.25 | 477.50 | 476.75 | 483.50 | 483.00 | 480.75 | 482.50 | 482.50 | 482.00 | 488.25 | 482.50 | 483.00 |
| process_scarce | 12 | 175.75 | 207.0 (greedy_distance) | 164.25 | 175.75 | 175.75 | 175.75 | 175.50 | 175.75 | 175.75 | 175.75 | 175.75 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.50 | 175.75 | 175.75 | 175.75 |
| process_scarce | 24 | 263.5 | 304.0 (greedy_unlock) | 250.0 | 263.25 | 263.25 | 270.25 | 270.25 | 270.25 | 275.75 | 275.75 | 275.75 | 275.75 | 275.75 | 275.75 | 275.75 | 275.25 | 275.00 | 275.00 | 275.00 | 275.00 | 275.00 | 275.00 | 275.00 |
| process_scarce | 42 | 332.25 | 346.5 (greedy_unlock) | 304.0 | 337.50 | 339.50 | 340.00 | 338.75 | 338.75 | 338.75 | 336.25 | 338.50 | 338.50 | 338.50 | 338.00 | 338.00 | 338.00 | 338.00 | 338.00 | 332.75 | 332.75 | 332.75 | 332.75 | 332.50 |
| process_scarce | 60 | 398.75 | 395.75 (greedy_unlock) | 356.0 | 397.75 | 396.25 | 404.00 | 405.75 | 393.00 | 396.25 | 397.00 | 398.25 | 411.50 | 412.50 | 398.75 | 398.25 | 410.50 | 401.75 | 401.75 | 401.75 | 397.50 | 404.00 | 404.25 | 402.25 |

## Secondary trends

Validation loss per epoch (index 0 = pretrained, ranking objective): [0.6794, 0.6627, 0.6511, 0.635, 0.632, 0.6291, 0.6077, 0.6221, 0.6064, 0.6051, 0.6032, 0.6022, 0.5928, 0.591, 0.5836, 0.5828, 0.5761, 0.5743, 0.58, 0.5733, 0.576]

Train loss per epoch: [0.6328, 0.6184, 0.6026, 0.5941, 0.5834, 0.5747, 0.568, 0.5668, 0.5579, 0.558, 0.5514, 0.5473, 0.5469, 0.5427, 0.5367, 0.5365, 0.5319, 0.5339, 0.5291, 0.5255]

### Val loss vs 60-task process_scarce makespan direction

* epoch 1: val_loss = 0.6627, 60-task process_scarce makespan = 397.75
* epoch 2: val_loss = 0.6511, 60-task process_scarce makespan = 396.25
* epoch 3: val_loss = 0.635, 60-task process_scarce makespan = 404.0
* epoch 4: val_loss = 0.632, 60-task process_scarce makespan = 405.75
* epoch 5: val_loss = 0.6291, 60-task process_scarce makespan = 393.0
* epoch 6: val_loss = 0.6077, 60-task process_scarce makespan = 396.25
* epoch 7: val_loss = 0.6221, 60-task process_scarce makespan = 397.0
* epoch 8: val_loss = 0.6064, 60-task process_scarce makespan = 398.25
* epoch 9: val_loss = 0.6051, 60-task process_scarce makespan = 411.5
* epoch 10: val_loss = 0.6032, 60-task process_scarce makespan = 412.5
* epoch 11: val_loss = 0.6022, 60-task process_scarce makespan = 398.75
* epoch 12: val_loss = 0.5928, 60-task process_scarce makespan = 398.25
* epoch 13: val_loss = 0.591, 60-task process_scarce makespan = 410.5
* epoch 14: val_loss = 0.5836, 60-task process_scarce makespan = 401.75
* epoch 15: val_loss = 0.5828, 60-task process_scarce makespan = 401.75
* epoch 16: val_loss = 0.5761, 60-task process_scarce makespan = 401.75
* epoch 17: val_loss = 0.5743, 60-task process_scarce makespan = 397.5
* epoch 18: val_loss = 0.58, 60-task process_scarce makespan = 404.0
* epoch 19: val_loss = 0.5733, 60-task process_scarce makespan = 404.25
* epoch 20: val_loss = 0.576, 60-task process_scarce makespan = 402.25

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
* epoch 13: 4/8 tiers not regressed vs baseline C0
* epoch 14: 4/8 tiers not regressed vs baseline C0
* epoch 15: 5/8 tiers not regressed vs baseline C0
* epoch 16: 5/8 tiers not regressed vs baseline C0
* epoch 17: 6/8 tiers not regressed vs baseline C0
* epoch 18: 5/8 tiers not regressed vs baseline C0
* epoch 19: 5/8 tiers not regressed vs baseline C0
* epoch 20: 5/8 tiers not regressed vs baseline C0

## Interpretation guidance

* If **any** checkpoint has 60-task process_scarce makespan <= 395.75 (i.e. beats greedy_unlock and improves >=3 timesteps over baseline C0 398.75), the ranking signal is directionally correct. Advance to Path C v2 (add continuation-makespan-based labels + listwise loss).
* If no checkpoint clears the primary, but *all* four checkpoints stay at least as good as Path A (405.2) / B1 (415.0) / B2 (413.5), the ranking loss is not enough on its own; continuation-value supervision is needed.
* If any checkpoint is worse than Path A/B1/B2 on 60-task process_scarce (makespan > 415), ranking loss is a strictly worse direction; abandon and pivot to candidate beta (add MILP-provided score margins) or candidate gamma.

