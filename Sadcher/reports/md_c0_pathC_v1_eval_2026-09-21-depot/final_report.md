# Path C v1 (margin ranking loss) evaluation, 2026-09-17

Fine-tuned from `reports/md_c0_baseline_2026-09-21-depot/training/best_checkpoint.pt` on `reports/md_c0_pathA_2026-09-21-depot/dataset` (9091 train / 1241 val samples), lr=5e-06 K=5 margin=1.0 seed=3101 epochs=20.

## Primary judgement (60-task process_scarce)

Target: mean makespan across the 4 seeds <= 395.75 (baseline C0 = 398.75, greedy_unlock = 395.75). Improvement threshold >= 3 timesteps.

| epoch | 60-task process_scarce makespan | vs baseline C0 | vs greedy_unlock | pass? |
|---:|---:|---:|---:|:---:|
| 5 | 1210.50 | +811.75 | +814.75 | no |
| 10 | 1178.75 | +780.00 | +783.00 | no |
| 15 | 1168.75 | +770.00 | +773.00 | no |
| 20 | 1128.75 | +730.00 | +733.00 | no |

**Primary verdict: FAILED.**
Best 60-task process_scarce makespan across epochs = 1128.75 (epoch 20); still worse than target 395.75 and baseline C0 398.75.

### Comparison vs prior attempts (60-task process_scarce)

| approach | 60-task process_scarce makespan |
|---|---:|
| baseline C0 (pre-Path A) | 398.75 |
| greedy_unlock | 395.75 |
| Path A (h=16 fine-tune) | 405.2 |
| Path B1 (h=48 scratch) | 415.0 |
| Path B2 (h=64 scratch) | 413.5 |
| Path C v1 epoch 5 | 1210.5 |
| Path C v1 epoch 10 | 1178.75 |
| Path C v1 epoch 15 | 1168.75 |
| Path C v1 epoch 20 | 1128.75 |

## Full per-tier table (all epochs)

For each (profile, task_count) the row shows baseline C0 (pre-Path A) then Path C v1 makespan per epoch.

| profile | tc | baseline C0 | best greedy | MILP | ep5 | ep10 | ep15 | ep20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dependency_deep | 12 | 245.5 | 245.75 (greedy_unlock) | 226.5 | 481.75 | 483.50 | 483.50 | 444.25 |
| dependency_deep | 24 | 284.0 | 281.5 (greedy_unlock) | 262.25 | 761.00 | 777.50 | 770.00 | 736.00 |
| dependency_deep | 42 | 383.25 | 392.25 (greedy_unlock) | 342.75 | 1089.00 | 1064.25 | 1078.50 | 1042.75 |
| dependency_deep | 60 | 500.5 | 492.5 (greedy_unlock) | 444.5 | 1726.50 | 1655.25 | 1591.25 | 1558.00 |
| process_scarce | 12 | 175.75 | 207.0 (greedy_distance) | 164.25 | 632.50 | 622.00 | 622.00 | 617.50 |
| process_scarce | 24 | 263.5 | 304.0 (greedy_unlock) | 250.0 | 844.00 | 824.75 | 825.75 | 807.50 |
| process_scarce | 42 | 332.25 | 346.5 (greedy_unlock) | 304.0 | 1033.25 | 1048.75 | 1022.00 | 1002.25 |
| process_scarce | 60 | 398.75 | 395.75 (greedy_unlock) | 356.0 | 1210.50 | 1178.75 | 1168.75 | 1128.75 |

## Secondary trends

Validation loss per epoch (index 0 = pretrained, ranking objective): [0.5414, 0.524, 0.4936, 0.489, 0.4787, 0.468, 0.4643, 0.4643, 0.463, 0.4413, 0.4624, 0.4466, 0.4388, 0.4243, 0.4329, 0.4279, 0.4313, 0.4316, 0.4232, 0.425, 0.4267]

Train loss per epoch: [0.4954, 0.4844, 0.474, 0.4693, 0.4574, 0.4487, 0.441, 0.4371, 0.4313, 0.4258, 0.4255, 0.4228, 0.42, 0.4027, 0.4077, 0.3989, 0.4005, 0.4003, 0.3985, 0.3863]

### Val loss vs 60-task process_scarce makespan direction

* epoch 5: val_loss = 0.468, 60-task process_scarce makespan = 1210.5
* epoch 10: val_loss = 0.4624, 60-task process_scarce makespan = 1178.75
* epoch 15: val_loss = 0.4279, 60-task process_scarce makespan = 1168.75
* epoch 20: val_loss = 0.4267, 60-task process_scarce makespan = 1128.75

### Tier non-regression count vs baseline C0 (out of 8)

* epoch 5: 0/8 tiers not regressed vs baseline C0
* epoch 10: 0/8 tiers not regressed vs baseline C0
* epoch 15: 0/8 tiers not regressed vs baseline C0
* epoch 20: 0/8 tiers not regressed vs baseline C0

## Interpretation guidance

* If **any** checkpoint has 60-task process_scarce makespan <= 395.75 (i.e. beats greedy_unlock and improves >=3 timesteps over baseline C0 398.75), the ranking signal is directionally correct. Advance to Path C v2 (add continuation-makespan-based labels + listwise loss).
* If no checkpoint clears the primary, but *all* four checkpoints stay at least as good as Path A (405.2) / B1 (415.0) / B2 (413.5), the ranking loss is not enough on its own; continuation-value supervision is needed.
* If any checkpoint is worse than Path A/B1/B2 on 60-task process_scarce (makespan > 415), ranking loss is a strictly worse direction; abandon and pivot to candidate beta (add MILP-provided score margins) or candidate gamma.

