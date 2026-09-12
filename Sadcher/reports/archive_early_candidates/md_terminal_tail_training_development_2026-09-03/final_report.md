# C0 Terminal-Tail Training Development

Development-only result; no confirmation or production claim.

## Change

Each legacy C0 checkpoint was fine-tuned for 100 epochs on 2,000 relational
pairs: 400 regular and 100 terminal-tail pairs per family. Terminal-tail labels
optimize joint assignment utility with a penalty on the maximum estimated robot
return time. The exact 2,000 selected pairs are stored in each run directory.

## Training Diagnostic

| Seed | Legacy regular | New regular | Legacy terminal | New terminal |
|---:|---:|---:|---:|---:|
| 3101 | 0.8878 | 0.8650 | 0.3450 | 0.6550 |
| 3102 | 0.9237 | 0.9062 | 0.3550 | 0.6800 |
| 3103 | 0.9831 | 0.9531 | 0.3375 | 0.7638 |

The added supervision was learned, with a small regular-label tradeoff. This is
training-set evidence only.

## Same-Instance End-To-End Diagnostic

All 90 new rollouts succeeded on the original 30 frozen instances.

| Seed | Makespan delta | Real completion delta | Return-tail delta | Better / tie / worse |
|---:|---:|---:|---:|---:|
| 3101 | -0.067 | -1.833 | +1.767 | 7 / 16 / 7 |
| 3102 | +3.067 | +0.667 | +2.400 | 3 / 20 / 7 |
| 3103 | -1.133 | +1.233 | -2.367 | 11 / 10 / 9 |
| Mean | +0.622 | +0.022 | +0.600 | 21 / 46 / 23 |

The current augmentation does not establish an end-to-end improvement. It
corrects terminal-aware synthetic labels, but two seeds have a worse return
tail and the cross-seed mean makespan is slightly worse.

## Interpretation

The legacy checkpoint lacked explicit terminal-return supervision. Adding that
supervision is learnable, but the synthetic one-step assignment objective does
not represent future task order, robot reuse, process coalitions, or the final
bottleneck robot well enough to improve full rollouts consistently. The next
dataset revision should label real late-stage simulator states using forced
first-action rollout or MILP completion cost, not further increase the weight
of this synthetic proxy.

## Reproducibility

New runs persist exact training pairs, CPU checkpoint state dictionaries,
epoch logs, deterministic-algorithm settings, and checkpoint reload checks.
Legacy C0 reproducibility status is documented under
`archives/c0_ticket46_legacy_2026-09-03/README.md`.
