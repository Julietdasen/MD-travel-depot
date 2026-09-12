# MD Ray PPO Phase A Diagnostic

Date: 2026-09-04

## Decision

**NO-GO.** None of the five preregistered cells improved validation makespan.
The three-seed confirmation and longer 20/100-iteration runs were therefore
not started.

## Protocol

- Fixed size: 5 robots x 12 tasks.
- Model seed: 5101.
- Training: 5 PPO iterations per cell.
- Selection/evaluation: fixed 16-instance validation split only.
- IL reference:
  `runs/md_offline_il_pilot_2026-08-27/training/best_checkpoint.pt`.
- IL and RL use the same coalition-aware autoregressive action semantics.
- Latency uses one warmup call per episode and alternates IL-first/RL-first
  order by seed index.

## Results

| Cell | IL/RL makespan | Gain | W/T/L | Success | Illegal IL/RL | Starvation IL/RL | P95 ratio | Action disagreement | Best iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 251.875 / 251.875 | 0.000% | 0/16/0 | 100% / 100% | 0 / 0 | 174.688 / 174.688 | 100.92% | 1 / 4030 | 5 |
| A1 | 251.875 / 251.875 | 0.000% | 0/16/0 | 100% / 100% | 0 / 0 | 174.688 / 174.688 | 94.37% | 1 / 4030 | 5 |
| A2 | 251.875 / 251.875 | 0.000% | 0/16/0 | 100% / 100% | 0 / 0 | 174.688 / 174.688 | 100.13% | 1 / 4030 | 5 |
| A3 | 251.875 / 251.875 | 0.000% | 0/16/0 | 100% / 100% | 0 / 0 | 174.688 / 174.688 | 100.08% | 0 / 4030 | 5 |
| A4 | 251.875 / 251.875 | 0.000% | 0/16/0 | 100% / 100% | 0 / 0 | 174.688 / 174.688 | 100.11% | 1 / 4030 | 5 |

All cells restored their selected checkpoint successfully, and all frozen
encoder parameters were bitwise unchanged.

Absolute latency across cells is not comparable because A2-A4 shared GPUs with
unrelated workloads. The within-cell alternating-order IL/RL ratios remain the
relevant safety measurement; all were below the 110% gate.

## Diagnostics

- A0 produced the largest actor drift (normalized L2 `1.396e-3`) but only one
  deterministic action disagreement and no schedule change.
- A1 and A2 were numerically nearly identical. Removing the frozen-IL KL did
  not create more policy movement or a validation improvement.
- A3 changed the reward/value scale substantially: final value loss was
  `0.0148`, but explained variance remained strongly negative (`-0.750`).
  Removing starvation shaping did not change a validation action.
- A4 reduced value loss relative to A1 in most iterations, but explained
  variance remained negative and the graph-aware critic did not change
  validation schedules.
- PPO old-policy KL and frozen-IL KL remained extremely small in all cells.
  The adaptive PPO KL coefficient fell from `0.2` to `0.0125` over five
  iterations, while policy entropy remained around `0.01-0.02`.

The failure mode is therefore not coalition legality or an obvious safety
regression. The actor remains effectively locked to the IL decision boundary.
The next experiment should move to the planned prefix-conditioned policy
diagnostic or decision-focused/offline-to-online supervision, rather than
extending these PPO cells.

## Artifacts

- `runs/md_ray_ppo_phase_a_A0_seed5101`
- `runs/md_ray_ppo_phase_a_A1_seed5101`
- `runs/md_ray_ppo_phase_a_A2_seed5101`
- `runs/md_ray_ppo_phase_a_A3_seed5101`
- `runs/md_ray_ppo_phase_a_A4_seed5101`
- `reports/md_ray_ppo_phase_a_2026-09-04/summary.json`
