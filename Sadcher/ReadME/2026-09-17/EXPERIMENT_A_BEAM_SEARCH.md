# Experiment A: inference-time beam@k on MILP-IL v2 C0

**Date:** 2026-09-17
**Report:** `reports/md_c0_beam_search_pilot_2026-09-17/`
**Script:** `experiments/md_c0_beam_search_pilot.py`
**Checkpoint:** `reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt` (unchanged)

## What was run
K=1, K=4, K=8 parallel rollouts on `process_scarce` × task_count {12, 24, 42, 60} × seeds {301..304}. K=1 is the deterministic policy; K>1 spawns K-1 additional branches whose scores are perturbed by per-branch Gumbel noise scaled by `0.5 * (score.max - score.min)` at every decision. Each branch runs the constrained decoder to completion; the branch with the minimum makespan wins. MILP / greedy baselines were reused from the 2026-09-14 scaled-target-profiles rows without re-running.

## Headline numbers (60-task tier, mean of 4 seeds)
| variant | makespan | vs MILP | vs greedy_unlock | wall (s) |
| --- | --- | --- | --- | --- |
| MILP incumbent | 356.0 | — | — | — |
| greedy_unlock | 395.8 | +11.2% | — | — |
| K=1 (baseline) | 395.0 | +11.0% | -0.2% | 4.5 |
| K=4 | 395.0 | +11.0% | -0.2% | 19.0 |
| K=8 | 393.5 | +10.5% | -0.6% | 38.3 |

## Verdict against pre-declared criteria
- Primary (60-task gap <+8% AND >=0% vs greedy): **FAIL**. K=8 only shaves 0.5pp off the MILP gap; both K=4 and K=8 remain around +10-11%.
- Secondary (12/24/42 no regression at K=4 vs K=1): **MET**. K=4 improves 12/42 tiers and ties 24.
- Failure (K=4 gain <1% or wall_time >10s at K=4): **wall_time trigger fires**. K=4 wall = 19.0s at tc=60.

## Why beam did not help
On every 60-task instance the deterministic branch 0 was tied for best or the sole winner. The 7 noisy branches in each K=8 trial produced makespans 5-15% *worse* than branch 0. Score-noise perturbation pushed the constrained decoder onto a worse-quality manifold rather than exposing higher-quality joints near the greedy pick. A true top-K joint decoder (Lawler k-best or one-swap enumeration) might yield different results, but that was not implemented here.

## Recommendation
- **Not worth production integration.** Serve the deterministic K=1 policy.
- If a beam knob is ever added for batch scenarios, K=4 is the least-bad choice — no regression in the mean and occasional 1-2% wins at tc<=42 — at ~4x compute. K=8 buys nothing over K=4 at this experiment's scale.
- Follow-ups worth trying (not run in this experiment): true top-K joint enumeration in the constrained decoder; a value head that lets beam nodes be pruned mid-rollout; direct retraining on larger instances to shrink the +12% MILP gap at 60 tasks, which appears to be a policy-quality issue rather than an inference-search issue.

## Files touched
- Added: `experiments/md_c0_beam_search_pilot.py`
- Added: `reports/md_c0_beam_search_pilot_2026-09-17/{rows.json, summary.json, final_report.md, run.log}`
- No modifications to checkpoints, training entrypoints, or historical reports.
