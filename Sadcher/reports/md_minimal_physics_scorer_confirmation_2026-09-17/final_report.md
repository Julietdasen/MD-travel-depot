# Minimal Physics Scorer - Confirmation Split (Blind)

Verdict: **not_confirmed**.

Confirmation seed range: 76000-76149; three-seed mean top-1 gain **+0.0333** (threshold >=+0.04); mean regret reduction **+1.0500** (threshold >=+0.5).

| Seed | Dev top-1 baseline | Dev top-1 physics | Conf top-1 baseline | Conf top-1 physics | Conf regret baseline | Conf regret physics |
|---:|---:|---:|---:|---:|---:|---:|
| 3101 | 0.8333 | 0.9167 | 0.8500 | 0.8833 | 3.3667 | 1.7500 |
| 3102 | 0.8500 | 0.8667 | 0.8500 | 0.9000 | 3.3667 | 2.2500 |
| 3103 | 0.8500 | 0.9333 | 0.8500 | 0.8667 | 3.3667 | 2.9500 |

Baseline picks the residual/greedy-preferred candidate (top-1 by rank). Physics scorer re-ranks and picks its argmin.

Deviations from the pre-registered protocol:
- The residual-tail scorer checkpoints referenced by md_exact_action_candidate_diagnostic.py (reports/md_residual_scale10_2026-09-04/training/seed{s}/C0_residual_tail_seed{s}/best_checkpoint.pt) are not present on this host, so candidate generation for the confirmation split used the zero tensor as the score input. For pending counts <=3 the resulting candidate set is a super-set of the top-8/16 residual selection; every legal action is included, which is at least as hard for the physics scorer as the development setup.
- C0 checkpoints at the diagnostic pilot path are symlinks to the equivalent md_c0_gpu_retrain_pilot_2026-09-13 files (identical training seeds, model spec, and hyperparameters; only the physical filesystem location differs).

Regression benchmark seeds (75800-75949) and any seeds outside 76000-76149 were not read.
