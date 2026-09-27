# Path B2 (hidden_dim=64) — scaled_target_profiles evaluation

**Date**: 2026-09-17

## Training summary

- Script: `experiments/md_c0_pathB_capacity_upgrade.py` (shared with B1).
- Dataset: `reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset` (300 scaled
  profile shards + 250 legacy symlinks from the C0 pilot).
- Capacity: `hidden_dim=64`, `embed_dim=64`, `ff_dim=128`,
  `transformer_layers=2`, `gat_layers=2`, `dropout=0.1`. Total params ≈ 257k.
- Optimisation: from scratch, Adam, cosine LR (peak 1e-3, warmup 5 epochs, min
  1e-5), `batch_size=32`, `seed=3101`.
- Ran 40 epochs (early stop; patience 15). **Best epoch 25, val loss 0.339690**.

## Evaluation table (mean makespan over 4 seeds)

| profile | tc | greedy_unlock | MILP | C0 baseline (2026-09-14) | Path A | B1 (48) | B2 (64) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dependency_deep | 12 | 245.75 | 226.50 | 245.50 | 245.50 | 249.25 | 265.00 |
| dependency_deep | 24 | 281.50 | 262.25 | 284.00 | 282.25 | 314.25 | 303.75 |
| dependency_deep | 42 | 392.25 | 342.75 | 383.25 | 374.75 | 384.75 | 387.75 |
| dependency_deep | 60 | 492.50 | 444.50 | 500.50 | 496.50 | 499.75 | 492.25 |
| process_scarce  | 12 | 207.00 | 164.25 | 175.75 | 175.25 | 174.50 | 174.25 |
| process_scarce  | 24 | 304.00 | 250.00 | 263.50 | 269.25 | 268.50 | 270.25 |
| process_scarce  | 42 | 346.50 | 304.00 | 332.25 | 338.50 | 339.25 | 346.00 |
| process_scarce  | 60 | 395.75 | 356.00 | 398.75 | 405.25 | 415.00 | 413.50 |

## Gap vs greedy_unlock (%, negative = C0 beats greedy)

| profile | tc | C0 baseline | Path A | B1 | B2 |
|---|---:|---:|---:|---:|---:|
| dependency_deep | 12 | -0.10 | -0.10 | +1.42 | **+7.83** |
| dependency_deep | 24 | +0.89 | +0.27 | +11.63 | +7.90 |
| dependency_deep | 42 | -2.29 | -4.46 | -1.91 | -1.15 |
| dependency_deep | 60 | +1.62 | +0.81 | +1.47 | **-0.05** |
| process_scarce  | 12 | -15.10 | -15.34 | -15.70 | **-15.82** |
| process_scarce  | 24 | -13.32 | -11.43 | -11.68 | -11.10 |
| process_scarce  | 42 | -4.11 | -2.31 | -2.09 | -0.14 |
| process_scarce  | 60 | +0.76 | +2.40 | +4.86 | **+4.49** |

## Success criteria

| criterion | target | B2 result | pass |
|---|---|---|---|
| Primary: 60-task process_scarce advantage vs greedy_unlock | ≥ 0 (i.e. gap ≤ 0%) | +4.49% (413.5 vs 395.75) | **FAIL** |
| process_scarce (12/24/42) no worse than baseline | ≥ 0 delta | tc=24 +6.75, tc=42 +13.75 worse | **FAIL** |
| dependency_deep (12/24/42/60) no worse than baseline | ≥ 0 delta | tc=12 +19.5, tc=24 +19.75 worse | **FAIL** |
| 60-task dependency_deep gap < +10% | < +10% | -0.05% | **PASS** |
| Trade-off eliminated (both profiles ≤ baseline) | both ≤ | violated on 5 of 8 tiers | **FAIL** |
| Capacity monotonic (B2 > B1) | B2 better | mixed; B2 only clearly better at process_scarce tc=60 (413.5 vs 415.0), dependency_deep tc=42/60 improved. B2 much worse than B1 on dependency_deep tc=12. | **INCONCLUSIVE** |

## Verdict

Path B2 **fails** the primary criterion. B2 does not eliminate the
process_scarce vs dependency_deep trade-off; on 60-task process_scarce the
scaled-up policy is worse than the baseline C0 checkpoint (398.75) and worse
than greedy_unlock (395.75) by +4.49%.

Notably, B2 also **regresses on dependency_deep tc=12/24** (+19.5 / +19.75
makespan vs baseline), which the baseline C0 checkpoint already handled well.
So B2 is not Pareto-dominated by baseline — it wins at dependency_deep tc=60
(-8.25) — but it loses badly enough elsewhere that "trade-off eliminated" is
clearly false.

Capacity monotonicity (B2 vs B1): B2 is only clearly better than B1 on
dependency_deep tc=60 (-7.5), and on process_scarce tc=60 by a hair (-1.5).
On dependency_deep tc=42 and process_scarce tc=42 B2 is worse than B1. This
does not support a "wider = better" story on this dataset.

**Diagnosis carried forward**: capacity alone does not fix the process_scarce
tc=60 policy quality. Combined with Path A's identical failure signature,
this strongly suggests that the augmented dataset (300 new scaled shards +
250 legacy symlinks) is not by itself enough — the two profile signatures
appear to induce conflicting gradient directions regardless of capacity.
Future work should try (a) profile-aware loss reweighting, (b) curriculum
that separates the two profiles, or (c) larger per-profile shard counts to
saturate the loss on 60-task instances specifically.

## Artifacts

- Training: `reports/md_c0_pathB2_capacity64_2026-09-17/`
- Eval: `reports/md_c0_pathB2_capacity64_eval_2026-09-17/rows.json` and
  `summary.json`
- Checkpoint: `reports/md_c0_pathB2_capacity64_2026-09-17/training/best_checkpoint.pt`
- Comparison baseline: `reports/md_c0_scaled_target_profiles_eval_2026-09-14/rows.json`
- Path A: `reports/md_c0_pathA_scaled_finetune_eval_2026-09-17/rows.json`
- Path B1: `reports/md_c0_pathB1_capacity48_eval_2026-09-17/rows.json`
