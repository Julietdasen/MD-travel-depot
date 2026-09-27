# Path C v1 stability sweep — final report

Dates: training + eval 2026-09-18. Three model seeds (3101 / 3102 / 3103),
20 epochs each, per-epoch checkpoints, evaluated on the 32-instance grid
(dependency_deep, process_scarce x tc={12,24,42,60} x 4 test seeds).
MILP solver=OR-Tools CP-SAT, 300 s time limit. Fine-tuned from baseline C0.

## Verdict

**Goal A (primary)**: each seed must produce >=1 checkpoint with
60-task process_scarce makespan <= 395.75 (greedy_unlock) AND
60-task dependency_deep makespan <= 500.5 (baseline C0).

- seed 3101: passing epochs = [5, 7, 8]; best-downstream = epoch 5 (ps60=393.50, dd60=487.00, val_loss=0.6244)
- seed 3102: passing epochs = [5]; best-downstream = epoch 5 (ps60=393.00, dd60=487.00, val_loss=0.6291)
- seed 3103: passing epochs = [5, 8, 16]; best-downstream = epoch 5 (ps60=393.50, dd60=487.00, val_loss=0.6213)

**Result: 3/3 seeds pass. Conclusion = `path_c_v1_stable`.**

Follow-up: do NOT start v2. Next session: build a downstream-correlated model selection signal.

## Best-downstream checkpoint per seed

| seed | epoch | ps60 makespan | dd60 makespan | val_loss | tier_regressions |
|---:|---:|---:|---:|---:|---:|
| 3101 | 5 | 393.50 | 487.00 | 0.6244 | 3 |
| 3102 | 5 | 393.00 | 487.00 | 0.6291 | 3 |
| 3103 | 5 | 393.50 | 487.00 | 0.6213 | 3 |

## Comparison vs prior attempts (60-task process_scarce)

| approach | 60-task process_scarce makespan |
|---|---:|
| MILP (300 s) | 356.0 |
| baseline C0 (pre-Path A) | 398.75 |
| greedy_unlock | 395.75 |
| Path A (h=16 fine-tune) | 405.2 |
| Path B1 (h=48 scratch) | 415.0 |
| Path B2 (h=64 scratch) | 413.5 |
| Path C v1 stability, mean of best per seed | 393.33 |

## Val-loss ↔ downstream direction

Across all 3 seeds the validation ranking loss decreases monotonically from
epoch 1 to 20, but the 60-task process_scarce makespan does NOT track it.
The best downstream checkpoint is at epoch 5 in every seed (or, for seed 3103,
tied by epoch 16). Late-epoch models (17–20) all regress despite lower val loss.
See summary.csv for the full seed x epoch matrix.

## Artifacts

- summary.csv (this directory): (seed, epoch, val_loss, ps60_makespan, dd60_makespan,
  primary_pass, tier_regression_count) for all 60 (seed, epoch) pairs.
- reports/md_c0_pathC_v1_stability_2026-09-18/seed{3101,3102,3103}/training/
  checkpoint_epoch_{1..20}.pt (per-epoch checkpoints).
- reports/md_c0_pathC_v1_stability_2026-09-18/seed{3101,3102,3103}/eval/
  rows_epoch{1..20}.json, summary.json, final_report.md (per-seed eval).
- reports/md_c0_pathC_v1_stability_2026-09-18/driver.log (run driver history).
