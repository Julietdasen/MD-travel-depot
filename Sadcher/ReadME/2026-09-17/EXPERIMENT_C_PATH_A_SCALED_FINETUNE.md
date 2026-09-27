# Path A scaled-profile fine-tune (2026-09-17)

## One-line conclusion

Fine-tuning C0 on 300 new scaled process_scarce + dependency_deep instances **failed the primary criterion** (60-task process_scarce vs greedy_unlock): the fine-tuned checkpoint is worse than the pre-tuning C0 at every process_scarce tier of size >= 24. Path A alone does not fix the tc=60 shortfall; escalate to Path B (model capacity upgrade).

## Numbers

Grid: 2 profiles x tc {12, 24, 42, 60} x seeds {301..304}. MRTA_MILP_SOLVER=ortools, 300s time limit. All 32 MILP solves optimal.

process_scarce (primary criterion is tc=60):

| tc | MILP | greedy_unlock | baseline C0 (09-14) | Path A C0 (09-17) | delta |
|---:|---:|---:|---:|---:|---:|
| 12 | 164.2 | 209 avg (gd 207) | 175.8 | 175.2 | -0.5 |
| 24 | 250.0 | 304.0 | 263.5 | 269.2 | **+5.75** |
| 42 | 304.0 | 346.5 | 332.2 | 338.5 | **+6.25** |
| 60 | 356.0 | 395.8 | 398.8 | 405.2 | **+6.50** |

dependency_deep:

| tc | MILP | greedy_unlock | baseline C0 | Path A C0 | delta |
|---:|---:|---:|---:|---:|---:|
| 12 | 226.5 | 245.8 | 245.5 | 245.5 |  0.0 |
| 24 | 262.2 | 281.5 | 284.0 | 282.2 | -1.75 |
| 42 | 342.8 | 392.2 | 383.2 | 374.8 | -8.50 |
| 60 | 444.5 | 492.5 | 500.5 | 496.5 | -4.00 |

Training: pretrained val loss 0.3549 -> fine-tuned val loss 0.3433 across 20 epochs (lr=2e-5, batch=32). Loss curve descended monotonically; downstream makespan on process_scarce did not follow.

## Should this ship?

**No.** The fine-tuned checkpoint regresses process_scarce at tc in {24, 42, 60} by roughly 6 timesteps each (2.0-2.4pp on MILP gap). The dependency_deep gains (2-8 timesteps at tc >= 24) are smaller and do not compensate. The tc=60 process_scarce case, which was the pilot's headline gap vs greedy, drifted from `+0.8%` above greedy to `+2.4%`.

Keep the production C0 as `reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt`.

## What this rules out

- Adding 300 more matched-profile instances at the current C0 capacity does not close the tc=60 gap; the loss signal decouples from downstream makespan.
- Profile-balanced training data cannot simultaneously improve both target profiles at this capacity (Pareto trade-off observed).

## Next step

Path B: increase model capacity (deeper GATN, larger `embed_dim`, more transformer layers, and/or explicit critical-path features). Rerun the same `scaled_target_profiles_eval` grid as the acceptance test. If Path B is deferred, consider a Path A v2 with a much lower lr (5e-6) or a process_scarce-weighted loss, plus tc=80 coverage, to avoid the current process_scarce drift.

## Artifacts

- Fine-tuned checkpoint: `reports/md_c0_pathA_scaled_finetune_2026-09-17/training/best_checkpoint.pt`
- Training summary: `reports/md_c0_pathA_scaled_finetune_2026-09-17/training/training_summary.json`
- Eval outputs: `reports/md_c0_pathA_scaled_finetune_eval_2026-09-17/{rows,summary}.json`
- Full report: `reports/md_c0_pathA_scaled_finetune_eval_2026-09-17/final_report.md`
- Baseline: `reports/md_c0_scaled_target_profiles_eval_2026-09-14/`
