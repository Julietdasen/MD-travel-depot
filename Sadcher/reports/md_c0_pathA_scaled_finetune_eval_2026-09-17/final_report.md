# Path A scaled-profile fine-tune evaluation

**Date**: 2026-09-17

**Fine-tuned checkpoint**: `reports/md_c0_pathA_scaled_finetune_2026-09-17/training/best_checkpoint.pt`

**Pretrained parent**: `reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt`

**Baseline for comparison**: `reports/md_c0_scaled_target_profiles_eval_2026-09-14/` (same 2 profiles x 4 task_counts x 4 seeds grid, MILP time limit 300s, MRTA_MILP_SOLVER=ortools).

## Training summary

- Dataset: 300 new scaled-profile instances (process_scarce + dependency_deep, tc in {24, 42, 60}, 50 seeds/tier) plus 250 legacy scale symlinks; 9838 train / 1291 dev snapshots.
- Regime: fine-tune from `md_c0_milp_supervised_scale_pilot_2026-09-13` checkpoint at lr=2e-5, batch=32, 20 epochs, no early stop trigger.
- Validation loss: `0.35486` (pretrained) -> `0.34327` (fine-tuned best @ epoch 20). Train loss `0.33901` -> `0.32433`.
- Training log: `reports/md_c0_pathA_scaled_finetune_2026-09-17/run.log`.

## Verdict

**Primary criterion (60-task process_scarce vs greedy_unlock 395.8): FAILED**

- Fine-tuned C0 makespan on 60-task process_scarce is **405.2**, worse than both the baseline greedy_unlock (395.8) and the pre-tuning C0 (398.8).
- The pre-tuning C0 was already fractionally behind greedy_unlock (+0.8%); the fine-tune widened this to +2.4%. Path A did not recover the greedy advantage at tc=60 process_scarce; it actively regressed.

**Secondary criteria**:

- 60-task dependency_deep gap vs MILP: `+12.6% -> +11.7%`. Improved but still above the `<+10%` target.
- process_scarce at tc in {24, 42}: **regressed** (see main table). 12 essentially unchanged.
- dependency_deep at tc in {24, 42}: marginal improvement, tc=12 unchanged.

Net: the fine-tune shifted quality **toward dependency_deep and away from process_scarce**, but did not cross the primary threshold in either direction on tc=60. Path A alone is not sufficient. Recommendation: escalate to Path B (model capacity upgrade) or expand data along the process_scarce dimension.

## Side-by-side main table

Columns: baseline row is the 2026-09-14 C0 pilot; Path A row is this fine-tune. delta_C0 is (Path A C0 - baseline C0) in absolute makespan. Negative delta_C0 = fine-tune improved.

### profile: `dependency_deep`

| tc | MILP | greedy_unlock (baseline greedy) | baseline C0 | Path A C0 | delta_C0 | baseline gap vs MILP | Path A gap vs MILP | baseline vs greedy | Path A vs greedy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 226.5 | 245.8 | 245.5 | 245.5 | 0.0    | +8.4% | +8.4% | -0.1% | -0.1% |
| 24 | 262.2 | 281.5 | 284.0 | 282.2 | -1.75  | +8.3% | +7.6% | +0.9% | +0.3% |
| 42 | 342.8 | 392.2 | 383.2 | 374.8 | -8.50  | +11.8% | +9.3% | -2.3% | -4.5% |
| 60 | 444.5 | 492.5 | 500.5 | 496.5 | -4.00  | +12.6% | +11.7% | +1.6% | +0.8% |

dependency_deep: fine-tune reduces gap to MILP at tc in {24, 42, 60}, largest gain at tc=42 (`-8.5` makespan, `-2.5pp` MILP gap). At tc=60 still `+11.7%` above MILP, above the `<+10%` secondary target. tc=12 unchanged.

### profile: `process_scarce`

| tc | MILP | best greedy | baseline C0 | Path A C0 | delta_C0 | baseline gap vs MILP | Path A gap vs MILP | baseline vs greedy | Path A vs greedy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 164.2 | 207.0 (gd) | 175.8 | 175.2 | -0.50   | +7.0% | +6.7% | -15.1% | -15.3% |
| 24 | 250.0 | 304.0 (gu) | 263.5 | 269.2 | **+5.75** | +5.4% | +7.7% | -13.3% | -11.4% |
| 42 | 304.0 | 346.5 (gu) | 332.2 | 338.5 | **+6.25** | +9.3% | +11.3% | -4.1% | -2.3% |
| 60 | 356.0 | 395.8 (gu) | 398.8 | 405.2 | **+6.50** | +12.0% | +13.8% | +0.8% | **+2.4%** |

process_scarce: fine-tune **regressed** at tc >= 24. Absolute makespan grew by 5.75-6.50 at every tc>=24; gap vs MILP grew by 2.0-2.3pp. The primary criterion tier (tc=60) drifted from `+0.8%` above greedy to `+2.4%` above greedy, and from `+12.0%` to `+13.8%` above MILP. tc=12 essentially flat (single seed noise, `-0.50`).

## Interpretation

- The validation loss curve monotonically decreased across 20 epochs, so the fine-tune did learn the training signal; the loss surface just does not track downstream makespan on process_scarce at scale.
- dependency_deep gained roughly `2-4` timesteps at tc >= 24, but the process_scarce loss (`+6` timesteps at each large tier) is larger and more consistent. The fine-tune shifted the decision boundary away from process_scarce-favorable dispatch heuristics.
- Consistent with the hypothesis that C0 model capacity (`embed_dim=16`, 1 GATN layer, 1 transformer layer) is the binding constraint: additional data increased the noise on one profile without meaningfully closing the tc=60 MILP gap.
- MILP is still `> 10%` optimal on both profiles at tc=60; the ceiling for pure supervised IL at this capacity is roughly `+11-12%` gap.

## Next steps

1. **Path A alone is not the winning move**. Do not deploy this checkpoint as the new production C0.
2. Retain both checkpoints on disk for regression comparisons.
3. Escalate to **Path B** (model capacity upgrade): larger `embed_dim`, deeper GATN, deeper transformer stack, or explicit critical-path / capacity features baked in earlier. Rerun the same scaled_target_profiles eval as the acceptance test.
4. As a data-side hedge, if Path B is delayed, consider expanding the process_scarce fine-tune set (more seeds and/or tc=80) with a much lower learning rate (`5e-6`) or profile-weighted loss to avoid the observed process_scarce drift.
5. Keep the current production C0 = `reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt` until Path B (or a Path A v2) passes the tc=60 process_scarce criterion.

## Artifacts

- Rows: `reports/md_c0_pathA_scaled_finetune_eval_2026-09-17/rows.json`
- Summary: `reports/md_c0_pathA_scaled_finetune_eval_2026-09-17/summary.json`
- Log: `reports/md_c0_pathA_scaled_finetune_eval_2026-09-17/run.log`
- Baseline for comparison: `reports/md_c0_scaled_target_profiles_eval_2026-09-14/summary.json`
