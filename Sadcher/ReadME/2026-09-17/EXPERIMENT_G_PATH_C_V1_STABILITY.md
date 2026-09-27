# Experiment G: Path C v1 stability sweep (2026-09-18)

## Hypothesis

Experiment F showed that Path C v1 (margin ranking) produced the first
downstream win since Path A: at seed 3101 the epoch-5 checkpoint hit
393.5 on 60-task `process_scarce` (baseline C0 = 398.75, greedy_unlock = 395.75).
Late epochs regressed to 401–402 despite lower val loss. That was a single seed
on 4 checkpoint dumps; the open question was whether the epoch-5 win reproduces
across independent training seeds, or whether it was a one-shot lottery.

## Setup

- Script: `experiments/md_c0_pathC_ranking_v1.py` unchanged. Reused
  `--checkpoint-epochs 1 2 ... 20` to dump every epoch (equivalent to the
  `--save-every-epoch` behaviour requested in Plan G, without touching code).
- Training seeds: 3101, 3102, 3103. Everything else fixed to v1: lr=5e-6,
  batch=32, K=5 negatives, margin=1.0, 20 epochs, cuda:0. Fine-tune source =
  baseline C0 checkpoint. Dataset = Path A dataset (9838 train / 1291 val).
- Driver: [reports/md_c0_pathC_v1_stability_2026-09-18/run_all.sh](../../reports/md_c0_pathC_v1_stability_2026-09-18/run_all.sh)
  runs the three seeds serially on cuda:0, then evaluates every checkpoint
  serially. Launched with `setsid nohup` so it survives ssh disconnect.
  Idempotent — restart re-uses completed training or eval outputs.
- Eval: `experiments/md_c0_pathC_ranking_v1_eval.py` unchanged. Per seed, 20
  checkpoints × the 32-instance grid (2 profiles × tc {12,24,42,60} × 4 test
  seeds), OR-Tools CP-SAT, 300 s time limit.
- Wallclock: 3 × 28 min training + 3 × 92 min eval, ~6 h end-to-end.

## Results

**Goal A (primary): PASS on 3/3 seeds.** Each seed produced at least one
checkpoint clearing both constraints (60-task `process_scarce` ≤ 395.75 AND
60-task `dependency_deep` ≤ 500.5):

| seed | passing epochs | best epoch | ps60 | dd60 | val_loss @best |
|---:|---|---:|---:|---:|---:|
| 3101 | 5, 7, 8 | 5 | 393.50 | 487.00 | 0.6244 |
| 3102 | 5 | 5 | 393.00 | 487.00 | 0.6291 |
| 3103 | 5, 8, 16 | 5 | 393.50 | 487.00 | 0.6213 |

Mean of per-seed best-downstream = **393.33** on 60-task `process_scarce`
(baseline C0 398.75; greedy_unlock 395.75; Path A 405.2; Path B1 415.0;
Path B2 413.5).

**Epoch 5 is the sweet spot in every seed.** For all three seeds, ps60 traces
the same pattern: 397–407 in epochs 1–4 → drops to 393–395 around epochs 5–8 →
climbs back to 397–411 in epochs 9–11 → oscillates in the 397–410 range through
epoch 20. dd60 stays in 477–493 (comfortably below 500.5) at every epoch. Full
matrix is in [summary.csv](../../reports/md_c0_pathC_v1_stability_2026-09-18/summary.csv).

**Goal B: val loss remains anti-correlated with downstream.** All three seeds
send validation ranking loss monotonically down from ~0.67 to ~0.57 across 20
epochs, but downstream ps60 stops improving after epoch 5–8 and then
oscillates. Val loss is still not a usable model-selection signal. Tier
regression count vs baseline C0 (out of 8) also does not track val loss.

**Goal C: mean makespan comparison.** Averaging each seed's best-downstream
checkpoint gives ps60 = 393.33, dd60 = 485.42 — a clean win over baseline C0,
greedy_unlock and all three prior fine-tune paths, at every checked point.

## Verdict

**`path_c_v1_stable`.** The Path C v1 result is reproducible: three seeds all
find a checkpoint that clears the primary judgement, and all three do so at
the same rough epoch band (5–8). This is not seed-3101 lottery.

Per the plan's decision path, **do not start Path C v2.** The next question is
worth its own session: build a model-selection signal correlated with
downstream, so we can pick epoch 5–8 checkpoints without running the full
32-instance grid each time. Candidates include top-1 match against re-solved
MILP first actions on a held-out pool, and preferred/negative score-gap
statistics. Val loss is still not it.

## Artifacts

- Training + checkpoints: `reports/md_c0_pathC_v1_stability_2026-09-18/seed{3101,3102,3103}/training/`
- Per-seed eval: `reports/md_c0_pathC_v1_stability_2026-09-18/seed{3101,3102,3103}/eval/`
- Aggregate: `reports/md_c0_pathC_v1_stability_2026-09-18/{summary.csv,final_report.md,aggregate.py}`
- Driver + logs: `reports/md_c0_pathC_v1_stability_2026-09-18/{run_all.sh,driver.log,nohup.log}`
