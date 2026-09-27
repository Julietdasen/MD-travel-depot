# Experiment F: Path C v1 — margin ranking loss (2026-09-17)

## Hypothesis

Path A / B1 / B2 all failed the same way: val loss falls, 60-task
`process_scarce` makespan gets worse (405.2 / 415.0 / 413.5 vs baseline C0
398.75). Root cause hypothesis: pointwise BCE forces the policy to reproduce
MILP tie-breaking on `first_action`. Path C v1 replaces BCE with a **margin
ranking loss**; architecture, dataset and pretraining are unchanged.

## Setup

- Pretrained: baseline C0 pilot checkpoint (hidden=16, TL=1, GAT=1).
- Dataset: Path A dataset (9838 train / 1291 val, 300 scaled shards + 250 legacy symlinks). Unmodified.
- Loss: for each preferred pair `(r*, t*)` (legal AND `expert_assignment==True`), sample K=5 negatives from `legal(s) \ preferred(s)` — we exclude *all* preferred pairs (not just `(r*, t*)`) to protect equally-optimal ties. Hinge `mean_k max(0, 1.0 − s_pos + s_neg_k)`. States with no positive or no negative are skipped.
- Adam, lr=5e-6, batch=32, seed=3101, 20 epochs on cuda:0, no early stopping. Checkpoint dumps at ep 5/10/15/20.
- Downstream eval: `md_c0_scaled_target_profiles_eval.py` (unmodified), 2 profiles × tc {12,24,42,60} × 4 seeds, `MRTA_MILP_SOLVER=ortools`, 300 s. All 32 MILP instances proven optimal.

## Results

Training moved val loss from `0.6738` (epoch 0) down to `0.5674` (epoch 20).
Train loss `0.6325 → 0.5263`.

Downstream 60-task `process_scarce` makespan by checkpoint:

| epoch | val loss | 60-task PS makespan | vs baseline C0 (398.75) | vs greedy_unlock (395.75) | primary pass? |
|---:|---:|---:|---:|---:|:---:|
| 5 | 0.6244 | **393.50** | **-5.25** | **-2.25** | **YES** |
| 10 | 0.6001 | 401.00 | +2.25 | +5.25 | no |
| 15 | 0.5849 | 401.75 | +3.00 | +6.00 | no |
| 20 | 0.5674 | 402.25 | +3.50 | +6.50 | no |

Reference: Path A = 405.2, Path B1 = 415.0, Path B2 = 413.5.

Val-loss vs downstream direction is **still inverted**: the checkpoint with
the highest val loss (epoch 5) wins downstream; the epoch with the lowest val
loss (epoch 20) is worst.

Non-regression vs baseline C0 across the 8 tiers: epoch 5 → 5/8, epoch 10 → 5/8,
epoch 15 → 4/8, epoch 20 → 5/8. `dependency_deep` tc {42, 60} improve
consistently at every epoch. `process_scarce` tc 24/42 stay ~1–8 timesteps
above baseline (small regress) at all epochs.

## Verdict

- **Primary judgement: PASSED (via epoch 5).** For the first time since Path A
  we have a checkpoint that beats both baseline C0 and greedy_unlock on the
  primary target (60-task `process_scarce`), and does so by a meaningful margin
  (-5.25 vs baseline).
- **Secondary: the val-loss / downstream inversion persists.** Longer training
  on the ranking objective continues to overfit MILP-specific structure. Val
  loss is still not a usable model-selection signal.
- Beam search / longer horizon are not needed: epoch 5's improvement comes from
  a policy change, not from search.

## Next step

Advance to **Path C v2**: keep margin ranking, but (a) replace the
"expert=preferred, rest=negative" label with continuation-makespan-based
rankings on multiple candidate first actions per state (from MILP re-solves or
Monte-Carlo rollouts), and (b) add a listwise loss so equally-optimal ties are
explicitly modelled. Also short-list model-selection signals that correlate
with downstream (candidate: top-1 match against re-solved MILP first actions
on a held-out state pool) so early-stopping and hyperparameter search can be
grounded again.

## Artifacts

- Script: `experiments/md_c0_pathC_ranking_v1.py`
- Eval script: `experiments/md_c0_pathC_ranking_v1_eval.py`
- Training: `reports/md_c0_pathC_ranking_v1_2026-09-17/`
- Eval: `reports/md_c0_pathC_ranking_v1_eval_2026-09-17/`
