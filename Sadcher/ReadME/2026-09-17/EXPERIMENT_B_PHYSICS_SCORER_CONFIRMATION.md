# Experiment B: Physics Scorer Confirmation Blind Test (2026-09-17)

## 结果

在 confirmation split(seeds 76000-76149, 60 snapshots × 3 model seeds)盲测 `minimal_physics_scorer`:

- **三 seed 平均 top-1 增益：+3.33pp**（0.850 → 0.883）— 未过 ≥+4pp 门槛
- **三 seed 平均 regret 下降：1.05**（3.367 → 2.317）— 过 ≥0.5 门槛
- 判决：**not_confirmed**（仅一半门槛达成）

Development 上的 +6.11pp 和 regret 2.10→1.08 无法在 confirmation 上完全复现；top-1 增益萎缩到 dev 的一半，regret 收益方向一致但绝对值降低。介于「dev 过拟合」（<2pp）与「confirmation 通过」（≥+4pp）之间。三个 seed 的 confirmation top-1 分别为 +3.33pp / +5.00pp / +1.67pp，方差比 dev 大。

## 结论

**不能接入生产 decoder**。特征方向有意义（regret 系统性下降），但泛化到 fresh confirmation seeds 的 top-1 增益不满足预注册门槛。下一步只能扩数据规模或改特征后重新走一遍 dev + 二次 confirmation split，不能在原 confirmation seeds 上再迭代。

## 协议偏差（诚实记录）

1. `md_residual_scale10_2026-09-04/training/seed*/C0_residual_tail_seed*/best_checkpoint.pt` 在本机不存在，confirmation 的 candidate generation 无法使用 residual scorer 排序，改用 zero-score 输入。当 pending≤3 时（本盲测唯一目标区间），legal action 数量 1-4 个，zero-score 生成的候选集包含全部 legal action，对 physics scorer 是更严格的 rerank 任务。
2. `md_c0_end_to_end_diagnostic_pilot_2026-09-01/training/C0_seed*/checkpoints/best_checkpoint.pt` 缺失，从 `md_c0_gpu_retrain_pilot_2026-09-13/training/C0_seed*/checkpoints/best_checkpoint.pt` symlink 而来（相同 model_seed / variant='C0' / 相同 loader 支路）。
3. 训练 checkpoint 缺失，用相同 TRAINING_SEEDS（5101/5102/5103）与相同 feature cache 重训一次，dev 结果与历史一致（+6.11pp）。

## 产物

- `reports/md_minimal_physics_scorer_confirmation_2026-09-17/final_report.md`
- `reports/md_minimal_physics_scorer_confirmation_2026-09-17/summary.json`
- `reports/md_minimal_physics_scorer_confirmation_2026-09-17/feature_package.jsonl`（180 rows）
- `reports/md_minimal_physics_scorer_confirmation_2026-09-17/candidate_diagnostic/`（120 exact-action labels + 180 candidate records + 60 cached snapshot solves）
