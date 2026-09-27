# MRTA 当前状态（2026-09-11）

本文是 MRTA（多机器人任务分配）相关工作的单一入口。原始实验报告仍保留在 `reports/`，本文只记录过程、证据和可使用的结论，避免新会话反复加载全部实验文件。

## 当前系统边界

- 问题包含异构机器人、process coalition、transport singleton、任务前置关系和动态可行性。
- 离线阶段使用 MILP/Gurobi 生成教师标签；在线阶段使用神经策略、集中式 hard mask、受约束 decoder 和 repair。
- 在线正常路径不调用 MILP；MILP 只作为离线 oracle、基准和显式 fallback。
- 当前正式 residual 范围是 stationary snapshots。动态 transport/process 状态尚未纳入 residual 数据协议。

### MILP solver 后端（2026-09-21 更新）

- **OR-Tools CP-SAT 是唯一后端**。2026-09-21 删除 Gurobi 后端（`baselines.gurobi_md_oracle` / `baselines.gurobi_md_residual_oracle` / `baselines.gurobi_md_exact_action_oracle`）及 WLS 许可证文件；共享 dataclass/helpers 移到 `baselines.md_oracle_types`，`MRTA_MILP_SOLVER` 只接受 `ortools`，其它值触发 `ValueError`。
- 2026-09-17 切换背景：Gurobi WLS 许可证 `2770405` 已过期，restricted license 变量上限约 2000，无法覆盖 Path A 的 tc ≥ 42 单元；CP-SAT 在 300s 时限、8 线程下把 Path A 全网格解到 proven-optimal，最慢 tc=60 dependency_deep 也在 20s 内完成。基础 oracle 与 residual forced-batch oracle 都完成移植，语义等价（delta=0 replay smoke、forbidden-idle 与 first_action pinning 均验证）。
- 统一入口是 `baselines.md_oracle_dispatch`（`solve_md_oracle`、`solve_residual_forced_batch`）。
- 详情与证据：`ReadME/2026-09-17/BASELINE_SWAP_TO_ORTOOLS.md`、`ReadME/2026-09-17/ORTOOLS_FEASIBILITY.md`、`reports/md_ortools_feasibility_2026-09-17/`。Path A（process_scarce / dependency_deep, tc ∈ {24, 42, 60}）可继续在此后端上启动，无需 Gurobi 许可证。

## 已验证结果

### Residual IL（主线）

过程：10 倍扩容，3 个训练 seed（3101–3103），每个 seed 使用独立 train/development 数据；随后在 150 个 test instances 上进行配对 rollout。

结果：三个 seed 的平均 makespan 均优于对应 Legacy C0 和 masked-greedy；改善约 1.22%–2.89%，success 均为 150/150，illegal assignment 均为 0。

限制：原计划的逐实例 latest-task-completion 硬验收未通过。因此只能声称“平均性能改善”，不能声称完整正式验收通过。

### Frozen joint correction

过程：冻结 residual base 和 edge heads，仅训练 joint head，correction 限制为 ±0.25，并保留 epoch-0 fallback。

结果：安全性验收通过（success 不下降、illegal 为 0、task-completion 约束通过），但 450 个 model/instance 配对中只有 1 个结果改善，收益可忽略。不能替换 residual 主模型。

### Exact-action candidate/scorer

Ticket 48 的 complete-action 标签在 150 train + 60 development snapshots 上全部通过 oracle、replay、schema 和 quota 检查；三个 residual seed 的 development recall@8 均为 100%，支持进入 scorer 消融。

随后进行的 linear/DeepSets scorer 消融未达到预注册门槛：DeepSets 平均 top-1 仅提升 1.67 个百分点（门槛为 2），regret 反而从 2.10 上升到 2.17；linear 下降。当时结论固定为 `scorer_not_supported`。

**更新（2026-09-08）**：`scorer_not_supported` 已被 minimal_physics_scorer 结果推翻。加入 16 维 physics features（remaining processing time、robot release time、capacity scarcity 等）后，三 seed development top-1 从 0.844 平均提升到 0.906（`+6.11pp`，过预注册 `+2pp` 门槛），regret 从 `2.10` 降到 `1.08`。之前失败的根因是输入特征太薄（只有 additive edge score），不是 scorer 容量或数据规模。证据：`reports/md_minimal_physics_scorer_2026-09-08/final_report.md`，状态 `state_feature_scorer_supported_for_fresh_validation`。下一步是在 confirmation split（seeds 76000–76149）盲测冻结验证；未通过 confirmation 前仍不得接入生产 decoder。

**更新（2026-09-17）**：confirmation split 盲测已跑完（seeds 76000-76149, 60 snapshots × 3 model seeds），结论 `not_confirmed`。三 seed 平均 top-1 增益 `+3.33pp`（0.850→0.883），未过预注册 `+4pp` 门槛，但也高于 `+2pp` overfit 阈值；平均 regret 从 `3.367` 降到 `2.317`（下降 `1.05`，过 `0.5` 门槛）。特征方向有意义（regret 系统性下降）但 top-1 泛化力度不足，不能接入生产 decoder。协议偏差：`md_residual_scale10_2026-09-04` 的 residual-tail 检查点在本机缺失，confirmation 的 candidate generation 用 zero-score 输入（pending≤3 时 legal-action 集合全被纳入候选，是更严格的 rerank 任务）；C0 检查点由 `md_c0_gpu_retrain_pilot_2026-09-13` symlink 至 diagnostic pilot 路径（相同 model_seed / variant='C0'）。证据：`reports/md_minimal_physics_scorer_confirmation_2026-09-17/final_report.md`、`ReadME/2026-09-17/EXPERIMENT_B_PHYSICS_SCORER_CONFIRMATION.md`。

### Path A scaled-profile fine-tune (2026-09-17)

过程：从 `md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt` 出发，在 300 新 scaled-profile 实例（process_scarce + dependency_deep, tc ∈ {24, 42, 60}, 50 seeds/tier）加 250 legacy scale symlinks 上 fine-tune 20 epoch（lr=2e-5, batch=32）。Val loss `0.3549 → 0.3433`，train loss `0.339 → 0.324`，无 early-stop。

Eval：与 2026-09-14 baseline 同网格（2 profiles × tc {12,24,42,60} × 4 seeds，MRTA_MILP_SOLVER=ortools，300s time limit），全部 32 个 MILP 均 optimal。

判决：**主要判据 FAILED**。tc=60 process_scarce 上 fine-tuned C0 makespan = `405.2`，比 baseline greedy_unlock（`395.8`）和 pre-tuning C0（`398.8`）都差；process_scarce 在 tc ∈ {24, 42, 60} 上一致 regress 约 6 timestep（+2.0–2.4pp MILP gap）。dependency_deep 在 tc ∈ {24, 42, 60} 上小幅改善（2–8 timestep），tc=60 MILP gap 从 `+12.6%` 收敛到 `+11.7%`，仍高于 `<+10%` 次要目标。

结论：Path A（同容量 fine-tune + 更多数据）不是收尾方案，profile-balanced 数据导致 process_scarce ↔ dependency_deep Pareto trade-off。生产 C0 仍保留 `md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt`，Path A checkpoint **不接入生产**。下一步走 Path B（模型容量升级）或 Path A v2（lr=5e-6 + process_scarce 加权 + tc=80 扩展）。证据：`reports/md_c0_pathA_scaled_finetune_eval_2026-09-17/final_report.md`、`ReadME/2026-09-17/EXPERIMENT_C_PATH_A_SCALED_FINETUNE.md`。

### Path C ranking v1 (2026-09-17)

过程：从同一 baseline C0 checkpoint 出发，在同一 Path A 数据集（9838 train / 1291 val）上 fine-tune 20 epoch（lr=5e-6, batch=32, seed=3101），把 pointwise `BCE with logits` 换成 **margin ranking loss**：对每个 state 抽取 preferred pair `(r*, t*)`，从 legal_actions \ preferred_pairs 均匀采样 K=5 个 negatives，hinge loss `max(0, margin=1.0 - s_pos + s_neg)`，K 内平均、batch 内平均。架构不动，dump epoch 5/10/15/20 checkpoint。

Eval：与 Path A / B1 / B2 同网格（2 profiles × tc {12,24,42,60} × 4 seeds, ortools, 300s），32 个 MILP 全部 optimal。

判决：**主要判据 PASSED（epoch 5）**。60-task process_scarce makespan 依次为 `393.5 / 401.0 / 401.75 / 402.25`（epoch 5/10/15/20）；epoch 5 打败 baseline C0（`398.75`，-5.25）和 greedy_unlock（`395.75`，-2.25），是自 Path A 起首个通过主要判据的 checkpoint。参照：Path A `405.2`、Path B1 `415.0`、Path B2 `413.5`。次要：val loss 与 downstream 仍然反向（epoch 20 val 最低但 downstream 最差），val loss 依然不能作为选型信号；8 个 tier 中 epoch 5/10/20 各不 regress 5 个、epoch 15 不 regress 4 个。`dependency_deep` tc {42, 60} 每个 epoch 都有 4-20 timestep 改善，`process_scarce` tc {24, 42} 依旧小幅 regress。

结论：margin ranking loss 方向正确，是候选 α 的核心信号。生产 C0 仍保留 pilot checkpoint（Path C v1 checkpoint **不接入生产**，因 ties 处理仍嫌粗糙、val loss 反向仍存在）。下一步走 Path C v2：加入 continuation-makespan 标签（MILP re-solve 或 rollout）+ listwise loss + 找与 downstream 同向的 model-selection signal。证据：`reports/md_c0_pathC_ranking_v1_eval_2026-09-17/final_report.md`、`reports/md_c0_pathC_ranking_v1_2026-09-17/training/training_summary.json`、`ReadME/2026-09-17/EXPERIMENT_F_PATH_C_RANKING_V1.md`。

### Path C v1 stability sweep (2026-09-18)

过程：Plan G 稳定性验证。同一 v1 训练脚本、同一 Path A 数据集,唯一变量是 model seed ∈ {3101, 3102, 3103}，每个 seed 训 20 epoch 并**每个 epoch 都 dump checkpoint**（通过给 `--checkpoint-epochs` 传 1..20 复用现有 CLI，未改脚本代码）。lr=5e-6, batch=32, K=5, margin=1.0 与 v1 完全一致。整套 driver 用 `setsid nohup` 后台跑，断链不中断；训练 3 seed 串行 + eval 60 checkpoint 串行,总 wallclock ~6 h。

Eval：与 Path C v1 同网格（2 profiles × tc {12,24,42,60} × 4 seeds, ortools, 300s），每 seed × epoch 32 个 MILP。

判决：**目标 A PASSED on 3/3 seeds** → 结论 `path_c_v1_stable`。每个 seed 都至少一个 checkpoint 同时满足 60-task process_scarce makespan ≤ 395.75 且 60-task dependency_deep makespan ≤ 500.5。best-downstream checkpoint 全部落在 epoch 5：seed 3101/3102/3103 分别 ps60 = 393.5 / 393.0 / 393.5, dd60 = 487.0 / 487.0 / 487.0。三 seed best-downstream mean ps60 = **393.33**（优于 baseline C0 398.75、greedy_unlock 395.75、Path A 405.2、Path B1 415.0、Path B2 413.5）。次要目标 B 复现：三 seed 的 val loss 都单调下降到 ~0.57 但 downstream ps60 在 epoch 5–8 后一律回升到 397–411，val loss 与 downstream 仍反向,不是选型信号。

结论：Path C v1 的 epoch-5 breakthrough 不是单 seed 单 epoch 的运气，可稳定复现。**不启动 Path C v2**。下一步单开一个 session 做 "downstream-correlated model-selection signal"（候选：top-1 match against re-solved MILP first actions on held-out pool、preferred/negative score-gap 统计）。生产 C0 仍保留 pilot checkpoint（stability 结果不改变生产接入判断）。证据：`reports/md_c0_pathC_v1_stability_2026-09-18/final_report.md`、`reports/md_c0_pathC_v1_stability_2026-09-18/summary.csv`、`ReadME/2026-09-17/EXPERIMENT_G_PATH_C_V1_STABILITY.md`。

### Profile-stratified pilot

已覆盖 balanced、process_scarce、transport_bottleneck、dependency_deep、mixed_hard、scale_medium 六类 profile，并完成 greedy baseline pilot。各 profile 的 pilot 均成功完成，但 C0 profile training 因入口要求 CUDA 且当前机器无 GPU，未产生新的 C0 checkpoint 或 profile score。

### Prefix / pairwise switch

在 50 个新状态、3226 个 exact actions、150 次决策上的 pairwise switch 验证没有改善（paired regret reduction 为 0，批准的 switch 为 0）。生产调度保持不变。

## 已确认的核心问题

旧 residual MILP 标签是“forced inclusion batch”：要求某些 assignment 出现在后续最优 schedule 中，但没有表达当前时刻的完整 first action、idle complement、其他 assignment 禁止条件或唯一并发动作。因此旧 joint 标签不是完整 online action，直接用于 joint decoder 会退化。

正确监督单位应为：`(state, complete first action, continuation value)`。在此之前，joint decoder 只能采用 local-safe evaluation，并在不严格优于 residual proposal 时回退。

## 当前可执行结论

1. residual IL 是当前主模型；可报告平均 makespan 改善和安全性结果。
2. frozen joint correction、旧 joint 端到端训练、prefix pairwise switch 均不应作为生产改进或正向主张。**旧 linear/DeepSets exact-action scorer 也不能作为生产改进**。minimal_physics_scorer 在 development 上通过 `+2pp` 门槛，但在 confirmation split 未过 `+4pp` 门槛（`+3.33pp`），只有 regret 下降通过 `0.5` 门槛（`1.05`）。**minimal_physics_scorer 不接入生产 decoder**；后续如继续该方向必须扩数据或改特征后再走一次完整 dev + 全新 confirmation split。
3. 下一阶段优先扩大 exact-action snapshot 数据规模或补足 physics features，然后重新走 dev 门槛与 confirmation 盲测；不再单独调 physics scorer 结构。
4. confirmation split（76000–76149）本轮已用于 minimal_physics_scorer 盲测，属于「一次性支付」；后续任何变体都必须使用未读取过的新 seeds。regression benchmark（75800–75949）仍未读取，只能用于回归评估。

## 原始证据索引

- `HANDOFF.md`：训练、评估、joint correction 和 exact-action 的详细交接记录。
- `reports/md_residual_scale10_2026-09-04/`
- `reports/md_frozen_joint_correction_2026-09-06/`
- `reports/md_exact_action_candidate_diagnostic_2026-09-06/`
- `reports/md_exact_action_scorer_ablation_2026-09-07/`
- `reports/md_minimal_physics_scorer_2026-09-08/`
- `reports/md_minimal_physics_scorer_confirmation_2026-09-17/`
- `reports/md_c0_pathA_scaled_finetune_2026-09-17/`
- `reports/md_c0_pathA_scaled_finetune_eval_2026-09-17/`
- `reports/md_c0_pathC_ranking_v1_2026-09-17/`
- `reports/md_c0_pathC_ranking_v1_eval_2026-09-17/`
- `reports/md_c0_pathC_v1_stability_2026-09-18/`
- `reports/md_profile_pilot_2026-09-11/final_report.md`
- `reports/md_prefix_pairwise_switch_validation_2026-09-10/final_report.md`
