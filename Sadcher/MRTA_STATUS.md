# MRTA 当前状态（2026-09-11）

本文是 MRTA（多机器人任务分配）相关工作的单一入口。原始实验报告仍保留在 `reports/`，本文只记录过程、证据和可使用的结论，避免新会话反复加载全部实验文件。

## 当前系统边界

- 问题包含异构机器人、process coalition、transport singleton、任务前置关系和动态可行性。
- 离线阶段使用 MILP/Gurobi 生成教师标签；在线阶段使用神经策略、集中式 hard mask、受约束 decoder 和 repair。
- 在线正常路径不调用 MILP；MILP 只作为离线 oracle、基准和显式 fallback。
- 当前正式 residual 范围是 stationary snapshots。动态 transport/process 状态尚未纳入 residual 数据协议。

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

随后进行的 linear/DeepSets scorer 消融未达到预注册门槛：DeepSets 平均 top-1 仅提升 1.67 个百分点（门槛为 2），regret 反而从 2.10 上升到 2.17；linear 下降。结论固定为 `scorer_not_supported`，不得接入生产 decoder，也不得读取 confirmation split。

### Profile-stratified pilot

已覆盖 balanced、process_scarce、transport_bottleneck、dependency_deep、mixed_hard、scale_medium 六类 profile，并完成 greedy baseline pilot。各 profile 的 pilot 均成功完成，但 C0 profile training 因入口要求 CUDA 且当前机器无 GPU，未产生新的 C0 checkpoint 或 profile score。

### Prefix / pairwise switch

在 50 个新状态、3226 个 exact actions、150 次决策上的 pairwise switch 验证没有改善（paired regret reduction 为 0，批准的 switch 为 0）。生产调度保持不变。

## 已确认的核心问题

旧 residual MILP 标签是“forced inclusion batch”：要求某些 assignment 出现在后续最优 schedule 中，但没有表达当前时刻的完整 first action、idle complement、其他 assignment 禁止条件或唯一并发动作。因此旧 joint 标签不是完整 online action，直接用于 joint decoder 会退化。

正确监督单位应为：`(state, complete first action, continuation value)`。在此之前，joint decoder 只能采用 local-safe evaluation，并在不严格优于 residual proposal 时回退。

## 当前可执行结论

1. residual IL 是当前主模型；可报告平均 makespan 改善和安全性结果。
2. frozen joint correction、旧 joint 端到端训练、prefix pairwise switch、exact-action scorer 均不应作为生产改进或正向主张。
3. 下一阶段优先完善 complete-action 标签和完整 simulator/task/robot physics features，再重新预注册 scorer/decoder 实验。
4. 不读取 confirmation split（76000–76149）；regression benchmark（75800–75949）只能用于回归评估。

## 原始证据索引

- `HANDOFF.md`：训练、评估、joint correction 和 exact-action 的详细交接记录。
- `reports/md_residual_scale10_2026-09-04/`
- `reports/md_frozen_joint_correction_2026-09-06/`
- `reports/md_exact_action_candidate_diagnostic_2026-09-06/`
- `reports/md_exact_action_scorer_ablation_2026-09-07/`
- `reports/md_profile_pilot_2026-09-11/final_report.md`
- `reports/md_prefix_pairwise_switch_validation_2026-09-10/final_report.md`
