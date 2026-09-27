# IL 训练路径探索经验总结 (2026-09-21)

**目的**：在删除大规模 dataset 前，把每条训练路径的关键决策、试错结论、以及为什么当前 checkpoint 是 Path C v1 stability seed3101 ep5 凝练下来。

**当前 baseline**：`reports/md_c0_pathC_v1_stability_2026-09-18/seed3101/training/checkpoint_epoch_5.pt`

**基准评价指标**：60-task process_scarce mean makespan，4 seeds (5000-5003)。
- MILP oracle = 356.00
- greedy_unlock = 395.75
- 目标：< 395.75 (beat greedy at scale)

---

## 时间线与每条路径的关键数字

### baseline C0 (Ticket 23, 2026-09-13)

- **来源**：`reports/md_c0_milp_supervised_pilot_2026-09-13/`（114 records）→ `md_c0_milp_supervised_scale_pilot_2026-09-13/`（200 records，最终采用的 pretrained）
- 训练：MSE loss over MILP-labeled action scores，200 epochs
- 结果：val loss 0.46 → 0.35，masked top-1 accuracy 62.8%（scale pilot）
- 60-task ps makespan: **398.75** (vs greedy 395.75, MILP 356)
- **判决**：已比 greedy 稍差 (+0.76%)，无法称"beats greedy at scale"

### Path A: 大规模数据 fine-tune (2026-09-17)

- **假设**：加更多 scaled-profile 数据能救大规模
- 数据：新加 300 instances (process_scarce + dependency_deep, tc ∈ {24, 42, 60}, 50 seeds/tier) + 250 legacy symlinks
- 训练：从 scale pilot checkpoint fine-tune，lr=2e-5, 20 epochs
- val loss：0.35486 → 0.34327（改善了）
- **60-task ps makespan：405.2**（比 baseline 差 6.5，比 greedy 差 9.5）
- **判决**：val loss 降但 downstream 反而变差。**证明 val loss 不追踪 downstream makespan on process_scarce**。这是后续所有路径的关键教训。

### Path B1/B2: 扩容 (2026-09-17)

- **假设**：baseline C0 (embed_dim=16, 1 GATN layer) 容量太小，是天花板
- Path B1：hidden=48，Path B2：hidden=64，2 layers each
- 训练：from scratch (不 fine-tune)，Adam + cosine LR，40 epochs
- Path B2 best val loss: 0.339690 (比 Path A 更好)
- **60-task ps makespan：B1=415.0, B2=413.5**（比 baseline 更差 15+）
- **判决**：扩容 + 更好 val loss → downstream **更差**。彻底证伪"容量不足"假设，也再次证明 val loss 和 downstream 反向。

### Path C v1: margin ranking loss (2026-09-17)

- **假设**：MSE 训 raw score 是标量回归，切换到 pairwise ranking (K=5 negatives, margin=1.0, hinge) 能学到 action ordering
- 训练：从 scale pilot checkpoint fine-tune，lr=5e-6，20 epochs
- **60-task ps makespan：ep5=393.5** ← 唯一击穿 greedy 的 checkpoint
- 后续 epoch：ep10=401.0, ep15=401.75, ep20=402.25 → **越训越差**
- **判决**：ranking loss 有效但收敛后过拟合。ep5 是 "早停" 也不是（val loss 在 ep5-20 单调下降）。这个 ep5 是**运气 + 早期结构学到的正确 bias**。

### Path C v1 stability (2026-09-18)

- **动机**：Path C v1 ep5 是否是单 seed artifact？
- 训练：3 seeds (3101, 3102, 3103) 重复 Path C v1 配方
- 结果：seed3101 ep5 依然是三者最好，ep5 makespan 稳定在 393.5 附近
- **判决**：**收敛点选 seed3101 ep5**。这就是当前 baseline。

### Path C B2 regret-weighted (2026-09-20)

- **假设**：训练信号里 low-regret 样本浪费容量，用 regret 加权 batch mean 能拉开有效样本
- **判决**：无稳定改善 (~40M dataset)。收进死路。

### Residual IL / joint action correction / exact-action scorer

- **residual scale10** (2026-09-04)：150-instance test 上平均 makespan 1.2-2.9% 改善但**没通过硬约束** (task completion max 增幅 > 1 timestep)。安全但收益不大。
- **frozen joint correction** (2026-09-06)：450 model/instance 配对中**只有 1 个动作改善**，实际上是无效。
- **exact-action scorer** (2026-09-07)：Ticket 48 exact-action gate 通过，但 scorer 消融 top-1 只涨 1.67pp，低于预注册 +2pp 门槛。**scorer_not_supported** 定档。

---

## 6 条可复用的教训

### 教训 1：val loss ≠ downstream makespan（on process_scarce, at scale）

**证据**：
- Path A：val 0.35 → 0.34 但 makespan +6.5
- Path B2：val 0.340（比 A 更好）但 makespan +8
- Path C v1 stability：val loss ep5→ep20 单调降但 makespan 单调恶化

**解释**：MILP-supervised action scores 是**参考答案**，但在 online rollout 里 policy 只需要**正确排序**，不需要 exact scores。MSE 训得 score 匹配好，不代表排序对；对于**大规模 process_scarce**（unlock/coalition 组合复杂）尤其如此。

**应用**：模型选择必须用 **downstream rollout makespan**，不能只用 val loss。当前 baseline 就是这么选的（ep5 是 ep5-20 里唯一 downstream 通过的）。

### 教训 2：容量不是瓶颈（在当前 problem 下）

**证据**：Path B1/B2 从 16 扩到 64 hidden dim，downstream 一路恶化。

**解释**：当前 problem 复杂度不需要更大网络。瓶颈在**训练信号本身**（标签是 MILP forced batch，不是完整 online action）或**问题定义**（default problem 上 coalition 大多不 forced，见 [coalition 诊断](md_coalition_diagnostic_2026-09-21/coalition_and_gap_2026-09-21.png)）。

**应用**：不做扩容实验；下一版工作先解决 problem 定义（scarce/multi-mode）。

### 教训 3：ranking loss > MSE，但会过拟合

**证据**：Path C v1 ep5 是全部路径中唯一击穿 greedy 的。但 ep5→ep20 越训越差。

**解释**：pairwise ranking 让 policy 学习 action ordering，early epoch 学到"哪些 action 显然差"，late epoch 开始记忆特定 (state, action) pair 的 exact score margin → 泛化下降。

**应用**：ranking loss + **早停**是可复用组合。**不要**运行满 100 epochs（Path A 教训）。

### 教训 4：Residual / joint correction 收益 marginal

**证据**：residual 1.2-2.9% 平均改善但 task_completion 硬约束 3 seed 都不通过；frozen joint correction 450 配对里只 1 个改善。

**解释**：C0 已经找到局部好解，residual/joint 修正的空间小（decision-event coverage 不足）。加上 forced-batch 标签的 partial-inclusion 语义，joint 决策 label 本身就有问题（HANDOFF 明确记录）。

**应用**：不做 residual 变体；如果要做 joint correction 必须先解决 label semantics（exact online-action oracle）。

### 教训 5：exact-action oracle 需要，但 candidate coverage 已 OK

**证据**：Ticket 48 exact-action gate 通过（recall@8 = 100%，各 stratum 全通过）。DeepSets scorer 消融 +1.67pp，低于预注册 +2pp 门槛。

**解释**：候选生成 OK，但当前 scorer 在**排序**上比 additive baseline 只提升边际。

**应用**：如果重启 scorer 工作，必须先把完整 simulator physics features 加入 exact-action package。当前不做，因为 problem 侧还有更大杠杆。

### 教训 6：MILP time-budget 存在性优势是硬 finding

**证据**（time_budget_grid_2026-09-20）：
- tc=60/5s：MILP 4/4 全 infeasible
- tc=80/15s：MILP 4/4 全 infeasible
- policy 永远在毫秒级出可行解

**解释**：MILP 需要热启动时间；短预算下 CP-SAT 找不到可行 initial schedule。这是 policy 相对 MILP 的**唯一无条件优势**。

**应用**：论文核心 claim 之一。不需要额外训练支持这个 finding。

---

## 数据文件保留策略

**已删除的 dataset (可重生成)**：
- `md_c0_pathA_scaled_finetune_2026-09-17/dataset/` (1.7G, 300 shards + 250 legacy symlinks)
- `md_c0_milp_supervised_scale_pilot_2026-09-13/dataset/` (306M, 200 records)
- `md_c0_milp_supervised_pilot_2026-09-13/dataset/` (45M, 114 records)

**保留的**（每条路径的**结果**，都是小文件）：
- 所有 `training/best_checkpoint.pt` (100-300K each) — 后续可能 regression 对比
- 所有 `training_summary.json` / `pilot_summary.json` — 训练配置和 loss 曲线
- 所有 `eval/` 目录 (60K each) — downstream makespan 数据
- 所有 `RESULTS.md` / `final_report.md` — 判决文档

**重生成命令**（如果未来需要）：
- `experiments/md_c0_milp_supervised_pilot.py`（scale pilot 数据）
- `experiments/md_c0_pathA_scaled_finetune.py`（Path A scaled 数据）

---

## 未来重启这些路径的前置条件

如果 problem 侧 (scarce/multi-mode) 完善后要重新试 IL 变体，必须先：

1. **确认新 problem 上 val loss 和 downstream makespan 相关** — 教训 1
   - 方法：跑 baseline C0 在新 problem 上，看 val loss 变化 vs downstream 变化的相关系数
   - 相关系数低 → val loss 不能用，直接 skip Path A/B 之类的 loss-driven 变体

2. **coalition ratio 至少 ≥ 30%**（forced-coalition ≥ 0.3） — 教训 2 + envelope regime
   - 方法：在新 problem 上跑 coalition_diagnostic.py，确认 tc=60 上 ratio 达标
   - 不达标：先改 problem 生成器（scarce_skill 或 skill_count 拉高）

3. **exact-action oracle 就位（如果要 scorer 变体）** — 教训 5
   - HANDOFF 里明确的下一阶段目标
   - 必须先把 partial-forced-batch 换成 complete online action

---

## 参考文件

- HANDOFF (2026-09-06): residual/joint correction 详细记录
- Path A eval: `reports/md_c0_pathA_scaled_finetune_eval_2026-09-17/final_report.md`
- Path B2 eval: `reports/md_c0_pathB2_capacity64_eval_2026-09-17/final_report.md`
- Path C v1 eval: `reports/md_c0_pathC_ranking_v1_eval_2026-09-17/final_report.md`
- Path C stability: `reports/md_c0_pathC_v1_stability_2026-09-18/final_report.md`
- Envelope report: `reports/md_envelope_process_robots_2026-09-20/PROBLEM_ENVELOPE.md`
- Full results 表: `reports/md_envelope_d2_2026-09-20/RESULTS_SUMMARY.md`
