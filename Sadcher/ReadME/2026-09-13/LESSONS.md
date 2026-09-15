# 错误案例复盘:三条已封存的支线

本页汇总 2026-09-05 至 2026-09-10 期间尝试过、最后以负面结果关闭的三条支线,提炼可迁移的经验,避免下一轮 IL 重设计([IL_REDESIGN.md](IL_REDESIGN.md))踩同样的坑。相关脚本已删除,对应报告移入 [../../archive/negative-results/](../../archive/negative-results/) 作为证据保留。

---

## 支线 A:端到端 joint 训练 / frozen-joint correction

**尝试的东西**:在 residual policy 基础上加一个 `JointResidualTailPolicy`,直接对"整批联合动作"打分(edge sum + joint_head correction),用 `joint_batch_ranking_loss` 端到端训练。冻结版(frozen-joint correction)则只训 correction head、其余参数冻结。

**结果**:
- 端到端 joint 训练:2 个 init × 3 seed × 100 epoch 全部不稳定,不能替换现有 residual policy。
- Frozen-joint correction:450 model/instance 配对中只有 **1 个** 结果改善,收益可忽略。

**根因**:监督标签是 `ResidualBatchLabel.legal_batches`(forced-batch partial-inclusion)——**只告诉模型"这个 batch 里的分配是可行/最优的一部分",没告诉模型另外那些机器人在 t0 必须 idle、不能同时启动别的任务**。joint head 试图学"整批打分"时,标签本身对"批的完整边界"是模糊的,模型自然学不到有信号的东西。

**经验**:
1. **不要在模糊的标签上加模型容量**。frozen-joint correction 的动机是"给一个 correction head 修正"——但如果标签本身缺信息,更多参数只会去拟合噪声。
2. **端到端 joint 训练翻车不是训练算法问题**,是标签层问题。别再回头调 lr / init / schedule。
3. **下一步该做的是补标签**,不是再加一层修正。这就是 [IL_REDESIGN.md](IL_REDESIGN.md) 用 MILP oracle 完整 action(含 `idle_robot_ids` + `forbidden_immediate_assignments`)重来的原因。

---

## 支线 B:prefix / rich_prefix / cost_gate / pairwise_switch

**尝试的东西**:一整套"在已有 residual 决策序列上事后打分/切换/rerank"的路径:
- `md_prefix_autoregressive_pilot`:把 rollout 切成前缀,训练一个 prefix-conditioned reranker;
- `md_prefix_cost_gate_*`:给 reranker 加一个成本 gate,阈值下才启用切换;
- `md_prefix_pairwise_switch_validation`:两两候选切换验证;
- `md_rich_prefix_*`:更丰富的前缀特征。

**结果**:
- `md_prefix_pairwise_switch_validation`:**0 accepted switches**(见归档报告)。
- 其他 cost_gate / rich_prefix 变体没有一个能通过预设的接受阈值。

**根因**:与支线 A 是同一个病症的不同表现——**事后打分只能重新排序模型已经产出的候选,如果模型本身在关键决策点上就没有正确的候选,rerank 无从改起**。frozen-joint 的 correction、prefix 的 reranker、pairwise switch 都属于"在决策链下游修补",不解决上游标签模糊。

**经验**:
1. **"事后修 policy"整条路径基本堵死**——joint correction、prefix reranker、pairwise switch 三种形式都试过,全部关掉。不要再起第四种。
2. **决策链下游的信号密度只会比上游更稀疏**(0 accepted switches 就是极端情况)——试图靠下游打分弥补上游的策略缺陷,先验就不成立。
3. **验证成本要早规划**:pairwise_switch 跑到最后接受 0 次,才发现整个方向不成立;应该先跑一个小规模 smoke(比如 30 个 rollout),看接受率是否至少非零,再决定是否投产线性扩展。

---

## 支线 C:exact-action scorer 消融

**尝试的东西**:Ticket 48 产出的 `exact-online-action-1.0` 标签(489 行,完整 first action + idle 补集 + forbidden set)——先用它做"scorer ablation":看单独把 scorer 换成从这些标签直接学的模型,能否超过现在的 C0。

**结果**:预注册阈值未达标,标记 `scorer_not_supported` 关闭。

**根因**:489 行 / 210 snapshots **只够诊断消融,不够重训生产 policy**。HANDOFF.md 明确写了这一点(`HANDOFF.md:26-28, 46-55`),但消融跑的时候还是撞到了数据量瓶颈:小数据 + 老架构 → 过拟合 / 泛化不足。

**经验**:
1. **诊断样本量与训练样本量是两个数量级**。别用诊断集当训练集,就算 schema 一样。
2. **不要因为标签"看起来对"就跳过扩量步骤**。`exact-online-action-1.0` 的 schema 是对的,但要真训练必须扩到 scale-10 级别(和现有 residual 数据集同量级)。
3. **[IL_REDESIGN.md](IL_REDESIGN.md) Step 1 的数据生成器**直接从这条经验来:先把标签扩到覆盖端到端 rollout(每 instance ~20 步),1000+ instance,才有资格进训练。

---

## 三条支线的共同教训

**监督信号是天花板,不是可以事后修补的地板**。

- 支线 A 想在模糊标签上加容量;
- 支线 B 想在下游 rerank;
- 支线 C 想在小样本上直接训。

三种失败路径都在回避"回头把标签本身搞对"这件事。C0 相对 MILP 最优差 15-36%(2026-09-13 评测),这不是模型架构问题——12-24 任务本来就在训练分布内,MILP 秒级出最优,C0 就是没学到——**问题在监督信号覆盖不全**。

下一轮 IL 重设计([IL_REDESIGN.md](IL_REDESIGN.md))**只做一件事:把 MILP 最优 `OracleAction` 序列变成完整 per-step 标签,直接监督**。不加下游修正,不加事后 rerank,不复用诊断集当训练集。

---

## 归档索引

- 脚本(experiments/、imitation_learning/、tests/、根 `run_*.sh`)已删除。
- 报告已移入 `archive/negative-results/`:
  - `md_joint_residual_2026-09-05/`
  - `md_frozen_joint_correction_2026-09-06/`
  - `md_exact_action_scorer_ablation_2026-09-07/`
  - `md_prefix_autoregressive_pilot_2026-09-10/`
  - `md_prefix_conditioned_reranker_2026-09-10/`
  - `md_prefix_cost_gate_pilot_2026-09-10/`
  - `md_prefix_cost_gate_large_validation_2026-09-10/`
  - `md_prefix_cost_gate_conservative_validation_2026-09-10/`
  - `md_prefix_pairwise_switch_validation_2026-09-10/`
