# IL 重设计方案:用 MILP 最优 assignment order 作为监督信号

**背景**:2026-09-13 的评测战役(见 [README.md](README.md))得到两个决定方向的证据:

1. C0 在训练分布内(12-24 任务)相对 MILP 最优就差 15-36%——**问题不在泛化,在监督信号本身**。
2. 现有 `ResidualBatchLabel.legal_batches` 是 forced-batch partial-inclusion:只说"这个 batch 是可行/最优的一部分",没告诉模型 t0 时刻另外那些机器人必须 idle、不能同时启动别的任务。frozen-joint correction 只改 1/450、端到端 joint 训练不稳定、exact-action scorer 消融未过阈值——**事后修 joint 的路子基本堵死,必须回到标签层**。

## 核心思路

`baselines/gurobi_md_oracle.py` 已经现成能做两件事:

- `solve_gurobi_md_oracle(instance, time_limit)` 返回 `GurobiOracleResult`,内含 `action_order: tuple[OracleAction, ...]`——**端到端最优的 assignment 顺序**,每个 `OracleAction` 带 `(order, robot_id, task_id, start_time, ...)`。
- `replay_oracle_actions(instance, actions)` 已验证能把这个顺序回放到 `MDDiscreteSimulator`,产出的 makespan 就是 `objective`(= latest_exit_return),与 C0/greedy 完全同定义。

`data_generation/md_expert_dataset.py:509+` 里已有把 `OracleAction` 序列消费成 dataset 行的骨架。

因此不必重造轮子——**把最优 `OracleAction` 序列切成"每一步的完整动作 + t0 idle 补集"**,直接作为 IL 监督信号。

## 与现有 forced-batch 标签的区别

| 维度 | 现有 forced-batch | 新方案(MILP oracle action) |
|---|---|---|
| 每步动作是否完整 | 否,partial-inclusion | 是,完整 `(assignments, idle_robot_ids)` |
| 是否含 t0 禁令 | 否 | 是,`forbidden = idle_robots × pending_tasks` |
| 是否绑定 regret | 有,但基于 forced-batch 的模糊比较 | 有,相对全局最优 continuation 的真实 regret |
| 覆盖范围 | 静止快照 + 强制某些机器人参与 | 端到端整条 rollout,每步都是最优 |
| 数据生成成本 | 中,需要 Gurobi 解 forced batch | 低(小规模):12-24 任务 MILP 秒级解出 |

Ticket 48 的 `exact-online-action-1.0`(489 行)已经验证了这个 schema 可以稳定生成,`idle_robot_ids` 和 `forbidden_immediate_assignments` 也已经在 schema 里——**新方案不是全新造轮,而是把这个 schema 从 210 snapshots 扩展到覆盖端到端 rollout,并把它接入生产端 dataloader**。

## 实施步骤

### Step 1 — 标签生成器(纯离线,不动模型)

新增 `data_generation/md_oracle_action_labels.py`:

- 输入:instance generator 参数(profile / seed 网格)、`time_limit`(小规模建议 30-60s,MILP 会秒级)。
- 对每个 instance 调 `solve_gurobi_md_oracle`,拿 `action_order`。
- 按 `start_time` 分桶:同一个 `start_time` 的 `OracleAction` 组成一个"完整动作",没参与的 available 机器人组成 `idle_robot_ids`。
- 对每个动作再调一次 Gurobi 解**限制该动作后**的续问题(`gurobi_md_residual_oracle.py` 已支持 `first_action` 和 `forbidden_immediate_robot_ids` 约束),拿到 `continuation_cost`,与全局最优 `objective` 对比得 `regret=0`(是最优动作时)或 `regret>0`(如果之后要生成 tie-optimal 候选)。
- 输出 JSONL,schema 复用 `exact-online-action-1.0`,新增字段 `is_oracle_optimal: true` 标记"这条来自端到端最优 rollout"。
- 存到 `reports/md_oracle_action_dataset_<date>/`。

**成本估算**:12-24 任务 MILP 秒级,加上每步续问题(通常 < 0.5s),单 instance ~10-30s。1000 instance × 20 步/instance ≈ 6-10 小时单机可完成,可与其他实验并行(不同 seed 不抢 license,可分批)。

**边界**:先只做 12-24 任务(MILP 100% 最优);42+ 任务因 MILP 超时,监督信号本身就不最优,暂不入库,只作为将来 warmup 后 RL 或 self-imitation 的输入。

### Step 2 — Dataloader shim(工程量小)

在 `data_generation/md_residual_dataset.py:_from_raw` 里加一个分支:

- 若 raw 有 `oracle_action_labels` 字段(新格式),按 `snapshot_id` 分组,把每条 oracle-action 包装成 `ResidualBatchLabel(assignments, makespan=continuation_cost, regret=regret, ...)`,拼成 `legal_batches`。
- 老格式路径原样保留,可通过 `dataset_type` 配置切换。

这样 `residual_batch_loss` 和 `joint_batch_ranking_loss` **不用改**,checkpoint、优化器、模型架构都不动。

### Step 3 — Masking hook(工程量中)

在 policy forward 里加一个可选 mask 通路:

- Dataloader 把 `forbidden_immediate_assignments`(如果新标签存在这个字段)也塞进 batch。
- Policy 打 edge score 后、进入 bipartite matching 前,用 `scores.masked_fill(forbidden_mask, -inf)` 处理。
- 默认关闭(`use_forbidden_mask=false`),训练脚本按 flag 打开——避免影响推理时的既有行为。

推理路径(部署)**不启用 mask**——这只是训练期让模型更快学到"哪些 t0 组合是不该出现的",部署时仍走原有的可行性 mask + fallback。

### Step 4 — 训练与验证协议

- 用 3 个 seed(3101/3102/3103 保持一致)分别训练:
  - 对照 A:老 forced-batch label(现状 baseline)
  - 对照 B:新 oracle-action label + shim only(不启用 forbidden mask)
  - 对照 C:新 oracle-action label + shim + forbidden mask
- 评测口径复用 [reports/md_c0_profile_eval_pilot_2026-09-13](../../reports/md_c0_profile_eval_pilot_2026-09-13/) 和 [reports/md_pure_milp_baseline_pilot_2026-09-13](../../reports/md_pure_milp_baseline_pilot_2026-09-13/) 的 seed 网格,直接比 makespan 相对 MILP 最优的差距。
- **成功标准**:6 profile 平均相对 MILP 最优的差距从当前 25% 缩到 **≤15%**;`greedy_unlock` 在 12-24 任务上不再战胜 C0。
- **失败保护**:若 B 与 A 没有可测差距,说明 shim 层丢的 idle 补集信息其实无关紧要,退回;若 C 与 B 也没差距,说明模型架构无法消费 forbidden 信息,才考虑动架构。

### Step 5 — 归档与迭代节奏

- 每一轮 3-seed 训练输出到独立 `reports/md_c0_oracle_label_<variant>_<date>/`。
- 完成一轮就把 log + run 脚本归档到 `archive/<date>-il-redesign/`,`ReadME/<date>/README.md` 只留一页结论。
- 与 2026-09-13 一样,不把日报直接堆在 `ReadME/`——按当天日期建目录,过长的原始记录归档。

## 与在跑实验的冲突

在跑:无(`md_pure_milp_baseline_pilot_2026-09-13` 已于 15:28 完成)。Step 1 可以立即启动,但会占用 Gurobi license,与将来任何 MIP 类实验有排他关系,建议单独一个 conda session 跑。

## 不做什么(明确划界)

- **不做端到端 joint 训练**:已被证明不稳定,`MRTA_STATUS.md:51` 明令禁止。
- **不做 frozen-joint correction 变体**:1/450 收益,负面结果已封存。
- **不做 exact-action scorer 消融**:已跑过,未过阈值。
- **不为 42+ 任务生成 oracle 标签**:MILP 会超时,监督信号不再最优,先做小规模。
- **不动现有 6 profile 评测口径**:新旧对照必须同 seed、同 profile,可比性优先。
