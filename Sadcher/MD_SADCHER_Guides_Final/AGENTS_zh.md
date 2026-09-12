# AGENTS.md 中文版

## 目的

将 SADCHER 从以任务为中心的异构多机器人调度，扩展为物料交付感知调度。

当前目标：**MD-SADCHER++ = SADCHER + 物料交付型 carry task + 下游感知 policy**。

除非明确要求，否则不要实现完整 M³-SADCHER。

## 文档职责

- `AGENTS.md`：短规则、范围和边界。
- `IMPLEMENT_GUIDE.md`：工程原则和模块级要求。
- `POLICY_GUIDE.md`：policy、网络、scoring 和 mask 指导。
- `IMPLEMENTATION_ROADMAP.md`：实现阶段、执行顺序和验收里程碑。
- `BACKGROUND.md`：科研背景、问题定义和数学建模。

范围冲突遵循 `AGENTS.md`。

工程实现细节遵循 `IMPLEMENT_GUIDE.md`。

Policy 细节遵循 `POLICY_GUIDE.md`。

执行顺序遵循 `IMPLEMENTATION_ROADMAP.md`。

科研定义遵循 `BACKGROUND.md`。

## 核心思想

不要把 `carry` 只当成一种技能任务。

在本项目中，carry 表示：

`取货 -> 运输 -> 送达 -> 解锁下游 process task`

贡献应表述为：

`carry-as-a-skill -> material-delivery-aware carry`

不要把本工作表述为“只是增加了一条前序边”。

## 当前范围

必须支持：

- carry task：`pickup`、`delivery`、`load`、`downstream_task_id`；
- process task：`normal_predecessors`、`material_predecessors`；
- robot：`capacity`、`loaded_speed`；
- 物料交付 ready condition；
- 运输感知 carry duration；
- 区分 normal/material 的 typed task edges；
- 下游感知 carry encoding；
- 运输感知 carry scoring；
- hard feasibility masks；
- makespan 和 material starvation 指标。

最小版本完成后，可选支持：

- rule-based cooperative carry。

## 当前不做

不要实现：

- 完整 M³-SADCHER；
- 显式 material nodes 或 lifecycle stages；
- 完整 task-material graph；
- robot-task-material-mode action space；
- learned mode selection；
- relay transport 或任意 handoff points；
- multi-trip transport；
- inventory、battery、charging、collision/path planning；
- low-level manipulation/control。

## 工程品味

遵循 Linux 设计哲学：小、清晰、可组合、显式、鲁棒。

必须遵守：

- 一次性修复根因，不要堆 fallback hack。
- 50 行能解决的，不要写 200 行。
- 优先使用简单数据结构和明显控制流。
- 可行时，让非法状态无法表示。
- 假设被破坏时要显式失败，不要隐藏错误逻辑。
- 在没有真实重复用例前，不要过度抽象。
- patch 要小、可 review、逻辑上原子化。
- 通过显式 config flag 保留旧行为，不要靠隐藏 fallback path。
- 永远不要用 `rm -rf` 清理仓库。

如果需要清理，先列出明确目标文件，只删除经过 review 的路径。

## 必须显式实现的约束

Process ready condition：

`ready(k) = 所有 normal predecessors 完成 AND 所有 material predecessors 完成`

Carry feasibility：

`机器人具备 carry skill AND robot.capacity >= carry.load`

Carry duration：

`robot_to_pickup/speed + load_time + pickup_to_delivery/loaded_speed + unload_time`

Material-delivery precedence：

`carry j 必须在 downstream process k 开始前完成`

这些是显式规则，不是神经网络预测。

## Policy 规则

使用结构化神经调度，不要使用纯端到端调度。

Policy 形态：

`显式状态/特征 + 学习式 scoring + hard masks + SADCHER-style decoder`

网络可以学习任务优先级、robot-task value 和 downstream unlock value。

网络不能学习 readiness、capacity feasibility、predecessor feasibility 或 carry duration formula。

尽量保留原 SADCHER backbone 和 decoder。

## Baselines 与 Metrics

Baselines 应包括：

- Original SADCHER；
- Carry-as-a-Skill SADCHER；
- Greedy Distance；
- Greedy Unlock；
- Material-Solo SADCHER；
- MD-SADCHER++；
- MILP oracle，如果可用。

Metrics 应包括：

- makespan；
- material starvation time；
- robot utilization；
- inference time；
- success rate。

## 代码规则

- 不要重写整个代码库。
- 保持 `material_delivery.enabled=false` 兼容。
- 保持 score matrix 与原 decoder 兼容。
- 为新约束添加测试。
- 不要悄悄改变旧实验行为。
- 记录新的 task、robot 和 config 字段。
- 使用确定性随机种子。
