# BACKGROUND.md 中文版

## 作用

本文档记录 MD-SADCHER++ 的科研背景、应用动机、问题定义和数学建模。

它用于人工理解、论文写作和高层项目参考。

它不是短规则式 agent 文件。

## 项目定位

当前项目：

**MD-SADCHER++：Material-Delivery-Aware SADCHER with Downstream-Aware Carry Encoding**

本项目将 SADCHER 从以任务为中心的异构多机器人调度扩展为物料交付感知调度。

当前版本不尝试解决完整 material-flow manufacturing scheduling，而是聚焦 material-delivery precedence，即 carry task 交付物料并解锁 downstream process task。

## 应用背景

广义应用类别是受物料约束的异构多机器人自动化系统。

代表性领域包括：

- 一般工厂自动化；
- 带 AGV/AMR 配送的柔性制造；
- 机器人装配和检测；
- 自动化建造；
- 远程无人化作业；
- 行星表面建造与探索。

主要叙事应放在一般自动化和制造系统上。行星探索可以作为高自主、资源受限的代表性强化场景。

## 为什么原 SADCHER 不够

原 SADCHER 处理：

- 异构机器人；
- 动态联盟形成；
- 任务前序；
- 图学习；
- robot-task matching。

但是在原任务抽象中，carry 通常被视为普通技能。

一个任务可以需要 carry skill，但框架没有显式表达：

- pickup location；
- delivery location；
- transported load；
- material arrival；
- delivery 解锁的 downstream task。

真实 carry operation 往往意味着：

```text
取货
-> 运输
-> 送达
-> downstream process 变为可执行
```

因此，本项目将 carry 从抽象技能转化为 material-delivery operation。

## 核心科研问题

当 carry task 物理交付物料并解锁 downstream process task 时，异构机器人团队应如何调度 carry 和 process tasks？

更具体地：

- 哪个 carry task 应优先执行？
- 哪个 robot 应执行 carry task？
- scheduler 如何权衡 transport cost 和 downstream unlock value？
- process task 如何因 missing material 被阻塞？
- 如何在不把方法变成端到端黑箱的情况下，把这些加入 SADCHER？

## 关键科学挑战

核心调度挑战是：

```text
transport cost vs downstream unlock value
```

短 carry task 可能只解锁非关键分支。

较长 carry task 可能解锁关键 downstream process。

好的 policy 不应只选择最近的 carry task，而应推理 delivery 如何改变未来 task readiness 和 coalition utilization。

## 任务集合

机器人集合：

```math
R = {r_1, r_2, ..., r_N}
```

任务集合：

```math
T = {t_1, t_2, ..., t_M}
```

任务分为 carry tasks 和 process tasks：

```math
T = T_C ∪ T_P
```

其中：

- `T_C`：carry tasks；
- `T_P`：process tasks。

二者不相交：

```math
T_C ∩ T_P = empty
```

## 机器人定义

每个机器人可表示为：

```math
r_i = (x_i(t), a_i(t), v_i, v_i^L, cap_i, s_i)
```

其中：

- `x_i(t)`：机器人位置；
- `a_i(t)`：最早可用时间；
- `v_i`：空载速度；
- `v_i^L`：负载速度；
- `cap_i`：运输容量；
- `s_i`：技能向量。

## Process Task 定义

每个 process task 可表示为：

```math
t_k^P = (l_k, p_k, q_k, Pred_N(k), Pred_M(k))
```

其中：

- `l_k`：任务位置；
- `p_k`：基础处理时间；
- `q_k`：所需技能向量；
- `Pred_N(k)`：普通工艺前序；
- `Pred_M(k)`：物料交付前序。

Process task 只有在两类前序都满足后才能开始。

## Carry Task 定义

每个 carry task 可表示为：

```math
t_j^C = (o_j, d_j, w_j, tau_j^load, tau_j^unload, delta(j))
```

其中：

- `o_j`：pickup location；
- `d_j`：delivery location；
- `w_j`：物料载荷；
- `tau_j^load`：装载时间；
- `tau_j^unload`：卸载时间；
- `delta(j)`：该 delivery 解锁的 downstream process task。

通常：

```math
d_j = l_delta(j)
```

即物料被送到 downstream process 的位置。

## 前序图

任务图包含两类边：

```math
E = E_N ∪ E_M
```

普通前序边：

```math
E_N = {(p,k) | p in Pred_N(k)}
```

物料交付边：

```math
E_M = {(j,k) | j in Pred_M(k)}
```

普通边含义：

```text
process p 必须在 process k 开始前完成
```

物料边含义：

```text
carry j 必须在 process k 开始前完成物料交付
```

## Ready Condition

对于 process task `k`，ready condition 为：

```math
ready(k,t) =
[所有 p in Pred_N(k): C_p <= t]
AND
[所有 j in Pred_M(k): C_j <= t]
```

等价地：

```math
ready(k) =
所有 normal predecessors DONE
AND
所有 material predecessors DONE
```

这是本项目的核心新约束。

## Carry Duration

如果 robot `i` 执行 carry task `j`，其 duration 为：

```math
P_ij^C =
dist(x_i(a_i), o_j) / v_i
+ tau_j^load
+ dist(o_j, d_j) / v_i^L
+ tau_j^unload
```

这包括：

- 移动到 pickup；
- 装载；
- 负载移动到 delivery；
- 卸载。

对应 ETA 可用相同项加上 robot available time 显式计算。

## Process Coalition Constraint

Process task 可能需要一个机器人联盟。

联盟必须覆盖所需技能：

```math
OR over robot skills in coalition >= required skills
```

等价地，对每个技能维度：

```math
sum_i s_im >= q_km
```

这应保留原 SADCHER 的 dynamic coalition 思想。

## Carry Feasibility

最小版本中，carry 可以由单机器人执行。

Robot `i` 可执行 carry task `j` 当且仅当：

```math
carry_i = 1
```

且：

```math
cap_i >= w_j
```

这些约束应通过 hard masks 保证。

## 目标函数

主目标是最小化 makespan：

```math
min C_max
```

其中：

```math
C_max = max_j C_j
```

Scheduler 必须满足：

- 技能约束；
- 容量约束；
- 普通前序约束；
- 物料交付前序约束；
- 机器人互斥；
- 任务 ready 约束。

## Material Starvation Time

Material starvation 衡量物料迟到造成的等待。

对于 process task `k`：

```math
C_N(k) = normal predecessors 的最大完成时间
```

```math
C_M(k) = material predecessors 的最大完成时间
```

则：

```math
ST_k = max(0, C_M(k) - C_N(k))
```

总 material starvation：

```math
T_starvation = sum_k ST_k
```

这应作为核心评价指标。

## Policy 原则

本方法不应是纯端到端黑箱调度器。

推荐原则：

```text
显式约束
+ 显式计算调度特征
+ 学习式 scoring
+ hard feasibility masks
+ SADCHER-style decoder
```

网络学习：

- task priority；
- robot-task value；
- downstream unlock value。

环境和 masks 保证：

- readiness；
- capacity；
- predecessors；
- carry duration；
- robot mutual exclusion。

## 推荐贡献表述

贡献可以表述为：

> We introduce material-delivery precedence and downstream-aware carry representation into a SADCHER-style learning-based heterogeneous multi-robot coalition scheduling framework.

中文：

> 我们在 SADCHER 式学习型异构多机器人动态联盟调度框架中，引入物料交付前置约束和下游感知 carry 表示。

## 不应声称

不要声称：

- full material-flow manufacturing scheduling；
- first multi-robot transportation scheduling method；
- 如果没有实现 explicit material nodes 和 mode selection，不要声称 full M³-SADCHER。

当前 claim 应聚焦于 SADCHER 的 material-delivery-aware extension。

## Diagnostic Scenarios

推荐场景：

1. 短距离非关键 carry vs 长距离关键 carry。
2. Material-blocked process coalition。
3. 具有稀缺技能的 robot opportunity cost。
4. 如果启用 cooperative carry，则加入 capacity-limited delivery。

这些场景应证明 policy 学到的是 downstream unlock value，而不只是 distance。
