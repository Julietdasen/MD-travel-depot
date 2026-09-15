# 方法价值分析:C0(+MIP fallback) 到底在哪里赢?

**这一页是给自己看的诚实评估,不是给评审看的辩护**。目的是回答一个简单而尖锐的问题:

> 如果我们在标准规模下打不过 MILP,在大规模下打不过启发式,那 method 有什么价值?

以下按证据、逻辑、结论三层组织。

---

## 第一层:证据(2026-09-13 得到的量化事实)

### 事实 1 — 标准规模,MILP 是最优,C0 追不上

6 个 profile(task_count 10-24),Gurobi MILP 4 线程 300s 时限,**24/24 全部秒级最优**。三方 makespan 对照:

| profile | MILP 最优 | 新 C0 gap | 最优贪心 gap |
|---|---:|---:|---:|
| balanced | 202.2 | +21.9% | +25.5% |
| process_scarce | 154.8 | +18.3% | +37.0% |
| transport_bottleneck | 389.0 | +12.6% | +12.5% |
| dependency_deep | 249.8 | +6.4% | +10.5% |
| mixed_hard | 262.0 | +13.5% | +16.4% |
| scale_medium | 243.5 | +17.6% | +21.7% |

- **C0 相对 MILP 差 6-22%**,平均 +15.1%。
- 相比 MILP 求解 makespan(0.02-1.7s),C0 决策 μs 级——**唯一可能的价值:延迟**。但延迟尚未测过,而且很少有 MRTA 部署场景会拒绝一个 < 2 秒的求解器。
- **结论:C0 在标准规模下不是最优选项**。MILP 秒级出最优,除非有硬延迟约束(比如 100ms 内必须响应),否则用户没理由选 C0。

### 事实 2 — 大规模,贪心比 C0 好

stress-test 配置的规模阶梯(task_count 12-150,transport_ratio=1/3,precedence_density=0.30,capacity_slack=0.10):

| task_count | 新 C0 vs 最优贪心 | 谁赢 |
|---:|---:|---|
| 12 | +7.9% | 贪心 |
| 24 | +3.7% | 贪心 |
| 42 | +1.5% | 贪心(勉强) |
| 60 | +10.7% | 贪心 |
| 90 | +4.2% | 贪心 |
| 114 | +18.2% | 贪心 |
| 150 | +7.4% | 贪心 |

- **7/7 档 C0 都输贪心**,差距 1.5-18.2%。
- 老 C0(合成关系对)输得更多(7.4-24.4%)。**多规模 IL 明显在追赶,但还没追上**。
- **结论:大规模下 C0 没有质量优势**。greedy_unlock 更简单(几十行 Python)、更快(纯 CPU、无神经网络前向)、更好(makespan 更小)。

### 事实 3 — 大规模 MILP 断供,IL 拿不到标签

| task_count | MILP 状态 |
|---:|---|
| 12/24 | 3/3 秒级最优 |
| 42 | 2 最优 + 1 超时(110s 平均) |
| 60 | 1 最优 + 2 超时 |
| 90 | 3/3 超时,gap 16.6-26.9% |
| 114+ | 3/3 超时,gap 12.2-40.2% |

- 90 任务以上 **100% 超时**,产出的只是 300s 内的次优解。用这种标签监督,等于教模型"逼近一个已经很差的次优解"——**监督信号本身失效**。
- 这不是"多加计算资源"就能解决的。MILP 复杂度对任务规模是超指数;42 → 60 → 90 每一档时间涨 3-10 倍;150 任务的最优解可能要跑几小时到几天,不现实。
- **结论:IL 路径在大规模上有物理上限**。加数据也没用,因为大规模数据本身就不是最优监督。

### 事实 4 — 多规模 IL 已经把 IL 能给的东西给到了

两次 MILP-IL 概念验证:

| 版本 | 训练数据 | 6 profile 平均 gap vs MILP | scale ladder 平均 vs 贪心 |
|---|---|---:|---:|
| 老 C0 | 2000 合成关系对(手工规则)| +25.1% | +15.0% |
| MILP-IL v1 | 150 balanced(task_count=12)| +19.5% | 未测(但显然更差)|
| MILP-IL v2 | 100+100+50 (12+24+42)| **+15.1%** | +7.7% |

从 v1 到 v2 加了 5/3 倍数据、覆盖 3 个 task_count 档,gap 只压了 4.4pp——**边际收益递减**已经非常明显。要继续压需要指数级更多训练数据。

- **结论:IL 天花板在望**。往下压 5pp 需要 4-5 倍数据 + 更多规模档 + 完整 IL_REDESIGN.md forbidden-mask schema;能不能到 ≤10% 目标不确定,而且大规模仍然断供。

---

## 第二层:逻辑

**Method 要有价值,必须在下面至少一个维度上不可替代**:

| 维度 | 现状 | 判定 |
|---|---|---|
| A. 标准规模上比 MILP 好 | 不可能——MILP 是最优 | 排除,除非目标不是 makespan |
| B. 标准规模上延迟低于 MILP | 未测量;MILP 0.02-1.7s,C0 μs 级 | 可能,但很少人为 100ms 约束选 C0 |
| C. 大规模上比启发式好 | 目前 7/7 档输给 greedy_unlock | 未达成 |
| D. 大规模上比 MILP 有解 | MILP 超时时 C0 秒级出解 | **达成,但贪心也一样做到,而且更好** |
| E. 生成 MILP 训练数据(> 60 任务) | MILP 100% 超时 | **物理断供** |
| F. 在线/动态场景响应 | 未测量 | 有理论价值但缺证据 |

- **A、C、D 三条同时被证伪** → method 目前没有质量优势。
- **B、F 是可能的价值维度但未量化** → 不能作为主张。
- **E 是 IL 路径的物理死结** → IL 不可能是终极答案。

**唯一还能救的路径是 C**——让 C0 在大规模上真的胜出贪心,变成"MILP 不够,C0 顶上,而且比启发式好"。这需要突破 IL 的天花板。

**IL 突破天花板不可行,因为**:
1. 大规模 MILP 超时,监督信号断供(事实 3)
2. 加数据边际收益递减,已经出现平台效应(事实 4)
3. IL 本质是"逼近老师"——如果老师(MILP)本身在大规模不可及,IL 学生学不到超越老师的东西

---

## 第三层:结论 — RL 是必要的下一步

**RL 解决 IL 的两个根本问题**:

**问题 1: 监督信号断供** → **RL 用 reward 代替 label**。
- reward = `-makespan`(或负 latest_exit_return,与 C0/MILP 用同一定义)。
- **不需要 optimal label**,仿真器就是老师。90 任务、150 任务、300 任务都能训。

**问题 2: IL 只能逼近老师** → **RL 可以超越老师**。
- 只要探索得足够,策略可以发现 MILP 300s 内没找到的解。
- 尤其在大规模(MILP 都是次优解)的场景下,RL 天花板 = 真最优,而不是 MILP 次优。

### 具体 RL 方案(继承 [PILOT_RESULTS.md](PILOT_RESULTS.md) 末尾草案)

**关键设计**:

- **Warm-start**:用本次 MILP-IL v2 的 checkpoint([reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt](../../reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt))作为策略初始化。**不是从随机开始**——scale ladder 上 C0 已经比贪心差 1.5-18%,但已经能产出合法调度,RL 从这里出发比从贪心/MILP 出发更容易训。
- **算法**:PPO(actor-critic,兼容当前 policy 的 `(N,M)` score 矩阵 → softmax action distribution;critic 加 `MDStateValueHead`——仓库现成有)。PPO 是研究项目里做 warm-start-RL 最稳的选择,资源利用率也可控。
- **Reward**:
  - Terminal: `-latest_exit_return`(与 MILP `objective`、C0 makespan 同定义)
  - Dense shaping: 每步 "本步 pending 减少数",上限 = 总任务数,避免游戏化
  - Success bonus: rollout 成功完成给 +K,失败给 -K
- **环境**:现有 `MDDiscreteSimulator` + `OnlineMDScheduler` 是完整可交互环境。`ExplicitMIPFallback` 保留作 exploration safety net——**训练时让 fallback 处理低置信度动作,避免采集大量非法轨迹污染 buffer**。
- **Curriculum**:12 → 24 → 42 → 60 → 90 → 150 逐档扩规模,每档收敛后再放大。**在同一 checkpoint 上继续训**,不重新初始化。
- **训练成本估算**:PPO 500k-2M environment steps,每 step 一次 simulator.step + 一次 network forward;在 42 任务规模上单机 GPU 约 8-24 小时/档。总预算约 3-7 天全 curriculum。
- **成功标准**:
  - **底线**:在 42-150 任务 stress-test 阶梯上,平均 makespan 胜出 greedy_unlock ≥ 5%
  - **理想**:在 12-24 任务上追平 MILP(gap < 5%)且延迟 < 100ms
  - **失败保护**:若 PPO 训不稳、或收益远不如 IL,回退到"IL 加更多数据 + 完整 forbidden-mask schema"路径

### 需要先做的准备(1-2 天)

1. **测 latency**:量化 C0 vs MILP vs greedy 在 12/24/42/60/90/150 上的单步/单实例决策时间。这是唯一没测过的价值维度,先坐实/证伪价值 B。
2. **读 `reinforcement_learning/` 目录**:仓库里有这个目录(第一次 pilot 时未深入检查),看是否已有 PPO 或类似基础设施可复用。**不重复造轮子**——若已有则接上,若没有再从零写。
3. **设计 curriculum 和 checkpoint 复用协议**:每档训完存到独立 `reports/md_c0_ppo_curriculum_<date>_stage<k>/`,失败保护回退可用。

### 明确划界:RL 也不做什么

- **不做无 warm-start 的从零 RL**——探索灾难成本太高
- **不为大 curriculum 追求最优 hyperparameter**——PPO 默认配 + minor tuning 即可,重点在验证方向可行
- **不在同一次训练里混多规模**——curriculum 分档更稳
- **不放弃 IL/MILP baseline**——继续作对照,RL 有优势才有意义

---

## 一句话总结

现有 C0(合成 IL / MILP-IL)在**质量维度上没有一个规模区间胜出**;IL 路径在大规模上因 MILP 超时而**物理断供**;继续压 IL 边际收益递减且不能突破 MILP 天花板。**RL 是唯一有希望让方法在大规模上真正胜出贪心、且不受 MILP 求解能力限制的路径**,应作为下一阶段的主线,MILP-IL v2 的 checkpoint 作 warm-start。
