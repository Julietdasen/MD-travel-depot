# Residual-Trajectory IL 实现方案

## 目标

让 C0 学习“当前联合动作对完整剩余任务和最终 depot return 的影响”，而不是只记忆某个 robot-task 配对编号。

首版只使用满足以下条件的静止 snapshot：所有机器人空闲、没有执行中的运输或 process coalition、剩余 1--3 个任务。每个 snapshot 的标签由 residual MILP 对所有合法 forced batch 求得。

## 数据流

```text
instance seed
  -> legacy C0 / masked_greedy rollout
  -> 每个静止决策点 snapshot
  -> residual state + legal mask
  -> 枚举全部合法 forced batches
  -> 每个 batch 的 residual MILP solve
  -> task horizon / return tail / makespan / regret
  -> edge 投影 + joint-batch 标签
  -> train/dev/test JSONL
```

每个完整 instance 只能属于一个 split。推荐首版固定使用既定 seed 划分，不根据 test 结果补 seed。

## Snapshot 采样

在 rollout 的每个决策点检查：

```python
def eligible_snapshot(sim):
    return (
        all(robot.activity is RobotActivity.AVAILABLE
            for robot in sim.robot_states.values())
        and all(task.status is not TaskStatus.IN_PROGRESS
                for task in sim.task_states.values())
        and 1 <= len(pending_tasks(sim)) <= 3
    )
```

同一 instance、同一 completed/pending 集合、同一机器人位置的 snapshot 只保留一次。snapshot 必须保存：

- `current_time`；
- 每个机器人的位置、类型、能力和可用性；
- completed/pending task IDs；
- 已满足的 material dependency；
- 原始 simulator snapshot 和合法 mask；
- 产生该状态的 rollout method。

当前实现中，snapshot 转成 residual domain 时删除 completed task，并删除已经满足的 precedence/material edge。未来扩展执行中状态时，必须额外保存 phase、remaining duration、携带材料和未完成 coalition，不能复用这个静止状态转换器。

## 联合动作标签

对每个 snapshot 枚举所有非空合法联合动作：

- transport：一个 robot 的 singleton；
- process：覆盖全部 required skills 的 inclusion-minimal coalition；
- 不重复使用 robot；
- 每个 forced task 必须是对应 robot 的第一个 residual action；
- 不做候选截断。

对 batch `a` 定义：

```text
task_cost(s, a) = residual task horizon
tail_cost(s, a) = synchronized terminal return tail
total_cost(s, a) = task_cost(s, a) + tail_cost(s, a)
regret(s, a) = total_cost(s, a) - min_a total_cost(s, a)
```

所有 forced solve 必须 `threads=1`、`time_limit=10s`。只有 snapshot 下所有 batch 都是 optimal 才能进入 train/dev/test；任何 timeout/infeasible/error 写入 exclusion JSONL。

`regret <= 1 timestep` 的 batch 统一设置 `tie_optimal=true`。训练 ranking 时，这些 batch 两两不排序；评估使用 tie-aware accuracy。

## Edge 投影

现有 C0 仍然输出 robot-task score matrix，因此 batch 标签需要投影到 edge：

```text
edge_best_total(r, t) = min total_cost(s, a), 其中 (r,t) 属于 a
edge_best_task(r, t)  = 对应 batch 的 task_cost
edge_best_tail(r, t)  = 对应 batch 的 tail_cost
```

如果多个 batch 并列，选择 lexicographically 最早的 batch 作为诊断代表，但训练 target 使用并列 batch 的平均分解代价，避免任意 batch 编号成为标签信号。

同时保存 batch 原始标签，不能只保存 edge 投影；否则无法检测 pairwise score 无法表达 coalition/joint action 的情况。

## 模型结构

保持线上接口不变：

```python
scores = model(robot_features, task_features, ..., md_inputs=inputs)
```

新增仅供训练和诊断的接口：

```python
scores, costs = model.forward_with_costs(...)
```

其中 `costs` 为 `[batch, robot, task]` 的三个 edge head：

- `total`：归一化 total cost；
- `task`：归一化 task horizon；
- `tail`：归一化 return tail。

三个 head 接在现有 robot/task pair representation 后，不修改 scheduler 或 constrained decoder。每个 snapshot 内分别做 min-max 或 robust scale；如果状态内方差为 0，target 置零并跳过该项的回归权重。

## 联合动作评分

训练时用 edge score 聚合 batch：

```python
def batch_score(batch, edge_scores):
    return sum(edge_scores[r, t] for r, t in batch.assignments)
```

对任意两个 batch `a,b`：

- 若 `abs(regret_a - regret_b) <= 1`，跳过 ranking pair；
- 否则要求更低 regret 的 batch 得分更高；
- 使用 margin ranking loss，margin 固定为 `0.1`。

对每个 snapshot 记录 additive edge cost 与真实 joint-batch cost 的平均相对误差 `joint_projection_error`。当全体样本均值超过 20% 时，manifest 只建议下一阶段增加 joint-batch head；本轮不改变 edge 聚合规则。

## 训练目标

从各自 legacy C0 checkpoint 初始化三个独立 seed，不覆盖旧 checkpoint：

```text
L = L_C0 + L_rank + 0.25 * (L_total + L_task + L_tail)
```

`L_C0` 保持原有 relational/imitation loss。每个 batch 按 50% 原始 C0 数据、50% residual snapshot 数据混合。固定 Adam、learning rate `0.002`、100 epochs。

每 10 epochs 在 development split 评估，并按以下顺序选择 checkpoint：

1. 最低平均 forced-action regret；
2. 若相同，最低平均 terminal return tail；
3. 若仍相同，选择更早 epoch。

Test split 在 checkpoint 固定前不可读取。

## 评估协议

对每个 test instance 记录完整事件序列，并比较：

- legacy C0 三个 seed；
- residual-tail 三个对应 seed；
- masked_greedy；
- full initial-domain MILP。

同时报告：

- success rate；
- illegal assignment count；
- 最晚真实任务完成时间；
- terminal return tail；
- final makespan；
- tie-aware forced-action accuracy；
- forced-action 平均 regret；
- task/tail regression error。

额外计算两个诊断量：

```text
decision_regret = 实际 batch 的 cost - 当前 snapshot 最优 cost
state_regret = 当前 snapshot 最优 cost
               - 同等任务进度下参考轨迹 snapshot 最优 cost
```

解释方式：

- `decision_regret` 高：当前 snapshot 的动作选择错误；
- `decision_regret` 低、`state_regret` 高：早期路径/分配已经把机器人带到不利位置，后期没有足够补救空间。

每个 split 保存改善最大、退化最大和接近中位数的三个 instance 事件序列，用于定位差距最早出现的决策点。

## 实现顺序

### 阶段 1：正确性基线

1. 修复并测试 forced-first residual MILP；
2. 验证 MILP schedule replay 与 simulator 的 task completion、tail、makespan 完全一致；
3. 在小 fixture 上验证 completed task、material/normal precedence 和 depot `(0,0)`。

### 阶段 2：多状态数据集

1. 为 C0 和 masked_greedy rollout 添加静止 snapshot 提取器；
2. 生成固定 train/dev/test JSONL；
3. 生成 exclusion JSONL 和 manifest；
4. 检查 instance/split 隔离及 duplicate snapshot 去重。

### 阶段 3：长程监督训练

1. 加载 legacy C0 checkpoint；
2. 添加三个 cost heads；
3. 混合 C0 relational 数据与 residual 数据；
4. 训练 100 epochs，保存 `C0_residual_tail_seed{seed}`；
5. 只用 development 选择 checkpoint。

### 阶段 4：独立测试与根因分析

1. 固定 checkpoint 后运行 test；
2. 计算 decision/state regret；
3. 输出 paired metrics 和代表性事件序列；
4. 只有三个 seed 同时满足预设门槛时才扩大规模。

## 扩展到完整长轨迹

首版不要求模型输入整条历史。只要 snapshot 是 Markov-complete，长期信息已经通过 residual cost-to-go 进入标签。

只有在以下情况才增加 trajectory encoder 或 recurrent state：

- 完整 snapshot 相同但历史不同会导致 simulator 未来行为不同；
- 执行中 phase、剩余 duration 或材料携带状态无法由当前输入恢复；
- edge projection error 长期超过阈值，说明 pair scorer 无法表达联合动作。

届时扩展为：`history encoder -> snapshot representation -> edge/joint cost heads`，并保持线上 decoder 的动作接口不变。

## 成功标准

首轮只有在三个 residual-tail seed 均满足以下条件时才宣布有效：

- test forced-action 平均 regret 低于对应 legacy C0；
- 平均 terminal return tail 更低；
- 平均 makespan 更低；
- 真实任务完成时间不增加超过 1 timestep；
- success rate 不下降；
- illegal assignment 始终为零。

若 train/dev 改善而 test 不改善，结论固定为过拟合或状态表达不足，不调整 loss、seed、tie threshold 后重新解释结果。

## 十倍扩容实现入口

当前正式 split 固定为：

```text
train       75000--75599
development 75600--75749
test        75800--75949
```

生成 train/development snapshot（默认 4 workers，每个 Gurobi solve 使用
`Threads=1`、`TimeLimit=10s`）：

```bash
python -m data_generation.md_residual_pipeline \
  --output-dir reports/md_residual_scale10/dataset_seed3101 \
  --model-seed 3101 \
  --c0-checkpoint 3101=reports/md_c0_end_to_end_diagnostic_pilot_2026-09-01/training/C0_seed3101/checkpoints/best_checkpoint.pt
```

输出采用 `batch_shards/` 和 `shards/` 两级缓存；中断后使用同一命令并增加
`--resume`。并发 license 探测失败时自动降为单 worker。`manifest.json`
记录 split、来源、pending 分布、全部 solve 状态、吞吐和
`joint_projection_error_mean`。

首轮 checkpoint 固定后，DAgger 只追加 train 状态：

```bash
python -m data_generation.md_residual_pipeline \
  --output-dir reports/md_residual_scale10/dataset_seed3101 \
  --model-seed 3101 \
  --splits train --resume \
  --c0-checkpoint 3101=/path/to/C0_seed3101/best_checkpoint.pt \
  --residual-checkpoint 3101=/path/to/C0_residual_tail_seed3101/best_checkpoint.pt
```

test snapshot 生成必须显式提供一个已固定 checkpoint：

```bash
python -m data_generation.md_residual_pipeline \
  --output-dir reports/md_residual_scale10/dataset_seed3101 \
  --model-seed 3101 \
  --splits test --resume \
  --unlock-test-checkpoint /path/to/fixed/best_checkpoint.pt \
  --c0-checkpoint 3101=/path/to/C0_seed3101/best_checkpoint.pt
```

最终 paired rollout：

```bash
python -m experiments.md_residual_evaluation \
  --output-dir reports/md_residual_scale10/evaluation \
  --c0-checkpoint 3101=/path/to/C0_seed3101/best_checkpoint.pt \
  --c0-checkpoint 3102=/path/to/C0_seed3102/best_checkpoint.pt \
  --c0-checkpoint 3103=/path/to/C0_seed3103/best_checkpoint.pt \
  --residual-checkpoint 3101=/path/to/C0_residual_tail_seed3101/best_checkpoint.pt \
  --residual-checkpoint 3102=/path/to/C0_residual_tail_seed3102/best_checkpoint.pt \
  --residual-checkpoint 3103=/path/to/C0_residual_tail_seed3103/best_checkpoint.pt
```

评估输出包括逐 rollout JSONL、按对应 model seed 配对的 summary、相对
masked-greedy/full MILP 的剩余 makespan gap，以及每个 seed 的最大改善、
最大退化和中位数附近各三个完整事件序列。现有 C0 dataset、训练入口和
checkpoint 只被读取，不会被覆盖。

三个 seed 分别训练，示例：

```bash
python -m imitation_learning.md_residual_train \
  --c0-dataset-root datasets/md_expert_v1_small \
  --residual-dataset reports/md_residual_scale10/dataset_seed3101 \
  --legacy-checkpoint /path/to/C0_seed3101/best_checkpoint.pt \
  --output-dir reports/md_residual_scale10/training \
  --seed 3101 --epochs 100 --batch-size 32
```
