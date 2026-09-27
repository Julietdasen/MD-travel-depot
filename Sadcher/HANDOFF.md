# Residual IL / Joint Action Correction Handoff

更新日期：2026-09-06

## Ticket 48 exact-action candidate coverage 结论

Ticket 48 已新增独立的 `exact-online-action-1.0` oracle/schema，并在固定的
150 train + 60 development stationary snapshots 上完成 pilot。旧 forced-batch API、
历史 residual JSONL、C0/residual checkpoint 和生产 decoder 均未迁移或覆盖。

正式输出位于：

```text
reports/md_exact_action_candidate_diagnostic_2026-09-06/
```

本轮共生成 489 个 exact complete-action labels 和 630 个三模型 candidate records，
Gurobi 全部为 optimal，replay/schema/quota 检查无 failure。三个 residual model seed 的
development recall@8 均为 100%，每个 model seed x pending-count stratum 也均为 100%。
最终 gate 为：

```text
candidate_coverage_supported_for_next_stage
```

该结论仅允许另行预注册 scorer 消融，不授权本轮训练 scorer、运行 DAgger 或读取
confirmation split。`75800--75949` 现仅作为 regression benchmark；`76000--76149`
为锁定的 confirmation split，本轮没有生成或读取。

旧 test 结果仍只支持“residual 平均 makespan 优于 C0/masked-greedy”，不支持声称旧
逐实例 task-completion 验收通过。旧 forced labels 是 partial inclusion constraints，
Ticket 48 labels 才是带 idle complement 和禁止 timestep-0 dispatch 的 complete actions。

## Exact-action scorer 消融结果（2026-09-07）

在 Ticket 48 gate 通过后，已按预注册协议仅使用 150 train snapshots 训练 linear 与
DeepSets action scorer，并使用 60 development snapshots 选择 checkpoint。结果位于：

```text
reports/md_exact_action_scorer_ablation_2026-09-07/
```

DeepSets 在三个 residual seed 上的 tolerance-optimal top-1 都提高 1/60（平均
`+1.67` percentage points），但低于预注册的 `+2` points 门槛，且跨 seed 平均
regret 从 additive baseline 的 `2.10` 上升为 `2.17`。Linear 平均 top-1 下降。
最终状态固定为：

```text
scorer_not_supported
```

不得读取 confirmation split，不得把这些 checkpoint 接入生产 decoder。训练曲线在
epoch 20 后明显过拟合；下一轮若继续，应先把完整 simulator/task/robot physics features
加入 exact-action package，并重新预注册更小容量或正则化 scorer，而不是在同一
development package 上事后调参。

## 新 Session 的任务上下文

当前工作已经完成 Residual IL 十倍扩容、三 seed 正式训练、150 个 test
instance 的配对评估，以及两轮 joint-action correction 实验。下一阶段不要继续盲目
增加 joint head 的训练 epoch；当前最主要的问题是 MILP 标签描述的是 partial
forced batch，而不是 simulator 当前时刻的完整联合动作。

建议新 session 先阅读本文，然后阅读：

- `AGENTS.md`
- `docs/residual_trajectory_il_implementation.md`
- `reports/md_residual_scale10_2026-09-04/evaluation/summary.json`
- `reports/md_frozen_joint_correction_2026-09-06/evaluation/summary.json`

## 不可破坏的约束

- 不修改或覆盖既有 C0 数据、C0 训练入口、C0 checkpoint 和历史报告。
- C0 checkpoint 只读路径：
  `reports/md_c0_end_to_end_diagnostic_pilot_2026-09-01/training/`。
- 当前 residual checkpoint 和历史评估结果继续保留，不原地覆盖。
- Train seeds：`75000--75599`。
- Development seeds：`75600--75749`。
- Regression benchmark seeds：`75800--75949`。
- Regression benchmark 不能用于训练、超参数选择或 checkpoint 选择。
- Confirmation seeds：`76000--76149`，在另行预注册前不得生成或读取。
- 当前正式范围只覆盖 stationary snapshots；动态 transport/process 状态尚未进入
  residual 数据转换协议。
- 工作区很脏，存在大量用户修改和未跟踪实验文件。不要清理、回滚或格式化无关文件。

## 已完成内容

### Residual IL 数据与训练

正式数据目录：

```text
reports/md_residual_scale10_2026-09-04/dataset_seed3101
reports/md_residual_scale10_2026-09-04/dataset_seed3102
reports/md_residual_scale10_2026-09-04/dataset_seed3103
```

数据统计：

| Model seed | Train snapshots | Development snapshots | Forced MILP solves | Exclusions | Joint projection error |
|---|---:|---:|---:|---:|---:|
| 3101 | 1884 | 460 | 4792 | 0 | 25.48% |
| 3102 | 1853 | 461 | 4727 | 0 | 25.61% |
| 3103 | 1861 | 454 | 4780 | 0 | 25.82% |

所有纳入数据的 forced MILP solves 均为 optimal。训练配置为 Adam、
`lr=0.002`、batch size 32、100 epochs，并保持 C0/residual 样本 50%/50%。

Residual checkpoint：

```text
reports/md_residual_scale10_2026-09-04/training/seed3101/C0_residual_tail_seed3101/best_checkpoint.pt
reports/md_residual_scale10_2026-09-04/training/seed3102/C0_residual_tail_seed3102/best_checkpoint.pt
reports/md_residual_scale10_2026-09-04/training/seed3103/C0_residual_tail_seed3103/best_checkpoint.pt
```

### Residual IL 正式测试

结果文件：

```text
reports/md_residual_scale10_2026-09-04/evaluation/summary.json
reports/md_residual_scale10_2026-09-04/evaluation/rollouts.jsonl
reports/md_residual_scale10_2026-09-04/evaluation/representative_traces.json
```

150 个 test instances 的平均 final makespan：

| Method | Seed 3101 | Seed 3102 | Seed 3103 |
|---|---:|---:|---:|
| Legacy C0 | 244.293 | 245.147 | 243.900 |
| Residual | 237.827 | 238.060 | 240.913 |

共同基线：

- Masked-greedy：`243.480`。
- Full initial-domain MILP：`198.580`。

Residual 相对对应 C0 的 paired mean 改善：

- Seed 3101：`-6.467` timestep，约 `2.65%`。
- Seed 3102：`-7.087` timestep，约 `2.89%`。
- Seed 3103：`-2.987` timestep，约 `1.22%`。
- Success 全部为 `150/150`，illegal assignments 全部为 `0`。

重要解释：三个 residual seed 的平均 makespan 都优于对应 C0，也优于
masked-greedy 的总体均值；但该轮原始 `acceptance.passed=false`，原因是三个 seed
均未满足“任一 instance 的 latest task completion 增幅不超过 1 timestep”这一硬约束。
因此可以说平均性能改善，但不能说原计划的全部正式验收条件均已满足。

### Joint-action correction

已增加的主要代码：

```text
experiments/joint_batch_features.py
experiments/md_joint_residual_training.py
experiments/md_frozen_joint_evaluation.py
imitation_learning/md_joint_residual_train.py
imitation_learning/md_joint_residual_train_v2.py
imitation_learning/md_frozen_joint_correction_train.py
schedulers/joint_batch_decoder.py
run_joint_residual_training.sh
run_joint_residual_v2_parallel.sh
run_frozen_joint_correction.sh
run_frozen_joint_test.sh
```

端到端 joint 训练完成了两条初始化路线、各三个 seed、100 epochs：

- Residual-init 三个 seed 在 development 上均退化。
- C0-init seed 3102/3103 改善，但 seed 3101 退化。
- 结论：端到端 joint 训练不稳定，不能替换当前 residual policy。

随后实现 frozen joint correction：冻结 residual base 和 edge heads，只训练
`joint_head.*`，correction 限制为 `+/-0.25`，并保留 epoch 0 fallback。

Development 选择结果：

| Seed | Selected epoch | Forced regret before | Forced regret after |
|---|---:|---:|---:|
| 3101 | 0 | 0.6152 | 0.6152 |
| 3102 | 30 | 0.6377 | 0.5336 |
| 3103 | 30 | 0.4097 | 0.3877 |

Frozen joint checkpoint：

```text
reports/md_frozen_joint_correction_2026-09-06/seed3101/best_checkpoint.pt
reports/md_frozen_joint_correction_2026-09-06/seed3102/best_checkpoint.pt
reports/md_frozen_joint_correction_2026-09-06/seed3103/best_checkpoint.pt
```

Full test 结果：

| Seed | Residual | Frozen joint | Delta | Win/Tie/Loss |
|---|---:|---:|---:|---:|
| 3101 | 237.827 | 237.700 | -0.127 | 1/149/0 |
| 3102 | 238.060 | 238.060 | 0.000 | 0/150/0 |
| 3103 | 240.913 | 240.913 | 0.000 | 0/150/0 |

测试结果位于：

```text
reports/md_frozen_joint_correction_2026-09-06/evaluation/summary.json
reports/md_frozen_joint_correction_2026-09-06/evaluation/rollouts.jsonl
```

Frozen joint 的安全验收通过：success 不下降、illegal 为 0、task completion
约束通过。但是实际收益可以忽略：450 个 model/instance 配对中只有一个动作结果发生
改善。因此不要把 frozen joint 宣称为有意义的总体性能提升，也不要用它替换 residual
主模型。

## 已确认的根因

当前 residual MILP 的 `forced batch` 语义是：这些 assignments 必须包含在最优
后续 schedule 中。它不保证：

- 这些 assignments 是当前 decision event 的完整动作集合；
- 其他当前可执行 assignments 必须被禁止；
- 未被分配的机器人应显式 idle/wait；
- forced batch 中的 assignments 是唯一同时执行的 first actions。

因此旧 joint 标签是 partial inclusion constraint，不是完整 online action。把它直接交给
joint decoder 后曾在 2-instance smoke 中产生严重退化。已采用的 local-safe evaluation
仅允许：

- 在 `eligible_snapshot(simulator)` 的静止状态应用 joint correction；
- 保留 residual proposal 的 task set 和 batch size；
- 只重排原 proposal 内的机器人 assignment；
- correction 只有严格优于原 proposal 时才采用，否则回退 residual；
- 动态状态直接使用 residual decoder。

`schedulers/joint_batch_decoder.py` 中的 unrestricted decoder 是实验代码，已知不能作为
默认线上路径使用。

## 下一阶段的正确目标

下一阶段应把监督单位从 forced inclusion batch 改为精确的
`(state, complete first action, continuation value)`：

```text
state s
  -> enumerate complete legal online actions A(s)
  -> execute/lock exactly one action a
  -> MILP solves the remaining continuation
  -> Q(s,a), task horizon, return tail, regret
  -> train a batch-aware ranking/value model
```

完整动作必须显式包含：

- 当前事件中所有机器人到任务的 assignment；
- coalition 的完整成员；
- 未动作机器人的 idle/wait；
- 对未选择且当前可执行 assignment 的禁止约束；
- simulator transition/replay 所需的所有状态变化。

对每个合法完整动作 `a` 定义：

```text
Q(s, a) = exact continuation final makespan after applying a
regret(s, a) = Q(s, a) - min_b Q(s, b)
```

`regret <= 1 timestep` 的多个动作应作为 tie-optimal set，不依赖 MILP 任意的
tie-breaking。

## 推荐的实施顺序

### 1. 先做标签协议 fixture，不立即重跑大数据

新增最小 fixture 覆盖：

- partial forced batch 与 complete first-action batch 的差异；
- 存在额外 unforced concurrent action；
- 显式 idle/wait；
- transport singleton；
- process coalition；
- material/normal precedence；
- dynamic state 必须被排除或拥有独立完整转换协议；
- MILP schedule replay 与 simulator 的 first action、task completion、tail、makespan
  完全一致。

成功标准：在小规模 domain 上，对每个候选完整动作，MILP 标注的 `Q(s,a)` 与
simulator 执行动作后 replay 的 final makespan 完全一致。

### 2. 实现 exact online-action oracle

建议增加独立接口，不改变旧 residual 数据的只读兼容：

```text
enumerate_complete_online_actions(...)
solve_exact_action_continuation(...)
label_complete_action_snapshot(...)
```

不要复用含糊的 `forced batch == complete action` 假设。旧 schema 继续可读，新 schema
显式增加 `complete_first_action`、`idle_robot_ids`、`forbidden_immediate_assignments`、
`continuation_*` 和 replay audit 字段。

### 3. 先测 candidate oracle recall

在 50--100 个 train snapshots 和一小批固定 development snapshots 上记录：

```text
oracle_action_recall@K
oracle_value_gap@K
candidate_count
candidate_generation_time
```

候选来源至少比较：residual top-K、masked-greedy、coalition-aware，以及 residual
proposal 的 swap/add/drop 邻域。如果 oracle action 不在候选集，优先修候选生成；如果
候选覆盖高但排序错误，再升级 joint scorer。

### 4. 用正确标签做受控模型消融

保持数据和候选一致，依次比较：

| Experiment | Label | Candidate set | Scorer |
|---|---|---|---|
| A | Old forced batch | Current | Current joint |
| B | Exact complete action | Current | Current joint |
| C | Exact complete action | Diverse top-K | Current joint |
| D | Exact complete action | Diverse top-K | DeepSets/attention |

模型目标建议使用状态内归一化 regret、pairwise/listwise ranking、tie-aware optimal-set
classification，以及 task/tail 辅助回归。绝对 makespan regression 不应成为唯一目标。

### 5. Development 通过后再做 active DAgger

- 只在 train seeds 上运行 residual/joint policy。
- 优先选择与专家分歧大、置信度低或 predicted regret 高的状态进行 MILP 标注。
- Development 和 test 保持冻结。
- 新标签单独落盘，不覆盖 `md_residual_scale10_2026-09-04`。

只有三个 seed 的 development 均稳定优于 edge-only residual，且 replay、安全指标通过，
才重新运行 150-instance test。

## 可继续改进但暂不应抢先做的模块

- 状态表示：precedence graph GNN、critical-path 特征、机器人 release time/position、
  terminal-return lower bound。
- Joint scorer：DeepSets、Set Transformer 或 graph attention。
- 推理解码：在 residual proposal 上进行有固定预算的 swap/add/drop 局部搜索。
- 动态状态：显式保存 phase、remaining duration、材料携带和 coalition 状态后，再扩展
  trajectory/event encoder。
- 规模工程：更大任务/机器人规模下的 candidate explosion、MILP 标注吞吐、内存和
  推理延迟。

这些工作应在 exact-action 标签和 candidate recall 诊断之后进行，否则无法区分标签、
候选和网络表达各自的贡献。

## 当前研究结论

1. Residual IL 的方向有效：三 seed 平均 makespan 相对 C0 改善约 `1.2%--2.9%`，
   success 100%，illegal assignment 为 0。
2. Residual 尚未通过原计划的全部逐实例 task-completion 硬约束，且距 MILP 仍约
   `39--42` timestep。
3. 约 `25%` joint projection error 说明纯 additive edge score 表达不足，但不能证明
   当前 forced-batch joint 标签是正确的。
4. End-to-end joint correction 不稳定；frozen/local-safe correction 安全但总体无效。
5. 当前首要瓶颈是 joint-action label semantics 和 candidate coverage，而不是 GPU、训练
   epoch 数或 joint head 宽度。

## 环境与复现实用信息

项目目录：

```text
/data/ZJZ/3D-Foundation-Model/MRTA/Sadcher
```

实验 Python：

```text
/data/ZJZ/miniconda3/envs/mrta-sadcher/bin/python
```

Frozen joint 正式评估入口：

```bash
bash run_frozen_joint_test.sh
```

该命令会写入既有 evaluation 目录。若需要重新实验，使用新的带日期输出目录，不要覆盖
本文引用的历史结果。

MILP solver 后端（2026-09-17 更新）：默认已切换到 OR-Tools CP-SAT，无需 Gurobi
license。所有 runtime 调用统一走 `baselines.md_oracle_dispatch`（`solve_md_oracle`、
`solve_residual_forced_batch`）；用环境变量 `MRTA_MILP_SOLVER=gurobi` 切回旧后端。
Residual forced-batch oracle 已完成 CP-SAT 移植（`baselines/md_ortools_residual_oracle.py`），
exact-action 数据生成也不再需要 Gurobi。细节见 `ReadME/2026-09-17/BASELINE_SWAP_TO_ORTOOLS.md`。

## 给新 Session 的建议首条指令

```text
阅读 AGENTS.md 和 HANDOFF.md。不要修改任何 C0 数据、训练流程、checkpoint 或历史报告。
先审计 residual oracle 中 forced batch 的实际约束与 simulator 的 apply/replay 语义，设计并
实现最小 complete-first-action fixture 和 exact continuation Q(s,a) 标签测试。在 fixture
通过前不要生成大规模数据，也不要启动新一轮 100-epoch 训练。
```
