# Problem envelope for Path C v1 stability checkpoint (2026-09-20)

## 目的

现有 stability eval 已经在 `process_scarce` profile 的 tc {12, 24, 42, 60} 上跑出规模轴 —— C0 相对 greedy_unlock 的优势从 tc12 的 −19pp 单调收窄到 tc60 的 −0.7pp。这份报告在**不改问题定义** (只加 `--override-*` CLI，不新增 profile) 的前提下，围绕 tc=24 (甜点区) 扫两个额外轴：

- **process robot 稀缺度** (Sweep 3)
- **process duration 异质性** (Sweep 2)

目的：刻画 policy 相对 greedy_unlock 结构性有利的 problem regime，把 tc=60 上"gap 消失"这个观察从"policy 变差"重新定位为"envelope 之外"。

## Sweep 结果

### Sweep 1 (规模轴，已有 stability eval)

`process_scarce` 默认 scaling，seeds 301-304，C0 = Path C v1 stability seed3101 epoch5。

| tc | MILP | C0 | greedy_unlock | C0−greedy |
|---:|---:|---:|---:|---:|
| 12 | 164.25 | 175.75 | 207.00 | **−15.1pp** (best greedy=distance) |
| 24 | 250.00 | 270.25 | 304.00 | **−11.1pp** |
| 42 | 304.00 | 338.75 | 346.50 | **−2.2pp** |
| 60 | 356.00 | 393.50 | 395.75 | **−0.6pp** |

**边界**：tc ≤ ~42。tc60 上 unlock 信号饱和，C0 学到的 non-unlock 结构无发挥空间。

### Sweep 3 (process robot 稀缺度，tc=24 固定)

Transport robot = 4 (tc=24 默认 scaling)，seeds 301-304。

| process_robot_count | MILP | C0 | best greedy (name) | C0 gap | greedy gap | **C0−greedy** |
|---:|---:|---:|---:|---:|---:|---:|
| 2  | 261.00 | 289.75 | 300.25 (distance) | +11.0% | +15.0% | **−3.5pp** |
| 3  | 245.25 | 270.25 | 286.75 (unlock)   | +10.2% | +16.9% | **−5.8pp** |
| **4**  | **250.00** | **270.25** | **304.00 (unlock)** | **+8.1%** | **+21.6%** | **−11.1pp** |
| 6  | 231.25 | 264.50 | 278.75 (unlock)   | +14.4% | +20.5% | **−5.1pp** |
| 8  | 228.75 | 255.75 | 264.00 (distance) | +11.8% | +15.4% | **−3.1pp** |

**形状：倒 U，pr=4 是甜点。**

- pr=2 (极稀缺)：只有 2 个 process robot 可选，"选谁"的动作空间坍缩，greedy 撞对的概率高，policy 无处发挥。
- pr=8 (充裕)：每个 process task 基本可以独占一个 robot，coalition 组成 tradeoff 消失，问题退化到 greedy 也能做好的 unlock 排序。
- pr=4 (甜点)：process_task_count / pr ≈ 5，coalition 组成有取舍但动作空间够大，policy 结构优势最大。

**边界**：pr / process_task_count 需要在 ~0.15–0.25 区间 (对应 tc=24 时的 pr=3-6)。

### Sweep 2 (process duration 异质性，tc=24 固定，pr=4)

`process_scarce` 默认 profile 不指定 duration range，走生成器 default (5-20)。三档 override：

| range | MILP | C0 | best greedy (name) | C0 gap | greedy gap | **C0−greedy** |
|---:|---:|---:|---:|---:|---:|---:|
| narrow (8-12) | 217.00 | 252.75 | 259.25 (eta)     | +16.5% | +19.5% | **−2.5pp** |
| **medium (5-20)** | **250.00** | **270.25** | **304.00 (unlock)** | **+8.1%**  | **+21.6%** | **−11.1pp** |
| wide (3-40)   | 345.25 | 391.75 | 393.25 (unlock) | +13.5% | +13.9% | **−0.4pp** |

**形状：也是倒 U 而不是单调，中等 spread 是甜点。**

反直觉：更宽的 duration spread 并没有让 policy 更有利。原因是极宽 spread (3-40) 下——

1. 长任务 (30-40) 决定了 critical path 上的 makespan floor，MILP 找到长任务提前调度的 schedule；
2. greedy_unlock 虽然看不到 duration，但只要**碰对**把长任务先做（unlock 计数或 upstream 排序恰好把它排前），gap 就消失；
3. 3-40 的 range 让单个长任务的影响非常大，尾部方差高，C0 学到的稳态 "先做 duration 大的" 策略反而没 MILP 的实例特定安排精确。

窄 spread (8-12) 反过来：duration 信号消失，C0 的 duration-aware 学习在 greedy_eta 面前也没优势——两者都在近似同一策略。

**边界**：duration 需要中等 spread，比值大约 max/min ∈ 3–5 之间；过窄失去信号，过宽被 MILP 精确排程碾压且 C0 稳态策略跟不上。

## 综合结论

Path C v1 stability epoch-5 checkpoint 的**问题 envelope**：

1. **规模** tc ≤ ~42。
2. **process robot 稀缺度** pr / process_task_count ∈ ~0.15–0.25 (tc=24 时 pr=3-6)。
3. **duration 异质性** max/min ∈ ~3–5 的中等 spread。

**tc=60 上 gap 缩窄是 envelope 外行为，不是 policy 退化。** 主 profile `process_scarce` 默认 scaling 在 tc=60 上把 pr scale 到 10，落在 Sweep 3 的 pr=8 邻域，正好是 policy 优势最弱的一档；这是"规模"和"稀缺度"两个 envelope 边界叠加的结果。

## 与前 session 结论的关系

- HANDOFF 建议"回到 downstream-correlated model-selection signal"仍然成立——这个 envelope 只说明 tc=60 的 gap 消失有结构原因，不改变 Path C v1 stability 是"最优可复现 checkpoint"这一判断。
- 上一次 layer-2 v2 profile 尝试通过更极端 problem 拉开 gap 失败，符合本报告的 pr=2 观察：极稀缺不是 policy 优势区。
- **不推荐**为了在 tc=60 上"多挤 5%"再改 profile；换到 envelope 内的 tc=12-42 部署是价值路径。

## 数据与代码

- Sweep 3 raw: `reports/md_envelope_process_robots_2026-09-20/pr{2,3,4,6,8}/{summary.json,rows.json,final_report.md}`
- Sweep 2 raw: `reports/md_envelope_duration_2026-09-20/{narrow,medium,wide}/{summary.json,rows.json,final_report.md}`
- Driver: `reports/md_envelope_process_robots_2026-09-20/run.sh`, `reports/md_envelope_duration_2026-09-20/run.sh`
- 代码改动: `experiments/md_c0_scaled_target_profiles_eval.py` 新增 `--tc-list`, `--override-process-robot-count`, `--override-transport-robot-count`, `--override-process-duration-min/max`。默认行为不变。
- Checkpoint: `reports/md_c0_pathC_v1_stability_2026-09-18/seed3101/training/checkpoint_epoch_5.pt`

## 未测但值得后续做的

- **3-seed 稳定性**：本报告只用 stability seed3101 ep5；重跑 seed3102/3103 应能确认曲线形状（seed3102/3103 ep5 与 seed3101 ep5 的 ps60 差 ≤0.5，形状不太可能翻转，但值得一小时确认）。
- **Transport robot 轴**：本报告固定 tr=4。tr 稀缺度是否也是倒 U 未验证。
- **pr × duration 交互**：本报告分开扫，未做 2D 网格。甜点区叠加是否进一步拉开 gap 未测。
