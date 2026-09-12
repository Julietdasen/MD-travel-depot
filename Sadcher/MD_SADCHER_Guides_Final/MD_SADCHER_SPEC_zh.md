# MD-SADCHER++ MVP 规格说明

状态：本地项目规格说明。本文档由 grill 已确认的决策和已接受的 ADR 综合而成，维护在仓库内，不发布到 issue tracker。

## Problem Statement

当前 SADCHER 主要表达异构机器人过程任务调度，尚未把物料运输建模为一等的任务依赖。因此，一个过程任务可能在输入物料尚未到达时被判断为可执行。运输能力与过程技能也属于不同语义，但 legacy 表示没有区分 transport robot 和 process robot。

项目需要一个最小可验证的 material-delivery（MD）runtime，在不修改神经策略的前提下明确任务/机器人类型、运输时间、ready 规则、hard feasibility、metrics 和确定性场景。第一版必须保留旧 SADCHER 实验和 checkpoint 的可运行性与可比性。

## Solution

增加环境级 MD 领域模型和离散 timestep 仿真器。真实任务分为互斥的 `PROCESS`、`TRANSPORT` 两类，真实机器人分为互斥的 `PROCESS_ROBOT`、`TRANSPORT_ROBOT` 两类。Transport task 携带一份物料，从显式的 pickup location 运到显式的 delivery location，并且恰好解锁一个 downstream process task。

所有 scheduler 和 baseline 共享集中式 hard-feasibility 服务。Transport 使用以下显式状态机：

```text
WAITING -> TO_PICKUP -> LOADING -> TO_DELIVERY -> UNLOADING -> COMPLETE
```

通过确定性手工场景、结构化 execution record、基础 metrics 和 Greedy baselines 验证语义。本阶段不修改 neural network、GAT 输入、checkpoint 或 decoder。上传的 `8t3r3s` optimal 数据集作为 legacy process-only 基准和回归夹具保留，不为其虚构 MD 字段。

## User Stories

1. 作为调度研究者，我希望声明任务是 `PROCESS` 还是 `TRANSPORT`，以便物料移动不会被误认为过程执行。
2. 作为调度研究者，我希望 process task 保留 SADCHER 的技能需求，以便过程 coalition 结果保持可比。
3. 作为调度研究者，我希望 transport task 显式保存 pickup 和 delivery location，以便运输时间和物料到达可观察。
4. 作为环境使用者，我希望每个 transport task 恰好解锁一个 process task，以便 MVP 中的物料前序具有确定语义。
5. 作为环境使用者，我希望每个 process task 至多有一个 material predecessor，以便第一版不隐藏物料聚合或库存策略。
6. 作为环境使用者，我希望 transport task 没有普通前序且物料在 `t=0` 可 pickup，以便 ready 不依赖尚未实现的库存子系统。
7. 作为仿真使用者，我希望 transport robot 与 process robot 互斥，以便 process-only robot 不会静默执行运输。
8. 作为过程 scheduler，我希望 process robot 可以动态组成 skill coalition，以便过程只有在全部技能被覆盖后才开始。
9. 作为运输 scheduler，我希望 MVP 中 transport assignment 是单机器人原子操作，以便单机无法搬运的重载物料明确报告 infeasible。
10. 作为运输 scheduler，我希望显式检查 capacity、transport capability、unloaded speed 和 loaded speed，以便只分配真正可完成任务的机器人。
11. 作为仿真使用者，我希望 transport robot 从 assignment 到 unloading 完成一直 busy 且不可抢占，以便占用和竞争具有确定语义。
12. 作为 metrics 使用者，我希望 task start 表示 loading 开始、task completion 表示 unloading 完成，以便 schedule 边界对应物理服务边界。
13. 作为 metrics 使用者，我希望保存结构化阶段时间和时长，以便审计空载移动、装载、负载运输、卸载、service 和 occupied time。
14. 作为 scheduler 作者，我希望所有策略使用统一的 typed hard-feasibility 结果，以便合法性判断一致。
15. 作为实例验证者，我希望静态无效实例在仿真前被拒绝，以便容量、类型、图结构和位置错误不会隐藏到运行时。
16. 作为仿真使用者，我希望区分 infeasible、deadlock 和 timeout，以便缺失 makespan 不被伪报为正常结果。
17. 作为过程负责人，我希望 downstream process 只有在 transport 完成后才 material-ready，以便运输完成成为明确的解锁事件。
18. 作为 baseline 研究者，我希望保留 process skill-coverage Greedy，并增加 Distance、ETA、Unlock 三个 transport Greedy，以便不依赖 learned policy 验证 MD 语义。
19. 作为可复现实验研究者，我希望 tie-breaking 和手工场景固定，以便 CPU 上能复现失败和 metrics 差异。
20. 作为 legacy SADCHER 使用者，我希望旧 `Q/R/T_e/T_t/task_locations/precedence_constraints` 实验继续运行，以便 MD rollout 不破坏已有 baseline。
21. 作为 legacy 数据使用者，我希望官方 `8t3r3s` optimal 数据集继续通过 process-only 路径加载，以便其 schedule 可以作为回归证据。
22. 作为数据接入维护者，我希望官方数据接入后可以直接修改 loader 和相关数据结构，以便不被偶然的文件布局永久束缚。
23. 作为 runtime 维护者，我希望领域模型和 MD 状态机在 loader 变化后保持稳定，以便 metrics 与测试含义不变。
24. 作为未来运输研究者，我希望预留 `SOLO`、`SMALL_COALITION`、`LARGE_COALITION` 三种模式，以便后续协同运输不必重定义 MVP。
25. 作为未来策略研究者，我希望 cooperative transport 的模式选择先 rule-based，以便在 solo 语义可信后再评估 learned mode selection。

## Implementation Decisions

- runtime 领域包含 `ProcessTask`、`TransportTask`、`ProcessRobot`、`TransportRobot`；`START`、`IDLE`、`EXIT` 仅作为可选 legacy sentinel，不进入真实任务和 material metrics。
- 实例内 task ID 连续且稳定。普通 process edge 与 material edge 分开保存；material relation 是唯一事实源，必须与两端派生字段一致。
- `TransportTask` 保存 `pickup_location`、`delivery_location`、downstream process ID、load requirement、loading duration、unloading duration 和 transport mode；MVP mode 为 `SOLO`。
- `ProcessTask` 保存普通前序、可选的 `material_predecessor` transport ID、位置、时长和原 SADCHER skill requirement vector。
- `TransportRobot` 保存 transport capability、capacity、当前位置、unloaded speed、loaded speed 和 runtime phase/occupancy；`ProcessRobot` 保存原能力向量和 coalition 状态；禁止 hybrid execution。
- Transport ready 条件为 `PENDING` 且没有普通前序。Process ready 条件为所有普通前序和 material predecessor（若有）均完成。
- 所有 assignment 经过 `is_assignment_feasible(robot, task) -> FeasibilityResult`。稳定 reason code 至少覆盖 task state、ready、robot type、capability、capacity、occupancy、invalid graph、invalid location 和 static infeasibility。
- Process assignment contribution 与 process coalition start feasibility 分离。机器人可以贡献尚未覆盖的技能，但 coalition 只有在全部需求覆盖后才开始。
- Transport assignment 为单机器人原子操作。MVP 禁止 transport coalition、capacity aggregation、同步等待、relay、multi-trip 和 preemption。
- 仿真采用离散 timestep。移动阶段按适用的 unloaded/loaded speed 使用 `ceil(distance / speed)`；装载和卸载时长由任务定义。
- Transport execution record 保存 assignment、pickup arrival、loading start/completion、delivery arrival、completion 及各阶段时长。Process record 保存 coalition、等待、开始和完成。
- Metrics 暴露 makespan、每个 process 的 material starvation、robot utilization、success、failure reason 和结构化 execution record。只有成功完成时 makespan 才有效；失败结果必须携带明确 reason。
- 所有 baseline 共享 hard-feasibility 和确定性 tie-breaking。Process baseline 保留原 skill-coverage Greedy；Transport baseline 为 Distance、ETA、Unlock。
- Legacy instance 解释为 process-only 且 material delivery disabled。Legacy `T_t` 保留给旧 solver；MD movement 初期按坐标和阶段速度计算。
- 官方数据集验收结论为 legacy 数据：231,634 对问题/解答，全部为 `8t3r3s`，索引得到 2,084,545 个决策样本。该数据没有 MD transport 字段，不进行转换。
- 本规格不包含 neural network、GAT、checkpoint、decoder 或 legacy feature dimension 的修改。

## Testing Decisions

- 测试只断言环境可观察行为：状态转移、ready、feasibility reason、metrics、execution record 和 legacy 兼容性，不断言私有 helper 或偶然的类布局。
- 统一测试 seam 采用 environment/simulator 边界：构造实例，经 scheduler-facing interface 提交 assignment，推进离散时间，再检查 terminal result 和 records。
- runtime 验证覆盖连续 ID、typed edge、位置一致性、无环性、数值范围和明确的 static infeasibility。
- feasibility 测试覆盖 robot type mismatch、skill 不足、单机器人 capacity 不足、robot 占用、task 未 ready 和合法 assignment，并检查稳定 reason code。
- 状态机测试覆盖全部 transport phase、阶段时长、不可抢占、位置更新、task start/completion 边界以及 unloading 后的 downstream unlock。
- metrics 测试覆盖 makespan、material starvation、utilization、success、failure reason 和结构化 process/transport record。
- 确定性场景覆盖 transport unlock、late material、capacity failure、type mismatch、location mismatch、process coalition、baseline 差异和 legacy regression。
- Legacy 回归复用已有 SADCHER simulator、Greedy 和 dataset loader seam，固定 schedule、makespan、可行 assignment，并验证 matching checkpoint 仍可在 CPU 加载。
- 官方数据集保持只读契约测试：文件配对、JSON shape、Q/R 语义、欧氏 `T_t`、前序无环、解答引用和 loader 索引数量。
- MVP 测试必须在 CPU 上确定性运行，不依赖 neural policy 或 GPU。

## Out of Scope

- Neural network、GAT、decoder、checkpoint、learned feature 修改。
- Transport coalition、capacity aggregation、同步等待、relay、multi-trip、preemption 或动态 transport reassignment。
- Inventory、material production state、material consumption accounting、batching 或多个 material predecessor。
- 同时执行 process 与 transport 的 hybrid robot。
- 非坐标 travel map、battery、charging、低层运动控制、碰撞规避或连续时间仿真。
- 在 solo transport 验证前进行 learned mode selection 或 learned cooperative transport。
- 将官方 legacy optimal 数据集转换为合成 MD 记录。

## Further Notes

官方数据集确认当前外部 benchmark 是 legacy process-only。未来数据集提供 MD 字段后，可以直接调整 loader，但所有 loader 都必须收敛到本规格的 runtime contract。本文档与英文版本、相关 ADR 和数据集验收记录一起作为 MVP 的本地实现依据；不需要 issue tracker 才能执行本地开发与测试。
