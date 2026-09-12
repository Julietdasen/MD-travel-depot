# MD-SADCHER++ 领域模型

状态：设计已确认，等待官方数据集接入验证。

## 范围

首个里程碑只实现环境级 material-delivery rollout：数据模型、仿真状态机、集中式 hard feasibility、基础 metrics、确定性手工场景和 Greedy baselines。首个里程碑不修改神经网络、GAT、checkpoint 或 decoder。

## 领域实体

真实任务只有两种大类型：

- `PROCESS`：使用原 SADCHER 的技能要求和动态技能联盟完成操作。
- `TRANSPORT`：由 TransportRobot 将物料从 pickup 运到 downstream process 的位置。

真实机器人也只有两种互斥类型：

- `PROCESS_ROBOT`：拥有 process capabilities，可组成 process coalition，不执行 transport。
- `TRANSPORT_ROBOT`：拥有 capacity、unloaded_speed、loaded_speed 和 transport capability，不参与 process skill coalition。

仿真内部仍可保留原 SADCHER 的 `START`、`IDLE`、`EXIT` sentinel；它们不是第三种真实任务类型，不进入领域任务统计、material graph 或论文指标。

## 任务关系

每个 TransportTask 恰好解锁一个 ProcessTask；第一版严格一对一：一个 ProcessTask 最多拥有一个 material predecessor。TransportTask 不允许普通前序，物料在 `t=0` 已可 pickup。ProcessTask 可以有零个或一个 material predecessor，并可有多个 normal process predecessors。

唯一关系事实源为：

```yaml
material_edges:
  - [transport_task_id, process_task_id]
```

runtime 加载后派生 `transport.downstream_process_id` 和 `process.material_predecessor_id`，并验证两者一致。Transport 的 `delivery_location` 必须等于 downstream ProcessTask 的 `location`。

## Transport 状态机

```text
WAITING -> TO_PICKUP -> LOADING -> TO_DELIVERY -> UNLOADING -> COMPLETE
```

Task start 是 `LOADING` 开始，task completion 是 `UNLOADING` 完成。TransportRobot 从 assignment 到 completion 一直 busy 且不可抢占。阶段记录必须结构化保存：空载移动、装载、负载运输、卸载、service time 和 robot occupied time。

离散 timestep 下，距离阶段使用：

```text
ceil(distance / speed)
```

理论 transport duration 为：

```text
empty travel + loading + loaded travel + unloading
```

## Ready 与 feasibility

ProcessTask ready：

```text
PENDING
and 所有 normal predecessors DONE
and material predecessor（若有）DONE
```

TransportTask ready：

```text
PENDING
and 无普通前序
```

集中接口 `is_assignment_feasible(robot, task)` 返回 `FeasibilityResult`。至少检查：类型匹配、task ready、task pending、未被占用、transport capacity、transport capability，以及 process 的 skill contribution。Process 的 assignment feasibility 与 coalition 的 start feasibility 分离：单个 ProcessRobot 可以贡献一个尚未覆盖的技能，完整 coalition 覆盖全部 requirement 后才开始。

MVP 禁止 transport coalition、capacity aggregation、同步等待和抢占。重载物料无法由单个 TransportRobot 搬运时，实例为 infeasible，不使用隐藏 fallback。

## Metrics

- `makespan`：所有任务完成且所有机器人到达 exit 的时间。
- `material_starvation(process)`：`max(0, material_ready_time - normal_ready_time)`；无 material predecessor 时为 0。
- `robot_utilization`：busy time / makespan。
- `success`：所有任务完成且无非法 assignment 或 deadlock。
- `failure_reason`：`static_infeasible`、`robot_type_mismatch`、`no_capable_transport_robot`、`invalid_graph`、`deadlock`、`timeout` 等稳定 reason code。

Process 和 Transport execution record 均结构化保存。Transport 记录 `assigned_at`、`arrived_pickup_at`、`loading_started_at`、`arrived_delivery_at`、`completed_at` 及各阶段时长；Process 记录 coalition、等待、开始和完成时间。

## 数据接入策略

旧 `Q/R/T_e/T_t/task_locations` schema 自动解释为 process-only、material delivery disabled 的 legacy instance。官方数据集到达后，可以直接修改数据加载和相关数据结构；但必须转换到稳定的 runtime domain model，不能让外部 schema 改变上述状态机和 feasibility 语义。旧实验至少保留固定实例的 schedule、makespan 和可行 assignment 回归。

`T_t` 在 legacy 中保留，表示任务位置之间的静态欧氏距离矩阵；MD runtime 暂按坐标和阶段速度计算 travel time。非欧氏地图或官方数据集自带 travel matrix 属于后续数据接入决策。

## 后续 cooperative transport

后续固定三种模式：`SOLO`、`SMALL_COALITION`、`LARGE_COALITION`。两种 coalition 只由 TransportRobot 组成，先采用 rule-based mode selection；人数更多可通过更高 loaded speed 提高运输效率，loading/unloading 第一版保持任务固定值。learned mode selection、relay 和 multi-trip 不属于 MVP。
