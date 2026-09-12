# ADR-001：任务与机器人类型

- 状态：已接受
- 范围：MD-SADCHER++ MVP

## 决策

真实任务统一为 `PROCESS` 和 `TRANSPORT` 两类；真实机器人统一为互斥的 `PROCESS_ROBOT` 和 `TRANSPORT_ROBOT` 两类。

Process task 沿用原 SADCHER skill requirements 和动态 coalition。Transport task 不使用 process skills，由 TransportRobot 执行，并检查 transport capability 与 capacity。MVP 不允许 hybrid robot、transport coalition 或 capacity aggregation。

每个 transport 恰好解锁一个 process；MVP 严格一对一。transport 不允许普通前序，所有物料在 `t=0` 可 pickup。delivery location 必须等于 downstream process location。

## 理由

将“操作能力”和“运输能力”分开，避免把 transport 错误建模成 process skill；一对一关系和无 transport 前序能在不引入 inventory/material nodes 的情况下表达最小 material-delivery 语义。

## 后续

Transport 后续支持固定 `SOLO`、`SMALL_COALITION`、`LARGE_COALITION` 模式，只由 TransportRobot 组成。先 rule-based，再考虑 learned mode selection。
