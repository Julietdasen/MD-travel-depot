# ADR-002：Transport 状态机与时间记录

- 状态：已接受
- 范围：MD-SADCHER++ MVP

## 决策

Transport 使用独立阶段：

```text
WAITING -> TO_PICKUP -> LOADING -> TO_DELIVERY -> UNLOADING -> COMPLETE
```

assignment 到 completion 期间 TransportRobot 忙碌且不可抢占。task start 是 loading 开始，task completion 是 unloading 完成。离散 timestep 的移动阶段使用 `ceil(distance / speed)`。

每次 execution record 保存：assignment、pickup arrival、loading start/completion、delivery arrival、completion，以及 empty travel、loading、loaded transport、unloading、service 和 occupied 时长。

## 理由

聚合 duration 无法表达 pickup 和 delivery 两个位置，也无法可靠计算运输利用率和 material arrival。显式阶段可以在不引入低层路径规划的情况下保持语义可验证。

## 非目标

MVP 不支持抢占、同步等待、relay、multi-trip 或 cooperative transport。
