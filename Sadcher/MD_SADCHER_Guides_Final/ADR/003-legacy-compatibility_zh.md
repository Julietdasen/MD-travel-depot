# ADR-003：Legacy 兼容与官方数据接入

- 状态：已接受
- 范围：MD-SADCHER++ MVP

## 决策

旧 `Q/R/T_e/T_t/task_locations/precedence_constraints` 实例继续作为 process-only、`material_delivery.enabled=false` 的 legacy instance。内部类和数据加载逻辑可以重构，但 legacy 模式必须保持旧实验可运行，并冻结固定实例的 schedule、makespan 和可行 assignment 回归。

官方数据集接入后，允许直接修改数据加载和相关数据结构，不要求永久保留独立 adapter 文件。无论外部格式如何变化，都必须转换到稳定 runtime domain model，并保持 MD 状态机、ready、feasibility 和 metrics 语义不变。

Legacy 中的 `T_t` 保留给旧 MILP、Greedy、可视化和数据集消费者。MD runtime 暂按坐标与阶段速度计算移动时间；官方数据集是否提供可复用 travel matrix，待数据接入时决定。

## 理由

既保护原始 baseline 的可比性，也避免在官方数据集尚未检查前冻结错误的外部 schema。稳定的是领域语义，不是输入文件的偶然布局。
