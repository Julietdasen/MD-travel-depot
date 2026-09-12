# 已确认设计决议

本文件是对现有 MD-SADCHER 指南的增量补充。它不替代 `AGENTS_zh.md`、`IMPLEMENT_GUIDE_zh.md`、`POLICY_GUIDE_zh.md`、`IMPLEMENTATION_ROADMAP_zh.md` 和 `BACKGROUND_zh.md`；实现时以本文件与 ADR 的最新决议共同为准。

## 已冻结

- 真实任务只有 `PROCESS` 和 `TRANSPORT`。
- 真实机器人为互斥的 `PROCESS_ROBOT` 和 `TRANSPORT_ROBOT`。
- ProcessRobot 使用原 SADCHER 技能组合；TransportRobot 使用 transport capability、capacity、unloaded speed 和 loaded speed。
- MVP 只实现 solo transport，禁止 transport coalition、capacity aggregation、同步等待和抢占。
- 每个 transport 恰好解锁一个 process；transport 没有普通前序；delivery location 必须等于 downstream process location。
- 采用显式 transport 状态机和结构化 execution records。
- 集中式 hard feasibility 和稳定 reason code 是所有 scheduler 的共同入口。
- 旧模式保持可运行并冻结 regression fixtures；内部结构可以调整。
- 官方数据集接入后允许直接修改 loader 和相关数据结构，但 runtime 领域模型与 MD 状态机语义不变。
- 中英文指南持续同步；外部数据 schema 在官方数据集检查前保持 provisional。

## 待后续

- `SOLO`、`SMALL_COALITION`、`LARGE_COALITION` cooperative transport。
- rule-based 后的 learned mode selection。
- typed edge encoder、downstream-aware scoring 和新的神经特征。
- `T_t` 是否能被官方数据集的 travel matrix 复用。
