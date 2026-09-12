# 官方数据集验收记录

## 数据集

- 压缩包：`dataset_optimal_8t3r3s.zip`
- SHA-256：`2aa489976b1150d4a6602ddd06c074c3f3abdb0a645002d69040b0aad3183029`
- 解压目录：`datasets/dataset_optimal_8t3r3s/`
- 压缩包完整性：`unzip -t` 通过
- 问题文件：`231,634`
- 解答文件：`231,634`

## 验收结果

全部问题和解答 JSON 均成功解析。排序后的文件名构成一一对应关系：

```text
problem_instance_<p>_<id>.json
optimal_schedule_<p>_<id>.json
```

每个实例都具有相同的 legacy 结构：

- `Q`：`3 x 3` 机器人-技能能力矩阵
- `R`：`10 x 3` 任务-技能需求矩阵（8 个真实任务，加 start 和 exit 行）
- `T_e`：长度 `10`（真实任务时长，加 start 和 exit）
- `T_t`：`10 x 10`
- `task_locations`：`10 x 2`
- 解答中的 `n_tasks`：`8`
- 解答中的 `n_robots`：`3`

文件名分组与前序约束数量完全一致：

| 分组 | 实例数 |
|---|---:|
| `1p` | 34,070 |
| `2p` | 31,996 |
| `3p` | 59,838 |
| `4p` | 37,479 |
| `5p` | 34,552 |
| `6p` | 33,699 |

所有前序边都是合法的、面向真实任务的 1-based ID，且图无环。所有解答中的机器人和任务引用均合法。`T_t` 与 `task_locations` 诱导出的欧氏距离矩阵完全一致（最大绝对误差：`0.0`）。

## Legacy 加载器兼容性

现有 `LazyLoadedSchedulingDataset` 可以直接加载解压目录，不需要 schema adapter：

- 决策样本数：`2,084,545`
- `n_robots=3`，`robot_dim=7`
- `n_tasks=8`，`task_dim=9`
- 第一个样本的张量成功返回

验收期间没有启动训练或 GPU 作业。

## 与 MD 的边界

该数据集只包含 legacy SADCHER 字段（`Q`、`R`、`T_e`、`T_t`、`task_locations`、`precedence_constraints`）。其中没有运输任务、pickup/delivery 位置、物料前驱、机器人类型标签或运输阶段状态。因此，该数据集作为官方 legacy 基准和回归夹具接入，但**不转换为 MD 数据**。

后续在数据语义最终确定后，可以直接调整 MD loader 及其外部数据结构；runtime 领域模型和 MD 运输状态机保持稳定，legacy loader 与旧实验保持不变。
