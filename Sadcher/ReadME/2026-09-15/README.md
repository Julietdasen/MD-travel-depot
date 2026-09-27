# 2026-09-15 会话摘要

**主题**:尝试把 RL 训练扩到 process_scarce tc≥24 的场景,让方法的价值边界从 tc≤42 推到更大规模。

## 一句话结论

**RL 走不通,不是慢的问题,是环境本身的合法性契约要求太严**。三个直觉都被证伪,深层根因是 `autoregressive_action_masks` 里嵌入了完整 CSP 搜索,单机并行绕不过。

## 三次尝试与结果

### 尝试 1: `num_env_runners=8` 并行 rollout
- 8 workers spawn,各 12% CPU
- 单 iter 时间 ≥ 20 min,几乎无加速
- **结论**:workers 各自在做 CSP 搜索,并行价值有限

### 尝试 2: `num_gpus_per_env_runner=0.1` 让 workers 上 GPU
- nvidia-smi 确认:driver 1.5 GB + 8 workers × 398 MB = 4.7 GB 使用中
- 但 iter 时间几乎没变(CPU workers 28 min vs GPU workers 24 min)
- **结论**:GPU 帮不上 CSP 搜索,memory 占用但 utilization 只有 1%

### 尝试 3: `MDIndependentActionDistribution` 换 non-autoregressive 分布
- 实现在 [../../models/md_rllib_model.py:99](../../models/md_rllib_model.py) —— 独立 Categorical for each robot,GPU-batched
- CLI 开关 `--use-independent-action-dist` 通过 monkey-patch shim 无侵入切换
- **RolloutWorker 立即抛异常**:`joint action prefix has no legal completion`
- **结论**:env 拒绝无效 joint action 是正确行为,独立采样破坏了 CSP 契约

## 深层根因(读代码后才明白)

`autoregressive_action_masks` 里的 `_action_masks` **不是简单唯一性过滤**,是**完整 CSP 搜索**:

- 判断:给定前面 robot 的选择,后面 robot 是否**存在**合法完成方式
- 涉及:skill coverage + process coalition 最小性 + material transport 前提
- 实现:递归 + memoization,必须序贯 + Python 递归

这不是资源分配问题,而是**训练环境合法性契约本身**的约束。

## 真正的下一步(1-2 周新工作,不是本 session 的范围)

- **选项 A**(1 周):env.step 里加 bipartite-matching repair 层,接受"policy 提议 → env 修正"的语义 mismatch
- **选项 B**(2 周):可微 constrained decoding 集成到策略前向,训练侧和部署侧一致
- **选项 C**(现在能做):不做 RL,先用 MILP-IL v2 checkpoint 部署到已证明有价值的 process_scarce 12-42 任务场景([../2026-09-14/SCALED_TARGET_PROFILES.md](../2026-09-14/SCALED_TARGET_PROFILES.md) 里 -13.7% 相对 greedy 优势)

**推荐选项 C**:RL 扩规模需要架构改造,不是几个小时能完成。MILP-IL v2 已经有明确、可复现、可交付的价值。

## 保留下来的产物

- [../../models/md_rllib_model.py::MDIndependentActionDistribution](../../models/md_rllib_model.py) — 未来若走选项 A 直接可用
- [../../experiments/md_c0_ppo_process_scarce_pilot.py](../../experiments/md_c0_ppo_process_scarce_pilot.py) — 含 `--num-gpus-per-env-runner` + `--use-independent-action-dist` 双 shim
- [../../experiments/md_rllib_to_il_adapter.py](../../experiments/md_rllib_to_il_adapter.py) — RLlib checkpoint → IL 格式转换器,已验证
- [../../run_c0_ppo_process_scarce.sh](../../run_c0_ppo_process_scarce.sh) — 环境变量参数化启动器
- 完整诊断:[RL_SCALED_PROCESS_SCARCE.md](RL_SCALED_PROCESS_SCARCE.md)

## 方法论上的三个教训

1. **"加 num_env_runners 就能并行采集"** ✗ — workers 各自的 CSP 搜索还在 CPU 上
2. **"把 workers 也上 GPU 就快"** ✗ — GPU 帮不上 CSP 搜索
3. **"换独立 Categorical 分布就能批处理"** ✗ — env 会拒绝无效 joint action

**真正的经验**:遇到吞吐瓶颈时,先读慢的部分**在做什么**,而不是先假设"再加并行就能突破"。
