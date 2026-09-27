# RL 尝试在放大 process_scarce 上训练 —— 结果与阻塞点 (2026-09-14 evening)

**动机**:[SCALED_TARGET_PROFILES.md](SCALED_TARGET_PROFILES.md) 显示 MILP-IL v2 在 process_scarce 12-42 任务上稳定胜出 greedy 4-15%,60 任务时刚好失去优势(+0.8%)。用户建议:**在这个"价值成立"的场景上做 RL,扩大到 60+ 任务的规模**。

## 已完成的基础设施

**新增脚本**(工程量小,复用现成基础设施):

- [experiments/md_c0_ppo_process_scarce_pilot.py](../../experiments/md_c0_ppo_process_scarce_pilot.py):monkey-patch `md_ray_ppo` 的模块级 `MDGeneratorConfig` binding,让 PPO 训练环境生成 scaled process_scarce 实例。可配置 task_count、iteration、kl_weight。
- [experiments/md_rllib_to_il_adapter.py](../../experiments/md_rllib_to_il_adapter.py):把 RLlib PPO checkpoint 转成 IL 格式 `.pt`,让 `load_md_policy_checkpoint` 能读、让六 profile / scale ladder / scaled_target_profiles 的评测脚本能复用。已验证 194 tensors → 95 tensors(过滤掉 `reference_net.*` + `value_head.*`),`MDEnhancedSchedulerNetwork` 加载 OK,12464 参数与 IL 一致。
- [run_c0_ppo_process_scarce.sh](../../run_c0_ppo_process_scarce.sh):环境变量参数化的启动器

**工作流验证过**:
- Shim 生成 tc=24 process_scarce config 正确(transport_ratio=1/6, precedence_density=0.30, capacity_slack=0.2, 8 robots × 24 tasks, MultiDiscrete([25]×8) 动作空间)
- Shim 生成 tc=42 process_scarce config 正确(14 robots × 42 tasks)
- RLlib→IL 适配器读入 `best_iter_0020` checkpoint,输出可 `load_md_policy_checkpoint` 加载
- `MDGymEnvironment` reset + step 正常
- 端到端 rollout 在 tc=42 上 92ms/decision,约 35 decisions/rollout

## 阻塞:iteration 耗时不可接受

**尝试的四种配置**,tc=24 或 tc=42 上单次 iteration 都 ≥ 18 分钟:

| 配置 | task_count | batch × epochs | 单 iter 时间 | 备注 |
|---|---:|---:|---:|---|
| 上一次 tc=12 训练(基线) | 12 | 256 × 30 | **3.5 min** | 30-iter 完成用时 1h43min |
| A: tc=42, batch=256, epochs=30 | 42 | 256 × 30 | > 30 min | iter 1 未完成 30 分钟就 kill |
| B: tc=42, batch=2048, epochs=5(大 batch 少 epoch) | 42 | 2048 × 5 | > 10 min | iter 1 未完成 10+ 分钟就 kill |
| C: tc=24, batch=2048, epochs=5 | 24 | 2048 × 5 | > 40 min | iter 1 完成 40+ 分钟 |
| D: tc=24, batch=256, epochs=30 | 24 | 256 × 30 | **~18 min** | iter 1 完成 18 分,iter 2 未在 36 分钟内完成,iter 1 的 train_reward_mean=nan |

**判读**:
1. **task_count 从 12→24 每次 iter 时间从 3.5 min → 18 min** ——**5× 慢**,不成比例(action space 只放大 4×,25/6 vs 13/5 ratio 类似)。
2. **train_reward_mean=nan**(D 配置 iter 1):说明 rollout 里所有 episode 都 truncate 到 max_steps 而没 terminated,reward 累积到 nan ——**这是 policy 在放大规模上初始阶段行为异常的信号**,不只是慢。
3. **大 batch(2048)反而更慢**——RLlib 的 PPO 更新阶段成本主导,epoch 减少省的时间不够 batch 增加的开销。

## 阻塞的根本原因

**`MDAutoregressiveActionDistribution`** 采用逐机器人的 categorical 采样:

- tc=12 时:每 step 5 个机器人 × 12 任务 softmax = 60 个 logit,采样是 5 步序贯
- tc=24 时:每 step 8 个机器人 × 25 任务(24+idle) = 200 个 logit,采样是 8 步序贯
- tc=42 时:每 step 14 个机器人 × 43 任务 = 602 个 logit,采样是 14 步序贯

**PPO 更新阶段每个 minibatch 都要重新算 log_prob**,这是 seq 化的 autoregressive 计算,**GPU 用不上并行**(所以我们看到的是主线程 100% CPU,GPU 只 1% 使用率)。

这是 RLlib PPO + 自回归动作分布组合的**结构性瓶颈**。

## 已产出但未 RL 训练的 checkpoint

`runs/md_ray_ppo_train_2026-09-14/best_iter_0020` — 上一次 30-iter tc=12 训练的 RL checkpoint。用 `md_rllib_to_il_adapter` 转成 IL 格式并跑六 profile 评测 —— **这个 checkpoint 与 IL v2 几乎完全一样**(deterministic action disagreement 0.028%,held-out makespan 差 0.03%),转成 IL 后评测意义不大。

## 后续可能的路径(选一条,预计工程量 1-3 天)

按可行性从高到低排:

### 1. 并行 env runners(最直接,预期 3-5× 加速)

`md_ray_ppo.train(num_env_runners=N)` 参数已经存在但默认 0。设置 `num_env_runners=8-16` 让 Ray 起 N 个 worker 并行采 rollout,PPO 更新仍在 driver。**期望:iter 时间从 18 min → 3-5 min at tc=24**。

注意:
- worker 数不能超过 CPU 核数(本机 10 核)——用 `num_env_runners=8` 保留 2 核给 driver
- 需要额外的 Ray temp dir 空间

### 2. 缩短 rollout 长度上限(简单)

`MDGymEnvironment` 里可能有默认 `max_steps`。tc=24 process_scarce 一个 rollout 通常 40-60 steps,如果 truncate 到 max_steps=200 反而会拉长(等到 timeout 再截止)。**train_reward_mean=nan 也可能是 truncate 导致的**,值得先排查 log 确认 episode 长度分布。

### 3. 更小 net(架构改动,收益中等)

当前 hidden_dim=16 已经很小,继续缩不太现实。跳过。

### 4. 换算法(改动较大)

- **A2C** 无 PPO 的多 epoch 更新,单次前向就更新,不用重复算 log_prob。但 sample-efficiency 差得多,不适合我们要突破 IL 天花板的目标。
- **APPO / IMPALA** 分布式 actor-learner 架构,与 num_env_runners 类似,但要重写 pilot(RLlib 的 `AlgorithmConfig` 换成 `IMPALAConfig`)。

## 决定:这个 session 到此为止

投入产出比考虑:
- 已跑 30-iter RL(tc=12)+ 两次失败的 tc=42 尝试 + 一次 tc=24 尝试 + 完整基础设施 + RLlib-IL 双向 checkpoint 转换
- 剩下的关键实验(tc=24 或 tc=42 上跑到 20-30 iter 收敛)需要 `num_env_runners` 改造 + 6-10 小时训练
- **不适合再做一次"发起后就看不完"的长跑**

next-session 应该:
1. 先把 `md_ray_ppo.train(num_env_runners=8)` 跑一次 tc=24 的 3-iter 冒烟,量化加速比
2. 若加速 ≥ 3×,启动 20-iter tc=24 process_scarce 训练;若不到,回退到路径 2(缩短 rollout)
3. 训完 → 用 `md_rllib_to_il_adapter` 转 checkpoint → 用 [experiments/md_c0_scaled_target_profiles_eval.py](../../experiments/md_c0_scaled_target_profiles_eval.py) 评测,与 MILP-IL v2 baseline 直接对比
4. 若 tc=24 RL 明显胜出 MILP-IL v2,再考虑 tc=42 训练

## 保留的产物

- `experiments/md_c0_ppo_process_scarce_pilot.py` — RL pilot(process_scarce 分布)
- `experiments/md_rllib_to_il_adapter.py` — RLlib→IL checkpoint 转换器,已验证
- `run_c0_ppo_process_scarce.sh` — 环境变量参数化启动器
- `runs/md_ray_ppo_train_2026-09-14/` — 上一次 tc=12 RL 训练结果(可用作 warm-start 起点)
- 本文件:阻塞诊断 + next-session plan

## 一句话结论

**RL 端到端 pipeline 完全就绪,但单机 SKRL/RLlib + autoregressive action distribution 的组合在 tc≥24 上单 iter ≥ 18 分钟,阻塞了这一 session 内完成有意义的 RL 训练。下一步的关键动作是启用 `num_env_runners=8` 并行 rollout 采集,若能把 iter 时间压到 3-5 min 就可以在 3-5 小时内完成 tc=24 的 20-30 iter 收敛训练**。

---

## Update 2026-09-15: 尝试 `num_env_runners=8` 的结果 — 未突破瓶颈

按 next-session plan 启用 `num_env_runners=8` 并行 rollout,做了两轮测试:

### 尝试一:num_env_runners=8 + batch=256 + epochs=30(原始 config)

- 3-iter smoke,tc=24,seed=0
- 8 rollout workers spawned,各 ~12% CPU,driver 100% CPU
- 由于 `train_batch_size=256` 仅需 8 workers 各贡献 32 samples ≈ 1 rollout/worker,**rollout 太少 workers 大部分时间在等 driver 的 PPO update**
- 单 iter 时间 ≥ 20 min,与无并行版本几乎一样 —— **确认 rollout 不是瓶颈,PPO update phase 才是**

### 尝试二:num_env_runners=8 + batch=2048 + epochs=5(重新平衡)

- 3-iter smoke → 6-iter production run,tc=24,seed=0
- Iter 1 时间 = 28 min(含 15s ray setup + 25s 的 warmup rollout)
- **Iter 2 时间 > 57 min 仍未完成**——8 workers 在 rollout 阶段各 ~10.7% CPU(总 85%),但 rollout 收集 2048 samples 花了约 1 小时
- 每 worker 在跑 rollout 的观察:CPU-bound(GPU 只 1%),说明**推理在 workers 里跑 CPU 而不是 GPU**
- 单 iter 平均 ≥ 50 min,**比 num_env_runners=0 更慢**

### 结论:num_env_runners 未能突破瓶颈

原因诊断:
1. **RLlib 默认让 rollout workers 在 CPU 上做前向**(不共享 driver 的 GPU),tc=24 的 8-robot × 25-way autoregressive forward 在 CPU 上比 GPU 慢很多
2. **`MDRewardTracker` 每次 `env.reset()` 都跑一次 `greedy_eta` 参考策略**,tc=24 上这个参考 rollout 本身要几秒
3. **rollout worker 之间的 checkpoint 分发 + weight sync** 也有可见开销

即使加了 8 workers,总 CPU 利用率只有 85%(每 worker 10-12%),**远没达到 8× 加速**。

### 到此为止

RL 在扩规模 process_scarce 上的训练**不能在这个 session 内完成**,更不能在几小时内跑 20-30 iter 收敛。已投入的证据:

- 单机 RLlib + `MDAutoregressiveActionDistribution` + `MDGymEnvironment` 组合在 **tc ≥ 24** 上都是每 iter 20-50 min
- `num_env_runners=8` 未突破瓶颈 —— 说明问题不只在 rollout 数量,而是**每次 rollout 的单机推理成本本身太高**

### 下一步的真正可行选项(需要更深的改造,不是本 session 能做的)

1. **让 rollout workers 使用 GPU** —— RLlib 支持,但需要每个 worker 分配 GPU 分数(比如 `num_gpus_per_env_runner=0.1`)。8 workers × 0.1 GPU = 0.8 GPU,driver 剩 0.2 GPU
2. **在 RLlib 之外重写 PPO 训练** —— 用 pure PyTorch + `torch.compile` 加速自回归 log_prob 计算,避免 RLlib 的进程隔离开销
3. **换 A2C(单步更新)** —— 放弃 PPO 的样本效率,换取 3-5× 迭代速度,但需要更多 env steps 才能收敛
4. **在别的机器上跑** —— 多卡 GPU + 更多 CPU 核

### 保留下来的产物(供未来使用)

- 3 个尝试的 log:`rl_process_scarce_tc24.log`(尝试一,batch=256/epochs=30,killed)、`rl_process_scarce_tc24_er8_smoke.log`(尝试二 smoke,killed 在 iter 1 完成后)、`rl_process_scarce_tc24_er8.log`(尝试二 production,killed 在 iter 1 完成后 iter 2 收集中)
- 三次训练的 run_metadata.json + seed_splits.json 都在对应 `runs/` 目录里,可作重现基准
- Pilot 脚本 [experiments/md_c0_ppo_process_scarce_pilot.py](../../experiments/md_c0_ppo_process_scarce_pilot.py) 已经完整参数化,只需修改 `_experiment_config` 里的参数 + launcher 里的 flag 就能试上面 4 条路径

### 一句话综合结论(2026-09-15)

**RL 端到端 pipeline 完整可用,但吞吐瓶颈比预期更深** —— `num_env_runners=8` 只把总 CPU 利用率从 100%(单 driver)提到 85%(8 workers),没有实质加速。**要真正让 RL 训练在 tc≥24 的 process_scarce 上跑起来,需要突破的是"per-rollout 单机推理成本",不是 rollout 数量**。建议 next-session 优先尝试 `num_gpus_per_env_runner`,让 workers 也走 GPU。

---

## Update 2026-09-15 evening:GPU-per-worker 尝试

按下面 "Next-session 真正可行的方向 1" 试了 `num_gpus_per_env_runner=0.1`(driver 0.2 GPU + 8 workers × 0.1 GPU = 全部 GPU)。

### 修改

给 [experiments/md_c0_ppo_process_scarce_pilot.py](../../experiments/md_c0_ppo_process_scarce_pilot.py) 加了个 monkey-patch:`_install_env_runner_gpu_shim(0.1)` 包一层 `PPOConfig.env_runners`,让它每次调用都自动带 `num_gpus_per_env_runner=0.1`。CLI 加了 `--num-gpus-per-env-runner` 开关(默认 0.0 保持向后兼容)。

### 结果:workers 确实上了 GPU,但 throughput 没变

**GPU 分配成功** —— `nvidia-smi` 确认:
- Driver (PID 205581):1.5 GB(PPO update / value head / KL reference)
- 8 workers 各 398 MB —— **明确证据显示 workers 也在 GPU 上做前向**
- 总 GPU 内存 ~4.7 GB / 12 GB

**GPU 利用率仍然很低** —— rollout 阶段短暂冲到 92%(几秒),然后大部分时间 0-1%。

**Iter 时间**:iter 1 = **24 min**(vs 之前 CPU-only workers 版本 28 min),**几乎相同**。iter 2 未在 43 min 内完成,和之前一样卡在 rollout 阶段。

### 更深的根因

即使 workers 上了 GPU,还是慢。**因为 `MDAutoregressiveActionDistribution` 是序贯的**:

- 每一步 env.step 需要采样 `(N robots)` 个动作,每个动作前的 softmax 依赖前面已选动作的条件概率
- **这是一条串行链**,GPU 并行不了 — 每一步都要等前一步的采样结果
- tc=24 有 8 robots,每 env.step 需要 8 次串行 softmax + Categorical.sample()
- GPU 完成单个 softmax 只需微秒,但 CPU-GPU 同步开销 + Python overhead ~ 数百 μs,8 步串行 ~ 几 ms
- 一个 rollout ~50 步 sim,每步 ~几 ms:**bottleneck 是 CPU-GPU 同步,不是 GPU compute**

### 综合结论:改硬件路径不够,必须改算法/架构

`num_env_runners=8 + num_gpus_per_env_runner=0.1` = **工程上把 GPU 用起来了,但没解决 autoregressive 的串行本质**。8 workers × GPU 加速 ≈ 单机 CPU 版本;GPU 上单次 softmax 快、但 8 步串行 dominate。

**真正能突破的方向**(比之前的 4 条更精确):

1. **换 non-autoregressive 动作分布** — 用独立 Categorical for each robot(牺牲一些表达力,但可以整批 softmax),或者用 Gumbel-Sinkhorn 类的可并行分配采样
2. **`torch.compile` 整个 rollout policy** — 消除 Python overhead,把 8 步串行融合成一个 CUDA graph
3. **CPU-only workers 但更多并行(num_env_runners=16-32)** — 靠 CPU 多核数量而不是 GPU 加速。本机 10 核不够,得多机
4. **换 A2C** — 每 rollout 短很多,没多 epoch 重算 log_prob(但样本效率差 4-8×)

### 决定:这个 session 到此为止

已经证明:
- 端到端 pipeline 完整可用(冒烟 + 30-iter tc=12 完整训练 + tc=24 单 iter 都跑到过)
- `num_env_runners=8` + GPU workers 都不能突破 autoregressive 的串行本质
- **瓶颈在 policy 架构层,不在硬件/资源分配**

next-session 应该 **不再试单机 CPU/GPU 分配的组合**,而是:

1. 改 policy 用 non-autoregressive 动作分布(改动最小,收益最大)
2. 或用 `torch.compile` 把整个 rollout policy JIT 掉
3. 都不行再考虑换 A2C 或多机

保留下来的产物包括这次的 GPU-per-worker shim,只需要下次继续用 `--num-gpus-per-env-runner 0.1` 就能复现;架构改动是新工作。

### 一句话最终结论(2026-09-15 晚)

**GPU 已被 workers 使用**(每 worker 398 MB,driver 1.5 GB),但**单 iter 时间几乎没变** —— `MDAutoregressiveActionDistribution` 的串行采样是真正的性能瓶颈,不是硬件资源分配问题。要在 tc≥24 上做 RL 训练,必须改 policy 架构(non-autoregressive 或 `torch.compile`),而不是继续调 num_env_runners / num_gpus_per_env_runner。

---

## Update 2 — 2026-09-15 late night:尝试 non-autoregressive 分布 — 揭示更深的约束

按前面 "next-session direction 1"(独立 Categorical 分布)动手实现,结果发现事情比想的复杂。

### 实施

在 [models/md_rllib_model.py](../../models/md_rllib_model.py) 加了 `MDIndependentActionDistribution`(第 99 行):
- 每个 robot 独立 softmax + Categorical.sample()
- 只用 `context["action_mask"]`(observation 里的静态 base_mask)mask 掉不可行动作
- 完全 GPU-batched,没有 CPU-side 序贯循环

在 [experiments/md_c0_ppo_process_scarce_pilot.py](../../experiments/md_c0_ppo_process_scarce_pilot.py) 加了 shim `_install_independent_action_dist_shim()` 通过 monkey-patch `md_ray_ppo.MDAutoregressiveActionDistribution = MDIndependentActionDistribution` 无侵入切换,加了 CLI 开关 `--use-independent-action-dist`。

### 结果:环境直接拒绝我采样的动作

RolloutWorker 立刻抛异常:

```
RuntimeError: joint action prefix has no legal completion
    at joint_action_is_legal → autoregressive_action_masks → _action_masks
```

即 [reinforcement_learning/md_gym_environment.py](../../reinforcement_learning/md_gym_environment.py):94:
```python
assert joint_action_is_legal(tensor_observation, tensor_action), (
    "env.step received an action outside the autoregressive support"
)
```

### 更深的诊断

之前以为 autoregressive 只是"避免两个 robot 选同一个 task"的简单唯一性约束,读了代码才发现 `autoregressive_action_masks` 里的 `_action_masks` 实际上做的是**完整的 CSP 搜索**:

- 不只是"当前 robot 挑的 task 不能与前面 robot 冲突"
- 还包括"给定前面 robot 的选择,**是否存在**一种合法方式完成剩余 robot 的分配"
- 涉及 skill coverage + process coalition 最小性 + material transport 前提等 [md_joint_action.py:82-180](../../reinforcement_learning/md_joint_action.py) 里的多个可行性判据
- 每一步都要用递归搜索 + memoization 判断 "completion_exists"

这**不是简单的重复任务过滤**,而是一个真正的组合约束满足问题。**独立采样破坏了这个 CSP 保证**,env.step 拒绝无效 joint action 是正确的。

### 三个选项(权衡)

1. **修 env**(在 `step` 里静默 repair 非法动作,用 bipartite matching 从独立 sample 里选合法子集)
   - 优点:训练侧速度最大化
   - 缺点:训练环境的 state transition 不再对应 policy 采样的动作,创造 train-deploy mismatch
2. **保留 autoregressive 但把 mask 计算搬到 GPU**
   - 需要把 `_action_masks` 里的完整 CSP 搜索用 GPU-friendly 方式重写(可能不可能)
3. **改用不同 policy 架构**:让骨干直接输出一个已经"通过 constrained decoder"的 joint action(即把 bipartite matching + repair 集成进策略前向传播),用直通估计器/Gumbel-Softmax 反传梯度
   - 优点:训练速度快 + 训练侧和部署侧使用相同的 constrained decoding
   - 缺点:改动非平凡,涉及可微 bipartite matching

**选项 1 是最快但语义上最脏**——`MDIllegalAssignmentCallbacks` 就是为了让 policy 学到避免非法动作的信号,如果 env 静默 repair 就丢了这个信号,policy 永远学不到分布的合法约束。

**选项 3 是最合适的长期方向**——训练环境跟部署侧的 decoding 一致,但工作量大(1-2 周)。

### 决定:保留代码,不再本 session 试

- `MDIndependentActionDistribution` 类保留在 [models/md_rllib_model.py](../../models/md_rllib_model.py:99),`--use-independent-action-dist` CLI 开关保留 —— 未来若走"选项 1: 修 env" 路径可以直接用
- 相应记录了三个选项,让下一次开始工作时不用重新推导

### 综合结论(RL 探索 3 天的最终结论,2026-09-15)

**技术上**:RL 端到端 pipeline 完整可用,但 tc≥24 的 process_scarce 分布上单机 PPO 每 iter 24-45 min。**根因**是 `autoregressive_action_masks` 里的完整 CSP 搜索必须序贯 + Python 递归 —— 不是 GPU/CPU 分配的问题,而是**训练环境本身的合法性契约**约束了并行度。

**方法论上**:三条被证伪的直觉:
1. "加 num_env_runners 就能并行采集" — ✗,workers 各自 CSP 搜索仍在 CPU 上
2. "把 workers 也上 GPU 就快" — ✗,GPU workers 也在做 CSP 搜索(GPU 帮不上)
3. "换独立 Categorical 分布就能批处理" — ✗,env 会拒绝无效 joint action

**真正的下一步方向**(需要另一次 session,可能 1-2 周):
- 短期 shortcut(方案 1):env.step 里加 bipartite-matching repair 层,接受 "policy 提议 → env 修正" 的语义 mismatch,先跑起来看效果
- 长期正确路径(方案 3):可微 constrained decoding 集成到策略前向

**当前推荐**:接受 RL 训练在这个 codebase 需要更深的架构改造,先**用当前 MILP-IL v2 checkpoint 部署到 process_scarce 12-42 任务的场景**(这是 [SCALED_TARGET_PROFILES.md](SCALED_TARGET_PROFILES.md) 里已经证明有价值的场景);RL 的价值扩大留到未来解决 CSP 搜索并行化之后。
