# MILP-supervised IL 概念验证结果

本页记录 [IL_REDESIGN.md](IL_REDESIGN.md) 方向的**第一次概念验证**——用户要求"将 IL 训练从 MILP 出发",在 `balanced` profile 上跑通一版,量化效果。RL 路径按用户要求本次不实现,只在末尾记录设计。

## 实施路径(比 IL_REDESIGN Step 1-5 简化)

IL_REDESIGN.md 里 Step 1-3 设计的是**新** label schema(`oracle_action_labels` + `forbidden_immediate_assignments` + masking hook)+ dataloader shim + policy 侧 masking——工程量较大。

概念验证阶段直接**复用仓库里已有的 MDDecisionSample 管线**:

- `baselines/gurobi_md_oracle.solve_gurobi_md_oracle(domain)` 求端到端最优 → `GurobiOracleResult.action_order`
- `data_generation/md_expert_dataset.generate_md_expert_record_from_oracle(instance, oracle)` 已经能把 `OracleAction` 序列在仿真器里逐步回放,每决策点抓一个 `MDDecisionSample`(含 pre-assign robot/task 快照 + `hard_feasibility_mask` + `expert_assignment` 二值矩阵)
- `imitation_learning/md_train.train_md_policy(dataset_root, ..., config)` 已经能用 BCE loss 在这些 sample 上训 `MDEnhancedSchedulerNetwork`(就是 C0 的模型类)

这条路径**不需要新 label schema、不需要 dataloader shim、不需要 masking hook**。缺点是没有 `forbidden_immediate_assignments` 的显式监督(每步只看 expert 的正例,反例来自 BCE 里的其它 feasible 未选中项)——但仍能验证"换 MILP 监督信号是否有收益"这个核心问题。

如果这版就已经明显缩小差距,说明 IL_REDESIGN.md 完整版会更好;如果这版没收益,再上完整 schema 也危险。

## 实施产物

- 训练编排脚本:[experiments/md_c0_milp_supervised_pilot.py](../../experiments/md_c0_milp_supervised_pilot.py) — 参数注入 + 调用现成管线
- 评测脚本:[experiments/md_c0_milp_supervised_profile_eval.py](../../experiments/md_c0_milp_supervised_profile_eval.py) — 六 profile × 4 seed × (新 C0 + 3 贪心),另拼入 MILP 最优与老合成 C0 的历史结果
- 后台启动:[run_c0_milp_supervised.sh](../../run_c0_milp_supervised.sh) — conda + PYTHONPATH + gurobi license,follow `run_c0_*.sh` 既有模式
- 训练输出:[reports/md_c0_milp_supervised_pilot_2026-09-13/](../../reports/md_c0_milp_supervised_pilot_2026-09-13/)
- 评测输出:[reports/md_c0_milp_supervised_profile_eval_2026-09-13/](../../reports/md_c0_milp_supervised_profile_eval_2026-09-13/)

## 训练配置

- **数据算例**:balanced profile 参数(task_count=12,transport_ratio=0.25,precedence_density=0.25,critical_path_length=3,capacity_slack=0.2,speed_ratio=0.8,3 process robot + 2 transport robot,3 skill)
- **种子**:`80000-80149`(远离 HANDOFF 保留边界 `75000-76149`)
- **规模**:150 个 instance(80/10/10 hash split → 113 train / 22 val / 15 test = 1008 train sample / 192 val sample)
- **MILP 求解**:60s 时限,4 线程 → **150/150 全部秒级最优**,0 skip
- **训练**:200 epoch,Adam lr=1e-3,batch=32,C0 架构(hidden_dim=16,pair-aware attention,no cross-attention,residual_bound=0.75,legacy_transport_bound=0.15,physics_scale=0.5)—— 12464 参数,与老 C0 (12459) 相近(差 5 个参数来自新版本的 `use_signed_opportunity_prior` 分支)
- **训练时长**:约 4 分钟(GPU RTX 3080 Ti)
- **train_loss**: 0.635 → 0.097(下降 6.5×,学到了)
- **val_loss**: 0.642 → 1.523,**best_val 0.491** — 后期过拟合,但保存的 checkpoint 走的是 best_validation 选择,评测用的是那份
- **solver_calls_during_training**: 0 ✓(纯离线 IL,训练期不调求解器)

## 结果:六 profile 三方对照

| profile | MILP 最优 | 老 C0(合成) | 老 C0 gap | **新 C0(MILP-IL)** | **新 C0 gap** | 最优贪心 | 贪心 gap | **新 C0 比老 C0 好** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 202.2 | 264.8 | +30.9% | **245.2** | **+21.3%** | 253.8 | +25.5% | **-9.6 pp** ✓ |
| process_scarce | 154.8 | 200.6 | +29.6% | 198.2 | +28.1% | 212.0 | +37.0% | -1.5 pp ≈ |
| transport_bottleneck | 389.0 | 449.8 | +15.6% | **418.8** | **+7.6%** | 437.8 | +12.5% | **-8.0 pp** ✓ |
| dependency_deep | 249.8 | 289.5 | +15.9% | **272.0** | **+8.9%** | 276.0 | +10.5% | **-7.0 pp** ✓ |
| mixed_hard | 262.0 | 356.2 | +36.0% | **328.8** | **+25.5%** | 305.0 | +16.4% | **-10.5 pp** ✓ |
| scale_medium | 243.5 | 298.2 | +22.5% | 305.0 | +25.3% | 296.2 | +21.7% | +2.8 pp ✗ |

**平均 gap 变化**:老 C0 平均 +25.1% → 新 C0 平均 +19.5%,**下降 5.6 个百分点**。

## 关键判读

1. **训练只用了 `balanced` profile,但改善迁移到了其他 5 个 profile 里的 3 个**(mixed_hard -10.5pp、balanced -9.6pp、transport_bottleneck -8.0pp、dependency_deep -7.0pp)。这不是过拟合,是 MILP 监督信号本身在教一些能跨 profile 迁移的调度决策模式。
2. **`scale_medium` 变差 2.8pp** — 唯一的负例。scale_medium 是六 profile 里规模最大的一档(task_count=24 vs balanced 的 12),训练分布外,泛化没兜住。这跟第 4.3、11 节看到的"C0 在超训练规模上 makespan 退化"的模式一致——**要覆盖 scale_medium 就必须在训练数据里包含 24 任务的算例**,不能只用 balanced。
3. **`process_scarce` 只改善 1.5pp** — 结构差异最大(2 process robot vs balanced 的 3、precedence_density 0.30 vs 0.25),泛化受限,但没变差。加入 `process_scarce` 训练数据后应该能进一步压。
4. **IL_REDESIGN.md 的成功标准是"6 profile 平均相对 MILP 最优的差距从当前 25% 缩到 ≤15%"** — 本次概念验证做到了 19.5%,**还没达标但方向对了**;继续做下去应该能达标(下面"接下来"说了怎么做)。
5. **对比"最优贪心 gap"**:新 C0 在 balanced/transport_bottleneck/dependency_deep/process_scarce 上已经**打平或胜过最优贪心**(-4.2pp / -4.9pp / -1.6pp / -8.9pp);mixed_hard 和 scale_medium 上仍输,但 mixed_hard 差距从老 C0 的 +19.6pp 缩到新 C0 的 +9.1pp。这从"C0 与最优贪心大致同量级"过渡到"C0 在多数 profile 上已胜过最优贪心",跟 [README.md](README.md) 里 2 号 TL;DR 相比是明显进步。

## 已跑但未上的东西

按 IL_REDESIGN.md Step 4 的"对照 A/B/C"完整协议,还差的是:

- **对照 A**(现状 baseline)—— 老 C0 已跑,数据在 `reports/md_c0_profile_eval_pilot_2026-09-13/`,复用
- **对照 B**(新 oracle-action label + shim only,不启用 forbidden mask)—— 本次实际跑的就是这个位置的**简化版**(用 `MDDecisionSample` 而非新 schema)。数据在 `reports/md_c0_milp_supervised_profile_eval_2026-09-13/`
- **对照 C**(新 oracle-action label + shim + forbidden mask)—— **尚未实现**,需要 IL_REDESIGN.md Step 1 完整的 `oracle_action_labels` 生成器 + Step 3 的 masking hook

也就是说:本次概念验证证实了 B 相对 A 有明确收益(-5.6pp 平均差距),下一步该做的是 IL_REDESIGN.md 的完整版看 C 是否比 B 又好一档。

## 接下来的自然延伸(按优先级)

1. **数据加规模档**:把 `process_scarce`、`scale_medium` 加进训练集(每 profile 150 instance,共 900),看两个未改善的 profile 是否也能拉下来
2. **训练更多 epoch + 更强 early stopping**:val_loss 过拟合是明确信号,当前用的 best_val 保存已经缓解,但可再加 patience-based early stop
3. **IL_REDESIGN.md 完整版(对照 C)**:实现 `oracle_action_labels` schema + `forbidden_immediate_assignments` masking,看是否比 MDDecisionSample 简化版又更好
4. **推到 42+ 任务规模**:MILP 会超时(参考 [README.md](README.md) 规模表),需要 RL 或 self-imitation 路径(见下节)

## RL 路径设计(本次不实现,按用户要求只记录)

**触发条件**:MILP 在目标规模上大量超时(参考本次会话第 12 节:42 任务 1/3 超时,90+ 100% 超时)。届时 IL 拿不到足够的 optimal 监督信号,继续走 IL 只会退化到 MILP 次优解。

**Warm-start**:本次 MILP-IL 训练得到的 checkpoint([reports/md_c0_milp_supervised_pilot_2026-09-13/training/best_checkpoint.pt](../../reports/md_c0_milp_supervised_pilot_2026-09-13/training/best_checkpoint.pt))作为 RL 起点,而不是从随机初始化开始。这样能:
- 避免 RL 早期探索灾难(纯 random policy 在 MRTA 上几乎不会产生合法调度)
- 把已经从 IL 学到的"跨 profile 迁移知识"带过来

**环境**:现有 `simulation_environment/md_discrete_simulator.MDDiscreteSimulator` + `schedulers/online_md_scheduler.OnlineMDScheduler` 已经是完整可交互环境;`ExplicitMIPFallback` 可作为 exploration safety net(超出神经网络置信度时走精确解,避免采集大量非法/低质轨迹)

**Reward**:
- 主奖励:`-latest_exit_return`(即负 makespan,跟 MILP 的 `objective` 同定义)
- 辅助 shaping:每次决策的"当前 pending task 减少量"作为 dense signal,缓解稀疏奖励;上限设为总 pending 以避免游戏化
- terminal bonus:若 rollout 成功完成(all task done)给 `+K`,失败给 `-K`

**算法选择**:
- **PPO**(推荐首选):离散动作(每步选 robot-task 对),actor-critic 结构直接兼容当前 policy 的 `(N,M)` score 矩阵——把 score 通过 softmax 变成 action distribution;critic 加一个 `MDStateValueHead`(仓库 `models/md_training_enhancements.py` 已有)。相对稳定,现成 warm-start 兼容。
- SAC / IMPALA 备选:如果 PPO 效率不够高,SAC 的 off-policy 特性能更好利用经验;IMPALA 在需要分布式加速时可选。

**训练目标规模**:先在 42-60 任务(MILP 部分超时区间)训一个 warmup RL policy,再往 90-150 推。**不建议直接在 150 上从头 RL** ——探索成本太高。

**Curriculum**:从 12→24→42→60→90 逐步放大,每档训到 makespan 收敛/持平前一档再放大。用本次 MILP-IL C0 作为 12/24 档的初始 policy 起步。

**评测**:
- 与 MILP(能解出来的规模)对比 makespan 差距,目标 ≤10%
- 与 IL C0 对比 makespan 差距,目标 20% 以上改善
- 与 greedy 对比,目标全场次胜出

**基础设施**:仓库里有 `reinforcement_learning/` 目录(本次会话未深入检查),实施前需先读一遍,决定复用现有 RL infra 还是新建。**不重复造轮子**,与 IL 侧一样先看现成能力再决定动手。

## 归档建议

本次会话没有归档需要——脚本、log、报告都在:
- 脚本:`experiments/md_c0_milp_supervised_*.py`(2 个)+ `run_c0_milp_supervised.sh`
- log:`c0_milp_supervised.log`
- 报告:`reports/md_c0_milp_supervised_pilot_2026-09-13/` + `reports/md_c0_milp_supervised_profile_eval_2026-09-13/`
- 本页作为结论页

按 IL_REDESIGN.md Step 5 建议的节奏,若下一轮迭代产生新一日的重构,可把本页与 `md_c0_milp_supervised_*` 目录一同归档到 `archive/2026-09-13-milp-il-pilot/`。
