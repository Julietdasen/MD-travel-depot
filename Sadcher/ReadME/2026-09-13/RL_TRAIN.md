# RL 30 迭代训练结果 (2026-09-14)

**背景**:[RL_SMOKE.md](RL_SMOKE.md) 冒烟通过后启动 30 迭代正式训练,warm-start from MILP-IL v2([reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt](../../reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt))。目标:验证 PPO 能否在训练分布内(balanced profile,12 任务)让 policy 收益超过 IL 起点。

## 命令

```bash
ITERATIONS=30 bash run_rl_train.sh
```

## 时间线

- Start: 16:57 (2026-09-14)
- 30 iter DONE: 18:40+ (确切时间从 `training_history.json`)
- **总耗时**:约 1 小时 43 分钟 = **~3.5 分钟/iter**(比冒烟单次估算的 10 min/iter 快很多——冒烟里 10 min 包含了 ray init 的固定成本)
- 无 crash,无 illegal action,无 policy 塌陷

## 训练动态(每 10 iter 一次日志)

| iter | train_reward_mean | ppo_kl | il_reference_kl | value_expl_var | entropy | actor drift(L2) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -3.44 | 2.2e-6 | 2.4e-6 | 0.063 | 0.026 | 2.1e-3 |
| 10 | -3.87 | 1.8e-5 | 2.0e-5 | -0.243 | 0.014 | 7.1e-3 |
| 20 | -3.74 | 3.7e-6 | 1.0e-5 | 0.295 | 0.014 | 8.6e-3(估) |
| 30 | -3.67 | 1.2e-6 | 1.1e-6 | 0.323 | 0.010 | ~9e-3(估) |

- **`il_reference_kl` 全程 < 2e-5**——KL guardrail (`il_kl_weight=0.02`) 生效,RL 几乎粘在 IL 附近
- **参数漂移微弱**——`normalized_l2` 只从 2.1e-3 涨到 ~9e-3,encoder frozen 生效,只有 head 在动
- **value_explained_variance 从 0.06 → 0.32**——value head 学到了信号,是训练在实际发生的最直接证据
- **entropy 从 0.026 → 0.010**——policy 越来越确定性,但仍非退化 (未到 0)

## 验证集(seed 100-115,与冒烟同)

| iter | IL makespan | RL makespan | 差 | W/T/L | go_no_go |
|---:|---:|---:|---:|---:|---:|
| 1 | 225.94 | 226.69 | +0.75(RL 差)| 0/15/1 | False |
| 10 | 225.94 | 226.69 | +0.75 | 0/15/1 | False |
| **20** | **225.94** | **225.38** | **-0.56(RL 胜)** | **1/15/0** | False |
| **30** | **225.94** | **225.38** | **-0.56** | **1/15/0** | False |

- **RL 从 iter 20 起 marginally 胜 IL**——16 个 seed 里 1 个胜、15 个持平、0 个输
- **胜的 seed 是 seed 115**:IL 254 vs RL 249(-5,约 -2%),其他 15 个 seed 输出完全一致
- **`go_no_go: False`**:内置阈值判定"改善不够部署"——makespan_gain_below_threshold

## 独立测试集(seed 200-215,held-out 从未训练/验证过的)

- **IL**: success 16/16, mean makespan **233.44**, 0 illegal, p95 latency 40.0 ms, mean material_starvation 180.88
- **RL**: success 16/16, mean makespan **233.38**, 0 illegal, p95 latency 40.1 ms, mean material_starvation 180.81

**RL 在 held-out 上比 IL 好 0.06 makespan(0.03%)**——统计意义上几乎为零。

## 判读:方向对、增益微弱

**已确认(正面)**:
1. **端到端 pipeline 100% 可用** —— `md_ray_ppo.py` + `MDRLlibModel` + `MDGymEnvironment` + `MDFineTuneConfig` 全部现成基础设施跑通
2. **IL warm-start 可靠** —— RL 从 IL checkpoint 起步不塌陷,`il_reference_kl` 全程受控
3. **PPO 训练本身工作** —— reward 上升、value head 学到、entropy 下降、参数微漂移
4. **RL 已经能 marginally 胜 IL** —— 训练集 seed 115 从 254 → 249(-2%),但这是在 IL 已经近饱和的分布内

**未达成(负面)**:
1. **增益太小** —— 训练集 -0.25%,held-out 只有 -0.03%。远小于 [ANALYSIS.md](ANALYSIS.md) 里定的"胜出 greedy_unlock ≥ 5%"底线
2. **`go_no_go: False`** —— 官方阈值判定"不够部署"
3. **未在 stress-test 规模阶梯上评测** —— 只在 12-任务 balanced profile 训练/验证/held-out,还没跑规模阶梯确认 RL 是否在**大规模上**能超越 greedy

## 为什么增益小(诚实分析)

**主要原因**:
1. **训练分布内 IL 已近饱和** —— 12-任务 balanced profile 上,MILP-IL v2 相对最优 MILP 差 15-36%,但相对 greedy 已经胜出;RL 从这个"接近局部最优"起步,PPO 300s 内可能的边际收益很小
2. **KL guardrail 太严** —— `il_kl_weight=0.02` 让 `il_reference_kl` 在整个训练中 < 2e-5;等于把 RL 锁在 IL 附近,不允许探索
3. **frozen_encoders=True** —— 只调 output head + value,表示层不动,策略变化的空间有限
4. **迭代太少** —— 30 iter 只训了 30 × 256 = 7680 环境步。真正 PPO 收敛通常需要 100k-1M 步
5. **训练分布单一** —— 只有 balanced profile,没有 curriculum;RL 学不到大规模上跟 greedy 的差距

**要看到实质性收益,必须做的调整**:

| 调整 | 现值 | 建议 | 期望效果 |
|---|---:|---:|---|
| 迭代数 | 30 | 200-500 | 让 PPO 真正收敛 |
| `il_kl_weight` | 0.02 | 0.005-0.001 | 放松 guardrail,允许 RL 探索 |
| `freeze_encoders` | True | False (阶段 2) | 让表示层也参与训练 |
| 训练分布 | balanced only(12 任务)| 加入 scale42 / scale60 | 覆盖 MILP 断供区间,让 RL 学"大规模" |
| 评测口径 | seed 100-115 balanced | + 六 profile + scale ladder | 与 IL v2 一致 |

## 已用去的成本

- Wall time: ~1h43min GPU 独占
- Iterations: 30
- Environment steps: ~7,680(每 iter 256 rollouts × 1 batch)
- Checkpoint 大小: 224KB(RLlib 序列化格式)

## 产物

- `runs/md_ray_ppo_train_2026-09-14/`:
  - `best_iter_0010/`, `best_iter_0020/` — 验证集 W/T/L 改善时的 checkpoint
  - `final_checkpoint/` — iter 30 终态
  - `training_history.json`, `run_metadata.json`, `seed_splits.json`, `held_out_comparison.json`
- `reports/md_ray_ppo_train_2026-09-14/`:
  - `train.log` — 完整训练 log
  - 上述 JSON 快照

## 下一步

**不要立即启动长训练**——先跑一次调整过的短训练验证方向,再决定是否投 5-10 小时/单卡的大 curriculum:

1. **调整 kl 试跑**:`--kl-weight 0.005 --iterations 60`(约 3.5 小时),看 il_reference_kl 是否明显上升、makespan 是否有实质改善
2. **同时观察 scale-ladder**:训完直接用 [experiments/md_c0_milp_supervised_scale_eval.py](../../experiments/md_c0_milp_supervised_scale_eval.py) 的框架 evaluate 新 checkpoint,看 42+ 任务上 RL 是否好于 IL
3. **若步骤 1-2 有正向信号**:再上完整 curriculum(参考 [RL_ROADMAP.md](RL_ROADMAP.md) Step 2)

## 一句话总结

**RL 训练 pipeline 完全可用,PPO 从 IL warm-start 30 iter 后 marginally 胜出(1/15/0),但 KL guardrail 太严 + 迭代太少 + 训练分布太窄,导致 held-out 增益仅 0.03%。方向对,但需要调整超参和放大 curriculum 才能验证是否真能突破 IL 天花板**。
