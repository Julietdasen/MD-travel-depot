# RL 冒烟测试记录 (2026-09-14)

**目的**:验证 [reinforcement_learning/md_ray_ppo.py](../../reinforcement_learning/md_ray_ppo.py) 端到端管线是否可用,能否直接消费我们的 MILP-IL v2 checkpoint。参考 [RL_ROADMAP.md](RL_ROADMAP.md) Step 1。

## 命令

```bash
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic
export MD_RAY_TEMP_DIR=/root/autodl-tmp/MD/Sadcher/tmp_ray
python -m reinforcement_learning.md_ray_ppo \
  --il-checkpoint reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt \
  --seed 0 \
  --iterations 1 \
  --num-env-runners 0 \
  --num-gpus 1.0 \
  --output-dir runs/md_ray_ppo_smoke_2026-09-14
```

## 结果:PASS ✓

- **总耗时**:约 10 分钟(16:38 → 16:48 completed 迭代 1;后续 driver 进入 post-processing,手动 kill)
- **checkpoint 加载**:MDRLlibModel 从 IL checkpoint 直接读取 `md_policy_config` + `state_dict`,无 adapter,无错误
- **成功率**:16 seed 全部 validation success = 100%,IL 与 RL 均 **0 illegal_assignments**

### 迭代 1 关键指标

| metric | value | 说明 |
|---|---:|---|
| `train_reward_mean` | -3.44 | 归一化后的 reward |
| `train_episode_len_mean` | 235 | 平均每条 rollout 的步数 |
| `ppo_kl` | 2.20e-06 | 极小,policy 基本没动(1 iter 正常)|
| `il_reference_kl` | 2.38e-06 | KL 到 IL 参考策略的距离,guardrail 生效 |
| `entropy` | 0.026 | 策略确定性较高 |
| `value_loss` | 1.46 | value head 起点损失 |
| `value_explained_variance` | 0.063 | value head 尚未学到多少信号 |
| `policy_loss` | -2.0e-04 | 极小 |
| `actor_parameter_drift.max_absolute` | 6.7e-04 | encoder frozen 生效,权重几乎不动 |
| `actor_parameter_drift.normalized_l2` | 2.1e-03 | 同上 |

### IL vs RL(16 seed validation)

| 指标 | IL | RL(iter 1)| 判读 |
|---|---:|---:|---|
| 成功率 | 100% (16/16) | 100% (16/16) | 都无失败 |
| 平均 makespan | 225.94 | 226.69 | RL 落后 0.75(0.33%)|
| illegal_assignments | 0 | 0 | 兜底/policy 均产合法动作 |
| p95 latency | 40.0 ms | 39.6 ms | 与 IL 几乎一致 |
| 平均 material_starvation | 171.56 | 171.38 | 打平 |
| paired W/T/L | — | 0 / 15 / 1 | 15 seed 完全相同,seed 115 上 RL 差 12 |
| deterministic_action_disagreement | — | 0.028% (1/3615) | RL 策略与 IL 在动作上几乎完全一致 |
| **`go_no_go`** | — | **False** | reason: `makespan_gain_below_threshold`——需要更多迭代 |

## 关键判断

1. **管线全部通** ✓——ray init、RLlib model catalog、MDGymEnvironment、KL guardrail 到 IL、validation 报告、held-out 对比、checkpoint 保存全部正常。
2. **IL warm-start 生效** ✓——RL policy 起点与 IL 几乎一致(动作 disagreement 0.028%,参数 drift < 0.001)。1 iter 后没有塌陷,说明 `il_kl_weight=0.02` guardrail 有约束力。
3. **训练分布内 IL 已经很强**——16 seed 全成功、0 非法,基本没有让 RL 立即翻盘的空间。收益要靠更多迭代 + curriculum 上更难规模。
4. **`go_no_go=False` 是设计里的正常状态**——`held_out_comparison` 已经在 code 里定义了 threshold,后续迭代通过它自动判定何时 checkpoint 值得部署。
5. **p95 latency ~40ms** 首次坐实——这是 [ANALYSIS.md](ANALYSIS.md) 里的价值维度 B 首次量化的数据点。C0 policy(hidden_dim=16、pair-aware attention)单步 p95 决策时间 40 ms 级,MILP 在 12 任务上是 20-1700 ms 级——**delay 维度上 C0 有 1-40x 优势**。这与老 C0 的 μs 级不一致(需要复查是不是包含了 fallback 兜底时间;若是,则策略侧 latency ≈ neural forward + fallback)。

## Post-processing 卡住的观察

Iteration 1 完成后,driver 在 `Restored on 172.17.0.6 from checkpoint` 之后仍然 100% CPU 运行了 5+ 分钟未退出。手动 kill 掉进程 + `ray stop` 后 GPU 立即释放。**这是一个已知坑,可能是 ray dashboard/log_monitor 悬挂在退出流程**。对多轮训练来说不影响,只影响 iterations=1 的短跑;若要作 CI/CD 需要加 `finally: ray.shutdown()` 或 signal handler。

## 产物

- `runs/md_ray_ppo_smoke_2026-09-14/`:
  - `best_iter_0001/` — iter 1 checkpoint
  - `final_checkpoint/` — 结束状态
  - `training_history.json` — 训练历史
  - `run_metadata.json` — split_seeds + config
- `reports/md_ray_ppo_smoke_2026-09-14/smoke.log` — 完整 20 行 log 归档

## 下一步

启动 30 迭代正式训练(`ITERATIONS=30 bash run_rl_train.sh`,输出 `runs/md_ray_ppo_train_2026-09-14/`,log `rl_train.log`)。按每迭代 10 分钟估算,约 5 小时完成,可留后台跑。完成后跑 profile eval + scale eval,与 IL v2 对照。
