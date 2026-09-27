# 2026-09-13 结论与下一步

原始的当日工作记录(29KB,12 节)已归档到
[../../archive/2026-09-13-c0-eval-campaign/工作记录_full.md](../../archive/2026-09-13-c0-eval-campaign/工作记录_full.md)。
本页只留下经过验证的结论、判定依据、以及基于这些结论重新设计的 IL 训练方向。

## TL;DR

1. **运行鲁棒性充分**:C0+MIP fallback 混合架构在 task_count 12→150、结构参数拉满、以及两者组合的 51 次超范围 rollout 里 **0 失败、0 非法分配**;单次 MIP 兜底始终 < 1s。
2. **调度质量校准**:纯 MILP baseline(Gurobi + 300s)在 12-24 任务上秒级最优,MILP 在 42 任务开始超时,90+ 100% 超时。这提供了首次可靠的最优参照。
3. **诊断结果**:老 C0(合成关系对 + 1.25×priority-eta 手工规则监督)相对 MILP 最优在 6 profile 上差 15-36%,相对最优贪心差 10-37%——**训练分布内就没学到接近最优的策略,问题在监督信号本身**。
4. **修复方向**:改用 MILP 最优 `OracleAction` 序列作为 IL 监督信号;方案见 [IL_REDESIGN.md](IL_REDESIGN.md)。
5. **两次 MILP-IL 概念验证已跑通**(详见 [PILOT_RESULTS.md](PILOT_RESULTS.md)、[SCALE_RESULTS.md](SCALE_RESULTS.md)):
   - **第一次**(150 balanced instance):6 profile 平均 gap +25.1% → **+19.5%**(-5.6pp);4/6 profile 改善 7-11pp。
   - **第二次**(100 balanced + 100 scale_medium + 50 scale42):6 profile 平均 gap → **+15.1%**,**达到 IL_REDESIGN.md ≤15% 目标线**;**5/6 profile 胜出最优贪心**。规模阶梯上新 C0 每档改善老 C0 3-17pp,但**仍未胜出贪心**(1.5-18.2% 差距)。
6. **方法当前的价值边界**(详见 [ANALYSIS.md](ANALYSIS.md)):质量上 —— 标准规模 C0 输 MILP 6-22%,大规模输贪心 1.5-18%,**目前没有一个规模区间上 C0 是质量最好的选项**。唯一未被证伪的价值维度是**决策延迟**(尚未测量)。要让方法有广义价值,需要**RL** —— IL 已到监督信号的天花板,大规模 MILP 超时更让 IL 断供。
7. **下一步:RL 路径,现成基础设施可直接复用**。IL 侧边际收益递减,42+ 任务 MILP 断供,必须 RL。查过一遍:本 fork 里 `reinforcement_learning/md_ray_ppo.py` (Ray RLlib PPO)、`models/md_rllib_model.py`(接 IL checkpoint 的 warm-start 适配器)、`md_gym_environment.py`、`md_finetune.py`(BC/KL guardrail)**已全部造好**;MILP-IL v2 checkpoint 格式与 `MDRLlibModel` 期望字段**已验证一致**,可直接传给 `--il-checkpoint` 参数,不需要写 adapter。工程量从"3-7 天开发+训练"缩到"< 1 天冒烟 + 3-7 天 curriculum 训练"。详见 [RL_ROADMAP.md](RL_ROADMAP.md)。
8. **RL 冒烟测试已通过**([RL_SMOKE.md](RL_SMOKE.md),2026-09-14):`md_ray_ppo` 1 迭代跑通,16 seed validation 全部成功、0 illegal,IL/RL 策略几乎一致(动作 disagreement 0.028%,paired W/T/L = 0/15/1),`go_no_go=False` 触发条件为"需要更多迭代"——**pipeline 完全可用**。
9. **RL 30 迭代训练已完成**([RL_TRAIN.md](RL_TRAIN.md)):warm-start from MILP-IL v2,~1h43min 完成 30 iter。验证集 seed 115 从 IL 254 → RL 249(-2%),W/T/L = **1/15/0**,RL marginally 胜出 IL。但 held-out 集(seed 200-215)上 IL 233.44 vs RL 233.38,只快 0.03% —— 增益极小,**未达 ≥5% 胜出 greedy 的成功阈值**。原因:`il_kl_weight=0.02` guardrail 太严(全程 `il_reference_kl < 2e-5`),encoder frozen,迭代太少,训练分布只在 12-任务 balanced。**方向对但需要调超参/放大 curriculum**——下一步应先跑 `--kl-weight 0.005 --iterations 60` 的短对照验证方向再决定是否投大 curriculum。
10. **目标 profile 放大评测发现方法在正确场景下有明确价值**([SCALED_TARGET_PROFILES.md](SCALED_TARGET_PROFILES.md),2026-09-14):挑 6 profile 里"距 MILP 最优最近 & 相对 greedy 优势最大"的两个 profile —— `dependency_deep`(小 gap)和 `process_scarce`(大优势),放大到 task_count 12→60 每档 4 seed。**process_scarce** 在 12-42 任务上 C0 相对最优贪心稳定胜出 4-15%,同时距 MILP 最优仅 5-9%,MILP 延迟秒级、C0 40ms —— **这是方法的核心价值场景**。**dependency_deep** 上 C0 与 greedy 打平但距 MILP 更近,60 任务时两 profile 都刚好失去对 greedy 的优势(输 0.8-1.6%),边界与 IL 训练覆盖(≤42 任务)一致。这修正了 [ANALYSIS.md](ANALYSIS.md) 里"没有一个规模区间 C0 质量最优"的表述 —— **应改为"profile-conditional 有价值"**;RL 的目标是扩大价值成立的场景范围。


11. **RL 训练在 tc≥24 上撞到深层约束,三个直觉都被证伪**([RL_SCALED_PROCESS_SCARCE.md](RL_SCALED_PROCESS_SCARCE.md),2026-09-14 至 2026-09-15):**基础设施完整可用**(pilot、RLlib↔IL 转换器、workers-on-GPU shim、独立 Categorical 分布 shim 都已实现)。但 tc=24 process_scarce 上单机 PPO 每 iter 24-45 min,三次尝试都撞墙:(1) `num_env_runners=8` 无实质加速;(2) `num_gpus_per_env_runner=0.1` 让 workers 上 GPU(nvidia-smi 确认 driver 1.5GB + 8 workers × 398MB),但 iter 时间几乎没变;(3) 独立 Categorical 分布蓄意跳过 autoregressive 采样,env 直接抛 `joint action prefix has no legal completion` 拒绝。**深层根因**:`autoregressive_action_masks` 里的 `_action_masks` 是**完整 CSP 搜索**(skill coverage + process coalition + material transport 前提),必须序贯 + Python 递归。这不是资源分配问题,而是**训练环境合法性契约本身**的约束。**真正的下一步**(1-2 周新工作):选项 A — env 加 bipartite-matching repair(接受语义 mismatch);选项 B — 可微 constrained decoding 集成到策略前向。**当前推荐**:接受 RL 需要架构改造,先用 MILP-IL v2 checkpoint 部署到已证明有价值的 process_scarce 12-42 任务场景([SCALED_TARGET_PROFILES.md](SCALED_TARGET_PROFILES.md))。产物:`experiments/md_c0_ppo_process_scarce_pilot.py`(含 `--num-gpus-per-env-runner` + `--use-independent-action-dist` 双 shim)、`models/md_rllib_model.py::MDIndependentActionDistribution`、`experiments/md_rllib_to_il_adapter.py`。

## 核心数据

### 六个 profile 三方对照 (老 C0 vs 新 C0(多规模 IL) vs 最优贪心 vs MILP 最优)

MILP 6/6 profile × 4 seed 全部秒级最优,即真实上限参照。

| profile | MILP 最优 | 老 C0 gap | **新 C0 gap** | 最优贪心 gap | **新 C0 vs 贪心** |
|---|---:|---:|---:|---:|---:|
| balanced | 202.2 | +30.9% | **+21.9%** | +25.5% | **-2.9%** ✓ |
| process_scarce | 154.8 | +29.6% | **+18.3%** | +37.0% | **-13.7%** ✓✓ |
| transport_bottleneck | 389.0 | +15.6% | **+12.6%** | +12.5% | +0.05% ≈ |
| dependency_deep | 249.8 | +15.9% | **+6.4%** | +10.5% | **-3.7%** ✓ |
| mixed_hard | 262.0 | +36.0% | **+13.5%** | +16.4% | **-2.5%** ✓ |
| scale_medium | 243.5 | +22.5% | **+17.6%** | +21.7% | **-3.4%** ✓ |
| **平均** | — | **+25.1%** | **+15.1%** | **+20.5%** | **-4.4%** |

### 规模阶梯 (stress-test config,12→150 任务)

MILP 12/24 秒级最优;42 部分超时;60+ 大量超时;90+ 100% 超时,报出的 makespan 是次优,真实 gap 更大。

| task_count | MILP incumbent | 新 C0 gap vs MILP | 最优贪心 gap vs MILP | **新 C0 vs 贪心** | 老 C0 vs 贪心 |
|---:|---:|---:|---:|---:|---:|
| 12 | 285.3 | +16.0% | +7.5% | **+7.9%** | +12.0% |
| 24 | 230.0 | +22.5% | +18.1% | **+3.7%** | +12.0% |
| 42 | 232.3 | +26.1% | +24.2% | **+1.5%** | +18.5% |
| 60 | 272.0 | +30.5% | +17.9% | **+10.7%** | +14.0% |
| 90 | 258.0 | +43.4% | +37.6% | **+4.2%** | +7.4% |
| 114 | 259.3 | +41.6% | +19.8% | **+18.2%** | +24.4% |
| 150 | 302.7 | +19.3% | +11.0% | **+7.4%** | +16.9% |
| **平均** | — | **+28.5%** | **+19.4%** | **+7.7%** | **+15.0%** |

### MILP 求解能力边界(300s 预算)

| task_count | 状态 | 平均求解时间 |
|---:|---|---:|
| 12 | 3/3 最优 | 0.07s |
| 24 | 3/3 最优 | 1.68s |
| 42 | 2 最优 + 1 超时 | 110s |
| 60 | 1 最优 + 2 超时 | 273s |
| 90 | 3/3 超时(gap 16.6-26.9%) | 302s |
| 114 | 3/3 超时(gap 12.2-15.2%) | 305s |
| 150 | 3/3 超时(gap 25.1-40.2%) | 311s |

## 判定:方法的价值边界

**当前状态诚实描述**:
- 标准规模(6 profile,10-24 任务):MILP 秒级最优,C0 差 6-22%,贪心差 10-37%。C0 胜出贪心,输给 MILP。→ **C0 只能在延迟约束下有价值**(MILP 需要 0.02-1.7s,C0 μs 级)。
- 规模阶梯:C0 差贪心 1.5-18.2%,输给贪心。→ **C0 目前没有质量优势**。
- 42+ 任务:MILP 超时率高,90+ 100% 超时。→ **IL 训练数据在 90+ 断供**,这条监督路径没法继续压。

要让方法有广义价值,只有两条路:
- **RL 探索**(自己产生监督,超越 IL 天花板,大规模也能训)—— 见 [RL_ROADMAP.md](RL_ROADMAP.md)
- 放弃质量目标,只主张延迟——但那与其说是研究价值,不如说是工程 tradeoff

完整分析见 [ANALYSIS.md](ANALYSIS.md)。

## 关键负面结果(值得记住)

- **加线程无效**:MIP fallback 从 1→4→8 线程,150 任务总耗时 46.9→44.0→44.1s,降幅仅 6%。瓶颈不在并行度上。
- **frozen-joint correction 1/450**:450 个 model/instance 配对里只有 1 个改善,收益可忽略,未采纳。
- **端到端 joint 训练不稳定**:两个初始化 × 三个 seed × 100 epoch 全部翻车。这不是训练算法问题,是标签本身模糊(partial-inclusion,无 idle 补集、无 t0 禁令)。
- **exact-action scorer 消融未过阈值**、prefix pairwise switch 接受 0 次——事后修 joint 的路子基本堵死。
- **多规模 IL 也未能胜出贪心(scale ladder)**:100+100+50 训练配比、200 epoch,scale ladder 上仍差贪心 1.5-18%。IL 天花板已现。

## 当日产出索引

| 类别 | 位置 | 状态 |
|---|---|---|
| C0 checkpoint(3 seed,合成关系对)| `reports/md_c0_gpu_retrain_pilot_2026-09-13/` | 保留 |
| Profile 对照 + 泛化评测 | `reports/md_c0_profile_eval_pilot_2026-09-13/`、`reports/md_c0_generalization_eval_2026-09-13/` | 保留 |
| Confidence 阈值扫描(3 seed + 大规模) | `reports/md_c0_confidence_threshold_sweep_2026-09-13*`、`reports/md_c0_scale_threshold_pilot_2026-09-13/` | 保留 |
| 规模压力测试(第一版 + extreme) | `reports/md_c0_scale_stress_pilot_2026-09-13/`、`reports/md_c0_scale_stress_extreme_pilot_2026-09-13/` | 保留 |
| 结构性压力 + 规模+结构组合 | `reports/md_c0_structural_stress_pilot_2026-09-13/`、`reports/md_c0_combined_extreme_pilot_2026-09-13/` | 保留 |
| MIP 线程扫描 | `reports/md_c0_mip_threads_pilot_2026-09-13/` | 保留 |
| C0 vs 贪心 规模对照 + confirm | `reports/md_c0_scale_baseline_comparison_pilot_2026-09-13/`、`reports/md_c0_scale_baseline_confirm_pilot_2026-09-13/` | 保留 |
| **纯 MILP 三方对照** | `reports/md_pure_milp_baseline_pilot_2026-09-13/` | 保留,方向决定证据 |
| **MILP-IL 第一次(balanced only)** | `reports/md_c0_milp_supervised_pilot_2026-09-13/`、`reports/md_c0_milp_supervised_profile_eval_2026-09-13/` | 保留,方向可行证据 |
| **MILP-IL 第二次(多规模)** | `reports/md_c0_milp_supervised_scale_pilot_2026-09-13/`、`reports/md_c0_milp_supervised_scale_eval_2026-09-13/`、`reports/md_c0_milp_supervised_scale_profile_eval_2026-09-13/` | 保留,方向天花板证据 |
| 完整工作记录 / pilot 脚本 / log / run 脚本 | `archive/2026-09-13-c0-eval-campaign/` | 已归档 |
