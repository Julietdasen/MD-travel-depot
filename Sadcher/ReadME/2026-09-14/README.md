# 2026-09-14 会话摘要

本日的会话主要围绕 **RL 路径的实施与验证**、以及 **目标 profile 放大评测** 展开。上一日([../2026-09-13/](../2026-09-13/))的诊断结论是"IL 已到监督信号天花板,大规模 MILP 断供,必须 RL",本日按此推进。

## 核心工作

### 1. RL 端到端 pipeline 冒烟([RL_SMOKE.md](RL_SMOKE.md))

- 用 MILP-IL v2 checkpoint 作 warm-start,`md_ray_ppo.py` 1 迭代跑通
- 16 seed validation 全部成功、0 illegal
- IL/RL 策略几乎一致(动作 disagreement 0.028%,paired W/T/L = 0/15/1)
- **结论:pipeline 完全可用**

### 2. RL 30 迭代正式训练([RL_TRAIN.md](RL_TRAIN.md))

- ~1h43min 完成 30 iter,warm-start from MILP-IL v2,tc=12 balanced 分布
- 验证集 seed 115 从 IL 254 → RL 249(-2%),W/T/L = **1/15/0**
- Held-out 集(seed 200-215):IL 233.44 vs RL 233.38,只快 0.03%
- **未达 ≥5% 胜出 greedy 的成功阈值**
- 原因:`il_kl_weight=0.02` guardrail 太严、encoder frozen、迭代太少、训练分布单一

### 3. 目标 profile 放大评测([SCALED_TARGET_PROFILES.md](SCALED_TARGET_PROFILES.md))**⭐ 本日最重要的发现**

选出 6 profile 里"距 MILP 最优最近 & 相对 greedy 优势最大"的两个 profile:
- **`dependency_deep`**(距 MILP +6.4%,绝对质量最优场景)
- **`process_scarce`**(-13.7pp 相对 greedy 优势,相对优势最大场景)

放大 task_count 12→60,每档 4 seed:

**process_scarce 关键结果**:

| task_count | MILP 最优 | C0 makespan | C0 vs MILP | **C0 vs 最优 greedy** |
|---:|---:|---:|---:|---:|
| 12 | 164.2 | 175.8 | +7.0% | **-15.1%** ✓✓ |
| 24 | 250.0 | 263.5 | +5.4% | **-13.3%** ✓✓ |
| 42 | 304.0 | 332.2 | +9.3% | **-4.1%** ✓ |
| 60 | 356.0 | 398.8 | +12.0% | +0.8% |

**结论**:
- **process_scarce 12-42 任务是 method 的核心价值场景** — C0 稳定胜 greedy 4-15%,距 MILP 最优仅 5-9%,MILP 延迟秒级、C0 40ms
- **dependency_deep 上 C0 与 greedy 打平但距 MILP 更近** — 价值主要在延迟
- **60 任务是训练分布外的可靠边界** — 两 profile 都刚好在这里失去对 greedy 的优势

修正了 [../2026-09-13/ANALYSIS.md](../2026-09-13/ANALYSIS.md) 里"没有一个规模区间 C0 质量最优"的表述 — **应改为"profile-conditional 有价值"**;RL 的目标是扩大价值成立的场景范围。

## 产物索引

| 类别 | 位置 |
|---|---|
| RL warm-start checkpoint (30 iter) | `runs/md_ray_ppo_train_2026-09-14/` |
| RL 冒烟报告 | `reports/md_ray_ppo_smoke_2026-09-14/` |
| RL 训练报告 | `reports/md_ray_ppo_train_2026-09-14/` |
| 目标 profile 放大评测 | `reports/md_c0_scaled_target_profiles_eval_2026-09-14/` |
| 评测脚本 | `experiments/md_c0_scaled_target_profiles_eval.py` |
| 启动脚本 | `run_c0_scaled_target_profiles_eval.sh` |
