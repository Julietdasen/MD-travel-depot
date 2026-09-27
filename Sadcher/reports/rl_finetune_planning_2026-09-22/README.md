# IL → RL Fine-tune Planning (Path D, depot+travel)

Date: 2026-09-22
Working dir: `/root/autodl-tmp/MD/Sadcher`
Author: Opus 4.7 planning + user

## Background: 为什么要做 RL fine-tune

### 今天做的事情
1. **修 bug**: `schedulers/md_constrained_decoder.py:_apply_ready_preference_guard`
   过严 guard 把「PENDING 但已有 coalition committed（robot 正在 travel）」和「PENDING 无人认领」都当 unsafe 一起 ban 掉，导致 process robot 大量 idle（wide 实例 +9520 idle → gap +130%）。
   Fix: 只 ban `PENDING + empty assigned_robot_ids` 的 precursor。
   同步 mirror 到 `schedulers/process_greedy_md.py:_has_pending_precursor`。

2. **验证**: feasibility gate 16/16 通过；Stage 2 checkpoint A/B eval 4/4 cell 下降：

   | Profile / tc | S2-old gap% | S2-tightened gap% |
   |---|---|---|
   | process_scarce tc=24 | +41.7% | +35.0% |
   | process_scarce tc=42 | +75.4% | +38.0% |
   | dependency_deep tc=24 | +63.6% | +17.1% |
   | dependency_deep tc=42 | +130.5% | +37.2% |

3. **Path C v1 eval 出结果 (`primary_verdict: FAILED`)**:
   p60 process_scarce makespan 从 epoch 5 的 1210.5 只降到 epoch 20 的 1128.75，`tier_non_regression_by_epoch = 0/4` for all epochs。
   → **BC 家族（ranking loss）marginal，第二次独立验证 `path_lessons` 里的结论**。

### Gap 拆解：makespan 差距来自哪里
基于 `reports/depot_travel_gap_analysis/analyze.py` 重跑（guard 修复后的最新数字）：

**tight (process_scarce tc=24, MILP=771 / C0=1097, +42%)**
- svc 完全一样（374 vs 374），travel +279，wait +323，idle +1906
- 分工乱：R2 本该闲置（MILP idle=771）却被派出去，R0 反而 idle=661
- routing 差：R3 travel 438 → 680

**wide (dependency_deep tc=42, MILP=422 / C0=587, +39%)**
- wait 翻倍：1613 → 3113
- coalition timing 不同步：C0 里 R2/R3/R4/R5/R6 全部 wait > 300，MILP 无一超过 267
- 早派 robot 却晚派它的 partners，先到的白等

**三类 policy quality 缺陷**：
1. worker selection 错位（scarce capacity 判断错）
2. coalition timing 不同步（frame-level BC 天然弱）
3. routing 顺序次优（可能是特征不足）

### 结论
BC 家族已经到顶（val loss ≠ downstream；ranking 二次 marginal）。
下一步：**IL → RL fine-tune，以 makespan 为主 reward，加 unlock bonus 和 coalition sync penalty**。

---

## Path D · IL → RL Fine-tune 规划

### 用户决策（2026-09-22 敲定）

1. **RL 框架**: 先用 Ray RLlib（`reinforcement_learning/md_ray_ppo.py` 已就绪），基建挂了再换手写 GRPO
2. **初期 tier**: process_scarce/dependency_deep × tc=24/42（tc=60 放最终 sweep，因为 MILP timeout 用 incumbent 会给 RL 噪声 baseline）
3. **Init checkpoint**: Stage 2 (`reports/md_c0_pathA_2026-09-21-depot/training/best_checkpoint.pt`)，不用 Path C v1（marginal）

### 现有 RL 基础设施（探索 agent 摸清）

**可直接复用**（当前）:
- `reinforcement_learning/md_ray_ppo.py` (529L): Ray RLlib PPO trainer, `MDGymEnvironment` 包装 `MDDiscreteSimulator`, `MDRLlibModel`, `MDAutoregressiveActionDistribution`, IL-KL 正则, phase-A 配置 (A0–A4), paired IL-vs-RL rollouts, go/no-go gate
- `reinforcement_learning/md_finetune.py` (152L): `MDFineTuneConfig` (bc_weight=0.10, kl_weight=0.05, illegal/completion/starvation coefficients), `MDRewardTracker.step()` (potential-based dense reward: time, completed_tasks, starvation, illegal, terminal makespan), `il_regularized_ppo_loss` (BC + KL-to-IL), `evaluate_go_no_go`
- `reinforcement_learning/md_gym_environment.py` (157L): Gym adapter
- `reinforcement_learning/md_joint_action.py` (282L): joint autoregressive action encoding
- `models/md_rllib_model.py`: RLlib TorchModelV2 wrapping IL `MDPolicy` + value head + IL-reference-KL

**已有 pilot（可参照）**:
- `experiments/md_c0_ppo_process_scarce_pilot.py`: 唯一 RL 入口示例
- `experiments/md_rllib_to_il_adapter.py`: RL checkpoint → IL checkpoint 转换

**legacy 遗留（不要用）**:
- `schedulers/sadcherRL.py`: 上一代 Sadcher 的 inference-only scheduler，不是 training loop
- `reinforcement_learning/ppo_train.py`: 用 pre-MD 环境的 SKRL PPO trainer

### ⚠️ 关键风险点

**`_apply_ready_preference_guard` 的 tightening 是否被 RL rollout 采用取决于 mask 从哪来。**
`LearnedConstrainedDecoder.decode` 接受 pre-computed hard mask；如果 `MDRLlibModel` / `MDAutoregressiveActionDistribution` 的 legal-action mask **不是** 来自 `simulator_hard_mask(simulator)`，PPO rollout 会绕过今天修的 guard，train 出的 policy 又会 deadlock/wrong-commit。

→ **Phase 0 必须先 audit 这个**。

---

## 分阶段计划

### Phase 0 · 安全 audit（Sonnet, 只读, ~1h）

Files to inspect:
- `models/md_rllib_model.py`
- `reinforcement_learning/md_joint_action.py` — `MDAutoregressiveActionDistribution`
- `reinforcement_learning/md_gym_environment.py`

Deliverable: <200 字报告，明确回答:
1. `MDAutoregressiveActionDistribution` 的 legal-action mask 是否来自 `simulator_hard_mask(simulator)`？
2. 如果不是，用的是什么？（`create_task_robot_mask` / `is_task_assignable` / ad-hoc）
3. Rollout 采样能否 commit process robot 到 uncommitted-PENDING precursor 的下游 task？

**若 audit 显示 guard 未被 RL path 采用 → Phase 1 第一步补 guard，不加 reward。**

### Phase 1 · Reward 建模（扩展 `MDRewardTracker`）

| 组件 | 源数据 | 位置 | 权重（初值） |
|---|---|---|---|
| A. Terminal makespan | `simulator.time` + Path C v1 per-instance baseline | 已存在，接 baseline | 1.0 |
| B. Unlock bonus | `_can_start_process(t)` True 时的下游 unblock 数 × `duration` | 新增 `MDRewardTracker.step` 分量 | 0.1 |
| C. Coalition sync penalty | `process_execution_records[t]["waiting_duration"]` | 新增分量，在 IN_PROGRESS→COMPLETE tick 结算 | 0.05 |
| D. Redundancy penalty | coalition members capability 覆盖比对 | Phase 2 再加 | — |

约束:
- 所有 dense 分量保持 potential-based（避免 shaping 破坏最优 policy）
- 保留 BC 辅助 loss 和 IL-KL 正则（`md_finetune.py` 已有），防止 RL 崩

### Phase 2 · Baseline 与实验设计

- Init: Stage 2 checkpoint
- Instances: 4 profile-tc × 4 seeds = 16 instances/tier
- Baseline 对照: Stage 2 IL / Path C v1 IL / Greedy_distance / Greedy_unlock / MILP
- Report dir: `reports/md_c0_pathD_rl_2026-09-22-depot/`
- Go/no-go gate: 任一 tier gap vs Stage 2 ≥5pp 改善 且 无 tier 恶化 >2pp

### Phase 3 · 落地顺序（一步一验证）

1. Phase 0 audit → 若有 gap，补 guard 到 RL mask path
2. 加 unlock reward → pilot 1 tier 100 iter，看 curve
3. 加 sync penalty → 同上
4. 全 4-tier sweep（3 组 reward 权重 × 2 seed）
5. Eval + Gantt 对照（复用 `reports/depot_travel_gap_analysis/analyze.py` 框架，加 Path D 列）

---

## Reference paths (绝对路径)

- `/root/autodl-tmp/MD/Sadcher/schedulers/md_constrained_decoder.py` (今日修的 guard)
- `/root/autodl-tmp/MD/Sadcher/schedulers/process_greedy_md.py` (mirror guard)
- `/root/autodl-tmp/MD/Sadcher/schedulers/md_greedy_baselines.py` (Greedy baselines)
- `/root/autodl-tmp/MD/Sadcher/schedulers/online_md_scheduler.py` (`OnlineNeuralScoreProvider`)
- `/root/autodl-tmp/MD/Sadcher/models/md_policy.py` (`MDPolicy.forward`)
- `/root/autodl-tmp/MD/Sadcher/models/md_rllib_model.py` (RL model wrap)
- `/root/autodl-tmp/MD/Sadcher/reinforcement_learning/md_ray_ppo.py` (主训练器 `train()` @ line 270)
- `/root/autodl-tmp/MD/Sadcher/reinforcement_learning/md_finetune.py` (reward tracker + guardrails)
- `/root/autodl-tmp/MD/Sadcher/reinforcement_learning/md_gym_environment.py` (Gym adapter)
- `/root/autodl-tmp/MD/Sadcher/reinforcement_learning/md_joint_action.py` (`MDAutoregressiveActionDistribution`)
- `/root/autodl-tmp/MD/Sadcher/simulation_environment/md_discrete_simulator.py` (rollout & reward hooks)
- `/root/autodl-tmp/MD/Sadcher/imitation_learning/md_train.py` (`load_md_policy_checkpoint` @ line 376)
- `/root/autodl-tmp/MD/Sadcher/experiments/md_c0_ppo_process_scarce_pilot.py` (RL pilot 示例)
- `/root/autodl-tmp/MD/Sadcher/reports/md_c0_pathA_2026-09-21-depot/training/best_checkpoint.pt` (Stage 2 init)
- `/root/autodl-tmp/MD/Sadcher/reports/md_c0_pathC_v1_eval_2026-09-21-depot/summary.json` (Path C FAILED verdict)
- `/root/autodl-tmp/MD/Sadcher/reports/md_c0_pathA_guard_tightened_eval_2026-09-21-depot/summary.json` (guard fix A/B eval)
- `/root/autodl-tmp/MD/Sadcher/reports/depot_travel_gap_analysis/summary.json` (gap 拆解)

## Env vars

- `MRTA_MILP_SOLVER=ortools`
- `MD_ORACLE_SOLVER=ortools`
- Conda env: `md`

## Memory 关联

- `feedback_cross_path_guard.md`: 已记录 simulator 语义放宽的 mirror-guard 教训
- `project_path_lessons.md`: BC 家族 marginal 的第二次独立验证
- `project_paper_frozen_2026-09-21.md`: process-only envelope 已冻结，depot+travel 是**另开一 paper**，Path D RL 是这条线的第一实验
