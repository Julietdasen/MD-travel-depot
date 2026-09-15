# RL 路径实施 roadmap — 复用现成基础设施

**为什么必须 RL**:详见 [ANALYSIS.md](ANALYSIS.md)。核心两点:
1. 大规模(90+ 任务)MILP 100% 超时,IL 拿不到最优标签
2. IL 本质是逼近老师,不能超越 MILP 上限;RL 用 simulator reward 代替 label,可探索超越

---

## 关键结论:不用造新轮子

用户提醒后查了一遍,发现**这条路的基础设施在这个 fork 里已经全部造好了**——不需要从零写。

### 上游(`jakbichler/Sadcher`)已有

- `reinforcement_learning/ppo_train.py`(269 行)—— **SKRL** PPO trainer
- `schedulers/sadcherRL.py`(245 行)—— 基础 SADCHER 的 RL 变体 policy
- `models/policy_value_discrete.py`(341 行)—— `SchedulerPolicy` 用 `MultiCategoricalMixin`,支持 IL warm-start 参数 `IL_pretrained=True`,含 checkpoint 加载逻辑

### 本地 fork(`Julietdasen/MD`)额外提供的 MD-specific 扩展

本仓库比上游多出几个 MD 专用的 RL 文件,而且**关键的 warm-start 适配器已经写好**:

| 文件 | 功能 | 关键点 |
|---|---|---|
| [reinforcement_learning/md_ray_ppo.py](../../reinforcement_learning/md_ray_ppo.py)(529 行)| Ray RLlib PPO trainer,MD 专用 | CLI 已就绪:`--il-checkpoint <path> --seed N --iterations N` |
| [models/md_rllib_model.py](../../models/md_rllib_model.py)(234 行)| `MDRLlibModel` — RLlib TorchModelV2 包裹 `MDEnhancedSchedulerNetwork` | **line 108-114**: 直接读 IL checkpoint 里的 `md_policy_config` + `state_dict`,自动重建 C0 架构模型并加载权重 |
| [reinforcement_learning/md_gym_environment.py](../../reinforcement_learning/md_gym_environment.py)| Gymnasium 适配器包裹 `MDDiscreteSimulator` | 用同一个 `build_md_policy_inputs_from_simulator` featurizer,与 C0 推理时 state 表示一致 |
| [reinforcement_learning/md_finetune.py](../../reinforcement_learning/md_finetune.py)| `MDFineTuneConfig` + `MDRewardTracker` | BC/KL guardrail、illegal_penalty、completion_bonus、starvation_penalty 已经就位 |
| [reinforcement_learning/md_joint_action.py](../../reinforcement_learning/md_joint_action.py)| `joint_action_is_legal` | RLlib callback `MDIllegalAssignmentCallbacks` 用它监控非法动作率 |

### 依赖已装好

- `ray 2.49.2` + `ray.rllib` ✓
- `gymnasium 1.1.1` ✓
- `skrl 2.1.0` ✓
- `torch 2.4.0+cu121` ✓

### Checkpoint 格式匹配已验证

已确认 MILP-IL v2 checkpoint(`reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt`)的 payload 结构 (`schema_version`, `model_kind='md_enhanced'`, `model_spec`, `md_policy_config`, `state_dict`) **与 `MDRLlibModel` 读取的字段完全一致**——直接可以传给 `md_ray_ppo` 的 `--il-checkpoint` 参数,不需要写任何 adapter。

## 现实的实施步骤(工程量骤减)

### Step 0 — 熟悉现有代码(半天)

- 读 [reinforcement_learning/md_ray_ppo.py](../../reinforcement_learning/md_ray_ppo.py) 的 `train()`(第 270 行起)与 `main()`(第 500 行起)
- 读 [models/md_rllib_model.py](../../models/md_rllib_model.py) 的 `MDRLlibModel.__init__`(line 97-)与 `MDAutoregressiveActionDistribution`
- 读 [reinforcement_learning/md_gym_environment.py](../../reinforcement_learning/md_gym_environment.py) 的 `step()` 和 reward 组装
- 检查 [reinforcement_learning/md_ray_ppo.py:51](../../reinforcement_learning/md_ray_ppo.py#L51) 的 `phase_a_experiment_configs()`——**里面可能已有本次 curriculum 需要的实验网格**

### Step 1 — 1 iteration 冒烟(< 1 天)

用最小配置跑一次 md_ray_ppo,只 1 iteration,证明 wire-up 通:

```bash
cd /root/autodl-tmp/MD/Sadcher
source /root/miniconda3/etc/profile.d/conda.sh && conda activate md
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export GRB_LICENSE_FILE=/root/.config/gurobi/gurobi.lic  # md_gym_environment 里的 fallback 需要
export MD_RAY_TEMP_DIR=/root/autodl-tmp/MD/Sadcher/tmp_ray

python -m reinforcement_learning.md_ray_ppo \
    --il-checkpoint reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt \
    --seed 0 \
    --iterations 1 \
    --num-env-runners 0 \
    --num-gpus 1.0 \
    --output-dir runs/md_ray_ppo_smoke
```

**验证点**:
- ray 启动、环境注册、模型创建、checkpoint 加载都不报错
- 1 iteration 完成,`runs/md_ray_ppo_smoke/` 下有 iteration log
- `MDIllegalAssignmentCallbacks` 报告的 illegal_assignment_rate 是有限数
- 用 [reinforcement_learning/rl_evaluation.py](../../reinforcement_learning/rl_evaluation.py)(或直接调 `held_out_comparison`)对 1-iter 结果做一次 held-out rollout,看 makespan 是否合理

如果冒烟通:说明 3-7 天 curriculum 训练可以正式启动。如果炸,**多半是 checkpoint 加载层的 state_dict key mismatch**——`MDRLlibModel` 用 `strict=False` 加载,允许部分 miss,但要看具体 skipped 列表判断是否影响 policy 前向。

### Step 2 — 小规模 curriculum(3-7 天)

用 `phase_a_experiment_configs()` 里的现成 experiment cell,或者显式指定超参 iterate:

- **Stage 1**: 12-24 任务(标准 profile 规模),iterations = 200-500,验证在训练分布上 PPO 稳定
- **Stage 2**: 42 任务(scale_medium 附近),继承 Stage 1 checkpoint,iterations = 500-1000
- **Stage 3**: 60/90 任务(MILP 部分超时到全超时),iterations = 1000-2000
- **Stage 4**(可选): 114/150 任务,继承 Stage 3 checkpoint,验证 RL 是否能真正超越贪心

每 Stage 独立输出 `runs/md_ray_ppo_stage{k}_tc{tc}_seed{seed}/`,失败可退回前一档 checkpoint。

### Step 3 — 评测协议(复用已有工具)

- 6 profile: 直接复用 [experiments/md_c0_milp_supervised_profile_eval.py](../../experiments/md_c0_milp_supervised_profile_eval.py) —— **需要写一个 adapter**,因为 `load_md_policy_checkpoint` 走的是 IL checkpoint 格式,RLlib checkpoint 格式不同。可以让 RLlib 训练结束时导出一个 IL 格式 checkpoint(`state_dict` 从 `MDRLlibModel.net` 提取,再包成同样 payload)
- Scale ladder: 同上,复用 [experiments/md_c0_milp_supervised_scale_eval.py](../../experiments/md_c0_milp_supervised_scale_eval.py)
- Held-out: `md_ray_ppo` 里的 `held_out_comparison(il_metrics, rl_metrics)` 已有

### Step 4 — 失败保护

**成功标准**:scale ladder 平均胜出 greedy_unlock ≥ 5%(当前 MILP-IL v2 是差 7.7%,即需要总提升 ≥ 12.7pp)

**回退方案**:
- Stage 1 训 500 iter 后不稳定:降 `learning_rate` 或提高 `il_kl_weight`(guardrail)
- Stage 2 无提升:检查 reward 是否被 shaping 项主导——调低 `completion_bonus`/`starvation_penalty`
- Stage 3 illegal_rate 高:检查 fallback 是否在环境里被禁用,或加大 `illegal_penalty`
- 全部 stage 都失败:退回 IL 路径,加更多 profile/规模数据

## 关键疑问(可能是坑,值得先查清)

1. **`md_ray_ppo` 训练时会不会调 Gurobi**?——`MDGymEnvironment` 用 `run_greedy_eta` 算 `greedy_makespan` 作参照(不调 Gurobi),但 fallback 在 online scheduler 里配置。**默认应该不调,但需要确认**——训练时调 MILP 会严重拖慢
2. **`freeze_encoders=True` 是不是必要**?—— `train()` 默认 `freeze_encoders=True`(line 277)。上游 `SchedulerPolicy` 也默认冻结 encoder(只训 output head 和 value)。对 MILP-IL warm-start 说得通(保住 IL 学到的表示),但如果 RL 需要探索大变化可能要放开
3. **`MDAutoregressiveActionDistribution` 的动作空间和 C0 推理时用的 bipartite matching 是否等价**?—— 如果不等价,训好的 RL policy 部署到 `OnlineMDScheduler` 时会有 train/deploy mismatch
4. **Ray 单机模式的资源占用**?—— `num_env_runners=0` 表示本地 driver 跑,单卡应该够用;如果开 parallel workers 会更快但也更耗资源

上面几个疑问应该在 Step 1 冒烟阶段回答清楚。

## 明确划界:不做什么

- **不做无 warm-start 的从零 RL**——探索灾难成本太高
- **不为大 curriculum 追求最优 hyperparameter**——PPO 默认配 + minor tuning 即可
- **不在同一次训练里混多规模**——curriculum 分档更稳
- **不放弃 IL/MILP baseline 对照**——RL 要有优势才有意义
- **不从零重写 PPO trainer / policy wrapper / gym adapter / checkpoint 加载**——上面全都已有

## 与 IL 路径的关系

**不放弃 IL**。IL v2 已达 6 profile ≤15% 目标,是可用的 baseline。RL 是**在此基础上**尝试突破天花板,不是替代。双线并存:

- IL:继续可加数据、加 profile 覆盖,推 gap 到 ≤10%——路径明确但边际收益递减
- RL:唯一有可能真正胜出 greedy 且不受 MILP 求解能力限制的路径

## 一句话总结

RL 路径的基础设施在 fork 里已经**全部造好且已验证 checkpoint 直连**,工程量从"3-7 天开发 + 3-7 天训练"缩到"< 1 天冒烟 + 3-7 天分档训练"。下一步应该直接跑 Step 1 冒烟,若通则展开 curriculum。
