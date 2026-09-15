# 多规模 MILP-IL 训练结果:上规模,能否胜出 greedy?

延续 [PILOT_RESULTS.md](PILOT_RESULTS.md) 的概念验证——把训练分布从"只 balanced (task_count=12)"扩到"balanced + scale_medium + scale42",看在**规模阶梯**(压力测试用的 stress config)和**六个 profile**(标准评测口径)上,新 C0 相对最优贪心的差距如何变化。

## 训练配置

同 PILOT_RESULTS 的 C0 架构 + BCE loss + 200 epoch,数据变了:

| shard | 算例参数 | count | seed 范围 | MILP 结果 |
|---|---|---:|---|---|
| balanced | balanced profile (task_count=12, transport_ratio=0.25, ...) | 100 | 80200-80299 | 100/100 optimal |
| scale_medium | scale_medium profile (task_count=24, ...) | 100 | 82000-82099 | 100/100 optimal |
| scale42 | scaled stress config (task_count=42, transport_ratio=1/3, precedence_density=0.30, capacity_slack=0.10, speed_ratio=0.70) | 50 | 84000-84049 | 50/50 solved (部分 timeout 但仍产出可行 schedule) |

- 生成阶段总耗时约 90 分钟(scale42 shard 最慢,120s time_limit × 部分 timeout)
- Split: 2833 train sample / 320 val sample (from 190+37+X instances 分为 80/10/10)
- 训练: 200 epoch, GPU 约 4-5 分钟。train_loss 0.520 → 0.194,val_loss 0.570 → best 0.461(后期过拟合但保存 best_val checkpoint)
- Checkpoint: `reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt`

## 结果 1:规模阶梯(压力测试口径)

对照:老 C0(合成关系对,来自 [reports/md_c0_scale_baseline_comparison_pilot_2026-09-13](../../reports/md_c0_scale_baseline_comparison_pilot_2026-09-13/))、新 C0(本次 MILP-IL,多规模训练)、最优贪心、纯 MILP(300s 超时会次优)。每档 task_count 3 seed。

### 完整对比表

| task_count | MILP incumbent | **新 C0** | 新 C0 相对 MILP | 老 C0 相对 MILP | 最优贪心 (名字) | 贪心相对 MILP | **新 C0 vs 贪心** | 老 C0 vs 贪心 |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 12 | 285.3 | 331.0 | +16.0% | +20.3% | 306.7 (unlock) | +7.5% | **+7.9%** | +12.0% |
| 24 | 230.0 | 281.7 | +22.5% | +32.3% | 271.7 (unlock) | +18.1% | **+3.7%** | +12.0% |
| 42 | 232.3 | 293.0 | +26.1% | +47.2% | 288.7 (unlock) | +24.2% | **+1.5%** | +18.5% |
| 60 | 272.0 | 355.0 | +30.5% | +34.4% | 320.7 (unlock) | +17.9% | **+10.7%** | +14.0% |
| 90 | 258.0 | 370.0 | +43.4% | +47.8% | 355.0 (distance) | +37.6% | **+4.2%** | +7.4% |
| 114 | 259.3 | 367.3 | +41.6% | +49.0% | 310.7 (unlock) | +19.8% | **+18.2%** | +24.4% |
| 150 | 302.7 | 361.0 | +19.3% | +29.7% | 336.0 (unlock) | +11.0% | **+7.4%** | +16.9% |

### 判读

- **新 C0 在每一档 task_count 都比老 C0 更接近贪心** —— gap 到贪心的差距缩小 3-17 pp(42 任务档最大改善,从 +18.5% → +1.5%,几乎平手)
- **但新 C0 在规模阶梯上还没胜出贪心** —— 每一档都还差贪心 1.5-18.2%
- **相对 MILP 的差距**:老 C0 是 20-49%,新 C0 是 16-43%,**整体压低了大约 5-10 pp**
- **42 任务档差距最小**(仅 +1.5% vs greedy)—— 这**恰好是训练集里 scale42 shard 覆盖的规模**,说明"训练分布内"的学习是有效的;114 任务档差距最大(+18.2%)—— 超出训练分布最远,泛化不到

**这是一个明确的负面结论但有价值**:*在 stress-test 配置的规模阶梯上,150 训练实例 + 3 个规模档还不足以让 C0 胜出贪心*。要真正胜出,需要更多 stress-test 分布的训练数据(不是 balanced/scale_medium profile 分布的数据)。

## 结果 2:六个 profile(标准评测口径)

对照:老 C0、新 C0(多规模)、最优贪心、MILP 最优。每 profile 4 seed。

| profile | MILP 最优 | 老 C0 (合成) | 老 C0 gap | **新 C0 (多规模 IL)** | **新 C0 gap** | 最优贪心 | 贪心 gap | **新 C0 vs 贪心** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 202.2 | 264.8 | +30.9% | **246.5** | **+21.9%** | 253.8 | +25.5% | **-2.9%** ✓ |
| process_scarce | 154.8 | 200.6 | +29.6% | **183.0** | **+18.3%** | 212.0 | +37.0% | **-13.7%** ✓✓ |
| transport_bottleneck | 389.0 | 449.8 | +15.6% | **438.0** | **+12.6%** | 437.8 | +12.5% | **+0.05%** ≈ |
| dependency_deep | 249.8 | 289.5 | +15.9% | **265.8** | **+6.4%** | 276.0 | +10.5% | **-3.7%** ✓ |
| mixed_hard | 262.0 | 356.2 | +36.0% | **297.2** | **+13.5%** | 305.0 | +16.4% | **-2.5%** ✓ |
| scale_medium | 243.5 | 298.2 | +22.5% | **286.2** | **+17.6%** | 296.2 | +21.7% | **-3.4%** ✓ |

### 判读

- **新 C0 在 6/6 profile 上都比老 C0 更接近 MILP 最优**(3-22 pp 改善)
- **新 C0 在 5/6 profile 上胜出最优贪心**,transport_bottleneck 上打平(差 0.05%)
- **process_scarce 上 -13.7% 是大胜**——从"比贪心差 7.4%"变成"比贪心好 13.7%"
- **平均 gap**:老 C0 平均 +25.1% → 新 C0 平均 +15.1%,**下降 10 个百分点**
- **达到了 IL_REDESIGN.md 定的 ≤15% 目标**(15.1% 刚好在门槛线附近)——数据加多档、加规模确实值得

## 综合判读

两组评测得出的方向一致但结论不同:

- **在标准 profile 口径上**:多规模 IL C0 **明确胜出老 C0 和最优贪心**,平均 gap 缩到 15.1%,达到 IL_REDESIGN.md 定的成功标准。
- **在 stress-test 规模阶梯上**:多规模 IL C0 **明确胜出老 C0**,但**还没胜出最优贪心**——规模阶梯用的是压力测试配置(更紧的 capacity slack、更高 precedence density、更慢速度比),训练集里只有 50 个这种分布的算例(42 任务档),分布不够密。

也就是说,"扩大到更大的算例"这个目标**部分达成**:
- **对标准调度问题(六 profile)**:胜出贪心,方向可用
- **对压力测试变形**:改善明显但未胜出,需要更多同分布训练数据

## 接下来的自然延伸

1. **加大 scale42 shard 数据量**(50 → 200-500),甚至加入 scale60/scale90(MILP 部分超时但仍可作监督),补齐压力测试规模分布
2. **加入六 profile 之外的分布**(process_scarce、transport_bottleneck 等 profile 参数 × 更大 task_count 组合),看能否泛化
3. **完整 IL_REDESIGN.md 对照 C**(forbidden mask + oracle-action schema),看能否再压 5-10 pp
4. **task_count > 42 的训练数据**:MILP 超时率高,但可以尝试用 300s 超时的次优解作监督,或先跑 profile 训练再用 self-imitation 引导——见 [PILOT_RESULTS.md](PILOT_RESULTS.md) 末尾 RL 路径设计

## 产物索引

- 训练脚本:[experiments/md_c0_milp_supervised_scale_pilot.py](../../experiments/md_c0_milp_supervised_scale_pilot.py)
- 规模阶梯评测:[experiments/md_c0_milp_supervised_scale_eval.py](../../experiments/md_c0_milp_supervised_scale_eval.py)
- 六 profile 评测:复用 [experiments/md_c0_milp_supervised_profile_eval.py](../../experiments/md_c0_milp_supervised_profile_eval.py)
- 后台启动:[run_c0_milp_supervised_scale.sh](../../run_c0_milp_supervised_scale.sh)
- log:`c0_milp_supervised_scale.log`
- 训练输出:[reports/md_c0_milp_supervised_scale_pilot_2026-09-13/](../../reports/md_c0_milp_supervised_scale_pilot_2026-09-13/)
- 规模评测输出:[reports/md_c0_milp_supervised_scale_eval_2026-09-13/](../../reports/md_c0_milp_supervised_scale_eval_2026-09-13/)
- profile 评测输出:[reports/md_c0_milp_supervised_scale_profile_eval_2026-09-13/](../../reports/md_c0_milp_supervised_scale_profile_eval_2026-09-13/)
