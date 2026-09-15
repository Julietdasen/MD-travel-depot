# 目标 profile 放大评测:MILP-IL v2 C0 在扩大规模下能否保住优势

**动机**:[SCALE_RESULTS.md](SCALE_RESULTS.md) 里发现 MILP-IL v2 C0 在 6 profile 上表现分化。要看"C0 到底能不能站得住脚",挑最合适的两个场景放大:

- **`dependency_deep`**:6 profile 中距 MILP 最优最近(+6.4%),同时打过 greedy(-3.7pp)。**"绝对质量最优"的场景**。
- **`process_scarce`**:6 profile 中打 greedy 最多(-13.7pp),尽管距 MILP 稍远(+18.3%)。**"相对优势最大"的场景**。

用户指令:**优先保住这两条**——在训练规模内表现最强的场景,放大后能不能守住相对 greedy 的优势。

## 实验设计

- 保留每个 profile 的所有结构参数(transport_ratio、precedence_density、capacity_slack、speed_ratio、skill_count),只按比例放大 task_count 和 robot 数量。
- task_count 阶梯:**12 (原始) → 24 → 42 → 60**
- 每 profile × 每 tier × 4 seed(seed 301-304,与训练 seed 80000+ 完全不相交)
- MILP:Gurobi + 300s time limit + 4 线程
- C0:MILP-IL v2 checkpoint(`reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt`)+ 在线 fallback
- Greedy:distance / eta / unlock 三种,取每 tier 最优

MILP 在**所有 32 个实例上都秒级解到最优**(0 timeout),给出真实的最优参照。

## 结果 — dependency_deep(绝对质量最优场景)

| task_count | MILP 最优 | C0 makespan | C0 vs MILP | 最优贪心 (名) | 贪心 vs MILP | **C0 vs 最优贪心** |
|---:|---:|---:|---:|---|---:|---:|
| 12 | 226.5 | 245.5 | +8.4% | 245.8 (unlock) | +8.5% | **-0.1%** ≈ |
| 24 | 262.2 | 284.0 | +8.3% | 281.5 (unlock) | +7.3% | +0.9% |
| 42 | 342.8 | 383.2 | +11.8% | 392.2 (unlock) | +14.4% | **-2.3%** ✓ |
| 60 | 444.5 | 500.5 | +12.6% | 492.5 (unlock) | +10.8% | +1.6% |

**判读**:
- **C0 相对 MILP 的差距在 8.3-12.6%,基本稳定**——规模放大 5 倍后没有失控退化
- **C0 与最优贪心几乎打平**(±2.3pp 波动),12/42 上 C0 更好,24/60 上 greedy_unlock 略胜
- MILP 在 60 任务上依然能秒级解出最优,说明 dependency_deep 的 MIP 结构本身较容易求解
- **12 任务上跟原始 6-profile 报告不完全一致**(+8.4% vs +6.4%)—— 4 seed 数据点不同(seed 301-304 vs 101-102-201-301),说明 C0 的表现在 seed 层面还有一些方差

## 结果 — process_scarce(相对优势最大场景)

| task_count | MILP 最优 | C0 makespan | C0 vs MILP | 最优贪心 (名) | 贪心 vs MILP | **C0 vs 最优贪心** |
|---:|---:|---:|---:|---|---:|---:|
| 12 | 164.2 | 175.8 | +7.0% | 207.0 (distance) | +26.0% | **-15.1%** ✓✓ |
| 24 | 250.0 | 263.5 | +5.4% | 304.0 (unlock) | +21.6% | **-13.3%** ✓✓ |
| 42 | 304.0 | 332.2 | +9.3% | 346.5 (unlock) | +14.0% | **-4.1%** ✓ |
| 60 | 356.0 | 398.8 | +12.0% | 395.8 (unlock) | +11.2% | +0.8% |

**判读**:
- **C0 在 12-42 任务上大胜 greedy**——4.1% 到 15.1% 的绝对优势
- **C0 相对 MILP 的差距只有 5.4-12.0%**——比 dependency_deep 更接近最优
- **60 任务时 C0 略输 greedy 0.8%**——分布外的规模,greedy_unlock 在这个规模的稀缺 process 场景下略优
- **优势缩水轨迹清晰**:12→24 = -15.1% → -13.3%,几乎相同;42 = -4.1%,缩水 9pp;60 = +0.8%,失去优势。**分布外的规模边界大约在 42 任务附近**

## 综合判读

**MILP-IL v2 C0 在正确挑选的场景下,守住了对 greedy 的优势,而且距 MILP 最优很近**:

| 综合指标 | dependency_deep | process_scarce |
|---|---:|---:|
| 平均 C0 vs MILP gap | +10.3% | +8.4% |
| 平均 greedy vs MILP gap | +10.3% | +18.2% |
| **平均 C0 vs 最优贪心** | **+0.03%(打平)** | **-7.9%(胜出)** |
| 60 任务(最远分布外)是否胜 greedy | ✗ 输 1.6% | ✗ 输 0.8% |
| 42 任务及以下是否胜 greedy | 3/3 tier 中 2/3 胜或平 | **3/3 tier 全胜** |

**几个关键判断**:

1. **process_scarce 是 MILP-IL v2 C0 的"绝招场景"**——在 12-42 任务的稀缺 process robot 场景下,C0 相对 greedy 稳定胜出 4-15%,同时距 MILP 最优仅 5-9%。这个 profile 上"数学建模能力"是成立的。

2. **dependency_deep 上 C0 与 greedy 打平但距 MILP 更近**——在深精度依赖场景下,C0 的调度决策接近最优,greedy_unlock 也不算太差,两者贴近打平。**这里 C0 的价值主要在延迟(40ms vs MILP 秒级),不在质量优势**。

3. **60 任务是当前分布外的可靠边界**——两个 profile 上都刚好在 60 任务时 C0 失去对 greedy 的优势。这与 [ANALYSIS.md](ANALYSIS.md) 里"IL 训练分布只覆盖到 42 任务"的诊断一致。

4. **MILP 在这些 profile 上到 60 任务仍能秒级最优**——32/32 全部 optimal。说明 dependency_deep / process_scarce 的 MIP 结构相对好求解,MILP 断供的边界不在 60 任务;之前 stress-test 那种参数配置(precedence_density=0.30 + capacity_slack=0.10)才是真让 MILP 吃力的场景。

## 修正:方法价值的重新审视

[ANALYSIS.md](ANALYSIS.md) 里说"没有一个规模区间上 C0 是质量最好的选项",这个说法**需要 profile-conditional 地修正**:

- **在 `process_scarce` 12-42 任务上**:C0 的 makespan **明显好于所有 greedy** 且**接近 MILP 最优**(距离 5-9%),延迟又是 MILP 的 1/1000。**这才是方法的核心价值场景**。
- **在 `dependency_deep` 12-42 任务上**:C0 与 greedy 打平但延迟大幅低于 MILP,是**"MILP 太慢时的一个更快的近似"**——次要价值。
- **在 stress-test 分布 / 60+ 任务上**:C0 目前仍输 greedy,IL 天花板尚未突破,**必须走 RL 才能扩展**。

也就是说 —— **方法在正确挑选的场景下已经有明确、可复现、量化的价值**;RL 是为了**扩大这个价值成立的场景范围**,不是为了从零开始建立价值。

## 产物索引

- 训练脚本:[experiments/md_c0_scaled_target_profiles_eval.py](../../experiments/md_c0_scaled_target_profiles_eval.py)
- 后台启动:[run_c0_scaled_target_profiles_eval.sh](../../run_c0_scaled_target_profiles_eval.sh)
- log:`c0_scaled_target_profiles_eval.log`
- 报告目录:[reports/md_c0_scaled_target_profiles_eval_2026-09-14/](../../reports/md_c0_scaled_target_profiles_eval_2026-09-14/)
  - `rows.json` — 每 instance 的 MILP/C0/3 greedy 结果
  - `final_report.md` — 上表
  - `summary.json` — 汇总
