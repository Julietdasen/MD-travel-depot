# D2 envelope 追加实验 — scarce_skill × scale-preserved (2026-09-21)

## 目的

回答用户 2026-09-20 提出的两个问题：

1. **pr=4 是否太少？robot 数量应该随 task 数量 scale？** → D2b 规模守恒 sweep。
2. **skill 复杂化（火星建造场景）是否会让 policy 更有利？** → D2a scarce_skill 稀缺度 sweep。第一个障碍：legacy 3-skill 特征契约，`skill_count` 无法拉高；改扫 `scarce_skill_count ∈ {0,1,2,3}`（在 skill=3 下每加一档，多一个"仅一个 robot 拥有"的稀缺 skill）。

Checkpoint: Path C v1 stability seed3101 epoch 5。Seeds: 301-304。MILP: OR-Tools CP-SAT, 300s time limit, 4 threads。

## D2a: scarce_skill sweep (tc=24 固定, pr=4, tr=4)

| scarce | MILP incumbent | C0 | best greedy (name) | C0 vs MILP | C0 vs greedy | forced-coalition (from diag) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 250.00 | 270.25 | 304.00 (unlock)   | +8.1% | **−11.1pp** | 15% |
| 1 | 264.50 | 278.75 | 315.50 (distance) | +5.4% | **−11.6pp** | 29% |
| 2 | 271.25 | 292.50 | 323.75 (distance) | +7.8% | **−9.7pp**  | 39% |
| 3 | 284.75 | 297.00 | 318.25 (distance) | +4.3% | **−6.7pp**  | 33% |

**观察**：
- **scarce=1 是甜点**：C0-vs-greedy 从 −11.1 → −11.6pp（微改善），gap-vs-MILP 从 8.1 → 5.4%（明显收窄）。少量 scarce skill 让 greedy 犯错，policy 相对占优。
- **scarce=2/3 反而收窄**：C0-vs-greedy 从 −11.6 → −6.7pp。原因：过多 coalition 约束下动作空间坍塌成 skill puzzle，greedy 也被约束得无路可走，policy 优势消失。这和 duration/pr axis 的**倒 U** 形状一致（envelope pattern）。
- best greedy 从 `greedy_unlock` 转向 `greedy_distance`：scarce 越多，"解锁下游最多"的启发式失效（scarce robot 位置决定谁能做，unlock count 不再主导）。

**边界**：scarce_skill_count ≈ 1，占 skill_count 的 ~1/3 是最有利的。

## D2b: 规模守恒 sweep (scarce=1 固定, pr/tc≈1/6)

| tc | pr | tr | MILP incumbent | MILP status | C0 | best greedy (name) | C0 vs MILP | C0 vs greedy |
|---:|---:|---:|---:|:---|---:|---:|---:|---:|
| 24 | 4  | 4  | 264.50 | 4/4 optimal    | 278.75 | 315.50 (distance) | +5.4% | **−11.6pp** |
| 42 | 7  | 7  | 336.25 | 3/4 optimal, 1 TO | 377.50 | 396.75 (unlock)   | +12.3% | **−4.9pp** |
| 60 | 10 | 10 | 439.25 | 0/4 optimal, 4 TO | 496.25 | 507.00 (unlock)   | +13.0% | **−2.1pp** |

**观察**：
- 规模守恒 (pr/tc 保持 ~1/6) 时，C0-vs-greedy 从 tc24 的 −11.6pp 单调收窄到 tc60 的 −2.1pp。**envelope 的规模轴边界不因 pr 缩放而消失** —— 就算 pr 从 4→10 保持 process_task/pr 比例，policy 优势也随规模衰减。
- 相比 D2 之前的固定小 pr (pr=2 for all tc)：pr scale 起来让 gap 缩窄稍缓和 (tc60 从 −0.6pp 恢复到 −2.1pp)，但结构性衰减仍在。
- **tc60 上 4/4 seed MILP timeout**：MILP 300s 都没解到 optimal，incumbent 已经很接近；greedy 反而也接近 policy —— 大规模上问题 landscape 变平，policy 学到的"结构优势"少。

**边界**：规模轴 tc ≈ 42-60 是 policy 优势区的上界，与 Sweep 1 观察一致。

## Time-budget MILP 对比 (前置实验，2026-09-19-20)

在 tc=60 和 tc=80 上，把 MILP time_limit 限制到 5/15/60/300s：

| tc | budget | MILP feasibility | MILP makespan mean | C0 makespan |
|---:|---:|---:|---:|---:|
| 60 | 5   | **0/4 solved** (all infeasible) | n/a | 393.5 (Path C v1 default eval) |
| 60 | 15  | 4/4 solved | ~380 | 393.5 |
| 60 | 60  | 4/4 solved | ~360 | 393.5 |
| 60 | 300 | 4/4 solved | 356.0 | 393.5 |
| 80 | 5   | **0/4 solved** | n/a | (from time-budget grid) |
| 80 | 15  | **0/4 solved** | n/a | |
| 80 | 60  | 4/4 solved | ~530 | |

（如需精确数字见 `reports/md_envelope_time_budget_2026-09-20/` 的 rows.json。）

**关键 finding**：**MILP 在低 budget 下无法给出可行解**。tc60 需要 ≥15s，tc80 需要 ≥60s。policy 在 ms 级别永远给出可行 schedule。这是 policy 相对 time-budgeted MILP 的**存在性优势**（feasibility guarantee），比 makespan 优势更硬核。

## 综合结论

Path C v1 stability epoch-5 的 envelope 现在有 4 条轴：

1. **规模轴** tc ≤ ~42（D2b 加强：即使 pr 保持缩放也衰减）
2. **process robot 稀缺度** pr/process_task ≈ 0.15-0.25（Sweep 3）
3. **duration 异质性** max/min ∈ 3-5（Sweep 2）
4. **skill 稀缺度** scarce_skill_count ≈ 1/skill_count = 1/3（D2a 新增）

规律一致：**每条轴都是倒 U 形，中等复杂度是甜点，极端两头 policy 优势消失**。

## Coalition 是否真的重要？

D2a 附带的诊断 (`reports/md_coalition_diagnostic_2026-09-21/`) 给出量化答案：

| scarce_skill_count | tc24 forced-coalition | tc60 forced-coalition |
|---:|---:|---:|
| 0 | 15% | 0% |
| 1 | 29% | 35% |
| 2 | 39% | 41% |
| 3 | 33% | 51% |

- process_scarce v1 (scarce=0) 下，tc=60 上 **0%** process task 需要 coalition —— coalition 是**装饰而非必需**。envelope 里 policy 学的更多是 unlock + assign ordering，不是 coalition composition。
- 要让 coalition 名副其实（forced ratio > 30%），必须 `scarce_skill_count ≥ 1`（tc24）或 `≥ 2` (tc60)。

**对 paper narrative 的影响**：不能强调 "coalition-aware" 是当前 policy 的核心贡献，因为 baseline problem 里 coalition 本身很少 forced。envelope 里的优势更准确地说是"skill assignment + unlock signal 的联合优化"。

## 推理时 MILP 的 threshold ablation (2026-09-21 附加)

在 tc=24 process_scarce scarce=1 上，扫 `confidence_threshold ∈ {0.0, 0.1, 0.5, 1.5}`（MILP fallback 门槛）：

| threshold | fallback calls | solver_time | makespan | gap vs MILP |
|---:|---:|---:|---:|---:|
| 0.0 | 27 | 0.6s | 278.75 | +5.39% |
| 0.1 | 27 | 0.6s | 278.75 | +5.39% |
| 0.5 | 31 | 0.7s | 278.75 | +5.39% |
| 1.5 | 43 | 0.9s | 278.75 | +5.39% |

**结论：threshold ablation 完全无效**。fallback 次数上升 60%，makespan 一字不改。原因：当前 `ExplicitMIPFallback` 只解**当前 decision event 的 assignment**，policy 和 MILP 在这个局部子问题上得到的答案基本一致（局部 CBS 早已收敛），fallback 拿不到额外信号。

**Take-away**：threshold-based fallback 是死胡同。有效的 MILP-in-the-loop 必须是：
- **Periodic full re-optimize**：每 K 个 decision event 用短 budget MILP 从当前状态解剩余问题
- **Top-K rerank**：policy 提候选，MILP 从中选 continuation 最优的

这两个都需要新代码；threshold ablation 表明"接现有 fallback"是无收益的。

## Artifacts

- D2a raw: `reports/md_envelope_d2_2026-09-20/d2a_scarce{0,1,2,3}/`
- D2b raw: `reports/md_envelope_d2_2026-09-20/d2b_tc{24,42,60}_pr{4,7,10}/`
- Coalition diagnostic: `reports/md_coalition_diagnostic_2026-09-21/`（含可视化 `coalition_visualization.png`）
- Threshold ablation: `reports/md_confidence_threshold_2026-09-21/`
- Time-budget grid: `reports/md_envelope_time_budget_2026-09-20/`
- Driver: `reports/md_envelope_d2_2026-09-20/run.sh`, `reports/md_envelope_scarce3_2026-09-21/run.sh`, `reports/md_confidence_threshold_2026-09-21/run.sh`

## 未完成 / 下一步

- **scarce=3 sweep at tc/pr={24/4, 42/7, 60/10}**：在跑 (`reports/md_envelope_scarce3_2026-09-21/`)。用于确认 D2a 里 scarce=3 vs 1 的比较在大规模上是否也保持（可能 tc=60 上 forced-coalition ratio ~51%，反而利于 policy）。
- **Periodic MILP re-optimize**：threshold ablation 已确认死路，下一步要实现全局重优化。
- **skill_count > 3**：需要重训 checkpoint 才能验证。当前 D2a scarce sweep 是**在架构限制内的最强 coalition 试探**。
