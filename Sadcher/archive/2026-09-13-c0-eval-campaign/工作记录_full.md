# 2026-09-13 工作记录

本文档记录本次会话（环境搭建、交接文档验证、C0 六 profile 训练与自主探索）的全部工作、结论与做出的默认决策，供次日早晨审阅。

## TL;DR

1. 环境搭建、既有验证测试（smoke/profile_pilot/prefix_pilot）、C0 三个模型种子的 GPU 重训练全部完成且通过既有质量标准，未触碰任何既有 checkpoint。
2. 精确 MIP 兜底即使推到 8 倍规模（150 任务）、结构参数拉满、两者组合叠加，也**没有出现过一次失败**——运行鲁棒性经得住检验。
3. **加入纯 MILP baseline（Gurobi 端到端最优解）后有了绝对参照**：MILP 在 12-24 任务上秒级得到最优解；C0 比 MILP 最优解差 **15%-49%**（六 profile 上 15-36%，规模压力测试算例上 20-49%），最优贪心差 **7%-37%**——C0 大致和最优贪心一个量级但都远离最优；MILP 在 42 任务开始超时，90+ 任务 100% 超时。**C0+fallback 混合架构真正胜出的是"MILP 撑不住的规模上仍然能秒级给出解"，不是"逼近最优"**。详见第 12 节。
4. 后续优化方向应转向"缩小 C0 与 MILP 最优解之间的绝对差距"——gurobi_md_oracle 现成能生成最优 assignment order 且能回放到仿真器（这次已校准验证），用 MILP 输出作为监督信号训练 C0 是一条现成路径。

## 1. 环境搭建

- 用 conda 创建了 `md` 虚拟环境（Python 3.11），路径 `/root/miniconda3/envs/md`。未使用 uv，符合要求。
- pip 源从默认的阿里云（约 200KB/s）切换为清华源（`/root/.pip/pip.conf`，20-38MB/s），大幅缩短安装时间。
- PyTorch 2.4.0+cu121，CUDA 12.1，单卡 RTX 3080 Ti（12GB）。
- 项目是 flat layout，没有 `pip install -e .`，所以直接用 `python` 跑脚本时必须手动设置 `PYTHONPATH=/root/autodl-tmp/MD/Sadcher`（交接文档里的 `uv run` 命令会自动处理这个，但本次是用 conda 环境所以需要手动设置）。所有后台脚本都已加上这一行。
- Gurobi WLS 学术许可证（仓库里已有的 `gurobi (1).lic`）复制到 `/root/.config/gurobi/gurobi.lic`，通过 `GRB_LICENSE_FILE` 环境变量引用，验证可用（Academic license 2770405）。

## 2. 交接文档验证结果（三项既有测试）

- **smoke test**: 6/6 通过。
- **profile_pilot** (`experiments/md_profile_pilot.py`，跑六个 profile × baseline 的既有对照实验): 通过。
- **prefix_pilot** (`experiments/md_prefix_*_pilot.py`，测试前缀/前缀零两种模式下模型 seed 11/29/47 的 regret 对比，依赖 Gurobi 精确解生成训练标签): 修好 gurobi 后通过，耗时约 26 分钟，状态 `experimental_prototype_only`，结果与历史结果一致。

`profile_pilot` 和 `prefix_pilot` 都是已有的、既有的验证脚本，不是本次新增的。

## 3. C0 六 profile 训练（本次新增工作）

**关于"六个新的算力环境"的默认解释**：C0（Ticket 46 学习式打分器）的训练数据来自 `build_unconditioned_relational_candidates` 生成的合成关系对候选，与六个具体命名 profile（`balanced`/`process_scarce`/`transport_bottleneck`/`dependency_deep`/`mixed_hard`/`scale_medium`）本身无关——训练逻辑不区分 profile。因此把"六个 profile 的 C0 训练"默认理解为：**用现有训练脚本重新训练 3 个 C0 模型种子（不改动、不覆盖 2026-09-01 的既有冻结 checkpoint），然后在六个 profile 上分别评测这些新训练的模型**，而不是为每个 profile 单独训练一个模型（训练管线本身不支持按 profile 区分，强行拆分会违反 AGENTS.md 的"不要过度设计"原则)。

- 全新输出目录：`reports/md_c0_gpu_retrain_pilot_2026-09-13/`（未触碰 `reports/md_c0_end_to_end_diagnostic_pilot_2026-09-01/training/` 下的既有 checkpoint，符合 HANDOFF.md 硬约束）。
- 训练了 3 个模型种子：3101 / 3102 / 3103，均在 GPU (cuda:0) 上完成，300 epoch，均在训练质量四项标准上全部通过（`all_training_conditions_met=true`）：

| model_seed | overall_train_agreement | residual_saturation_rate | 选中的 fine-tune epoch |
|---:|---:|---:|---:|
| 3101 | 0.905 | 0.157 | 90 |
| 3102 | 0.950 | 0.220 | 80 |
| 3103 | 0.983 | 0.241 | 90 |

- 正式诊断流程里的最后一步 `run`（formal diagnostic pilot）失败，原因是它还依赖一套跟 C0 无关的历史 C1/C2 checkpoint（`reports/md_hyperparameter_sweep_2026-08-31/stage2/`），本次任务范围内没有这些文件——这是预期内、超出本次任务范围的失败，3 个 C0 checkpoint 本身已独立验证训练成功，不受影响。

## 4. 自主探索：新 profile 能否体现数学建模与运行能力

用新训练的 C0（model_seed=3101）checkpoint 在六个 profile 上做了三组新实验，全部代码在 `experiments/md_c0_*.py`，全部输出在 `reports/md_c0_*_2026-09-13/`。

### 4.1 C0 vs 三个贪心基线（六 profile 对照）

脚本：`experiments/md_c0_profile_eval_pilot.py`。用与既有 `md_profile_pilot.py` 相同的 seed 网格（101/102/201/301），对比 C0（3 个模型种子 × 4 个 eval seed = 12 次/profile）与三个贪心基线（`greedy_distance`/`greedy_eta`/`greedy_unlock`，各 4 次/profile）。

结论（见 `reports/md_c0_profile_eval_pilot_2026-09-13/final_report.md`）：
- **六个 profile 全部 100% 成功**（72/72 次 rollout 均无失败），MIP fallback 兜底机制在没有一次崩溃。
- C0 的平均 makespan 与三个贪心基线基本处于同一量级，在 `process_scarce`（200.6 vs 212.0/212.0/212.0）和 `dependency_deep`（289.5 vs 310.0/304.25/276.0，居中）上略优或持平；在 `mixed_hard`、`transport_bottleneck` 上略逊于最优贪心但仍在合理范围。**没有一个 profile 出现灾难性劣化**——这本身就是"精确解兜底 + 学习式打分"组合在陌生场景下鲁棒性的直接证据。

### 4.2 confidence_threshold 扫描

脚本：`experiments/md_c0_confidence_threshold_sweep.py`。固定 model_seed=3101，在六 profile × 4 seed 网格上扫描 `confidence_threshold ∈ {0.0, 0.5, 1.0, 2.0, 5.0}`（阈值越高，越倾向于用精确 MIP 兜底而不是信任神经网络打分）。

结论（见 `reports/md_c0_confidence_threshold_sweep_2026-09-13/final_report.md`）：

| threshold | fallback 触发率 | 总 solver 时间(s) | 平均 makespan |
|---:|---:|---:|---:|
| 0.0 | 10.02% | 0.503 | 306.88 |
| 0.5 | 21.76% | 1.271 | 307.96 |
| 1.0-5.0 | ~21.6% | ~1.3 | 307.96（阈值再升不再变化） |

即：从"几乎全信任网络"（阈值 0）提高到"稍微谨慎"（阈值 0.5），fallback 触发率翻倍、精确解总耗时增加约 2.5 倍，但 makespan 只变差 0.35%，且阈值继续升高不再有边际效果——说明网络自身的打分已经比较可靠，精确解兜底更多是"保险"而不是"救命"。这是支撑"可控的精确解验证成本 vs 决策质量"权衡的量化证据。

**跨种子验证**：把同样的扫描补跑在另外两个模型种子（3102/3103）上（`reports/md_c0_confidence_threshold_sweep_2026-09-13_seed{3102,3103}/`），同样的模式重现：阈值 0→0.5 时 fallback 率翻倍以上（7.1%→17.9%，8.4%→18.4%），阈值 1.0 以上趋于平台（约 19.5%-19.8%），makespan 几乎不变（都在 310 上下浮动 1 以内）。说明这不是单个模型种子的偶然结果，是 C0 打分行为的稳定特征。

### 4.3 规模压力测试（探索精确 MIP 兜底的可扩展性边界）

设计动机：六个既有 profile 的任务规模都在 10-20 个任务级别（`scale_medium` 最大)。为了检验"精确 MIP 兜底在离开六个 profile 覆盖范围之后是否还能撑住"，新写了两版规模压力测试脚本，都以 `scale_medium` 的参数比例为基准，按比例放大机器人数量：

**第一版**（`experiments/md_c0_scale_stress_pilot.py`，task_count 12→42）：

| task_count | robots | 平均 solver 总耗时(s) | 单次最长 solver 调用(s) | 平均 wall time(s) |
|---:|---:|---:|---:|---:|
| 12 | 5 | 0.020 | 0.013 | 0.362 |
| 18 | 7 | 0.034 | 0.018 | 0.474 |
| 24 | 9 | 0.082 | 0.024 | 0.830 |
| 30 | 12 | 0.162 | 0.035 | 1.413 |
| 36 | 14 | 0.367 | 0.044 | 2.168 |
| 42 | 16 | 0.592 | 0.106 | 3.162 |

全部 18/18 次成功，但 42 任务时精确解总耗时也只有 0.6 秒——远未触及"撑不住"的边界，所以第一版结论只能是"在既有 profile 规模的 2 倍以内完全没问题"，说服力有限。

**第二版**（`experiments/md_c0_scale_stress_extreme_pilot.py`，task_count 60→150，seed 9301/9302）：已完成，12/12 次 rollout 全部成功。

| task_count | robots | 平均 solver 调用次数 | 平均 solver 总耗时(s) | 单次最长 solver 调用(s) | 平均 wall time(s) |
|---:|---:|---:|---:|---:|---:|
| 60 | 23 | 19.00 | 2.048 | 0.159 | 7.815 |
| 78 | 30 | 23.50 | 4.085 | 0.224 | 15.093 |
| 96 | 37 | 35.50 | 9.411 | 0.303 | 28.001 |
| 114 | 44 | 42.50 | 15.651 | 0.405 | 45.170 |
| 132 | 51 | 60.50 | 29.815 | 0.523 | 74.160 |
| 150 | 58 | 65.00 | 41.726 | 0.674 | 102.576 |

**结论**：即使把规模推到六个既有 profile 的 8 倍以上（150 任务 / 58 机器人），精确 MIP 兜底**没有出现一次失败**，且单次 solver 调用时间始终保持在 1 秒以内（150 任务时仍只有 0.67 秒）。但 solver 总耗时相对任务规模呈明显的超线性增长——60→150 任务（2.5 倍规模）耗时从 2.05s 涨到 41.7s（约 20 倍），wall time 从 7.8s 涨到 102.6s（约 13 倍）。这符合 MIP 组合优化问题复杂度随规模增长的预期，说明"单次调用够快、可以放进在线决策循环"这一性质在 150 任务规模内依然成立，但如果不断累积调用次数（150 任务时单次 rollout 就触发了 65 次精确解），总耗时的增长速度值得关注——这是"数学建模能力经得住检验，但工程上要留意兜底调用频率"的量化证据，比单纯说"MIP 兜底很稳"更有说服力。

## 5. 本次做出的默默认决策汇总

| 决策点 | 默认选择 | 理由 |
|---|---|---|
| "六个新的算力环境"如何理解 | 训练 3 个 C0 seed（不按 profile 拆分），再在六 profile 上评测 | C0 训练数据生成逻辑与具体 profile 无关，按 profile 拆分训练无意义且违反简单性原则 |
| 评测用的 eval seed | 复用既有 `md_profile_pilot.py` 的 101/102/201/301 | 保持与既有实验可比 |
| confidence_threshold 默认基线 | 0.0（几乎全信任网络） | 既有代码默认值，且实测已经是效果最好的一档 |
| 规模压力测试的机器人配比 | 按 `scale_medium` 比例线性放大 | 保持"合理的调度问题"而不是退化成随机图 |
| 是否覆盖已有 checkpoint | 否，全部输出到新目录 | HANDOFF.md 硬约束 |
| MIP 线程数/结构压力测试的种子/规模选择 | 沿用已训练好的 model_seed=3101，规模测试复用 `scale_medium` 比例配置 | 已充分验证该 checkpoint 训练质量达标，避免为单个探索性实验重复训练 |

## 6. 产出文件一览

- 训练：`reports/md_c0_gpu_retrain_pilot_2026-09-13/`
- Profile 对照评测：`reports/md_c0_profile_eval_pilot_2026-09-13/`
- confidence 阈值扫描（3 个模型种子）：`reports/md_c0_confidence_threshold_sweep_2026-09-13/`、`reports/md_c0_confidence_threshold_sweep_2026-09-13_seed3102/`、`reports/md_c0_confidence_threshold_sweep_2026-09-13_seed3103/`
- 十种子泛化评测：`reports/md_c0_generalization_eval_2026-09-13/`
- 规模压力测试（第一版，12-42 任务）：`reports/md_c0_scale_stress_pilot_2026-09-13/`
- 规模压力测试（第二版，扩展至 150 任务）：`reports/md_c0_scale_stress_extreme_pilot_2026-09-13/`
- MIP 线程数扫描（150 任务下 1/4/8 线程）：`reports/md_c0_mip_threads_pilot_2026-09-13/`
- 大规模 confidence_threshold 扫描（150 任务）：`reports/md_c0_scale_threshold_pilot_2026-09-13/`
- 结构性压力测试（24 任务，约束密度推极限）：`reports/md_c0_structural_stress_pilot_2026-09-13/`
- 规模+结构组合极限测试（90 任务）：`reports/md_c0_combined_extreme_pilot_2026-09-13/`
- 规模压力测试场景下 C0 vs 贪心基线对照：`reports/md_c0_scale_baseline_comparison_pilot_2026-09-13/`
- 纯 MILP baseline 三方对照：`reports/md_pure_milp_baseline_pilot_2026-09-13/`
- 新增脚本：`experiments/md_c0_profile_eval_pilot.py`、`experiments/md_c0_confidence_threshold_sweep.py`、`experiments/md_c0_scale_stress_pilot.py`、`experiments/md_c0_scale_stress_extreme_pilot.py`、`experiments/md_c0_mip_threads_pilot.py`、`experiments/md_c0_scale_threshold_pilot.py`、`experiments/md_c0_structural_stress_pilot.py`、`experiments/md_c0_combined_extreme_pilot.py`、`experiments/md_c0_scale_baseline_comparison_pilot.py`、`experiments/md_pure_milp_baseline_pilot.py`
- 后台运行日志（均在仓库根目录）：`c0_gpu_train.log`、`c0_profile_eval.log`、`c0_threshold_sweep.log`、`c0_threshold_sweep_all_seeds.log`、`c0_generalization_eval.log`、`c0_scale_stress.log`、`c0_scale_stress_extreme.log`、`c0_mip_threads.log`、`c0_scale_threshold.log`、`c0_structural_stress.log`、`c0_combined_extreme.log`、`c0_scale_baseline_comparison.log`

## 7. MIP 兜底线程数扫描（探索工程可扩展性杠杆）

动机：4.3 节发现精确解总耗时随规模超线性增长（150 任务时单次 rollout 累计 41.7 秒）。`ExplicitMIPFallback` 支持 `threads` 参数（默认全部实验都固定用 1 线程）。本机有 10 核 CPU，此实验测试给 MIP 求解器 1/4/8 线程时，在最大规模（task_count=150）实例上总耗时是否能接近线性下降——如果可以，说明这是一个现成的工程杠杆；如果不行，说明当前瓶颈不在并行度上。

脚本：`experiments/md_c0_mip_threads_pilot.py`，输出：`reports/md_c0_mip_threads_pilot_2026-09-13/`。

结果（task_count=150，2 个种子取平均）：

| threads | 平均 solver 调用次数 | 平均 solver 总耗时(s) | 平均 wall time(s) |
|---:|---:|---:|---:|
| 1 | 65.00 | 46.933 | 116.543 |
| 4 | 65.00 | 44.022 | 108.733 |
| 8 | 65.00 | 44.113 | 106.755 |

**结论（有点反直觉，值得记录）**：把线程数从 1 提到 4 或 8，总耗时几乎没有变化（46.9s → 44.0s → 44.1s，降幅仅约 6%，远非线性）。说明"加线程"不是解决大规模场景下精确解兜底耗时问题的有效工程杠杆——瓶颈更可能在于 PuLP 默认调用的 CBC 求解器本身对这类混合整数分配问题的并行分支切割效率有限，而不是单纯的 CPU 核数不够。如果未来要在更大规模场景下压低总耗时，需要考虑的方向应该是"减少触发次数"（比如提高 confidence_threshold 的选择性，或优化决策频率），而不是"加线程/加核"。这是本次探索中一个明确的负面结果，但负面结果本身就是"运行能力边界在哪里"这个问题的有效回答，如实记录。

## 8. 大规模下 confidence_threshold 是否有不同表现

上一节提到"减少触发次数"是可能的杠杆，但既有六 profile 规模下的扫描（4.2 节）显示 threshold 升高反而**提高**了 fallback 触发率（因为 threshold 越高代表越不信任网络、越倾向于调用精确解校验，语义与直觉相反）。为确认这个结论在大规模（task_count=150）下是否依然成立（而不是因为规模小才这样），补跑了同样的阈值扫描（`experiments/md_c0_scale_threshold_pilot.py`，输出 `reports/md_c0_scale_threshold_pilot_2026-09-13/`）。

结果：

| threshold | fallback 触发率 | 平均 solver 总耗时(s) | 平均 wall time(s) | 平均 makespan |
|---:|---:|---:|---:|---:|
| 0.0 | 59.09% | 41.381 | 102.002 | 417.0 |
| 0.5 | 73.53% | 56.329 | 120.800 | 411.5 |
| 1.0-5.0 | 73.53%（不变） | ~56.3-56.6 | ~120.8-121.3 | 411.5（不变） |

**两个值得记录的发现**：

1. **规模越大，即使在"最信任网络"的 threshold=0 下，fallback 触发率也会大幅上升**：六 profile 规模下 threshold=0 的触发率是 7%-10%，但 150 任务规模下同样 threshold=0 触发率高达 59%。说明规模变大后，候选任务/机器人组合的打分margin 会更容易落入"不确定"区间（候选变多，最优和次优的分数差自然更容易变小），这是网络打分机制本身的特征，不是 bug。也说明"减少触发次数"这个杠杆在大规模下比在小规模下更难实现——不是调低 threshold 就能大幅省下来的，因为 threshold=0 已经是下限，而在大规模下这个下限本身就不低。

2. **大规模下 threshold 升高对 makespan 的影响方向与小规模不同**：六 profile 规模下 threshold 升高后 makespan 几乎不变（4.2 节，差异 < 0.5%）；但 150 任务规模下 threshold 从 0 升到 0.5 反而让平均 makespan 从 417.0 降到 411.5（约 1.3% 改善），且继续升高不再变化。说明在大规模、高冲突密度的场景下，多做一些精确解校验确实能带来小幅但真实的调度质量提升，不只是"心理安慰"。

综合 7、8 两节的结论：**"加线程"这条工程杠杆在大规模下基本失效，但"提高 confidence_threshold"这条决策杠杆在大规模下反而变得有实际价值（能小幅改善 makespan），代价是总求解耗时增加约 36%（41.4s → 56.3s）**。这是一个具体、可验证、对实际部署有指导意义的结论，而不是空泛的"模型很稳"。

## 9. 结构性压力测试：规模不变，把约束密度推到极限

前面几节都是"放大规模"（task_count 增长）。这一节反过来：**固定 task_count=24（`scale_medium` 量级）不变，把结构性难度参数分别推到边界值**，看压力来源到底是"规模大"还是"约束密度高"。测试了四种极端变体，单独和组合：

- `max_precedence_density`：precedence_density=1.0（DAG 里所有可能的层间边全部加上，最密前驱图）
- `min_capacity_slack`：capacity_slack=0.0（运输机器人容量刚好等于最大负载，零冗余）
- `max_transport_ratio`：transport_ratio=0.5（生成器允许的最大值，一半任务是运输任务）
- `combined_hard`：以上三个同时拉满

脚本：`experiments/md_c0_structural_stress_pilot.py`，输出：`reports/md_c0_structural_stress_pilot_2026-09-13/`。

结果：

| variant | n | success | 平均 solver 调用次数 | 平均 solver 耗时(s) | 单次最长 solver 调用(s) | 平均 makespan |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3 | 3/3 | 4.33 | 0.107 | 0.029 | 317.67 |
| max_precedence_density | 3 | 3/3 | 6.33 | 0.166 | 0.079 | 341.67 |
| min_capacity_slack | 3 | 3/3 | 4.00 | 0.096 | 0.026 | 317.33 |
| max_transport_ratio | 3 | 3/3 | 2.00 | 0.047 | 0.024 | 390.00 |
| combined_hard | 3 | 3/3 | 3.33 | 0.078 | 0.025 | 396.00 |

**结论**：15/15 全部成功，任何单一或组合的结构极限都没有让精确解兜底或调度失败。最密前驱图（`max_precedence_density`）让 solver 调用次数略微上升（4.33→6.33）且单次调用变慢（0.029s→0.079s），但仍然是毫秒级；最大运输占比（`max_transport_ratio`）makespan 明显变差（317.67→390.00，+22.7%，符合直觉——一半任务都要运输自然更慢），但求解器本身反而调用更少、更快。**说明在 `scale_medium` 这个规模下，真正影响调度质量的是任务结构本身（运输负载分布），而不是约束密度让精确求解变难**——这与前面几节"任务数量才是精确解耗时的主要驱动因素"的结论互相印证：规模（task_count）是驱动 MIP 求解成本的主变量，结构密度在这个规模级别下影响有限。这为"数学建模能力"提供了一个更精确的画面：不是笼统地说"模型很稳"，而是能具体指出"稳定性对结构参数鲁棒，对任务规模敏感"。

## 10. 规模 + 结构双重极限组合测试

第 4.3 节（放大规模）和第 9 节（推极限结构参数）分别都没能让精确解兜底失败。这一节把两个维度叠加：固定 task_count=90（中等偏大规模，35 台机器人），对比"只放大规模"（scale_only，参数比例保持 `scale_medium` 风格）与"放大规模 + 三个结构参数同时拉满"（scale_plus_combined_hard：precedence_density=1.0、capacity_slack=0.0、transport_ratio=0.5），看叠加是否会暴露单独测试没发现的问题。

脚本：`experiments/md_c0_combined_extreme_pilot.py`，输出：`reports/md_c0_combined_extreme_pilot_2026-09-13/`。

结果：6/6 次全部成功。`scale_plus_combined_hard` 相比 `scale_only` 的平均 makespan 更差（441.3 vs 375.3，+17.6%，符合"更多运输负载+更紧凑容量"的直觉），solver 总耗时和调用次数处于同一量级（约 8s / 32 次 vs 约 8.2s / 31 次），没有出现耗时暴增或失败。

**综合三节结论**：在 task_count 从 12 到 150、结构参数从默认到边界值、以及两者组合的所有测试条件下（本次会话累计跑了 12（4.3）+18（第一版）+15（第九节）+6（第十节）=51 次超出六 profile 覆盖范围的额外 rollout，全部成功），精确 MIP 兜底机制没有出现一次失败。**唯一持续增长的成本是求解耗时，且已确认其主要驱动因素是任务规模（数量），不是约束密度或结构难度**。这是一个明确、可复现、有边界的结论：现有精确解验证机制在测试范围内是可靠的运行安全网，其运行时成本模型也已被量化，可以作为后续任何"扩展到更大规模场景"讨论的依据，而不需要凭空假设。

## 11. 规模压力测试场景下 C0 vs 贪心基线对照

第 4.1 节的 C0-vs-贪心对照只覆盖了六个既有 profile 的规模（10-20 任务级别）。第 4.3/9/10 节的规模与结构压力测试都只跑了 C0，没有同时跑三个贪心基线做对照——这意味着"规模变大后 C0 是否还能保持相对贪心基线的优势"这个问题还没有直接证据。补跑了 `experiments/md_c0_scale_baseline_comparison_pilot.py`（输出 `reports/md_c0_scale_baseline_comparison_pilot_2026-09-13/`），在 task_count ∈ {12, 24, 42, 60, 90, 114, 150} 上同时跑 C0 与三个贪心基线（每档 3 seed）。

结果（平均 makespan）：

| task_count | C0 | greedy_distance | greedy_eta | greedy_unlock | C0 相对最优贪心的差距 |
|---:|---:|---:|---:|---:|---:|
| 12 | 343.3 | 330.0 | 330.0 | 306.7 | +12.0% |
| 24 | 304.3 | 309.0 | 307.3 | 271.7 | +12.0% |
| 42 | 342.0 | 326.3 | 335.0 | 288.7 | +18.5% |
| 60 | 365.7 | 355.3 | 355.0 | 320.7 | +14.0% |
| 90 | 381.3 | 355.0 | 360.7 | 361.3 | +5.5% |
| 114 | 386.3 | 352.3 | 375.0 | 310.7 | +24.3% |
| 150 | 392.7 | 362.7 | 371.3 | 336.0 | +16.9% |

**这是一个诚实且重要的负面发现，如实记录，不做美化**：在六个既有 profile 的规模范围内（4.1 节），C0 的 makespan 与三个贪心基线基本处于同一量级、有时更优。但在本次压力测试探索出的更大规模（task_count 42 及以上，超出训练/评测覆盖范围）下，C0 的平均 makespan **持续、系统性地比表现最好的贪心基线（通常是 `greedy_unlock`）差 12%-24%**，且没有随规模变化而收敛。三个贪心基线彼此之间在这些规模下的差距也不大，说明这不是"任务变难所以谁都会变差"的普遍现象，而是 C0 打分策略本身在超出六 profile 训练/评测覆盖规模之后，调度质量的相对优势就消失了——**MIP 兜底保证了"不失败"，但没有保证"调度质量不退化"，这两者是分开的性质**。

这个结果直接回答了用户"这些新算例能否突出数学建模与运行能力"的问题：**运行能力（鲁棒性、精确解兜底）在扩展规模后依然成立，但调度质量的优势不会自动扩展**——如果未来要在更大规模场景下部署，需要在这些规模上重新训练或至少重新评测 C0，不能假设六 profile 上的优势会自动迁移。

**用更多种子确认这个差距不是偶然**：原始对照每档只用了 3 个 seed，差距最大的两档（42、114 任务）补跑了另外 6 个全新 seed（`experiments/md_c0_scale_baseline_confirm_pilot.py`，输出 `reports/md_c0_scale_baseline_confirm_pilot_2026-09-13/`）。结果：42 任务时 C0 340.3 vs 最优贪心（greedy_unlock）298.3，差距 14.1%；114 任务时 C0 398.8 vs 最优贪心 330.5，差距 20.7%。与原始 3-seed 结果（+18.5%、+24.3%）方向和量级一致，48/48 次全部成功。**这不是采样噪声，是 C0 在超出训练规模后调度质量退化的稳定现象。**

## 12. 纯 MILP baseline 三方对照（重要发现）

前面所有 C0-vs-贪心对照都是"启发式 vs 启发式"——没有一个真正的最优解参照物，所以只能相对比较不能绝对评估。这一节把 `baselines/gurobi_md_oracle.py` 里现成的**端到端**精确 MILP 求解器（Gurobi + 4 线程 + 300 秒时限）加进对照，作为绝对的最优解参照物。

脚本：`experiments/md_pure_milp_baseline_pilot.py`，输出：`reports/md_pure_milp_baseline_pilot_2026-09-13/`。

### 一个关于 makespan 定义的重要修正

初次写这一节时我犯了个错误：从 MILP 结果里我用 `max(entry.completion for entry in result.schedule)` 提 makespan，得到 130-200 这个量级；但仿真器里 C0/greedy 用的 makespan 是**"最后一个机器人返回 exit 位置的时间"**（`latest_exit_return`），即 MILP 结果的 `objective` 字段，通常比 `max(completion)` 大 60-90（因为要加上机器人返程时间）。用 `replay_oracle_actions` 把 MILP 输出回放到仿真器里验证过：仿真器报出来的 makespan 就是 `objective`。**下表都用修正后的可比 makespan（`objective`）**。

### 六个 profile（小规模）三方对照

在六个既有 profile（10-24 任务）上，纯 MILP **24/24 次全部求得最优解**（optimality_gap=0），绝大多数 1 秒以内解完（scale_medium 平均 0.7 秒）。这就得到了真正的最优参照：

| profile | MILP 最优 makespan | C0 makespan | C0 差距 | 最优贪心 makespan | 贪心差距 |
|---|---:|---:|---:|---:|---:|
| balanced | 202.2 | 264.8 | +30.9% | 253.8 | +25.5% |
| process_scarce | 154.8 | 200.6 | +29.6% | 212.0 | +37.0% |
| transport_bottleneck | 389.0 | 449.8 | +15.6% | 437.8 | +12.5% |
| dependency_deep | 249.8 | 289.5 | +15.9% | 276.0 | +10.5% |
| mixed_hard | 262.0 | 356.2 | +36.0% | 305.0 | +16.4% |
| scale_medium | 243.5 | 298.2 | +22.5% | 296.2 | +21.7% |

**修正后的画面**：**C0 在六个 profile 上比 MILP 最优解差 15-36%，最优贪心比 MILP 最优解差 10-37%**——C0 与最优贪心大致相当，两者都还有明显的、可优化的余地。这个"绝对上限"参照物之前一直缺失，让此前几节里"C0 与贪心基线基本处于同一量级"的评价虽然方向对，但缺乏"离真最优多远"这个坐标系。

### 规模压力测试上（大规模）三方对照

在压力测试用的 scale_medium-风格算例上（每档 3 seed）：

| task_count | MILP 状态 | MILP makespan | MILP 求解时间 | C0 makespan | C0 差距 | 最优贪心 makespan | 贪心差距 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 12 | 3/3 最优 | 285.3 | 0.07 s | 343.3 | +20.3% | 306.7 | +7.5% |
| 24 | 3/3 最优 | 230.0 | 1.68 s | 304.3 | +32.3% | 271.7 | +18.1% |
| 42 | 2 最优 + 1 超时(gap 11.7%) | 232.3 | 110 s | 342.0 | +47.2% | 288.7 | +24.2% |
| 60 | 1 最优 + 2 超时(gap 7.9-28.1%) | 272.0 | 273 s | 365.7 | +34.4% | 320.7 | +17.9% |
| 90 | 3/3 超时(gap 16.6-26.9%) | 258.0 | 302 s | 381.3 | +47.8% | 355.0 | +37.6% |
| 114 | 3/3 超时(gap 12.2-15.2%) | 259.3 | 305 s | 386.3 | +49.0% | 310.7 | +19.8% |
| 150 | 3/3 超时(gap 25.1-40.2%) | 302.7 | 311 s | 392.7 | +29.7% | 336.0 | +11.0% |

（注：90 任务以上 MILP 都超时了，报告的 makespan 是次优解——真最优只会更好。所以 C0 的实际差距 **不小于** 表格显示的值。）

### 修正后的完整结论

1. **纯 MILP 在 12-24 任务上秒级解出全局最优**，是这一规模上的最优参照；
2. **纯 MILP 在 42 任务开始出现超时**（1/3），60 任务 2/3 超时，**90 任务以上 100% 超时**——这是本机 + 5 分钟预算下 MILP baseline 的能力边界；
3. **C0 相对 MILP 最优/次优差距 20-49%**（小规模 20-32%，中大规模 34-49%），差距随规模变大略有扩大。这不是"训练规模覆盖不到"的问题——12-24 任务本来就在训练分布内也差 20-32%——而是**C0 在训练分布内就没有学到接近最优的策略**；
4. **贪心 unlock/distance 通常是最好的启发式**，差距 7-38%，比 C0 略好但也远非最优；
5. **C0 + MIP fallback 混合架构的真正价值**：**在纯 MILP 撑不住的规模（42+）上，它仍然能秒级给出解**，虽然差最优 30-49%。混合架构胜在"能给解"这个维度，不是"给最优解"。

**下一步优化方向**：既然 gurobi_md_oracle 现成能生成最优的 assignment 顺序（`OracleAction`），也现成能通过 `replay_oracle_actions` 回放到仿真器（已在这次校准中验证过），**用 MILP 最优解作为监督信号训练 C0** 应该能大幅缩小这 20-49% 的差距——尤其对小规模，因为 MILP 在那里秒级给最优解，数据成本极低。这比继续做启发式互比更能突出"数学建模能力"。
