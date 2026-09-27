# Travel + Depot + Coalition MRTA 文献调研

调研日期: 2026-09-22
参照基线: SADCHER (IEEE MRS 2025, Bichler et al., TU Delft) 与 HeteroMRTA (marmotlab, RA-L 2025)
目标 setting: 异构 skill、动态 coalition、任务间 precedence、depot + travel time、makespan 最小化、learning-based (IL / RL / GNN / Transformer)

---

## 1. 直接相似的工作 (setting 几乎完全重合)

这一组工作的 problem 定义与本项目基本一致 (异构 skill + coalition + travel + depot + precedence + makespan)。目前只找到少数几篇。

### 1.1 SADCHER — Bichler et al., IEEE MRS 2025 (TU Delft)
- 全称: Scheduling using Attention-based Dynamic Coalitions of Heterogeneous Robots in Real-Time
- Setting: 异构 skill matching、动态 coalition、travel time、depot (start/end at depot)、可选 precedence、makespan 目标
- Method: **Imitation Learning from MILP optimal schedules** + Transformer encoder + bipartite (Sinkhorn / matching) head，coalition-aware
- 代码: 开源 (jakbichler/Sadcher)
- 链接: [arXiv](https://arxiv.org/abs/2510.14851) · [IEEE](https://doi.org/10.1109/mrs66243.2025.11357250) · [Project page](https://autonomousrobots.nl/paper_websites/sadcher_MRTA/) · [Code](https://github.com/jakbichler/Sadcher)

### 1.2 HeteroMRTA — Dutta / MARMotLab, RA-L 2025
- 全称: Heterogeneous Multi-robot Task Allocation and Scheduling via Reinforcement Learning
- Setting: 异构 capability、depot、travel time、coalition (多机器人协作单任务)、precedence-like temporal 约束、makespan / task-completion 目标
- Method: **RL + graph attention** (heterogeneous graph)，attention-based policy over robot-task graph
- 代码: 开源 (marmotlab/HeteroMRTA)
- 链接: [Paper PDF](https://marmotlab.org/publications/73-RAL2025-HetMRTA.pdf) · [ADS](https://ui.adsabs.harvard.edu/abs/2025IRAL...10.2654D/abstract) · [Code](https://github.com/marmotlab/HeteroMRTA)

### 1.3 DCMRTA — MARMotLab, ICRA 2024 (HeteroMRTA 的前身)
- 全称: Dynamic Coalition Formation and Routing for Multirobot Task Allocation
- Setting: coalition + routing (含 travel + depot) + 异构 capability，无严格 precedence
- Method: RL + attention
- 代码: 开源 (marmotlab/DCMRTA)
- 链接: [Paper PDF](https://www.marmotlab.org/publications/56-ICRA2024-MRTA.pdf) · [Code](https://github.com/marmotlab/DCMRTA)

### 1.4 Heterogeneous Graph Transformers for Simultaneous Mobile MRTA — Altundas, Chen, Gombolay et al., NeurIPS 2025 (Georgia Tech)
- Setting: mobile 机器人 (含 travel)、heterogeneous、temporal (deadline/precedence-类) 约束、simultaneous allocation + scheduling
- Method: Heterogeneous Graph Transformer，RL/IL 混合训练 (依 OpenReview 描述)
- 代码: 未见公开仓库
- 链接: [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/82a227f51354b7ebafea51023d547d29-Paper-Conference.pdf) · [OpenReview](https://openreview.net/forum?id=k1fbdnwjCH)

### 1.5 CF-HMRTA — Verma, Gautam, Dutta et al., JIRS 2025
- 全称: Coalition Formation for Heterogeneous Multi-Robot Task Allocation
- Setting: 异构 capability、coalition、travel + depot、precedence 有讨论、makespan / completion time
- Method: **启发式/搜索式** (非 learning-based)，用作对照实验很合适
- 链接: [Springer](https://doi.org/10.1007/s10846-025-02287-4)

### 1.6 Learning Policies for Dynamic Coalition Formation in MRTA — Bezerra et al., IEEE LRA 2025 (KAUST)
- Setting: coalition formation、异构、travel、多任务分派 (grid world 版，depot 不严格)
- Method: MARL (multi-agent RL) + attention
- 代码: 开源 (lcdbezerra/marl_mrt2a)
- 链接: [arXiv](https://arxiv.org/abs/2412.20397) · [IEEE](https://doi.org/10.1109/lra.2025.3592080) · [Code](https://github.com/lcdbezerra/marl_mrt2a)

---

## 2. 部分相似的工作 (少一到两个维度)

### 2.1 Q-ITAGS — Liu et al., 2024 (GT STAR Lab)
- 全称: Quality-Optimized Spatio-Temporal Heterogeneous Task Allocation with a Time Budget
- Setting: 异构 skill、time-budget (makespan 上限约束)、travel、depot、precedence via temporal network — 缺 learning
- Method: 优化 / heuristic (非 learning)，可作 MILP-替代基线
- 代码: 开源 (GT-STAR-Lab/Q-ITAGS)
- 链接: [arXiv](https://arxiv.org/abs/2404.07902) · [Code](https://github.com/GT-STAR-Lab/Q-ITAGS)

### 2.2 EITAGS — Liu et al., 2026 (GT STAR Lab)
- 全称: Learning and Optimizing the Efficacy of Spatio-Temporal Task Allocation under Temporal and Resource Constraints
- Setting: 与 Q-ITAGS 相同 + learning-based efficacy 估计
- 链接: [arXiv](https://arxiv.org/abs/2601.02505) · [Project page](https://star-lab.cc.gatech.edu/papers/Liu-EITAGS/)

### 2.3 Multi-Robot Task Allocation with Spatiotemporal Constraints via Edge-Enhanced Attention Networks — MDPI Appl. Sci. 2026
- Setting: 异构、spatiotemporal (含 travel)、temporal window (类似 precedence)、单机器人执行 (无 coalition)
- Method: Edge-enhanced graph attention + RL
- 链接: [MDPI](https://doi.org/10.3390/app16020904)

### 2.4 A Constraint-Aware Heterogeneous Transformer for Real-Time Multi-Robot Scheduling — Preprints, 2026-03
- Setting: 异构、real-time、含 constraints (precedence + resource)，coalition 支持不明确
- Method: Heterogeneous Transformer + IL
- 链接: [Preprints](https://www.preprints.org/manuscript/202603.1389)

### 2.5 Online Multi-Robot Coordination and Cooperation with Task Precedence Relationships — Malencia et al., arXiv 2509.15052 (2025)
- Setting: 强调 precedence + cooperation (coalition-like)，异构、travel 有考虑
- Method: online planning / decentralized
- 链接: [arXiv](https://arxiv.org/abs/2509.15052)

### 2.6 Heterogeneous MRTA for Long-Endurance Missions in Dynamic Scenarios — arXiv 2411.02062 (v3 2025-11-26)
- Setting: 异构、长时段、depot + travel、动态任务，coalition 弱
- 链接: [arXiv](https://arxiv.org/abs/2411.02062)

### 2.7 MAGNNET — Autonomous vehicle task allocation, arXiv 2502.02311 (2025)
- Setting: GNN + RL，含 travel，无 coalition，无 precedence
- 链接: [arXiv](https://arxiv.org/abs/2502.02311)

### 2.8 ScheduleNet — Wang, Gombolay et al., RSS 2020
- Setting: 异构 graph attention scheduler、precedence、无 travel (工厂调度语境)
- 意义: SADCHER / HeteroMRTA 均引用的经典 learning-based scheduler
- 链接: [PDF](https://core-robotics.gatech.edu/wp-content/uploads/sites/958/2020/06/RSS20_ScheduleNet.pdf)

### 2.9 Sampling-Based Heterogeneous Coalition Scheduling with Temporal Uncertainty — RSS 2023
- Setting: coalition + heterogeneous + temporal，非 learning
- 链接: [DOI](https://doi.org/10.15607/rss.2023.xix.107)

---

## 3. "Travel vs no-travel" ablation 的显式研究

搜索显式"process-only vs with-travel"的 MRTA ablation，几乎**没有独立论文**在做这件事。绝大部分工作要么假设 travel 一定包含 (routing / MRTA-CD 传统)，要么假设 travel 忽略 (工厂 job-shop / flexible scheduling 传统)。两侧几乎没有 head-to-head 的对比。

唯一比较接近的角度是 Q-ITAGS / EITAGS 内部 ablation 中比较过"忽略 travel"版本的 quality drop (STAR Lab 论文里的 baseline 之一)，以及 ScheduleNet vs. mobile-scheduler 之类的间接对比。**独立、专门做"process-only vs full MRTA"的实验论文目前是空白**，这是一个明确的可切入方向。

---

## 4. 总结: travel-depot MRTA 现在有多"卷"?

**结论: 卷，但只卷了核心几篇；仍有明确空白。**

**已经卷的部分:**
- Learning-based coalition + heterogeneous + travel + precedence 的组合，2024-2025 有 SADCHER、HeteroMRTA、DCMRTA、CF-HMRTA、Bezerra LRA'25、GT NeurIPS'25 六篇，方法覆盖 IL、RL、Heuristic、MARL、Heterogeneous Transformer 五种范式。
- 每篇 setting 都自成一格 (是否强制 coalition、precedence 是硬约束还是软偏好、depot 是否 identical start/end、makespan 还是 total-completion-time)，**几乎没有 apples-to-apples 的公共 benchmark**。
- 主要出口都是 IEEE MRS / ICRA / RA-L / NeurIPS，2025 下半年到 2026 上半年集中出现，且大多开源。

**明显空白:**
1. **"Travel 影响 learning-based scheduler 表现"的系统实验缺失。** 没有独立论文比较过同一模型在 process-only vs. travel-inclusive 版本上的行为差异 (作为 MRTA envelope 诊断，这是本项目的天然机会)。
2. **Coalition + precedence + travel 三者交互的 diagnostic 分析很少。** 现有论文多是"整体 outperform baseline"，很少系统研究哪个约束在什么规模下是瓶颈。
3. **Time-budget infeasibility (MILP 求不出解) 作为 finding** — 现有 learning-based MRTA 论文的 MILP teacher 都在"能解得动"的规模内报数据；本项目已有的 time-budget grid 结果 (project_path_lessons) 在这个方向上有独立价值。
4. **SADCHER 之后 (2026) 目前只有小体量 preprint (Preprints.org 的 Constraint-Aware Heterogeneous Transformer 等)**，没有明显"完全超越"SADCHER + coalition + IL 的新工作。HeteroMRTA 也没有直接后续。这个位置暂时是 open 的。

**对本项目 (paper_frozen_2026-09-21) 的意义:**
- Process-only envelope 论文的 "no-travel, no-depot" simplification 在文献中是罕见 setting；它是 diagnostic 工具而非 competitor，narrative 上不冲突。
- Travel-depot 版本 (Sadcher 原生 setting) 已被 SADCHER 本身占据；如果后续要开 transport-dynamics paper，切入点应该是 (a) travel vs no-travel 的显式 ablation，或 (b) time-budget infeasibility 作为 envelope 的第五轴，而不是再刷一遍"我们的 coalition scheduler 更强"。

---

## 参考文献 (集中列表)

- SADCHER (Bichler et al., MRS 2025): https://arxiv.org/abs/2510.14851 · https://doi.org/10.1109/mrs66243.2025.11357250 · code https://github.com/jakbichler/Sadcher
- HeteroMRTA (Dutta et al., RA-L 2025): https://marmotlab.org/publications/73-RAL2025-HetMRTA.pdf · code https://github.com/marmotlab/HeteroMRTA
- DCMRTA (ICRA 2024): https://www.marmotlab.org/publications/56-ICRA2024-MRTA.pdf · code https://github.com/marmotlab/DCMRTA
- Heterogeneous Graph Transformers for Mobile MRTA (NeurIPS 2025): https://openreview.net/forum?id=k1fbdnwjCH
- CF-HMRTA (JIRS 2025): https://doi.org/10.1007/s10846-025-02287-4
- Learning Policies for Dynamic Coalition Formation (LRA 2025): https://arxiv.org/abs/2412.20397 · code https://github.com/lcdbezerra/marl_mrt2a
- Q-ITAGS (2024): https://arxiv.org/abs/2404.07902 · code https://github.com/GT-STAR-Lab/Q-ITAGS
- EITAGS (2026): https://arxiv.org/abs/2601.02505
- Edge-Enhanced Attention MRTA (MDPI 2026): https://doi.org/10.3390/app16020904
- Constraint-Aware Heterogeneous Transformer (Preprints 2026-03): https://www.preprints.org/manuscript/202603.1389
- Online MR Coordination with Task Precedence (arXiv 2509.15052): https://arxiv.org/abs/2509.15052
- Heterogeneous MRTA for Long-Endurance Missions (arXiv 2411.02062): https://arxiv.org/abs/2411.02062
- ScheduleNet (RSS 2020): https://core-robotics.gatech.edu/wp-content/uploads/sites/958/2020/06/RSS20_ScheduleNet.pdf
- Sampling-based Heterogeneous Coalition Scheduling (RSS 2023): https://doi.org/10.15607/rss.2023.xix.107
