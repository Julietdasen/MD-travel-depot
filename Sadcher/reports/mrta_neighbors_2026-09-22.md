# MRTA 近邻工作调研 (2026-09-22)

聚焦 setting：异构机器人 + skill matching + dynamic coalition + task precedence + 2D travel time + depot start/end + makespan 最小化 + IL from MILP。以 SADCHER 和 HeteroMRTA 为两个锚点，梳理其引用邻居与 2024–2026 的近邻方法。

## 1. SADCHER (Bichler, Matoses Gimenez, Alonso-Mora — IEEE MRS 2025)

- arXiv：2510.14851（v1，2025-10-16），IEEE MRS 2025 已录用；代码 https://github.com/jakbichler/Sadcher。
- 问题类别：ST-MR-TA-XD + precedence，含 skill、位置、task duration、robot remaining time。
- 方法：Imitation Learning 从 MILP 最优解学 reward matrix，GAT + Transformer encoder，relaxed bipartite matching 生成可行调度；提供 stochastic 变体 S-Sadcher；有 PPO fine-tune 分支。
- Baseline：MILP（Aswale & Pinciroli 2023 formulation 加 precedence）、HeteroMRTA / S-HeteroMRTA（Dai et al. RA-L 2025）、Greedy skill-reduction。
- 数据/代码：250k 最优 schedules 公开（4TU 平台），完整代码开源。
- 特征：travel ✓ / heterogeneous ✓ / coalition ✓ / precedence ✓ / depot start-end ✓ / IL 主 + RL 微调。

## 2. HeteroMRTA (Dai, Rai, Chiun, Cao, Sartoretti — IEEE RA-L 2025)

- 论文 PDF：marmotlab.org/publications/73-RAL2025-HetMRTA.pdf，代码 https://github.com/marmotlab/HeteroMRTA。
- 问题类别：ST-MR heterogeneous + skill requirements + coalition，decentralized 顺序决策；start-end depot 相同（SADCHER 用 `--start_end_identical` 复现）。原论文未强调 precedence，SADCHER adaptor 用 mask 加上。
- 方法：REINFORCE + attention network + flash-forward 重排机制解决 RL 死锁；Boltzmann sampling 得 S-HeteroMRTA。
- Baseline：OR-Tools、CTAS-D (Fu et al. TRO 2021)、以及 marmotlab 自家 DCMRTA。
- 特征：travel ✓ / heterogeneous ✓ / coalition ✓ / precedence ✗（原生不支持，需 mask）/ RL / 开源。

## 3. DCMRTA (Dai, Bidwai, Sartoretti — ICRA 2024)

- 代码：https://github.com/marmotlab/DCMRTA。HeteroMRTA 的前身。
- 问题：同构机器人 dynamic coalition + routing；ST-MR；travel time 是 core；depot 有。
- 方法：Attention + REINFORCE (Ray)。Baselines：OR-Tools、CTAS-D。
- 特征：travel ✓ / heterogeneous ✗ / coalition ✓ / precedence ✗ / RL / 开源。

## 4. LVWS — Learning for Voluntary Waiting & Subteaming (Jose & Zhang, ICRA 2024, UMass HCR Lab)

- SADCHER refs [30]。项目页 hcrlab.gitlab.io/project/lvws/。
- 问题类别：Heterogeneous ST-MR，coalition 全程考虑（不像 CSAF 只在失败后再形成），引入 voluntary waiting 提升未来 coalition 质量。
- 方法：IL + reward-prediction network + bipartite matching（与 SADCHER 同框架，但 SADCHER 指出其 network **未显式建模位置与 duration**，故隐式假设 travel time / duration 与训练分布匹配）。
- 特征：travel ✗（隐式 / 忽略）/ heterogeneous ✓ / coalition ✓ / precedence 有限 / IL / 代码半开源。

## 5. CSAF (Gao, Siva, Micciche, Zhang, ICRA 2023)

- SADCHER refs [29]。IL + reward network + bipartite matching 的 **原始版本**，coalition 仅在 single-robot 失败时形成，因此对 travel/duration 建模粗糙。
- 特征：travel ✗ / heterogeneous ✓ / coalition (reactive) / precedence ✗ / IL。

## 6. HGAT (Wang, Liu, Gombolay, Autonomous Robots 2022)

- SADCHER refs [23]。Heterogeneous Graph Attention Network for scalable MRTS with temporospatial constraints；IL；mild heterogeneity（robots 效率不同、能力同）；ST-SR。
- 特征：travel ✓（temporospatial）/ heterogeneous 弱 / coalition ✗ / precedence ✓ / IL / 开源。

## 7. CTAS / CTAS-D (Fu, Smith, Rizzo et al., TRO 2021)

- MILP 框架，异构 team + capability uncertainty；含 skill、travel、precedence、makespan。属于 exact baseline 家族；HeteroMRTA/DCMRTA 都用它做 optimality baseline。
- 特征：travel ✓ / heterogeneous ✓ / coalition ✓ / precedence ✓ / MILP / 开源（C++）。

## 8. Aswale & Pinciroli 2023 — Heterogeneous Coalition Formation and Scheduling

- SADCHER refs [16]，MILP formulation 直接被 SADCHER 借用后加 precedence。同组 [10] Babincsak 2023 是 ACO 求解器。
- 特征：travel ✓ / heterogeneous ✓ / coalition ✓ / precedence 弱 / exact + heuristic。

## 9. CF-HMRTA (2025, JIRS)

- Coalition Formation for Heterogeneous MRTA，Journal of Intelligent & Robotic Systems 2025。启发式 coalition formation，包 skill；travel/precedence 覆盖需查全文。
- 特征：travel likely ✓ / heterogeneous ✓ / coalition ✓ / precedence 未确认 / heuristic / 代码未见。

## 10. CASH (Capability-Aware Shared Hypernetworks, arXiv 2501.06058, 2025)

- Flexible heterogeneous multi-robot coordination；hypernetwork 根据 capability 生成 policy。偏 MARL 学 policy sharing，非 scheduling 主线。
- 特征：travel 视 env / heterogeneous ✓ / coalition 弱 / precedence ✗ / RL (MARL) / 开源。

## 11. Q-ITAGS (Neville et al., arXiv 2404.07902, 2024–25)

- Quality-optimized spatio-temporal heterogeneous task allocation with time budget。含 travel + skill + temporal constraints。规划器基线家族，非 learning。
- 特征：travel ✓ / heterogeneous ✓ / coalition 依 spec / precedence ✓ / search + planning。

## 12. Calvo & Capitan (arXiv 2411.02062, RA-L 2025)

- Heterogeneous MRTA for long-endurance missions in dynamic scenarios。MILP + heuristic + online replanning，允许 recharge、task 分裂/中继、coalition。空中 inspection 场景。
- 特征：travel ✓ / heterogeneous ✓ / coalition ✓ / precedence 有限 / MILP + heuristic。

## 13. Auction-Consensus with Learned Bidding (Rodriguez, Tarawneh, Koenig et al., 2026)

- arXiv 2605.21932 / UR 2026。SADCHER 已被引用 1 次即此篇。方向偏 CBBA + learned bid function，setting 更偏 SR-ST。
- 特征：travel ✓ / heterogeneous 可 / coalition ✗ / IL/RL bid model。

## 14. Gosrich et al. 2023 — Precedence coordination

- SADCHER refs [2]，多机 precedence relationships 协同，图神经网络策略；无 skill/coalition。
- 特征：travel ✓ / heterogeneous 弱 / coalition ✗ / precedence ✓ / RL。

## 15. 其它 SADCHER 引用中的 learning baseline

- Paul, Ghassemi, Chowdhury 2022（Capsule Attention Nets, ST-SR mildly hetero, RL）；Altundas, Wang, Gombolay 2022（RNN schedule propagation, human-robot）；Deng et al. 2022（priority constraints + coalition, homogeneous, RL）。
- 这些都是 SADCHER 明确列出的近邻，但均**缺失 heterogeneous + coalition + precedence + travel** 中的至少一项。

## 空白总结（travel-depot 版本 MRTA）

综合来看，在 **异构 skill + dynamic coalition + precedence + 显式 travel time + depot start/end + makespan** 六件套齐全的 learning-based 方法里，公开工作实际上只有 **SADCHER 一家做到全部显式建模**：

- 前身 CSAF/LVWS 缺 travel & duration；HeteroMRTA 缺 precedence 原生支持且 depot 强制 start=end；DCMRTA 缺 heterogeneous；HGAT 缺 coalition；CTAS/Aswale/Calvo 是 exact + heuristic 而非 learning。
- 空白 1：**precedence + travel + coalition + heterogeneous 上的 non-IL 方法**（RL 目前只有 HeteroMRTA + precedence-mask hack，其原生 RL 训练不带 precedence loss）——SADCHER 的 PPO 微调分支是唯一路径。
- 空白 2：**start ≠ end depot 的 heterogeneous coalition scheduling**：HeteroMRTA/DCMRTA 家族默认 start=end；SADCHER 支持两者但只在训练分布上评估，跨 depot 分布的 OOD 表现未系统检验。
- 空白 3：**time-budget infeasibility / scarce-resource 场景**下 IL from MILP 的样本效率——SADCHER 训练集集中在 medium complexity，稀缺资源与 tight deadline 的 envelope 尚未有近邻覆盖。
- 空白 4：**process-only（travel≈0）与 travel-dominant 的 transport dynamics 对比研究**：目前无公开工作系统比较 travel time 权重对最优 coalition 结构的影响。
- 空白 5：**多种子稳定性与 out-of-distribution generalization 报告**：多数近邻只报单-run 或少 seed，缺 systematic robustness study。

参考：
- SADCHER arXiv 2510.14851 https://arxiv.org/abs/2510.14851
- HeteroMRTA RA-L 2025 https://marmotlab.org/publications/73-RAL2025-HetMRTA.pdf
- DCMRTA https://github.com/marmotlab/DCMRTA
- LVWS https://par.nsf.gov/servlets/purl/10574244
- CTAS TRO 2021 (Fu et al.)
- Aswale & Pinciroli 2023 (arXiv)
- Calvo & Capitan arXiv 2411.02062
- CASH arXiv 2501.06058
- Q-ITAGS arXiv 2404.07902
- CF-HMRTA JIRS 2025 https://doi.org/10.1007/s10846-025-02287-4
