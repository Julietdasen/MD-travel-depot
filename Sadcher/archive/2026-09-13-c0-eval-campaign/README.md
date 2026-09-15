# 2026-09-13 C0 评测战役归档

本目录归档 2026-09-13 凌晨完成的 C0 checkpoint 重训 + 大规模评测战役的原始产物。当日凝练结论见 [../../ReadME/2026-09-13/README.md](../../ReadME/2026-09-13/README.md)。

## 归档理由

这一批工作把"policy 到底赢在哪、输在哪"这个问题从模糊状态推到了明确结论:

- **C0+fallback 混合架构的卖点是"MILP 撑不住的规模上仍能秒级给出解",不是"逼近最优"**。
- **在 6 profile 上 100% 成功、0 非法分配、单次 MIP 兜底 < 1s**——运行鲁棒性充分。
- **相对 MIP 最优解差 15%-49%,相对 `greedy_unlock` 从 12 任务差 12% 涨到 150 任务差 17%**——训练分布内就没学到接近最优的策略。
- 结论指向**用 MILP 最优 assignment order 作为 IL 监督信号重训 C0**(新 IL 方案见 `ReadME/2026-09-13/IL_REDESIGN.md`)。

结论已凝练进日报和 IL 方案后,战役期间产生的 14 个 log、14 个 run 脚本、11 个 pilot 脚本和 29KB 工作记录不再需要挂在根目录污染工作区,故整体下沉到本归档。`reports/md_c0_*_2026-09-13/` 和 `reports/md_pure_milp_baseline_pilot_2026-09-13/` 保持原位,是后续 IL 重设计的实证依据。

## 目录内容

- `工作记录_full.md` — 完整的当日工作记录(29KB,12 节,含全部表格与决策依据)。
- `logs/` — 14 个战役期间的后台运行日志。
- `run_scripts/` — 14 个后台启动脚本(含 `run_pure_milp_baseline.sh`)。
- `pilot_scripts/` — 11 个新写的 pilot 脚本(`experiments/md_c0_*_pilot.py` + `md_pure_milp_baseline_pilot.py`)。

## 复现说明

若需重跑本战役中的任意 pilot:

1. 脚本本身用 `PYTHONPATH=/root/autodl-tmp/MD/Sadcher` 直接调用即可,不依赖 `experiments/` 目录位置。
2. 报告输出位置(`reports/md_c0_*_2026-09-13/`)已固化在脚本内,不必修改。
3. 需要 Gurobi license(`gurobi (1).lic`,已在仓库根目录)。
