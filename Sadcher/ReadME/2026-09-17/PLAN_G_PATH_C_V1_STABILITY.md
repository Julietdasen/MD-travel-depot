# Plan G: Path C v1 stability sweep (handoff for next session)

本文档是给下一个 session 的执行手册。目标是**验证 Path C v1 的 ep5 突破是否稳定**,
再决定是否投入 12–20 h 做 Path C v2(continuation-makespan aware ranking)。

不要修改本文件之外的任何 baseline / dataset / eval 脚本;下面写清楚了每一项需要新
增或复用的文件路径。

## 上游前置条件(session 开始前必须成立)

以下都已在 `MRTA_STATUS.md` 与 `HANDOFF.md` 中登记,直接引用即可,勿重跑:

- MILP 求解后端已切到 OR-Tools CP-SAT。所有 MILP 调用走
  `baselines.md_oracle_dispatch.solve_md_oracle`,默认 `MRTA_MILP_SOLVER=ortools`。
  证据:`ReadME/2026-09-17/BASELINE_SWAP_TO_ORTOOLS.md`、`ORTOOLS_FEASIBILITY.md`。
- Path A 数据集已生成并落盘,位置:
  `reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset/`(9838 train / 1291 val)。
  勿再生成、勿改动。
- Path C v1 训练脚本:`experiments/md_c0_pathC_ranking_v1.py`(unchanged,已验证)。
- Path C v1 eval 脚本:`experiments/md_c0_pathC_ranking_v1_eval.py`。
- Baseline C0 checkpoint:
  `reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt`(勿动)。
- Path C v1 seed=3101 已完成的证据(不要覆盖):
  `reports/md_c0_pathC_ranking_v1_2026-09-17/` 与
  `reports/md_c0_pathC_ranking_v1_eval_2026-09-17/`。

## 已知的实验前提(不是猜测)

- Path A(fine-tune, h=16)、Path B1(from scratch, h=48)、Path B2(from scratch, h=64)
  三次实验均出现"val loss 单调下降 + 60-task process_scarce makespan 反向退化"。
- Path C v1(margin ranking loss, seed=3101):首次在 60-task process_scarce 上打赢
  baseline C0(393.5 vs 398.75, −5.25 timestep)与 greedy_unlock(395.75),但只有
  epoch 5 通过;epoch 10/15/20 都退化到 401–402。val loss ↔ downstream 仍然反向。
- 尚不清楚 ep5 是不是单 seed 单 epoch 的运气。**本次 Plan G 就是验证这一点**。

## 目标(严格,可判决)

**目标 A(主要):** 在 3 个 model seed(3101, 3102, 3103)上重跑 Path C v1,证明:
每个 seed 都能在 20 epoch 里找到至少一个 checkpoint,同时满足
- 60-task `process_scarce` makespan ≤ 395.75(greedy_unlock),且
- 60-task `dependency_deep` makespan ≤ 500.5(baseline C0)。

**目标 B(次要,信息导向):** 记录每个 seed × 每个 epoch 的 (val_loss, 60-task
process_scarce makespan) 对,分析是否存在能预测 downstream 的 auxiliary signal
(可选候选:val loss 一阶差分、gradient norm 均值、preferred/negative score gap)。

**目标 C(轻,允许失败):** 尝试挑出每个 seed 里的 best-downstream checkpoint,
对比三 seed 的 mean makespan 与 baseline C0 / Path A / B1 / B2。这一步是"若目标 A
成立,总结整体收益"的输出。

## 判决路径(session 结束前必须落地一个)

- 若 3 seed 都达标(目标 A PASS)→ 结论 `path_c_v1_stable`,**不启动 v2**。转向
  "为 v1 建立 downstream-correlated model selection signal"(单独一个 session)。
- 若 2/3 seed 达标 → 结论 `path_c_v1_partial`,**启动 v2a**(60-task subset
  continuation label)。
- 若 ≤ 1/3 seed 达标 → 结论 `path_c_v1_fluke`,**启动 v2a 或 v2b**。

上述判决要写入本次 session 的 final_report,并同步到 `MRTA_STATUS.md` "已验证结果"
下的新子节 "Path C v1 stability sweep (2026-09-17 或后续日期)"。

## 具体执行步骤

### Step 1 — sanity check(<5 min)

在 `md` conda 环境下:

```bash
cd /root/autodl-tmp/MD/Sadcher
export PYTHONPATH=/root/autodl-tmp/MD/Sadcher
export MRTA_MILP_SOLVER=ortools
python -c "from baselines.md_oracle_dispatch import active_solver_name; print(active_solver_name())"
# 期望输出: ortools
ls reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset | head -5
ls reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt
```

如果 dataset 目录或 baseline checkpoint 缺失,不要重跑 Path A;直接停并汇报。

### Step 2 — 训练 3 seed(~6–8 h,可并行两个 seed on cuda:0)

**不要新写训练脚本**。复用 `experiments/md_c0_pathC_ranking_v1.py`,只改 CLI:

```bash
mkdir -p reports/md_c0_pathC_v1_stability_<DATE>/{seed3101,seed3102,seed3103}

for SEED in 3101 3102 3103; do
  OUT=reports/md_c0_pathC_v1_stability_<DATE>/seed${SEED}
  nohup /root/miniconda3/envs/md/bin/uv run python -m experiments.md_c0_pathC_ranking_v1 \
    --output ${OUT} \
    --dataset reports/md_c0_pathA_scaled_finetune_2026-09-17/dataset \
    --pretrained reports/md_c0_milp_supervised_scale_pilot_2026-09-13/training/best_checkpoint.pt \
    --epochs 20 \
    --learning-rate 5e-6 \
    --negatives-per-state 5 \
    --margin 1.0 \
    --seed ${SEED} \
    --save-every-epoch \
    --device cuda:0 > ${OUT}/run.log 2>&1 &
  echo "SEED ${SEED} PID=$!"
done
```

**注意:**
- 当前脚本可能没有 `--save-every-epoch`。**需要小改动**:把 v1 里 ep 5/10/15/20 dump
  改成每个 epoch 都 dump(加 CLI flag,默认 False,兼容既有 ep-5/10/15/20 行为)。
- 若两个 seed 同时占 cuda:0 显存不足,串行跑;seed=3101 已有结果,可只跑 3102 与
  3103,若目标 A 需要就补 3101。
- 单 seed 训练 ~90 分钟,3 seed 串行 ~4.5 小时,并行两 seed ~3 小时。

### Step 3 — Eval 全部 checkpoint(~4 分钟/checkpoint,~80 分钟总)

对每个 seed 的每个 checkpoint 跑 `md_c0_pathC_ranking_v1_eval.py`。**不要改 eval 脚本**。

```bash
for SEED in 3101 3102 3103; do
  OUT=reports/md_c0_pathC_v1_stability_<DATE>/seed${SEED}
  for EP in $(seq 1 20); do
    CKPT=${OUT}/training/checkpoint_epoch_${EP}.pt
    [ -f "${CKPT}" ] || continue
    /root/miniconda3/envs/md/bin/uv run python -m experiments.md_c0_pathC_ranking_v1_eval \
      --checkpoint ${CKPT} \
      --output ${OUT}/eval_epoch_${EP} \
      --device cuda:0 \
      2>&1 | tee -a ${OUT}/eval.log
  done
done
```

**Eval 网格固定**:`md_c0_scaled_target_profiles_eval` 内部常量 2 profiles × tc
{12,24,42,60} × 4 seeds = 32 instance,MILP 300 s time limit,MRTA_MILP_SOLVER=ortools。
勿修改脚本常量。

### Step 4 — 汇总与判决(~30 分钟)

写脚本 `reports/md_c0_pathC_v1_stability_<DATE>/aggregate.py`,合并 60 个
`rows.json`,输出:

- `summary.csv`:每行 (seed, epoch, val_loss, ps60_makespan, dd60_makespan,
  primary_pass, tier_regression_count)
- `final_report.md`:判决(目标 A/B/C)+ 表格(三 seed 各自的 best-downstream
  checkpoint)+ 与 baseline / Path A / B1 / B2 对比
- **建议图**:三 seed × 20 epoch 的 `val_loss ↔ ps60_makespan` 散点,用来看反向
  是否稳定

### Step 5 — 文档同步

- 写 `ReadME/2026-09-17/EXPERIMENT_G_PATH_C_V1_STABILITY.md`(风格对齐
  `EXPERIMENT_F_PATH_C_RANKING_V1.md`:Hypothesis / Setup / Results / Verdict /
  Next step / Artifacts,~200–400 词)。
- 更新 `MRTA_STATUS.md` "已验证结果" 下新增 "Path C v1 stability sweep (<DATE>)"
  子节,格式对齐 "Path A scaled-profile fine-tune (2026-09-17)"。
- 原始证据索引末尾追加两个 report 路径(training + eval)。
- **不要改** `HANDOFF.md`(遵循 Path A/B/C 的"失败或诊断类实验不动 HANDOFF"约定;
  只有生产 checkpoint 产生变更才动)。

## 硬约束(不能碰)

- 不修改 baseline C0 checkpoint、Path A 数据集、`md_c0_scaled_target_profiles_eval.py`。
- 不使用 Gurobi(license 已过期)。
- 不消费 confirmation split(76000–76149)或 regression benchmark(75800–75949)。
  Path A 已经把 86000–86449、87000–87449 用掉;本次沿用 Path A 数据,不额外分配 seed。
- 训练超参与 Path C v1 一致:lr=5e-6, batch=32, K=5, margin=1.0, epochs=20。
  唯一新增自由度是 model seed 与 checkpoint dump 频率。
- 每个 checkpoint 落盘大小 ~115 KB × 3 seed × 20 epoch ≈ 7 MB,可忽略。

## 兼容性 / 工程注意

- 使用 `uv run python -m ...`,勿混用 `pip install`,勿改 `pyproject.toml`。
- 若脚本需要小改(加 `--save-every-epoch`),改动限制在
  `experiments/md_c0_pathC_ranking_v1.py`:
  1. 加 argparse flag `--save-every-epoch`,default False,兼容原逻辑
  2. 若 True,每个 epoch 结束都 `torch.save(model.state_dict(),
     training/checkpoint_epoch_{ep}.pt)`
  3. 不改 loss、data pipeline、模型 build 逻辑
- 训练 log 与 eval log 落在 `run.log` / `eval.log`,不要输出到 stdout 之外的地方。
- checkpoint 命名必须匹配 `checkpoint_epoch_{ep}.pt`(与 v1 已有的 ep-5/10/15/20
  格式一致),eval 才能循环遍历。

## 预期耗时(从新 session 计)

| 阶段 | 时长 | 备注 |
|---|---|---|
| sanity + 小改脚本 | 20 min | 加 `--save-every-epoch` |
| 训练 3 seed(串行) | ~4.5 h | 单卡 cuda:0 |
| Eval 60 checkpoint | ~4 h | 每 checkpoint ~4 min,可并行 2 |
| 汇总 + 文档 | 1 h | aggregate + final_report + status/README |
| **合计** | **~10 h** | |

## 若在 session 内卡住

按 Path A 那次的"卡住就报告 + 停"约定:发现问题(如显存不足、脚本 crash、
checkpoint 保存失败),写清楚已完成什么、卡在哪、建议怎么处理,不要盲修其他文件。

## 完成后的下一步(下一个 session 的输入)

- 若 `path_c_v1_stable`:下一个 session 做 "downstream-correlated model selection
  signal" 的小实验(内容待另行规划,不属于本 handoff)。
- 若 `path_c_v1_partial` 或 `_fluke`:下一个 session 直接启动 Path C v2a
  (continuation-makespan aware ranking on 60-task subset)。详细设计已在
  `EXPERIMENT_F_PATH_C_RANKING_V1.md` 的 "Next step" 节。
