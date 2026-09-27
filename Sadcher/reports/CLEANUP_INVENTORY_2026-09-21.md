# 清理清单 v2 (2026-09-21) — 更新版

**变更**：Path A / MILP supervised pilot 里的 **训练结果**（checkpoint、summary、RESULTS.md、eval）**全部保留**（经验已凝练到 `PATH_HISTORY_LESSONS_2026-09-21.md`，但原始数据保留以便 regression 对比）。

**只删 dataset 子目录**（原始数据，可用 driver 重生成）。

## 保留（全部，含所有训练结果）

| 目录 | 大小 | 内容 |
|---|---:|---|
| md_c0_pathC_v1_stability_2026-09-18/ | 13M | **当前 baseline checkpoint (3 seeds ep5)** |
| md_c0_pathA_scaled_finetune_2026-09-17/{training,pilot_summary.json,run.log} | 148K + logs | Path A 训练结果 |
| md_c0_pathA_scaled_finetune_eval_2026-09-17/ | 60K | Path A eval |
| md_c0_milp_supervised_scale_pilot_2026-09-13/{training,RESULTS.md,pngs} | 124K + logs | 当前 baseline 的 pretrained parent |
| md_c0_milp_supervised_pilot_2026-09-13/{training,RESULTS.md,pngs} | 保留结果 | 更早的 pilot |
| md_c0_pathB1_capacity48_2026-09-17/ | 704K | Path B1 |
| md_c0_pathB2_capacity64_2026-09-17/ | 1.2M | Path B2 |
| md_c0_pathB1/B2_eval_2026-09-17/ | 60K each | Path B eval |
| md_c0_pathC_b2_regret_weighted_2026-09-20/ | 3.7M | Path C B2 regret 结果 |
| md_c0_pathC_ranking_v1_2026-09-17/ | 604K | Path C v1 训练 |
| md_c0_pathC_ranking_v1_eval_2026-09-17/ | 204K | Path C v1 eval |
| md_residual_scale10_2026-09-04/ | 5.0M | Residual IL 结果 |
| md_frozen_joint_correction_2026-09-06/ | 5.0M | Joint correction 结果 |
| md_exact_action_candidate_diagnostic_2026-09-06/ | 1.1M | Exact-action gate 结果 |
| md_minimal_physics_scorer_*/ | 1.2M total | Physics scorer 结果 |
| md_envelope_d2_2026-09-20/ | 152K | D2 sweep + RESULTS_SUMMARY.md |
| md_envelope_process_robots_2026-09-20/ | 100K | Sweep 3 |
| md_envelope_duration_2026-09-20/ | 60K | Sweep 2 |
| md_envelope_scarce3_2026-09-21/ | 60K | scarce=3 headline |
| md_coalition_diagnostic_2026-09-21/ | 304K | Coalition 诊断 |
| md_time_budget_grid_2026-09-20/ | 116K | Time-budget infeasibility |
| md_3seed_stability_verification_2026-09-21/ | 在跑 | 3-seed 验证 |
| md_confidence_threshold_2026-09-21/ | 60K | Threshold ablation |
| md_confidence_dist_2026-09-21/ | 8K | Confidence 分布 |
| PATH_HISTORY_LESSONS_2026-09-21.md | 12K | **6 条路径教训凝练** |
| HANDOFF.md (在项目根) | — | 历史 handoff |

## 建议删除

### A. Dataset 目录（大头，可重生成）

| 路径 | 大小 | 说明 | 重生成 |
|---|---:|---|---|
| md_c0_pathA_scaled_finetune_2026-09-17/dataset/ | **1.7G** | 300 shards + 250 legacy symlinks | `experiments/md_c0_pathA_scaled_finetune.py` |
| md_c0_milp_supervised_scale_pilot_2026-09-13/dataset/ | **306M** | 200 records | `experiments/md_c0_milp_supervised_pilot.py --scale` |
| md_c0_milp_supervised_pilot_2026-09-13/dataset/ | **45M** | 114 records | `experiments/md_c0_milp_supervised_pilot.py` |

**空间释放：~2.05 GB**

### B. 早期 pilot / 探索目录（都是死路，凝练已到 lessons doc）

| 路径 | 大小 | 说明 |
|---|---:|---|
| archive_early_candidates/ | 21M | 明确是 archive |
| md_ray_ppo_train_2026-09-14/ | 84K | RL 死路 |
| md_ray_ppo_smoke_2026-09-14/ | 24K | |
| md_ray_ppo_joint_2026-09-03/ | 16K | |
| md_ray_ppo_phase_a_2026-09-04/ | 4K | |
| md_c0_gpu_retrain_pilot_2026-09-13/ | 932K | 早期 GPU pilot |
| md_c0_end_to_end_diagnostic_pilot_2026-09-01/ | 6.7M | 早期诊断 |
| md_c0_event_trigger_regression_2026-09-02{,_smoke}/ | 576K | 早期 regression |
| md_c0_confidence_threshold_sweep_2026-09-13*/ | 108K | 早期 threshold (被 2026-09-21 取代) |
| md_c0_scaled_target_profiles_eval_2026-09-14/ | 44K | 早期 profile eval (被 envelope 取代) |
| md_c0_generalization_eval_2026-09-13/ | 96K | 早期 gen eval |
| md_c0_scale_*_pilot_2026-09-13/ (7 个目录) | ~100K total | scale pilots |
| md_c0_structural_stress_pilot_2026-09-13/ | 16K | |
| md_c0_combined_extreme_pilot_2026-09-13/ | 12K | |
| md_c0_mip_threads_pilot_2026-09-13/ | 12K | |
| md_c0_beam_search_pilot_2026-09-17/ | 64K | Beam search pilot |
| md_c0_milp_supervised_*_eval_2026-09-13/ (3 个) | ~76K | 早期 eval (被 Path A eval 取代) |
| md_c0_profile_eval_pilot_2026-09-13/ | 48K | |
| md_profile_pilot_2026-09-11/ | 52K | |
| md_ortools_feasibility_2026-09-17/ | 32K | OR-tools 切换验证 |
| md_pure_milp_baseline_pilot_2026-09-13/ | 24K | |
| md_rich_prefix_*/ | 1.4M total | Rich prefix 探索死路 |
| md_process_scarce_v2_pilot_2026-09-20/ | 52K | v2 profile 死路 (LAYER2 verdict 已凝练) |
| md_rich_exact_action_exploration_2026-09-08/ | 0 | 空 |

**空间释放：~30M**

### C. 2026-08-2X 早期审计 (都是 4K 级别记录)

- md_unconditioned_relational_candidate_audit_*
- md_train_only_optimization_diagnostic_*
- md_train_error_diagnosis_*
- md_task_process_context_ablation_*
- md_policy_signal_gate_*
- md_pickup_high_margin_error_attribution_*
- md_pair_structured_train_diagnostic_*
- md_independent_held_out_*
- md_frozen_relational_evaluation_*
- md_context_ablation_development_package_*

**空间释放：~50K**（省的是数量，不是空间）

## 汇总

- 保留：**所有训练结果、checkpoint、eval、RESULTS 文档** —— 便于未来对比
- 删：**dataset 子目录（2.05 GB）+ 早期 pilot（30M）+ 早期审计（50K）**
- 净释放：**~2.08 GB**

## 需要你决定

1. **只删 dataset + 早期 pilot（B、C），保留所有训练结果**（推荐，安全）？
2. 还是**只删 dataset**（更保守）？
3. 删前 `git status` 检查手改？（数据 dir 不在 git，但 code 目录可能有）
4. 删除方式：直接 `rm -rf` 还是先打 tarball？

我不主动删。等你说。
