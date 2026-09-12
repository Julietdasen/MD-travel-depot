import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.md_task_process_context_models import (
    CONTEXT_ABLATION_VARIANTS,
    build_context_ablation_model,
)
from experiments.md_task_process_context_ablation import (
    _resolve_device,
    render_task_process_context_ablation_report,
    run_task_process_context_ablation,
    select_context_ablation_outcome,
)
from models.md_policy import MDPolicyInputs


FORMAL_SUMMARY = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "md_task_process_context_ablation_2026-08-30"
    / "task_process_context_ablation.json"
)
FORMAL_REPORT = FORMAL_SUMMARY.with_suffix(".md")


class MDTaskProcessContextAblationTests(unittest.TestCase):
    def test_reduced_comparison_is_deterministic_complete_and_not_authorizing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "train_pairs_per_family": 20,
                "pairs_per_family_per_batch": 10,
                "pretrain_epochs": 1,
                "fine_tune_epochs": 1,
                "checkpoint_interval": 1,
                "model_seeds": (3101,),
                "device": "cpu",
            }
            first = run_task_process_context_ablation(root / "first", **kwargs)
            second = run_task_process_context_ablation(root / "second", **kwargs)
            first_summary = json.loads(first.summary_path.read_text())
            second_summary = json.loads(second.summary_path.read_text())

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(
                first.report_path.read_text(), second.report_path.read_text()
            )
            self.assertEqual(
                first.report_path.read_text(),
                render_task_process_context_ablation_report(first_summary),
            )
            self.assertEqual(first_summary["schema_version"], "1.2.0")
            self.assertEqual(
                first_summary["status"], "development_diagnostic_only"
            )
            self.assertEqual(
                first_summary["runtime"]["resolved_device"], "cpu"
            )
            self.assertTrue(
                first_summary["runtime"]["single_device_execution"]
            )
            self.assertFalse(first_summary["formal_protocol_run"])
            self.assertFalse(first_summary["ticket_44_authorized"])
            self.assertEqual(first_summary["package_replay_mismatch_count"], 0)
            self.assertEqual(
                first_summary["training_development_template_overlap"], 0
            )
            self.assertEqual(
                first_summary[
                    "ticket_40_held_out_development_template_overlap"
                ],
                0,
            )
            self.assertEqual(set(first_summary["variants"]), set(CONTEXT_ABLATION_VARIANTS))
            for variant in first_summary["variants"].values():
                self.assertEqual(variant["parameter_count"], 12459)
                self.assertEqual(
                    variant["parameter_budget_evidence"],
                    {
                        "inert_padding_parameter_count": 0,
                        "padding_parameters_used": False,
                        "trainable_parameter_count": 12459,
                    },
                )
                self.assertEqual(len(variant["per_seed"]), 1)
                records = variant["per_seed"][0]["development_records"]
                self.assertEqual(len(records), 800)
                self.assertEqual(len({row["pair_id"] for row in records}), 800)
                self.assertTrue(
                    all(set(row) >= {"before", "after"} for row in records)
                )
                self.assertEqual(
                    variant["aggregate"]["overall"]["pair_count"], 800
                )
                self.assertEqual(
                    sum(
                        row["pair_count"]
                        for row in variant["aggregate"]["by_family"].values()
                    ),
                    800,
                )
                self.assertEqual(
                    sum(
                        row["pair_count"]
                        for row in variant["aggregate"][
                            "by_margin_stratum"
                        ].values()
                    ),
                    800,
                )
                self.assertEqual(
                    sum(
                        row["pair_count"]
                        for row in variant["aggregate"][
                            "by_oracle_flip_status"
                        ].values()
                    ),
                    800,
                )
            controls = first_summary["controls"]
            self.assertFalse(controls["ticket_40_per_pair_model_outputs_read"])
            self.assertFalse(
                controls["ticket_40_held_out_used_for_candidate_selection"]
            )
            self.assertFalse(controls["inert_padding_parameters_used"])
            for comparison in first_summary[
                "comparisons_to_current_pair_aware"
            ].values():
                self.assertEqual(len(comparison["paired_comparison_by_seed"]), 1)
                per_seed = comparison["paired_comparison_by_seed"][0]
                self.assertEqual(
                    sum(per_seed["paired_pair_win_tie_loss"].values()), 800
                )
                self.assertEqual(
                    sum(per_seed["state_transitions"].values()), 1600
                )
                self.assertEqual(
                    sum(per_seed["exact_pair_transitions"].values()), 800
                )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_task_process_context_ablation(root / "first", **kwargs)

    def test_model_builder_and_nested_inputs_support_explicit_device(self):
        robot_features, task_features, task_adjacency, md_inputs = (
            self._synthetic_batch()
        )
        model = build_context_ablation_model(
            "current_pair_aware", seed=3101
        )
        self.assertEqual(next(model.parameters()).device, torch.device("cpu"))
        moved = md_inputs.to("cpu")
        for tensor in (
            moved.robot_metadata,
            moved.task_metadata,
            moved.pair_metadata,
            moved.task_is_transport,
            moved.typed_adjacency,
            moved.downstream_task_index,
            moved.hard_feasibility_mask,
            moved.opportunity_context,
        ):
            self.assertIsNotNone(tensor)
            self.assertEqual(tensor.device, torch.device("cpu"))
        diagnostics = model.forward_with_diagnostics(
            robot_features,
            task_features,
            task_adjacency,
            md_inputs=moved,
        )
        self.assertEqual(diagnostics.scores.device, torch.device("cpu"))

    def test_device_resolution_rejects_multi_device_and_unavailable_cuda(self):
        requested, resolved = _resolve_device("auto")
        self.assertEqual(requested, "auto")
        self.assertEqual(
            resolved,
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _resolve_device("cuda:0,1")
        with patch(
            "experiments.md_task_process_context_ablation.torch.cuda.is_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                _resolve_device("cuda:0")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_single_cuda_forward_matches_cpu(self):
        robot_features, task_features, task_adjacency, md_inputs = (
            self._synthetic_batch()
        )
        cpu_model = build_context_ablation_model(
            "current_pair_aware", seed=3101, device="cpu"
        )
        cuda_model = build_context_ablation_model(
            "current_pair_aware", seed=3101, device="cuda:0"
        )
        with torch.no_grad():
            cpu_scores = cpu_model.forward_with_diagnostics(
                robot_features,
                task_features,
                task_adjacency,
                md_inputs=md_inputs,
            ).scores
            cuda_scores = cuda_model.forward_with_diagnostics(
                robot_features.to("cuda:0"),
                task_features.to("cuda:0"),
                task_adjacency.to("cuda:0"),
                md_inputs=md_inputs.to("cuda:0"),
            ).scores.cpu()
        self.assertTrue(
            torch.allclose(cpu_scores, cuda_scores, atol=1e-5, rtol=1e-5)
        )
        self.assertEqual(
            next(cuda_model.parameters()).device, torch.device("cuda:0")
        )

    def test_outcome_mapping_uses_pre_registered_literals(self):
        names = CONTEXT_ABLATION_VARIANTS[1:]

        def comparisons(*passing, late_gain=0.04):
            rows = {}
            for name in names:
                rows[name] = {
                    "all_acceptance_criteria_met": name in passing,
                    "pickup_exact_pair_gain": (
                        late_gain
                        if name == "transport_process_late_fusion"
                        else 0.05
                    ),
                    "high_margin_state_agreement_gain": 0.03,
                    "overall_exact_pair_accuracy_gain": 0.0,
                    "mean_development_residual_saturation": 0.2,
                    "pickup_state_agreement_difference_by_seed": [0.01] * 3,
                }
            return rows

        cases = (
            (False, comparisons(), "development_comparison_invalid", None),
            (True, comparisons(), "current_architecture_retained", None),
            (
                True,
                comparisons("process_context_only"),
                "process_context_supported_for_confirmation",
                "process_context_only",
            ),
            (
                True,
                comparisons("transport_context_only"),
                "transport_context_supported_for_confirmation",
                "transport_context_only",
            ),
            (
                True,
                comparisons("transport_calibration_control"),
                "transport_calibration_supported_for_confirmation",
                "transport_calibration_control",
            ),
            (
                True,
                comparisons("transport_process_late_fusion", late_gain=0.08),
                "late_fusion_supported_for_confirmation",
                "transport_process_late_fusion",
            ),
        )
        for valid, rows, outcome, selected in cases:
            with self.subTest(outcome=outcome):
                result = select_context_ablation_outcome(
                    rows, comparison_valid=valid
                )
                self.assertEqual(result["outcome"], outcome)
                self.assertEqual(result["selected_variant"], selected)

        rows = comparisons(
            "process_context_only", "transport_calibration_control"
        )
        rows["transport_calibration_control"]["pickup_exact_pair_gain"] = 1.0
        result = select_context_ablation_outcome(rows, comparison_valid=True)
        self.assertEqual(
            result["outcome"], "process_context_supported_for_confirmation"
        )
        self.assertEqual(result["selected_variant"], "process_context_only")
        self.assertNotIn(
            "transport_calibration_control", result["eligible_candidates"]
        )

    def test_formal_artifact_has_three_seed_and_package_completeness(self):
        summary = json.loads(FORMAL_SUMMARY.read_text())
        self.assertEqual(
            FORMAL_REPORT.read_text(),
            render_task_process_context_ablation_report(summary),
        )
        self.assertEqual(summary["schema_version"], "1.1.0")
        self.assertTrue(summary["formal_protocol_run"])
        self.assertEqual(summary["model_seeds"], [3101, 3102, 3103])
        self.assertEqual(summary["development_pair_count"], 800)
        self.assertEqual(summary["development_state_count"], 1600)
        self.assertEqual(summary["unique_development_pair_count"], 800)
        self.assertEqual(len(summary["development_cells"]), 16)
        self.assertTrue(
            all(cell["pair_count"] == 50 for cell in summary["development_cells"])
        )
        self.assertEqual(summary["package_replay_mismatch_count"], 0)
        self.assertEqual(
            summary["corrected_held_out_replay_mismatch_count"], 0
        )
        self.assertEqual(summary["training_development_template_overlap"], 0)
        self.assertEqual(
            summary["ticket_40_held_out_development_template_overlap"], 0
        )
        for variant in summary["variants"].values():
            self.assertEqual(variant["parameter_count"], 12459)
            self.assertEqual(
                {row["model_seed"] for row in variant["per_seed"]},
                {3101, 3102, 3103},
            )
            for row in variant["per_seed"]:
                records = row["development_records"]
                self.assertEqual(len(records), 800)
                self.assertEqual(len({record["pair_id"] for record in records}), 800)
            seed_range = variant["aggregate"]["seed_range"]
            self.assertEqual(
                set(seed_range),
                {
                    "overall",
                    "by_family",
                    "by_margin_stratum",
                    "by_oracle_flip_status",
                },
            )
            overall_metrics = [
                row["metrics"]["overall"] for row in variant["per_seed"]
            ]
            for metric, bounds in seed_range["overall"].items():
                values = [row[metric] for row in overall_metrics]
                self.assertEqual(
                    bounds,
                    {"minimum": min(values), "maximum": max(values)},
                )
            for grouping in (
                "by_family",
                "by_margin_stratum",
                "by_oracle_flip_status",
            ):
                self.assertEqual(
                    set(seed_range[grouping]),
                    set(variant["aggregate"][grouping]),
                )
                for group, ranges in seed_range[grouping].items():
                    grouped_metrics = [
                        row["metrics"][grouping][group]
                        for row in variant["per_seed"]
                    ]
                    for metric, bounds in ranges.items():
                        values = [row[metric] for row in grouped_metrics]
                        self.assertEqual(
                            bounds,
                            {"minimum": min(values), "maximum": max(values)},
                        )
        for comparison in summary[
            "comparisons_to_current_pair_aware"
        ].values():
            per_seed = comparison["paired_comparison_by_seed"]
            self.assertEqual(
                {row["model_seed"] for row in per_seed}, {3101, 3102, 3103}
            )
            for row in per_seed:
                self.assertEqual(
                    sum(row["paired_pair_win_tie_loss"].values()), 800
                )
                self.assertEqual(sum(row["state_transitions"].values()), 1600)
                self.assertEqual(
                    sum(row["exact_pair_transitions"].values()), 800
                )
            pooled = comparison["pooled_paired_comparison"]
            self.assertEqual(pooled["seed_count"], 3)
            for counter in (
                "paired_pair_win_tie_loss",
                "state_transitions",
                "exact_pair_transitions",
            ):
                expected = {
                    key: sum(row[counter][key] for row in per_seed)
                    for key in pooled[counter]
                }
                self.assertEqual(pooled[counter], expected)
            self.assertEqual(
                sum(pooled["paired_pair_win_tie_loss"].values()), 2400
            )
            self.assertEqual(sum(pooled["state_transitions"].values()), 4800)
            self.assertEqual(
                sum(pooled["exact_pair_transitions"].values()), 2400
            )
            self.assertEqual(
                comparison[
                    "pooled_paired_state_agreement_difference_ci95"
                ],
                pooled["pooled_paired_state_agreement_difference_ci95"],
            )
            self.assertEqual(
                comparison[
                    "pooled_pickup_paired_state_agreement_difference_ci95"
                ],
                pooled[
                    "pooled_pickup_paired_state_agreement_difference_ci95"
                ],
            )
        selected = summary["selection"]["selected_variant"]
        self.assertEqual(summary["ticket_44_authorized"], selected is not None)
        if not summary["comparison_valid"]:
            self.assertEqual(summary["status"], "development_comparison_invalid")
            self.assertFalse(summary["ticket_44_authorized"])

    def test_variants_match_parameter_budget_and_fixed_score_components(self):
        robot_features, task_features, task_adjacency, md_inputs = (
            self._synthetic_batch()
        )
        diagnostics = {}

        for variant in CONTEXT_ABLATION_VARIANTS:
            model = build_context_ablation_model(variant, seed=3101)
            diagnostics[variant] = model.forward_with_diagnostics(
                robot_features,
                task_features,
                task_adjacency,
                md_inputs=md_inputs,
            )
            self.assertEqual(
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                12459,
            )
            self.assertEqual(tuple(diagnostics[variant].scores.shape), (2, 3, 8))
            self.assertEqual(
                diagnostics[variant].combined_score[0, 0, 0].item(),
                -1.0e9,
            )

        baseline = diagnostics["current_pair_aware"]
        for variant in CONTEXT_ABLATION_VARIANTS[1:]:
            self.assertTrue(
                torch.equal(
                    diagnostics[variant].legacy_component,
                    baseline.legacy_component,
                )
            )
            if variant == "transport_calibration_control":
                self.assertTrue(
                    torch.all(diagnostics[variant].physics_utility <= 0)
                )
            else:
                self.assertTrue(
                    torch.equal(
                        diagnostics[variant].physics_utility,
                        baseline.physics_utility,
                    )
                )

    def test_visibility_late_fusion_gradients_and_calibration_are_observable(self):
        robot_features, task_features, task_adjacency, md_inputs = (
            self._synthetic_batch()
        )
        process_inputs = replace(
            md_inputs,
            task_metadata=md_inputs.task_metadata + 0.75,
            opportunity_context=md_inputs.opportunity_context + 0.5,
            typed_adjacency=md_inputs.typed_adjacency * 2.0,
        )
        downstream_task_features = task_features.clone()
        downstream_task_features[:, 4:] += 1.25
        transport_inputs = replace(
            md_inputs,
            robot_metadata=md_inputs.robot_metadata + 0.6,
            pair_metadata=md_inputs.pair_metadata + 0.4,
        )
        competitor_robot_features = robot_features + 0.8
        transport_model = build_context_ablation_model(
            "transport_context_only", seed=3101
        )
        process_model = build_context_ablation_model(
            "process_context_only", seed=3101
        )

        transport_raw = transport_model.forward_with_diagnostics(
            robot_features,
            task_features,
            task_adjacency,
            md_inputs=md_inputs,
        ).raw_residual
        process_raw = process_model.forward_with_diagnostics(
            robot_features,
            task_features,
            task_adjacency,
            md_inputs=md_inputs,
        ).raw_residual
        self.assertTrue(
            torch.equal(
                transport_raw,
                transport_model.forward_with_diagnostics(
                    robot_features,
                    task_features,
                    task_adjacency,
                    md_inputs=process_inputs,
                ).raw_residual,
            )
        )
        self.assertTrue(
            torch.equal(
                transport_raw,
                transport_model.forward_with_diagnostics(
                    robot_features,
                    downstream_task_features,
                    task_adjacency,
                    md_inputs=process_inputs,
                ).raw_residual,
            )
        )
        self.assertFalse(
            torch.equal(
                transport_raw,
                transport_model.forward_with_diagnostics(
                    robot_features,
                    task_features,
                    task_adjacency,
                    md_inputs=transport_inputs,
                ).raw_residual,
            )
        )
        self.assertTrue(
            torch.equal(
                process_raw,
                process_model.forward_with_diagnostics(
                    robot_features,
                    task_features,
                    task_adjacency,
                    md_inputs=transport_inputs,
                ).raw_residual,
            )
        )
        self.assertTrue(
            torch.equal(
                process_raw,
                process_model.forward_with_diagnostics(
                    competitor_robot_features,
                    task_features,
                    task_adjacency,
                    md_inputs=transport_inputs,
                ).raw_residual,
            )
        )
        self.assertFalse(
            torch.equal(
                process_raw,
                process_model.forward_with_diagnostics(
                    robot_features,
                    task_features,
                    task_adjacency,
                    md_inputs=process_inputs,
                ).raw_residual,
            )
        )

        differentiable_inputs = replace(
            md_inputs,
            robot_metadata=md_inputs.robot_metadata.clone().requires_grad_(),
            task_metadata=md_inputs.task_metadata.clone().requires_grad_(),
            pair_metadata=md_inputs.pair_metadata.clone().requires_grad_(),
            opportunity_context=(
                md_inputs.opportunity_context.clone().requires_grad_()
            ),
        )
        late_fusion = build_context_ablation_model(
            "transport_process_late_fusion", seed=3101
        )
        late_fusion.forward_with_diagnostics(
            robot_features,
            task_features,
            task_adjacency,
            md_inputs=differentiable_inputs,
        ).raw_residual.sum().backward()
        self.assertGreater(
            differentiable_inputs.pair_metadata.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            differentiable_inputs.task_metadata.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            differentiable_inputs.opportunity_context.grad.abs().sum().item(), 0
        )

        eta = md_inputs.pair_metadata.clone()
        eta[:, :, 0, 0] = 0.5
        eta[:, :, 1, 0] = 2.0
        calibration_inputs = replace(md_inputs, pair_metadata=eta)
        calibration = build_context_ablation_model(
            "transport_calibration_control", seed=3101
        ).forward_with_diagnostics(
            robot_features,
            task_features,
            task_adjacency,
            md_inputs=calibration_inputs,
        )
        self.assertTrue(torch.all(calibration.physics_utility <= 0))
        self.assertTrue(
            torch.all(
                calibration.physics_utility[:, :, 1]
                < calibration.physics_utility[:, :, 0]
            )
        )

    @staticmethod
    def _synthetic_batch():
        generator = torch.Generator().manual_seed(43)
        robot_features = torch.randn(2, 3, 7, generator=generator)
        task_features = torch.randn(2, 8, 9, generator=generator)
        task_adjacency = torch.zeros(2, 8, 8)
        typed_adjacency = torch.zeros(2, 2, 8, 8)
        for task in range(4):
            task_adjacency[:, task, task + 4] = 1
            typed_adjacency[:, 1, task, task + 4] = 1
        hard_mask = torch.zeros(2, 3, 8, dtype=torch.bool)
        hard_mask[:, :, :4] = True
        hard_mask[0, 0, 0] = False
        md_inputs = MDPolicyInputs(
            robot_metadata=torch.randn(2, 3, 4, generator=generator),
            task_metadata=torch.randn(2, 8, 8, generator=generator),
            pair_metadata=torch.rand(2, 3, 8, 5, generator=generator),
            task_is_transport=torch.tensor([[True] * 4 + [False] * 4] * 2),
            typed_adjacency=typed_adjacency,
            downstream_task_index=torch.tensor(
                [list(range(4, 8)) + [-1] * 4] * 2,
                dtype=torch.long,
            ),
            hard_feasibility_mask=hard_mask,
            opportunity_context=torch.randn(2, 3, 8, 5, generator=generator),
        )
        return robot_features, task_features, task_adjacency, md_inputs


if __name__ == "__main__":
    unittest.main()
