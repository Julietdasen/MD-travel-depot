import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.gurobi_md_oracle import OracleAction
from data_generation.md_dataset import MDInstanceRecord
from data_generation.md_expert_dataset import (
    ExpertQuality,
    generate_md_expert_record,
    save_md_expert_record,
)
from data_generation.md_instance_generator import GeneratedMDInstance, MDGeneratorConfig
from experiments.md_offline_il_pilot import run_md_offline_il_pilot
from experiments.protocol import DatasetSplit, task_level_split
from imitation_learning.md_train import MDILTrainingConfig
from tests.fixtures.md_diagnostic_scenarios import LATE_MATERIAL


def _group_for(split: DatasetSplit) -> str:
    for index in range(10_000):
        candidate = f"offline-il-pilot-{split.value}-{index}"
        if task_level_split(candidate) is split:
            return candidate
    raise AssertionError(f"unable to find deterministic {split.value} group")


def _late_material_expert_record(group: str, suffix: str):
    config = MDGeneratorConfig(
        seed=LATE_MATERIAL.seed,
        task_count=3,
        transport_ratio=1 / 3,
        precedence_density=1,
        critical_path_length=2,
        process_robot_count=1,
        transport_robot_count=1,
        skill_count=1,
    )
    instance = MDInstanceRecord(
        instance_id=f"late-material-pilot-{suffix}",
        task_group_id=group,
        seed=LATE_MATERIAL.seed,
        generated=GeneratedMDInstance(LATE_MATERIAL.domain, config),
    )
    return generate_md_expert_record(
        instance,
        (
            OracleAction(0, 0, 2, 0, 0),
            OracleAction(1, 1, 3, 0, 0),
            OracleAction(2, 0, 1, 7, 7),
        ),
        quality=ExpertQuality.OPTIMAL,
        exit_location=LATE_MATERIAL.exit_location,
        max_steps=30,
    )


class MDOfflineILPilotTests(unittest.TestCase):
    def test_pilot_is_solver_free_finite_and_writes_auditable_artifacts(self):
        train_record = _late_material_expert_record(
            _group_for(DatasetSplit.TRAIN), "train"
        )
        validation_record = _late_material_expert_record(
            _group_for(DatasetSplit.VALIDATION), "validation"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "md_offline_il_pilot_2026-08-27"
            save_md_expert_record(dataset / "train.json", train_record)
            save_md_expert_record(dataset / "validation.json", validation_record)

            with (
                patch("baselines.gurobi_md_oracle.solve_gurobi_md_oracle") as gurobi,
                patch("pulp.LpProblem.solve") as pulp_solve,
            ):
                paths = run_md_offline_il_pilot(
                    dataset,
                    output,
                    config=MDILTrainingConfig(
                        epochs=1,
                        batch_size=2,
                        seed=47,
                        hidden_dim=16,
                    ),
                    run_date="2026-08-27",
                )

            gurobi.assert_not_called()
            pulp_solve.assert_not_called()
            self.assertEqual(paths["summary"].parent, output)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            self.assertGreater(paths["training_loss_plot"].stat().st_size, 1_000)
            self.assertGreater(paths["dataset_split_plot"].stat().st_size, 1_000)

            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "pilot")
            self.assertEqual(summary["ticket"], 23)
            self.assertTrue(summary["limitations"]["formal_benchmark_pending"])
            self.assertEqual(summary["dataset"]["record_counts"]["train"], 1)
            self.assertEqual(summary["dataset"]["record_counts"]["validation"], 1)
            self.assertEqual(summary["dataset"]["sample_counts"]["train"], 2)
            self.assertEqual(summary["dataset"]["sample_counts"]["validation"], 2)
            self.assertEqual(summary["training"]["solver_calls"], 0)
            self.assertTrue(summary["training"]["all_losses_finite"])
            self.assertTrue(summary["validation"]["all_scores_finite"])
            self.assertEqual(summary["validation"]["evaluated_samples"], 2)
            self.assertGreaterEqual(
                summary["validation"]["masked_top1_expert_pair_accuracy"], 0.0
            )
            self.assertLessEqual(
                summary["validation"]["masked_top1_expert_pair_accuracy"], 1.0
            )
            report = paths["report"].read_text(encoding="utf-8")
            self.assertIn("Pilot", report)
            self.assertIn("not a formal benchmark", report)


if __name__ == "__main__":
    unittest.main()
