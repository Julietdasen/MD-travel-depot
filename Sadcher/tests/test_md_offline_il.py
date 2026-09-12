import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from baselines.gurobi_md_oracle import OracleAction
from data_generation.md_dataset import MDInstanceRecord
from data_generation.md_expert_dataset import (
    ExpertQuality,
    generate_md_expert_record,
    save_md_expert_record,
)
from data_generation.md_instance_generator import (
    GeneratedMDInstance,
    MDGeneratorConfig,
)
from experiments.protocol import DatasetSplit, task_level_split
from imitation_learning.md_train import (
    MDILTrainingConfig,
    load_md_policy_checkpoint,
    train_md_policy,
)
from models.md_legacy_features import legacy_policy_inputs_from_samples
from models.md_policy_features import md_policy_inputs_from_samples
from tests.fixtures.md_diagnostic_scenarios import LATE_MATERIAL


def _group_for(split: DatasetSplit) -> str:
    for index in range(10_000):
        candidate = f"offline-il-{split.value}-{index}"
        if task_level_split(candidate) is split:
            return candidate
    raise AssertionError(f"unable to find deterministic {split.value} group")


def _record(group: str, suffix: str):
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
        instance_id=f"late-material-{suffix}",
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


class MDOfflineILTests(unittest.TestCase):
    def test_training_rejects_task_group_leakage_across_splits(self):
        train_record = _record(_group_for(DatasetSplit.TRAIN), "group-train")
        validation_record = _record(
            _group_for(DatasetSplit.VALIDATION), "group-validation"
        )
        validation_payload = validation_record.to_dict()
        validation_payload["task_group_id"] = train_record.task_group_id
        for sample in validation_payload["decision_samples"]:
            sample["task_group_id"] = train_record.task_group_id

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            save_md_expert_record(dataset / "train.json", train_record)
            (dataset / "validation.json").write_text(
                json.dumps(validation_payload), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "task groups must be disjoint"):
                train_md_policy(
                    dataset,
                    root / "output",
                    config=MDILTrainingConfig(epochs=1, hidden_dim=16),
                )

    def test_training_rejects_split_label_that_disagrees_with_task_group(self):
        train_record = _record(_group_for(DatasetSplit.TRAIN), "label-train")
        mismatched_group = next(
            candidate
            for index in range(10_000)
            if (
                (candidate := f"offline-il-mislabeled-train-{index}")
                != train_record.task_group_id
                and task_level_split(candidate) is DatasetSplit.TRAIN
            )
        )
        validation_record = _record(
            _group_for(DatasetSplit.VALIDATION), "label-validation"
        )
        validation_payload = validation_record.to_dict()
        validation_payload["task_group_id"] = mismatched_group
        for sample in validation_payload["decision_samples"]:
            sample["task_group_id"] = mismatched_group

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            save_md_expert_record(dataset / "train.json", train_record)
            (dataset / "validation.json").write_text(
                json.dumps(validation_payload), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "disagree with task-level split"):
                train_md_policy(
                    dataset,
                    root / "output",
                    config=MDILTrainingConfig(epochs=1, hidden_dim=16),
                )

    def test_cpu_tiny_train_is_split_safe_solver_free_and_reloadable(self):
        train_record = _record(_group_for(DatasetSplit.TRAIN), "train")
        validation_record = _record(
            _group_for(DatasetSplit.VALIDATION), "validation"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "output"
            save_md_expert_record(dataset / "train.json", train_record)
            save_md_expert_record(
                dataset / "validation.json", validation_record
            )
            config = MDILTrainingConfig(
                epochs=2,
                batch_size=2,
                learning_rate=1e-3,
                seed=41,
                hidden_dim=16,
            )

            with (
                patch(
                    "baselines.gurobi_md_oracle.solve_gurobi_md_oracle"
                ) as gurobi_solve,
                patch("pulp.LpProblem.solve") as pulp_solve,
            ):
                result = train_md_policy(dataset, output, config=config)

            gurobi_solve.assert_not_called()
            pulp_solve.assert_not_called()
            self.assertEqual(result.train_instance_ids, (train_record.instance_id,))
            self.assertEqual(
                result.validation_instance_ids,
                (validation_record.instance_id,),
            )
            self.assertTrue(
                set(result.train_instance_ids).isdisjoint(
                    result.validation_instance_ids
                )
            )
            self.assertEqual(result.train_sample_count, 2)
            self.assertEqual(result.validation_sample_count, 2)
            self.assertTrue(result.checkpoint_path.is_file())
            self.assertTrue(result.summary_path.is_file())
            self.assertTrue(torch.isfinite(torch.tensor(result.train_losses)).all())
            self.assertTrue(
                torch.isfinite(torch.tensor(result.validation_losses)).all()
            )

            model, checkpoint = load_md_policy_checkpoint(result.checkpoint_path)
            validation_samples = validation_record.samples
            legacy = legacy_policy_inputs_from_samples(validation_samples)
            md_inputs = md_policy_inputs_from_samples(validation_samples)
            with torch.no_grad():
                scores = model(
                    legacy.robot_features,
                    legacy.task_features,
                    legacy.task_adjacency,
                    md_inputs=md_inputs,
                )
            self.assertTrue(torch.isfinite(scores).all())
            self.assertEqual(checkpoint["dataset"]["train_instance_ids"], [train_record.instance_id])
            self.assertEqual(
                checkpoint["dataset"]["validation_instance_ids"],
                [validation_record.instance_id],
            )
            self.assertEqual(checkpoint["solver_calls_during_training"], 0)

    def test_optional_structured_and_value_losses_are_solver_free_and_checkpointed(self):
        train_record = _record(_group_for(DatasetSplit.TRAIN), "enhanced-train")
        validation_record = _record(
            _group_for(DatasetSplit.VALIDATION), "enhanced-validation"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            save_md_expert_record(dataset / "train.json", train_record)
            save_md_expert_record(dataset / "validation.json", validation_record)
            config = MDILTrainingConfig(
                epochs=1,
                batch_size=2,
                seed=43,
                hidden_dim=16,
                use_structured_loss=True,
                use_value_head=True,
                max_feasible_assignments=100,
            )

            with (
                patch("baselines.gurobi_md_oracle.solve_gurobi_md_oracle") as gurobi,
                patch("pulp.LpProblem.solve") as pulp_solve,
            ):
                result = train_md_policy(dataset, root / "output", config=config)

            gurobi.assert_not_called()
            pulp_solve.assert_not_called()
            _, checkpoint = load_md_policy_checkpoint(result.checkpoint_path)
            self.assertTrue(checkpoint["training_enhancements"]["use_structured_loss"])
            self.assertTrue(checkpoint["training_enhancements"]["use_value_head"])
            self.assertIsNotNone(checkpoint["value_head_state_dict"])
            self.assertEqual(checkpoint["solver_calls_during_training"], 0)


if __name__ == "__main__":
    unittest.main()
