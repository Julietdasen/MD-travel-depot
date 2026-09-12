import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from data_generation.md_dataset import (
    MD_DATASET_SCHEMA_VERSION,
    MDDatasetLoader,
    MDInstanceRecord,
    load_md_domain,
    load_md_instance,
    save_md_instance,
)
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from experiments.legacy_regression import LEGACY_INSTANCE_FIELDS, load_legacy_problem
from experiments.protocol import DatasetSplit, task_level_split


class MDDatasetLoaderTests(unittest.TestCase):
    def setUp(self):
        self.config = MDGeneratorConfig(
            seed=23,
            task_count=10,
            transport_ratio=0.2,
            precedence_density=0.5,
            critical_path_length=3,
            capacity_slack=0.25,
            speed_ratio=0.7,
        )
        generated = generate_md_instance(self.config)
        self.record = MDInstanceRecord(
            instance_id="md-task10-seed23",
            task_group_id="md-task10",
            seed=self.config.seed,
            generated=generated,
            solution={"status": "unavailable", "assignments": []},
            execution_records={
                "process": [{"task_id": 1, "start": 0.0, "end": 1.0}],
                "transport": [],
            },
            metrics={"makespan": None, "success": False},
        )

    def test_instance_round_trip_preserves_domain_and_traceability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instance.json"
            save_md_instance(path, self.record)
            payload = json.loads(path.read_text())

            self.assertEqual(payload["schema_version"], MD_DATASET_SCHEMA_VERSION)
            self.assertEqual(payload["instance_id"], self.record.instance_id)
            self.assertEqual(payload["task_group_id"], self.record.task_group_id)
            self.assertEqual(payload["seed"], self.record.seed)
            self.assertEqual(payload["generation_config"]["seed"], self.record.seed)
            self.assertEqual(payload["solution"], {"status": "unavailable", "assignments": []})
            self.assertEqual(payload["execution_records"], {"process": [{"task_id": 1, "start": 0.0, "end": 1.0}], "transport": []})
            self.assertEqual(payload["metrics"], {"makespan": None, "success": False})

            loaded = load_md_instance(path)
            self.assertEqual(load_md_domain(path), loaded.generated.domain)
            self.assertEqual(loaded, self.record)
            self.assertEqual(loaded.generated.domain, self.record.generated.domain)
            self.assertEqual(
                loaded.generated.generation_config,
                self.record.generated.generation_config,
            )

    def test_loader_assigns_stable_task_level_split_and_rejects_group_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_md_instance(root / "first.json", self.record)
            same_group = MDInstanceRecord(
                instance_id="md-task10-seed99",
                task_group_id=self.record.task_group_id,
                seed=99,
                generated=generate_md_instance(
                    MDGeneratorConfig(
                        seed=99,
                        task_count=self.config.task_count,
                        transport_ratio=self.config.transport_ratio,
                        precedence_density=self.config.precedence_density,
                        critical_path_length=self.config.critical_path_length,
                        capacity_slack=self.config.capacity_slack,
                        speed_ratio=self.config.speed_ratio,
                    )
                ),
            )
            save_md_instance(root / "second.json", same_group)

            loader = MDDatasetLoader(root)
            records = loader.load()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].split, task_level_split(self.record.task_group_id))
            self.assertEqual(records[0].split, records[1].split)
            self.assertEqual(loader.load(split=records[0].split), records)

            self.assertEqual(loader.load(task_group_id="different-group"), ())

    def test_loader_filters_by_split_without_decision_sample_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_md_instance(root / "one.json", self.record)
            other = MDInstanceRecord(
                instance_id="another-instance",
                task_group_id="another-group",
                seed=4,
                generated=generate_md_instance(
                    MDGeneratorConfig(seed=4, task_count=10, transport_ratio=0.2)
                ),
            )
            save_md_instance(root / "two.json", other)
            loader = MDDatasetLoader(root)

            all_records = loader.load()
            self.assertEqual(len(all_records), 2)
            chosen_split = task_level_split(self.record.task_group_id)
            chosen = loader.load(split=chosen_split)
            self.assertTrue(all(record.split is chosen_split for record in chosen))
            self.assertEqual(len(chosen), sum(record.split is chosen_split for record in all_records))

    def test_loader_rejects_duplicate_instance_ids_and_missing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_md_instance(root / "one.json", self.record)
            duplicate = MDInstanceRecord(
                instance_id=self.record.instance_id,
                task_group_id="other-group",
                seed=24,
                generated=generate_md_instance(
                    MDGeneratorConfig(seed=24, task_count=10, transport_ratio=0.2)
                ),
            )
            save_md_instance(root / "two.json", duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate MD instance_id"):
                MDDatasetLoader(root).load()

            payload = self.record.to_dict()
            payload.pop("split")
            path = root / "missing-split.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "missing required MD instance fields"):
                load_md_instance(path)

    def test_execution_records_require_process_and_transport_arrays(self):
        with self.assertRaisesRegex(
            ValueError, "execution_records must contain exactly process and transport arrays"
        ):
            MDInstanceRecord(
                instance_id="invalid-records",
                task_group_id="invalid-group",
                seed=self.config.seed,
                generated=generate_md_instance(self.config),
                execution_records={"events": []},
            )

    def test_legacy_loader_still_loads_the_official_fixture(self):
        fixture = (
            Path(__file__).parent
            / "fixtures"
            / "legacy_8t3r3s"
            / "problem_instances"
            / "problem_instance_1p_000000.json"
        )

        loaded = load_legacy_problem(fixture)

        self.assertEqual(set(loaded), LEGACY_INSTANCE_FIELDS)
        self.assertEqual(loaded["R"].shape[0], 10)

    def test_records_and_loader_are_immutable_and_json_safe(self):
        with self.assertRaises(FrozenInstanceError):
            self.record.instance_id = "changed"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instance.json"
            save_md_instance(path, self.record)
            loaded = load_md_instance(path)
            with self.assertRaises(FrozenInstanceError):
                loaded.generated = None
            self.assertEqual(json.loads(path.read_text()), json.loads(path.read_text()))

    def test_malformed_or_unsupported_documents_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"schema_version": "1.0.0"}))
            with self.assertRaisesRegex(ValueError, "missing required MD instance fields"):
                load_md_instance(path)

            path.write_text(
                json.dumps(
                    {
                        "schema_version": "9.9.9",
                        "instance_id": self.record.instance_id,
                        "task_group_id": self.record.task_group_id,
                        "seed": self.record.seed,
                        "generation_config": {},
                        "domain": {},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "unsupported MD schema_version"):
                load_md_instance(path)


if __name__ == "__main__":
    unittest.main()
