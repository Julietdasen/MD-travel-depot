import json
import tempfile
import unittest
from pathlib import Path

from experiments.md_results_package import rebuild_results_package
from experiments.protocol import DatasetSplit, ExperimentResult, FailureReason


class MDResultsPackageTests(unittest.TestCase):
    def test_one_command_rebuilds_auditable_tables_plot_and_failure_classes(self):
        success = ExperimentResult.succeeded(
            run_id="learned-a-0",
            method="learned",
            instance_id="a",
            seed=0,
            split=DatasetSplit.TEST,
            makespan=12,
            all_real_tasks_completed=True,
            all_robots_at_exit=True,
            material_starvation={"1": 2},
            robot_utilization={"0": 0.75},
            inference_time_seconds=0.01,
            metadata={"gurobi_gap": 0.02},
        )
        timeout = ExperimentResult.failed(
            run_id="learned-b-1",
            method="learned",
            instance_id="b",
            seed=1,
            split=DatasetSplit.TEST,
            reason=FailureReason.TIMEOUT,
            all_real_tasks_completed=False,
            all_robots_at_exit=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw.json").write_text(
                json.dumps([success.to_dict(), timeout.to_dict()]), encoding="utf-8"
            )
            manifest = {
                "schema_version": "1.0.0",
                "status": "complete",
                "seeds": [0, 1],
                "generation_config": {"task_count": 8},
                "baseline_config": {"masked_greedy": {}},
                "checkpoint": {"identifier": "sha256:test", "path": "checkpoint.pt"},
                "environment": {"conda_env": "mrta-sadcher"},
                "raw_results": "raw.json",
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            paths = rebuild_results_package(manifest_path, root / "rebuilt")

            self.assertTrue(all(path.is_file() for path in paths.values()))
            summary = json.loads(paths["summary"].read_text())
            self.assertEqual(summary["methods"]["learned"]["success_rate"], 0.5)
            self.assertEqual(summary["methods"]["learned"]["makespan"]["mean"], 12)
            failures = json.loads(paths["failures"].read_text())
            self.assertEqual(len(failures["timeouts"]), 1)
            self.assertEqual(failures["gurobi_gaps"][0]["gurobi_gap"], 0.02)
            self.assertGreater(paths["plot"].stat().st_size, 1000)

    def test_rejects_private_absolute_manifest_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema_version": "1.0.0",
                "status": "formal_experiments_pending",
                "seeds": [],
                "generation_config": {},
                "baseline_config": {},
                "checkpoint": {"identifier": None, "path": None},
                "environment": {},
                "raw_results": "/private/raw.json",
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "portable relative paths"):
                rebuild_results_package(path, root / "output")


if __name__ == "__main__":
    unittest.main()

