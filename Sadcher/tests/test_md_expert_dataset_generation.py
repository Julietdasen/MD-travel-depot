import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.gurobi_md_oracle import (
    GurobiOracleResult,
    GurobiOracleStatus,
    OracleAction,
    OracleScheduleEntry,
    build_gurobi_md_model,
)
from data_generation.md_expert_dataset import MDExpertDatasetLoader
from data_generation.md_expert_dataset_generation import (
    MDExpertGenerationConfig,
    _generation_exit_code,
    generate_md_expert_dataset,
)
from data_generation.md_instance_generator import GeneratedMDInstance
from tests.fixtures.md_diagnostic_scenarios import LATE_MATERIAL


def _actions():
    return (
        OracleAction(0, 0, 2, 0.0, 0.0),
        OracleAction(1, 1, 3, 0.0, 1.0),
        OracleAction(2, 0, 1, 7.0, 7.0),
    )


def _schedule():
    return (
        OracleScheduleEntry(0, 2, 0.0, 1.0),
        OracleScheduleEntry(1, 3, 1.0, 7.0),
        OracleScheduleEntry(0, 1, 7.0, 9.0),
    )


class MDExpertDatasetGenerationTests(unittest.TestCase):
    def test_batch_generation_replays_oracle_and_writes_records(self):
        calls = []

        def fake_solver(domain, **kwargs):
            calls.append((domain, kwargs))
            return GurobiOracleResult(
                status=GurobiOracleStatus.OPTIMAL,
                feasible=True,
                solve_time_seconds=0.01,
                timeout=False,
                optimality_gap=0.0,
                objective=13.0,
                action_order=_actions(),
                schedule=_schedule(),
                model=build_gurobi_md_model(domain),
                message="fake optimal oracle",
            )

        config = MDExpertGenerationConfig(
            count=2,
            seed_start=31,
            task_group_prefix="family",
            instance_prefix="md",
            max_steps=30,
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "data_generation.md_expert_dataset_generation.generate_md_instance",
                side_effect=lambda item: GeneratedMDInstance(
                    LATE_MATERIAL.domain, item
                ),
            ):
                summary = generate_md_expert_dataset(
                    directory, config=config, oracle_solver=fake_solver
                )

            self.assertEqual(summary.generated_count, 2)
            self.assertEqual(summary.skipped_count, 0)
            self.assertTrue(summary.manifest_path.is_file())
            self.assertEqual(
                tuple(path.name for path in summary.record_paths),
                ("md-seed-000031.json", "md-seed-000032.json"),
            )
            records = MDExpertDatasetLoader(directory).load_records()
            self.assertEqual(tuple(record.seed for record in records), (31, 32))
            self.assertEqual(
                tuple(record.terminal_makespan for record in records),
                (13.0, 13.0),
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1]["time_limit_seconds"], 60.0)

    def test_seed_based_ids_allow_disjoint_shards_in_one_directory(self):
        def fake_solver(domain, **kwargs):
            return GurobiOracleResult(
                GurobiOracleStatus.OPTIMAL,
                True,
                0.01,
                False,
                0.0,
                13.0,
                _actions(),
                _schedule(),
                build_gurobi_md_model(domain),
                "fake optimal oracle",
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "data_generation.md_expert_dataset_generation.generate_md_instance",
                side_effect=lambda item: GeneratedMDInstance(
                    LATE_MATERIAL.domain, item
                ),
            ):
                first = generate_md_expert_dataset(
                    directory,
                    config=MDExpertGenerationConfig(
                        count=1, seed_start=3, max_steps=30
                    ),
                    oracle_solver=fake_solver,
                )
                second = generate_md_expert_dataset(
                    directory,
                    config=MDExpertGenerationConfig(
                        count=1, seed_start=4, max_steps=30
                    ),
                    oracle_solver=fake_solver,
                )

            self.assertNotEqual(first.record_paths, second.record_paths)
            self.assertEqual(
                len(MDExpertDatasetLoader(directory).load_records()), 2
            )

    def test_schedule_mismatch_is_audited_without_writing_labels(self):
        def mismatched_solver(domain, **kwargs):
            return GurobiOracleResult(
                GurobiOracleStatus.FEASIBLE,
                True,
                0.01,
                True,
                0.2,
                13.0,
                _actions(),
                (
                    *_schedule()[:2],
                    OracleScheduleEntry(0, 1, 7.0, 10.0),
                ),
                build_gurobi_md_model(domain),
                "mismatched incumbent",
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "data_generation.md_expert_dataset_generation.generate_md_instance",
                side_effect=lambda item: GeneratedMDInstance(
                    LATE_MATERIAL.domain, item
                ),
            ):
                summary = generate_md_expert_dataset(
                    directory,
                    config=MDExpertGenerationConfig(count=1, max_steps=30),
                    oracle_solver=mismatched_solver,
                )

            self.assertEqual(summary.generated_count, 0)
            self.assertEqual(
                summary.skipped[0].status, "replay_schedule_mismatch"
            )
            self.assertEqual(tuple(Path(directory).glob("*.json")), ())

    def test_unavailable_oracle_is_skipped_without_pseudo_labels(self):
        def unavailable_solver(domain, **kwargs):
            return GurobiOracleResult(
                status=GurobiOracleStatus.UNAVAILABLE,
                feasible=None,
                solve_time_seconds=0.0,
                timeout=False,
                optimality_gap=None,
                objective=None,
                action_order=(),
                schedule=(),
                model=build_gurobi_md_model(domain),
                message="license unavailable",
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "data_generation.md_expert_dataset_generation.generate_md_instance",
                side_effect=lambda item: GeneratedMDInstance(
                    LATE_MATERIAL.domain, item
                ),
            ):
                summary = generate_md_expert_dataset(
                    directory,
                    config=MDExpertGenerationConfig(count=1, max_steps=30),
                    oracle_solver=unavailable_solver,
                )

            self.assertEqual(summary.generated_count, 0)
            self.assertEqual(summary.skipped_count, 1)
            self.assertEqual(summary.skipped[0].status, "unavailable")
            self.assertEqual(tuple(Path(directory).glob("*.json")), ())
            with self.assertRaises(FileExistsError):
                generate_md_expert_dataset(
                    directory,
                    config=MDExpertGenerationConfig(count=1, max_steps=30),
                    oracle_solver=unavailable_solver,
                )

    def test_config_rejects_invalid_count(self):
        with self.assertRaisesRegex(ValueError, "count"):
            MDExpertGenerationConfig(count=0)

    def test_config_rejects_path_like_prefix_and_invalid_generator_values(self):
        with self.assertRaisesRegex(ValueError, "instance_prefix"):
            MDExpertGenerationConfig(count=1, instance_prefix="../escape")
        with self.assertRaisesRegex(ValueError, "task_count"):
            MDExpertGenerationConfig(count=1, task_count=0)

    def test_all_output_conflicts_are_detected_before_oracle_calls(self):
        calls = []

        def unused_solver(domain, **kwargs):
            calls.append(domain)
            raise AssertionError("oracle must not be called")

        with tempfile.TemporaryDirectory() as directory:
            conflict = Path(directory) / "md-instance-seed-000001.json"
            conflict.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_md_expert_dataset(
                    directory,
                    config=MDExpertGenerationConfig(count=2),
                    oracle_solver=unused_solver,
                )
            self.assertEqual(calls, [])
            self.assertFalse(
                (Path(directory) / "md-instance-seed-000000.json").exists()
            )

    def test_partial_batch_requires_explicit_cli_opt_in(self):
        from data_generation.md_expert_dataset_generation import (
            MDExpertGenerationSummary,
        )

        summary = MDExpertGenerationSummary(Path("out"), 2, 1, (), ())
        self.assertEqual(_generation_exit_code(summary, allow_partial=False), 2)
        self.assertEqual(_generation_exit_code(summary, allow_partial=True), 0)


if __name__ == "__main__":
    unittest.main()
