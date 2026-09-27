import tempfile
import unittest
from pathlib import Path

from baselines.md_oracle_types import (
    GurobiMDModel,
    GurobiOracleResult,
    GurobiOracleStatus,
    OracleAction,
    OracleScheduleEntry,
    replay_oracle_actions,
)
from data_generation.md_dataset import MDInstanceRecord
from data_generation.md_expert_dataset import (
    ExpertQuality,
    MDExpertDatasetLoader,
    generate_md_expert_record,
    generate_md_expert_record_from_oracle,
    load_md_expert_record,
    save_md_expert_record,
)
from data_generation.md_instance_generator import (
    GeneratedMDInstance,
    MDGeneratorConfig,
)
from experiments.protocol import DatasetSplit
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
)
from tests.fixtures.md_diagnostic_scenarios import LATE_MATERIAL


def _actions() -> tuple[OracleAction, ...]:
    return (
        OracleAction(0, 0, 2, 0.0, 0.0),
        OracleAction(1, 1, 3, 0.0, 1.0),
        OracleAction(2, 0, 1, 7.0, 7.0),
    )


def _instance() -> MDInstanceRecord:
    config = MDGeneratorConfig(
        seed=LATE_MATERIAL.seed,
        task_count=3,
        transport_ratio=1 / 3,
        precedence_density=1.0,
        critical_path_length=2,
        process_robot_count=1,
        transport_robot_count=1,
        skill_count=1,
    )
    return MDInstanceRecord(
        instance_id=LATE_MATERIAL.name,
        task_group_id="late-material-family",
        seed=LATE_MATERIAL.seed,
        generated=GeneratedMDInstance(LATE_MATERIAL.domain, config),
    )


class MDExpertDatasetTests(unittest.TestCase):
    def test_oracle_replay_emits_auditable_decision_samples(self):
        record = generate_md_expert_record(
            _instance(),
            _actions(),
            quality=ExpertQuality.OPTIMAL,
            exit_location=LATE_MATERIAL.exit_location,
            max_steps=30,
            expected_makespan=13.0,
        )

        self.assertEqual(record.quality, ExpertQuality.OPTIMAL)
        self.assertEqual(record.terminal_makespan, 13.0)
        self.assertEqual(len(record.samples), 2)
        first, second = record.samples
        self.assertEqual(first.decision_time, 0)
        self.assertEqual(first.robot_ids, (0, 1))
        self.assertEqual(first.task_ids, (1, 2, 3))
        self.assertEqual(first.typed_graph.normal_edges, ((2, 1),))
        self.assertEqual(first.typed_graph.material_edges, ((3, 1),))
        # Under the arrival-contract, robot 0 may commit to task 1 while its
        # precursors are still in progress; simulator.is_task_ready still gates
        # actual service start.
        self.assertEqual(
            first.hard_feasibility_mask,
            ((True, True, False), (False, False, True)),
        )
        self.assertEqual(
            first.capacity_feasibility,
            ((False, False, False), (False, False, True)),
        )
        self.assertEqual(first.expert_actions, ((0, 2), (1, 3)))
        self.assertEqual(first.expert_assignment, first.expert_reward)
        process_one = next(state for state in first.tasks if state.task_id == 1)
        self.assertFalse(process_one.ready)
        self.assertFalse(process_one.material_ready)
        process_one_later = next(state for state in second.tasks if state.task_id == 1)
        self.assertTrue(process_one_later.ready)
        self.assertTrue(process_one_later.material_ready)
        self.assertEqual(second.decision_time, 7)
        self.assertEqual(second.expert_actions, ((0, 1),))
        self.assertEqual(first.remaining_makespan, 13.0)
        self.assertEqual(second.remaining_makespan, 6.0)

        canonical = replay_oracle_actions(
            LATE_MATERIAL.domain,
            _actions(),
            exit_location=LATE_MATERIAL.exit_location,
            run_id="canonical",
            instance_id=LATE_MATERIAL.name,
            seed=LATE_MATERIAL.seed,
            split=record.split,
            max_steps=30,
        )
        self.assertEqual(record.terminal_result["metrics"]["makespan"], canonical.makespan)
        canonical_result = canonical.to_dict()
        self.assertEqual(
            record.terminal_result["execution_records"]["process"],
            canonical_result["execution_records"]["process"],
        )
        self.assertEqual(
            record.terminal_result["execution_records"]["transport"],
            canonical_result["execution_records"]["transport"],
        )

    def test_round_trip_loader_preserves_instance_level_split(self):
        record = generate_md_expert_record(
            _instance(),
            _actions(),
            quality=ExpertQuality.HEURISTIC,
            exit_location=LATE_MATERIAL.exit_location,
            max_steps=30,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.json"
            save_md_expert_record(path, record)

            self.assertEqual(load_md_expert_record(path), record)
            loader = MDExpertDatasetLoader(directory)
            self.assertEqual(loader.load_records(split=record.split), (record,))
            self.assertEqual(
                loader.load_records(quality=ExpertQuality.OPTIMAL), ()
            )
            samples = loader.load_samples(split=record.split)
            self.assertEqual(samples, record.samples)
            self.assertTrue(all(sample.split is record.split for sample in samples))
            self.assertTrue(
                all(sample.instance_id == record.instance_id for sample in samples)
            )

    def test_oracle_quality_is_explicit_and_legacy_sources_are_rejected(self):
        instance = _instance()
        model = GurobiMDModel(
            domain=instance.generated.domain,
            exit_location=LATE_MATERIAL.exit_location,
            normal_precedence=((2, 1),),
            material_precedence=((3, 1),),
            transport_eligible_robots={3: (1,)},
            process_eligible_robots={1: (0,), 2: (0,)},
            process_skill_coverage={1: {0: (0,)}, 2: {0: (0,)}},
            objective="minimize_latest_robot_exit_return",
            big_m=100,
        )
        oracle = GurobiOracleResult(
            status=GurobiOracleStatus.FEASIBLE,
            feasible=True,
            solve_time_seconds=0.1,
            timeout=True,
            optimality_gap=0.2,
            objective=13.0,
            action_order=_actions(),
            schedule=(
                OracleScheduleEntry(0, 2, 0.0, 1.0),
                OracleScheduleEntry(1, 3, 1.0, 7.0),
                OracleScheduleEntry(0, 1, 7.0, 9.0),
            ),
            model=model,
            message="time-limited incumbent",
        )

        record = generate_md_expert_record_from_oracle(
            instance,
            oracle,
            exit_location=LATE_MATERIAL.exit_location,
            max_steps=30,
        )
        self.assertEqual(record.quality, ExpertQuality.TIME_LIMITED_FEASIBLE)
        self.assertEqual(record.expert_metadata["optimality_gap"], 0.2)
        self.assertEqual(len(record.expert_metadata["oracle_schedule"]), 3)

        legacy_domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=False),
            tasks=(ProcessTask(1, (0, 0), 1, (True,)),),
            robots=(ProcessRobot(0, (0, 0), (True,)),),
        )
        legacy = MDInstanceRecord(
            instance_id="legacy-process-only",
            task_group_id="legacy",
            seed=1,
            generated=GeneratedMDInstance(
                legacy_domain,
                MDGeneratorConfig(
                    seed=1,
                    task_count=1,
                    transport_ratio=0,
                    precedence_density=0,
                    critical_path_length=1,
                    process_robot_count=1,
                    transport_robot_count=0,
                    skill_count=1,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "enabled MD instance"):
            generate_md_expert_record(
                legacy,
                (OracleAction(0, 0, 1, 0, 0),),
                quality=ExpertQuality.HEURISTIC,
                max_steps=5,
            )


if __name__ == "__main__":
    unittest.main()
