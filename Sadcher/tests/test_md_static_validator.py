import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from data_generation.md_dataset import MDInstanceRecord, save_md_instance
from data_generation.md_instance_generator import (
    MDGeneratorConfig,
    generate_md_instance,
)
from data_generation.md_validated_loader import load_validated_md_instance
from experiments.protocol import DatasetSplit, ExperimentResult, FailureReason
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)
from simulation_environment.md_static_validator import (
    StaticValidationCode,
    StaticValidationError,
    require_valid_md_domain,
    validate_md_domain,
)


def valid_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=True),
        tasks=(
            ProcessTask(1, (4, 0), 2, (True, False)),
            ProcessTask(2, (8, 0), 3, (False, True)),
            TransportTask(3, (0, 0), (4, 0), 5, 1, 1),
        ),
        robots=(
            ProcessRobot(0, (0, 0), (True, True)),
            TransportRobot(1, (0, 0), True, 5, 2),
        ),
        normal_edges=((1, 2),),
        material_edges=((3, 1),),
    )


class MDStaticValidatorTests(unittest.TestCase):
    def test_valid_generated_domain_passes_and_loader_enforces_the_gate(self):
        generated = generate_md_instance(
            MDGeneratorConfig(seed=31, task_count=10, transport_ratio=0.2)
        )
        result = validate_md_domain(generated.domain)

        self.assertTrue(result.is_valid)
        self.assertIsNone(result.reason_code)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(require_valid_md_domain(generated.domain), generated.domain)

        record = MDInstanceRecord(
            instance_id="md-static-valid",
            task_group_id="md-static",
            seed=31,
            generated=generated,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instance.json"
            save_md_instance(path, record)
            self.assertEqual(load_validated_md_instance(path), record)

    def test_structural_errors_have_stable_reason_codes(self):
        domain = valid_domain()
        process_one, process_two, transport = domain.tasks
        cases = (
            (
                "non-continuous task IDs",
                replace(domain, tasks=(replace(process_one, task_id=4), process_two, transport)),
                StaticValidationCode.INVALID_ID,
            ),
            (
                "invalid task entity type",
                replace(domain, tasks=(object(), process_two, transport)),
                StaticValidationCode.INVALID_ENTITY_TYPE,
            ),
            (
                "unknown edge reference",
                replace(domain, normal_edges=((1, 99),)),
                StaticValidationCode.INVALID_REFERENCE,
            ),
            (
                "duplicate material target",
                replace(
                    domain,
                    tasks=domain.tasks
                    + (TransportTask(4, (1, 0), (4, 0), 1, 1, 1),),
                    material_edges=((3, 1), (4, 1)),
                ),
                StaticValidationCode.INVALID_MATERIAL_RELATION,
            ),
            (
                "delivery location mismatch",
                replace(
                    domain,
                    tasks=(
                        process_one,
                        process_two,
                        replace(transport, delivery_location=(5, 0)),
                    ),
                ),
                StaticValidationCode.INVALID_LOCATION,
            ),
            (
                "normal cycle",
                replace(
                    domain,
                    tasks=(
                        replace(process_one, normal_predecessors=(2,)),
                        replace(process_two, normal_predecessors=(1,)),
                        transport,
                    ),
                    normal_edges=((1, 2), (2, 1)),
                ),
                StaticValidationCode.INVALID_GRAPH,
            ),
        )

        for name, invalid_domain, expected_code in cases:
            with self.subTest(name=name):
                result = validate_md_domain(invalid_domain)
                self.assertFalse(result.is_valid)
                self.assertIs(result.reason_code, expected_code)
                self.assertIs(result.failure_reason, FailureReason.INVALID_GRAPH)

    def test_numeric_values_are_rechecked_at_the_validation_boundary(self):
        domain = valid_domain()
        object.__setattr__(domain.tasks[0], "duration", float("nan"))

        result = validate_md_domain(domain)

        self.assertFalse(result.is_valid)
        self.assertIs(result.reason_code, StaticValidationCode.INVALID_NUMERIC_VALUE)
        self.assertIs(result.failure_reason, FailureReason.INVALID_GRAPH)

    def test_value_and_feature_dimension_codes_are_precise(self):
        invalid_value = valid_domain()
        object.__setattr__(invalid_value.tasks[0], "requirements", (1, False))

        value_result = validate_md_domain(invalid_value)

        self.assertIs(value_result.reason_code, StaticValidationCode.INVALID_VALUE)
        self.assertIs(value_result.failure_reason, FailureReason.INVALID_GRAPH)

        dimension_domain = valid_domain()
        dimension_domain = replace(
            dimension_domain,
            robots=(
                replace(dimension_domain.robots[0], capabilities=(True,)),
                dimension_domain.robots[1],
            ),
        )

        dimension_result = validate_md_domain(dimension_domain)

        self.assertIs(
            dimension_result.reason_code,
            StaticValidationCode.INCOMPATIBLE_FEATURE_DIMENSION,
        )
        self.assertIs(dimension_result.failure_reason, FailureReason.INVALID_GRAPH)

    def test_empty_domain_is_static_infeasibility(self):
        empty = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=False),
            tasks=(),
            robots=(),
        )

        result = validate_md_domain(empty)

        self.assertFalse(result.is_valid)
        self.assertIs(result.reason_code, StaticValidationCode.EMPTY_DOMAIN)
        self.assertIs(result.failure_reason, FailureReason.STATIC_INFEASIBLE)

    def test_transport_capability_and_solo_capacity_are_static_infeasibility(self):
        domain = valid_domain()
        process_robot, transport_robot = domain.robots

        no_capable_robot = replace(
            domain,
            robots=(process_robot, replace(transport_robot, transport_capable=False)),
        )
        no_capable_result = validate_md_domain(no_capable_robot)
        self.assertIs(
            no_capable_result.reason_code,
            StaticValidationCode.NO_CAPABLE_TRANSPORT_ROBOT,
        )
        self.assertIs(
            no_capable_result.failure_reason,
            FailureReason.NO_CAPABLE_TRANSPORT_ROBOT,
        )

        insufficient_capacity = replace(
            domain,
            robots=(process_robot, replace(transport_robot, capacity=4)),
        )
        capacity_result = validate_md_domain(insufficient_capacity)
        self.assertIs(
            capacity_result.reason_code,
            StaticValidationCode.INSUFFICIENT_SOLO_CAPACITY,
        )
        self.assertIs(capacity_result.failure_reason, FailureReason.STATIC_INFEASIBLE)

    def test_uncoverable_process_skills_are_static_infeasibility(self):
        domain = valid_domain()
        invalid = replace(
            domain,
            robots=(
                replace(domain.robots[0], capabilities=(True, False)),
                domain.robots[1],
            ),
        )

        result = validate_md_domain(invalid)

        self.assertFalse(result.is_valid)
        self.assertIs(
            result.reason_code,
            StaticValidationCode.NO_CAPABLE_PROCESS_COALITION,
        )
        self.assertIs(result.failure_reason, FailureReason.STATIC_INFEASIBLE)

    def test_invalid_domain_is_rejected_before_rollout_without_fake_makespan(self):
        domain = valid_domain()
        invalid = replace(
            domain,
            robots=(domain.robots[0], replace(domain.robots[1], capacity=1)),
        )
        validation = validate_md_domain(invalid)

        with self.assertRaises(StaticValidationError) as raised:
            require_valid_md_domain(invalid)
        self.assertIs(
            raised.exception.result.reason_code,
            StaticValidationCode.INSUFFICIENT_SOLO_CAPACITY,
        )

        failed_run = ExperimentResult.failed(
            run_id="validator-invalid-0",
            method="validator",
            instance_id="invalid",
            seed=0,
            split=DatasetSplit.TEST,
            reason=validation.failure_reason,
            all_real_tasks_completed=False,
            all_robots_at_exit=False,
            metadata={"static_validation_code": validation.reason_code.value},
        )
        self.assertFalse(failed_run.success)
        self.assertIsNone(failed_run.makespan)


if __name__ == "__main__":
    unittest.main()
