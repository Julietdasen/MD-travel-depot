import unittest

from baselines.md_oracle_types import OracleAction
from data_generation.md_dataset import MDInstanceRecord
from data_generation.md_expert_dataset import (
    ExpertQuality,
    generate_md_expert_record,
)
from data_generation.md_instance_generator import (
    GeneratedMDInstance,
    MDGeneratorConfig,
)
from experiments.md_policy_smoke import (
    independent_ablation_configs,
    run_md_policy_smoke,
)
from models.md_policy import MDPolicyConfig
from models.md_legacy_features import legacy_policy_inputs_from_samples
from tests.fixtures.md_diagnostic_scenarios import LATE_MATERIAL


def _samples():
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
        instance_id=LATE_MATERIAL.name,
        task_group_id="late-material-family",
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
    ).samples


class MDPolicySmokeTests(unittest.TestCase):
    def test_legacy_feature_contract_is_7_by_9_and_finite(self):
        inputs = legacy_policy_inputs_from_samples(_samples())

        self.assertEqual(tuple(inputs.robot_features.shape), (2, 2, 7))
        self.assertEqual(tuple(inputs.task_features.shape), (2, 3, 9))
        self.assertEqual(tuple(inputs.task_adjacency.shape), (2, 3, 3))
        self.assertTrue(inputs.robot_features.isfinite().all())
        self.assertTrue(inputs.task_features.isfinite().all())

    def test_each_ablation_disables_only_its_target_module(self):
        base = MDPolicyConfig(hidden_dim=16)
        configs = independent_ablation_configs(base)

        self.assertEqual(
            set(configs), {"typed_edges", "downstream", "eta", "capacity"}
        )
        for name, config in configs.items():
            changed = [
                field
                for field in (
                    "use_typed_edges",
                    "use_downstream_encoding",
                    "use_transport_eta",
                    "use_capacity_features",
                )
                if getattr(config, field) != getattr(base, field)
            ]
            self.assertEqual(len(changed), 1, name)
            self.assertFalse(getattr(config, changed[0]))

    def test_deterministic_cpu_train_validation_and_masked_rollout_are_finite(self):
        samples = _samples()
        configs = {
            "full": MDPolicyConfig(hidden_dim=16),
            **independent_ablation_configs(MDPolicyConfig(hidden_dim=16)),
        }
        results = {
            name: run_md_policy_smoke(samples, md_config=config, seed=31)
            for name, config in configs.items()
        }

        full_repeat = run_md_policy_smoke(
            samples, md_config=configs["full"], seed=31
        )
        self.assertEqual(results["full"], full_repeat)
        self.assertEqual({result.sample_count for result in results.values()}, {2})
        self.assertEqual({result.illegal_assignment_count for result in results.values()}, {0})
        self.assertEqual(len({result.hard_mask_digest for result in results.values()}), 1)
        self.assertEqual(len({result.split_digest for result in results.values()}), 1)
        for result in results.values():
            self.assertTrue(result.forward_finite)
            self.assertTrue(result.backward_finite)
            self.assertTrue(result.validation_finite)
            self.assertGreaterEqual(result.train_loss, 0.0)
            self.assertGreaterEqual(result.validation_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
