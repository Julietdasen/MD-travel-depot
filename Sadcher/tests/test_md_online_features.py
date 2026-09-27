import tempfile
import unittest
from pathlib import Path

import torch

from baselines.md_oracle_types import OracleAction
from data_generation.md_dataset import MDInstanceRecord
from data_generation.md_expert_dataset import (
    ExpertQuality,
    generate_md_expert_record,
)
from data_generation.md_instance_generator import GeneratedMDInstance, MDGeneratorConfig
from experiments.protocol import DatasetSplit
from models.md_legacy_features import legacy_policy_inputs_from_samples
from models.md_policy_features import md_policy_inputs_from_samples
from models.md_online_features import (
    build_md_policy_inputs_from_simulator,
    build_md_policy_state_from_simulator,
)
from schedulers.md_constrained_decoder import simulator_hard_mask
from schedulers.online_md_scheduler import OnlineNeuralScoreProvider
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from tests.fixtures.md_diagnostic_scenarios import COALITION, LATE_MATERIAL


def _late_material_record():
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
    )


class MDOnlineFeatureTests(unittest.TestCase):
    def test_online_and_offline_features_are_equal_for_the_same_state(self):
        record = _late_material_record()
        simulator = MDDiscreteSimulator(
            LATE_MATERIAL.domain, exit_location=LATE_MATERIAL.exit_location
        )

        online = build_md_policy_inputs_from_simulator(simulator)
        offline_legacy = legacy_policy_inputs_from_samples((record.samples[0],))
        offline_md = md_policy_inputs_from_samples((record.samples[0],))

        self.assertTrue(torch.equal(online[0], offline_legacy.robot_features))
        self.assertTrue(torch.equal(online[1], offline_legacy.task_features))
        self.assertTrue(torch.equal(online[2], offline_legacy.task_adjacency))
        self.assertTrue(torch.equal(online[3].robot_metadata, offline_md.robot_metadata))
        self.assertTrue(torch.equal(online[3].task_metadata, offline_md.task_metadata))
        self.assertTrue(torch.equal(online[3].pair_metadata, offline_md.pair_metadata))
        self.assertTrue(torch.equal(online[3].typed_adjacency, offline_md.typed_adjacency))
        self.assertTrue(
            torch.equal(
                online[3].hard_feasibility_mask,
                offline_md.hard_feasibility_mask,
            )
        )

    def test_blocked_task_is_context_visible_but_not_a_legal_action(self):
        simulator = MDDiscreteSimulator(
            LATE_MATERIAL.domain, exit_location=LATE_MATERIAL.exit_location
        )
        state = build_md_policy_state_from_simulator(simulator)
        inputs = build_md_policy_inputs_from_simulator(simulator)[3]

        blocked_task_index = state.task_ids.index(LATE_MATERIAL.process_task_id)
        self.assertGreater(float(inputs.task_metadata[0, blocked_task_index, 4]), 0.0)
        self.assertGreater(float(inputs.task_metadata[0, blocked_task_index, 5]), 0.0)
        self.assertFalse(bool(inputs.hard_feasibility_mask[:, :, blocked_task_index].any()))

    def test_online_mask_is_the_centralized_simulator_mask(self):
        simulator = MDDiscreteSimulator(
            LATE_MATERIAL.domain, exit_location=LATE_MATERIAL.exit_location
        )
        inputs = build_md_policy_inputs_from_simulator(simulator)[3]
        self.assertTrue(torch.equal(inputs.hard_feasibility_mask, simulator_hard_mask(simulator).unsqueeze(0)))

    def test_zero_distance_and_heterogeneous_shapes_are_finite(self):
        coalition_simulator = MDDiscreteSimulator(
            COALITION.domain, exit_location=COALITION.exit_location
        )
        coalition_inputs = build_md_policy_inputs_from_simulator(coalition_simulator)
        self.assertEqual(tuple(coalition_inputs[0].shape), (1, 2, 7))
        self.assertEqual(tuple(coalition_inputs[1].shape), (1, 1, 9))
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in coalition_inputs[:3]))
        self.assertTrue(torch.isfinite(coalition_inputs[3].pair_metadata).all())

        config = MDGeneratorConfig(
            seed=701,
            task_count=6,
            transport_ratio=0.5,
            precedence_density=0.4,
            critical_path_length=2,
            process_robot_count=4,
            transport_robot_count=1,
            skill_count=3,
        )
        from data_generation.md_instance_generator import generate_md_instance

        generated = generate_md_instance(config)
        heterogeneous_simulator = MDDiscreteSimulator(generated.domain)
        heterogeneous_inputs = build_md_policy_inputs_from_simulator(heterogeneous_simulator)
        self.assertEqual(tuple(heterogeneous_inputs[0].shape), (1, 5, 7))
        self.assertEqual(tuple(heterogeneous_inputs[1].shape), (1, 6, 9))
        self.assertTrue(torch.isfinite(heterogeneous_inputs[0]).all())
        self.assertTrue(torch.isfinite(heterogeneous_inputs[1]).all())
        self.assertTrue(torch.isfinite(heterogeneous_inputs[3].pair_metadata).all())

    def test_real_provider_accepts_the_bridge_and_returns_finite_scores(self):
        from experiments.md_task_process_context_models import build_context_ablation_model

        model = build_context_ablation_model("current_pair_aware", seed=3101)
        provider = OnlineNeuralScoreProvider(
            model,
            build_md_policy_inputs_from_simulator,
            device="cpu",
        )
        simulator = MDDiscreteSimulator(
            LATE_MATERIAL.domain, exit_location=LATE_MATERIAL.exit_location
        )
        output = provider(simulator)
        self.assertEqual(tuple(output.scores.shape), (2, 3))
        self.assertTrue(torch.isfinite(output.scores).all())
        self.assertGreaterEqual(output.encoder_time_seconds, 0.0)
        self.assertGreaterEqual(output.scoring_time_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
