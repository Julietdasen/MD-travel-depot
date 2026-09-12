import unittest

import torch

from baselines.gurobi_md_oracle import OracleAction
from data_generation.md_dataset import MDInstanceRecord
from data_generation.md_expert_dataset import (
    ExpertQuality,
    generate_md_expert_record,
)
from data_generation.md_instance_generator import (
    GeneratedMDInstance,
    MDGeneratorConfig,
)
from models.md_policy import MD_OPPORTUNITY_FEATURE_COUNT, MDOpportunityFeature
from models.md_policy_features import md_policy_inputs_from_samples
from tests.fixtures.md_diagnostic_scenarios import LATE_MATERIAL


class MDPolicyFeatureTests(unittest.TestCase):
    def test_decision_sample_builds_explicit_transport_and_graph_tensors(self):
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
        record = generate_md_expert_record(
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

        inputs = md_policy_inputs_from_samples((record.samples[0],))

        self.assertEqual(tuple(inputs.robot_metadata.shape), (1, 2, 4))
        self.assertEqual(tuple(inputs.task_metadata.shape), (1, 3, 8))
        self.assertEqual(tuple(inputs.pair_metadata.shape), (1, 2, 3, 5))
        self.assertEqual(
            tuple(inputs.opportunity_context.shape),
            (1, 2, 3, MD_OPPORTUNITY_FEATURE_COUNT),
        )
        self.assertEqual(inputs.pair_metadata[0, 1, 2, 0], 6.0)
        self.assertEqual(inputs.pair_metadata[0, 1, 2, 1], 5.0)
        self.assertEqual(inputs.pair_metadata[0, 1, 2, 2], 0.0)
        self.assertEqual(inputs.pair_metadata[0, 1, 2, 3], 1.0)
        # The fifth transport pair feature is the physical loaded leg distance.
        self.assertEqual(inputs.pair_metadata[0, 1, 2, 4], 4.0)
        self.assertEqual(inputs.downstream_task_index.tolist(), [[-1, -1, 0]])
        # Criticality, last-blocker, coalition availability/scarcity, and
        # transport capacity scarcity are explicit current-state proxies.
        transport_context = inputs.opportunity_context[0, 1, 2]
        self.assertGreater(
            float(transport_context[MDOpportunityFeature.CRITICAL_PATH_PROXY]), 0.0
        )
        self.assertEqual(
            float(transport_context[MDOpportunityFeature.LAST_MATERIAL_BLOCKER]), 0.0
        )
        self.assertEqual(
            float(transport_context[MDOpportunityFeature.COALITION_AVAILABILITY]), 1.0
        )
        self.assertEqual(
            float(transport_context[MDOpportunityFeature.COALITION_SCARCITY]), 1.0
        )
        self.assertEqual(
            float(transport_context[MDOpportunityFeature.CAPACITY_SCARCITY]), 1.0
        )
        self.assertTrue(
            torch.equal(
                inputs.opportunity_context[0, :, :2],
                torch.zeros(2, 2, MD_OPPORTUNITY_FEATURE_COUNT),
            )
        )
        self.assertEqual(inputs.typed_adjacency[0, 0, 1, 0], 1.0)
        self.assertEqual(inputs.typed_adjacency[0, 1, 2, 0], 1.0)
        self.assertTrue(
            torch.equal(
                inputs.hard_feasibility_mask,
                torch.tensor(record.samples[0].hard_feasibility_mask).unsqueeze(0),
            )
        )


if __name__ == "__main__":
    unittest.main()
