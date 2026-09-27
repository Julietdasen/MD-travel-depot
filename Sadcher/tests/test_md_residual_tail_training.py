import unittest

import torch
from torch import nn

from baselines.md_oracle_types import (
    ForcedAssignmentBatch, ResidualMDState, ResidualOracleResult,
    ResidualOracleStatus, enumerate_forced_batches, residual_domain,
)
from data_generation.md_residual_dataset import (
    ResidualBatchLabel, build_batch_label, project_edge_labels, robust_scale,
)
from experiments.md_residual_tail_training import (
    ResidualCostPredictions, ResidualTailPolicy, batch_margin_ranking_loss,
    residual_tail_loss, select_development_checkpoint,
)
from simulation_environment.domain_model import (
    MaterialDeliveryConfig, ProcessRobot, ProcessTask, SchedulingDomain,
    TransportRobot, TransportTask,
)


class _PairPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, robot_features, task_features, task_adjacencies=None, *, md_inputs=None):
        return torch.zeros(robot_features.shape[0], robot_features.shape[1], task_features.shape[1])

    def pair_representation(self, robot_features, task_features, task_adjacencies=None, *, md_inputs=None):
        return torch.ones(robot_features.shape[0], robot_features.shape[1], task_features.shape[1], 5) * self.weight


class ResidualTailTrainingTests(unittest.TestCase):
    def test_cost_heads_use_pair_representation_and_keep_forward_unchanged(self):
        base = _PairPolicy()
        policy = ResidualTailPolicy(base, pair_feature_dim=5, hidden_dim=4)
        robots = torch.zeros(2, 3, 7)
        tasks = torch.zeros(2, 4, 9)
        self.assertTrue(torch.equal(policy(robots, tasks), base(robots, tasks)))
        scores, costs = policy.forward_with_costs(robots, tasks)
        self.assertEqual(scores.shape, (2, 3, 4))
        self.assertEqual(costs.total.shape, (2, 3, 4))

    def test_total_cost_drives_regret_and_tied_edge_decomposition_is_averaged(self):
        a = ForcedAssignmentBatch(((0, 10),))
        b = ForcedAssignmentBatch(((0, 10), (1, 20)))
        result_a = ResidualOracleResult(ResidualOracleStatus.OPTIMAL, 99, 5, 4, 0, 0, 0)
        result_b = ResidualOracleResult(ResidualOracleStatus.OPTIMAL, 20, 7, 3, 1, 0, 0)
        la = build_batch_label(a, result_a, 9)
        lb = build_batch_label(b, result_b, 9)
        self.assertEqual((la.regret, lb.regret), (0, 1))
        edge = project_edge_labels((la, lb))[0]
        self.assertEqual(edge.best_batch, a)
        self.assertEqual((edge.total_cost, edge.task_cost, edge.tail_cost), (9.5, 6, 3.5))

    def test_state_scaling_and_masked_regression_skip_constant_heads(self):
        self.assertEqual(robust_scale((4,)), (0,))
        self.assertEqual(robust_scale((4, 4, 4)), (0, 0, 0))
        predicted = ResidualCostPredictions(*(torch.tensor([3.0], requires_grad=True) for _ in range(3)))
        target = ResidualCostPredictions(*(torch.tensor([0.0]) for _ in range(3)))
        loss = residual_tail_loss(predicted, target, regression_masks=ResidualCostPredictions(
            torch.tensor([False]), torch.tensor([True]), torch.tensor([False])))
        self.assertAlmostEqual(float(loss), 0.625)

    def test_ranking_is_tie_aware_and_prefers_lower_regret_batch(self):
        labels = (
            ResidualBatchLabel(ForcedAssignmentBatch(((0, 0),)), 0, 0, 0, None, 0),
            ResidualBatchLabel(ForcedAssignmentBatch(((1, 1),)), 0, 0, 0, None, 0.5),
            ResidualBatchLabel(ForcedAssignmentBatch(((0, 1),)), 0, 0, 0, None, 3),
        )
        scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        self.assertEqual(float(batch_margin_ranking_loss(scores, labels)), 0.0)
        self.assertGreater(float(batch_margin_ranking_loss(-scores, labels)), 0.0)

    def test_checkpoint_selection_uses_regret_then_tail_then_earlier_epoch(self):
        rows = ({"epoch": 20, "forced_action_regret": 2, "terminal_return_tail": 3},
                {"epoch": 10, "forced_action_regret": 2, "terminal_return_tail": 3},
                {"epoch": 30, "forced_action_regret": 2, "terminal_return_tail": 2})
        self.assertEqual(select_development_checkpoint(rows)["epoch"], 30)

    def test_satisfied_pending_material_edge_is_removed_from_residual_domain(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=True),
            tasks=(TransportTask(1, (0, 0), (1, 0), 1, 0, 0, downstream_process_task_id=2), ProcessTask(2, (1, 0), 1, (True,))),
            robots=(TransportRobot(1, (0, 0), True, 2, 1), ProcessRobot(0, (0, 0), (True,))),
            material_edges=((1, 2),),
        )
        state = ResidualMDState(0, {0:(0,0),1:(0,0)}, (1,), (2,), ((1,2),))
        self.assertEqual(residual_domain(domain, state).material_edges, ())

    def test_all_inclusion_minimal_coalition_sizes_are_retained(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=True),
            tasks=(ProcessTask(1, (0, 0), 1, (True, True)),),
            robots=(ProcessRobot(0, (0, 0), (True, True)), ProcessRobot(1, (0, 0), (True, False)), ProcessRobot(2, (0, 0), (False, True))),
        )
        state = ResidualMDState(0, {0:(0,0),1:(0,0),2:(0,0)}, (), (1,))
        batches = {x.assignments for x in enumerate_forced_batches(domain, state)}
        self.assertEqual(batches, {((0, 1),), ((1, 1), (2, 1))})

    def test_enumerates_transport_singletons_and_minimal_process_coalitions(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=True),
            tasks=(TransportTask(1, (0, 0), (1, 0), 1, 0, 0, downstream_process_task_id=2), ProcessTask(2, (1, 0), 1, (True, True))),
            robots=(TransportRobot(2, (0, 0), True, 2, 1), ProcessRobot(0, (0, 0), (True, False)), ProcessRobot(1, (0, 0), (False, True))),
            material_edges=((1, 2),),
        )
        state = ResidualMDState(0, {0:(0,0),1:(0,0),2:(0,0)}, (), (1,2))
        batches = {x.assignments for x in enumerate_forced_batches(domain, state)}
        self.assertEqual(batches, {((2, 1),)})


if __name__ == "__main__":
    unittest.main()
