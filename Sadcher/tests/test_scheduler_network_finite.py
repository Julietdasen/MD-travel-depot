import unittest

import torch

from models.scheduler_network import SchedulerNetwork


class SchedulerNetworkFiniteTests(unittest.TestCase):
    def _network(self) -> SchedulerNetwork:
        torch.manual_seed(7)
        return SchedulerNetwork(
            robot_input_dimensions=7,
            task_input_dimension=9,
            embed_dim=16,
            ff_dim=32,
            n_transformer_heads=4,
            n_transformer_layers=1,
            n_gatn_heads=4,
            n_gatn_layers=1,
            use_idle=False,
        )

    def _assert_finite_forward_backward(
        self, robot_features: torch.Tensor, task_features: torch.Tensor
    ) -> None:
        network = self._network()
        scores = network(robot_features, task_features)

        self.assertTrue(torch.isfinite(scores).all())
        scores.sum().backward()
        gradients = [
            parameter.grad
            for parameter in network.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_random_forward_and_backward_are_finite(self):
        self._assert_finite_forward_backward(
            torch.randn(2, 3, 7), torch.randn(2, 5, 9)
        )

    def test_zero_distance_forward_and_backward_are_finite(self):
        robot_features = torch.zeros(2, 3, 7)
        task_features = torch.zeros(2, 5, 9)

        self._assert_finite_forward_backward(robot_features, task_features)


if __name__ == "__main__":
    unittest.main()
