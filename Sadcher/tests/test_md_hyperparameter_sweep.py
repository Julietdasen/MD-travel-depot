import unittest

import torch

from experiments.md_hyperparameter_sweep import (
    SWEEP_CONFIGS,
    _scheduler,
    _validate_config,
)


class HyperparameterSweepTests(unittest.TestCase):
    def test_registered_configs_are_valid_and_c0_preserves_defaults(self):
        for name, config in SWEEP_CONFIGS.items():
            _validate_config(name, config)
        self.assertEqual(SWEEP_CONFIGS["C0"]["weight_decay"], 0.0)
        self.assertEqual(SWEEP_CONFIGS["C0"]["scheduler"], "none")

    def test_scheduler_none_is_noop_and_cosine_changes_learning_rate(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        self.assertIsNone(_scheduler(optimizer, name="none", epochs=2))
        scheduler = _scheduler(optimizer, name="cosine", epochs=2)
        self.assertIsNotNone(scheduler)
        before = optimizer.param_groups[0]["lr"]
        scheduler.step()
        self.assertLess(optimizer.param_groups[0]["lr"], before)


if __name__ == "__main__":
    unittest.main()
