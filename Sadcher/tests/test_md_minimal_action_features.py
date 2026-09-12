import torch

from baselines.exact_online_action_oracle import CompleteOnlineAction
from experiments.md_minimal_action_features import FEATURE_DIM, build_minimal_action_features
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from tests.fixtures.md_diagnostic_scenarios import COALITION, UNLOCK


def test_features_are_order_invariant_and_compact():
    simulator = MDDiscreteSimulator(COALITION.domain, exit_location=COALITION.exit_location)
    first = CompleteOnlineAction(((0, 1), (1, 1)), ())
    second = CompleteOnlineAction(((1, 1), (0, 1)), ())
    left = build_minimal_action_features(simulator, first)
    right = build_minimal_action_features(simulator, second)
    assert tuple(left.shape) == (2, FEATURE_DIM)
    assert torch.equal(left, right)
    assert bool(torch.isfinite(left).all())


def test_transport_features_include_observable_eta_and_return_context():
    simulator = MDDiscreteSimulator(UNLOCK.domain, exit_location=UNLOCK.exit_location)
    features = build_minimal_action_features(simulator, ((1, 2),))
    assert tuple(features.shape) == (1, FEATURE_DIM)
    assert float(features[0, 1]) == 1.0
    assert float(features[0, 3]) > 0.0
    assert float(features[0, 5]) > float(features[0, 4])
    assert float(features[0, 13]) > 0.0
