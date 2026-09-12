from __future__ import annotations

import torch

from experiments.md_prefix_conditioned_reranker import (
    PrefixCostScorer,
    load_states,
    normalization,
)
from experiments.md_prefix_features import EDGE_FEATURE_DIM, PREFIX_FEATURE_DIM


def test_cached_loader_reads_only_frozen_development_ranges() -> None:
    states = load_states()
    assert len(states) == 630
    assert {state.split for state in states} == {"train", "development"}
    assert all(
        75000 <= state.seed <= 75599
        if state.split == "train"
        else 75600 <= state.seed <= 75749
        for state in states
    )


def test_model_forward_backward_is_finite() -> None:
    torch.manual_seed(1)
    model = PrefixCostScorer(hidden_dim=8)
    sequence = torch.randn(3, EDGE_FEATURE_DIM + PREFIX_FEATURE_DIM)
    output = model(sequence)
    output.backward()
    assert torch.isfinite(output)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_normalization_uses_training_rows_and_is_finite() -> None:
    states = load_states()
    mean, std = normalization(states)
    assert mean.shape == (EDGE_FEATURE_DIM,)
    assert std.shape == (EDGE_FEATURE_DIM,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert torch.all(std >= 1)
