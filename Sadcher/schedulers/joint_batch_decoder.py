"""Batch-aware decoder for the optional residual joint head."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch

from baselines.gurobi_md_residual_oracle import enumerate_forced_batches
from data_generation.md_residual_generation import eligible_snapshot, residual_state
from schedulers.md_constrained_decoder import (
    DecoderResult,
    LearnedConstrainedDecoder,
    simulator_hard_mask,
)


class JointBatchDecoder:
    """Choose one legal forced batch using a caller-provided joint scorer."""

    name = "joint_batch"

    def decode(
        self,
        scores: torch.Tensor,
        simulator,
        *,
        domain,
        batch_scorer: Callable[[tuple[tuple[int, int], ...]], float],
        hard_feasibility_mask: torch.Tensor | None = None,
    ) -> DecoderResult:
        mask = simulator_hard_mask(simulator) if hard_feasibility_mask is None else hard_feasibility_mask
        if not eligible_snapshot(simulator):
            return LearnedConstrainedDecoder().decode(scores, simulator, hard_feasibility_mask=mask)
        if not bool(torch.any(mask)):
            return LearnedConstrainedDecoder().decode(scores, simulator, hard_feasibility_mask=mask)
        state = residual_state(type("Candidate", (), {"simulator": simulator, "domain": domain})())
        batches = enumerate_forced_batches(domain, state)
        if not batches:
            return LearnedConstrainedDecoder().decode(scores, simulator, hard_feasibility_mask=mask)
        selected = min(batches, key=lambda batch: float(batch_scorer(batch.assignments)))
        repaired = LearnedConstrainedDecoder().decode(
            scores,
            simulator,
            hard_feasibility_mask=mask,
        ) if not selected.assignments else _result_for_batch(selected.assignments, scores, simulator, mask)
        return replace(repaired, decoder=self.name)


def _result_for_batch(assignments, scores, simulator, mask):
    # Reuse the existing repair and validation implementation by applying the
    # selected legal batch through a decoder-shaped result.
    from schedulers.md_constrained_decoder import FastAssignmentRepair
    return FastAssignmentRepair().repair(assignments, scores, simulator, hard_feasibility_mask=mask)

