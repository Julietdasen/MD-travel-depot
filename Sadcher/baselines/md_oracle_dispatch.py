"""Solver-agnostic dispatch for the MD MILP oracle.

The project ships a single implementation of the MD MILP formulation:

* ``baselines.md_ortools_scheduler.solve_ortools_md_oracle`` (default,
  license-free CP-SAT).

This module is the single entry point every runtime caller should use.
The Gurobi backend (``baselines.gurobi_md_oracle``) was removed on
2026-09-21 — the WLS license had expired and CP-SAT has been the
production backend since 2026-09-17.  See
``ReadME/2026-09-17/BASELINE_SWAP_TO_ORTOOLS.md`` for the swap history
and feasibility evidence.

The environment variable ``MRTA_MILP_SOLVER`` is retained for
backwards-compatible call sites; only ``ortools`` is accepted.  Any
other value raises ``ValueError`` so stale scripts fail loudly instead
of silently falling back.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from baselines.md_oracle_types import GurobiOracleResult

# Public alias so callers can refer to the shared return type without depending
# on the legacy module name.
OracleResult = GurobiOracleResult

DEFAULT_SOLVER = "ortools"
_ENV_FLAG = "MRTA_MILP_SOLVER"
_VALID = ("ortools",)


def active_solver_name() -> str:
    """Return the solver name selected by the environment (validated)."""

    raw = os.environ.get(_ENV_FLAG, DEFAULT_SOLVER).strip().lower()
    if raw not in _VALID:
        raise ValueError(
            f"{_ENV_FLAG}={raw!r} is not supported; expected one of {_VALID}. "
            "The Gurobi backend was removed on 2026-09-21; only OR-Tools "
            "CP-SAT is available."
        )
    return raw


def _load(name: str) -> Callable[..., GurobiOracleResult]:
    if name == "ortools":
        from baselines.md_ortools_scheduler import solve_ortools_md_oracle

        return solve_ortools_md_oracle
    raise ValueError(f"unknown solver {name!r}")


def solve_md_oracle(*args, **kwargs) -> GurobiOracleResult:
    """Dispatch to the configured MD MILP solver.

    Accepts the same arguments as the individual solver functions.  The active
    solver is chosen by :func:`active_solver_name`.  Callers should treat the
    return value as a :class:`GurobiOracleResult` regardless of the backend.
    """

    solver = _load(active_solver_name())
    return solver(*args, **kwargs)


def _load_residual(name: str) -> Callable[..., Any]:
    if name == "ortools":
        from baselines.md_ortools_residual_oracle import (
            solve_residual_forced_batch as _fn,
        )

        return _fn
    raise ValueError(f"unknown solver {name!r}")


def solve_residual_forced_batch(*args, **kwargs):
    """Dispatch to the configured residual forced-batch oracle."""

    solver = _load_residual(active_solver_name())
    return solver(*args, **kwargs)


__all__ = [
    "DEFAULT_SOLVER",
    "OracleResult",
    "active_solver_name",
    "solve_md_oracle",
    "solve_residual_forced_batch",
]
