# Baseline MILP Swap: Gurobi → OR-Tools CP-SAT (2026-09-17)

## Motivation

The project's Gurobi WLS license (`2770405`) expired.  The restricted license
that ships with the pip package caps model size around ~2000 variables, which
is well below what Path A instances (tc ∈ {24, 42, 60}) require.  Without a
license refresh, every runtime path that touched the MD MILP oracle would
either time out inside the restricted solver or crash with a size limit
error.  Rather than block on the license renewal we ported the formulation
to Google OR-Tools' CP-SAT solver (installed with `uv add ortools`), which
is license-free.

The feasibility evidence — CP-SAT hits proven-optimal in seconds across the
whole Path A grid — is captured in
`ReadME/2026-09-17/ORTOOLS_FEASIBILITY.md` and
`reports/md_ortools_feasibility_2026-09-17/`.

## Scope of the swap

1. **Base MD MILP oracle** — new module
   `baselines/md_ortools_scheduler.py::solve_ortools_md_oracle` re-implements
   the formulation with integer time variables, per-robot Circuit-based
   sequencing, and the same objective (`makespan_weight * makespan +
   sum(assignment)`).  Return type is the shared `GurobiOracleResult`
   dataclass so downstream callers do not change.
2. **Dispatch layer** — `baselines/md_oracle_dispatch.py` picks the active
   backend at import time via `MRTA_MILP_SOLVER` (default `ortools`, opt-in
   `gurobi`).  Both `solve_md_oracle` and the new
   `solve_residual_forced_batch` dispatch functions live here.
3. **Residual forced-batch oracle** — new module
   `baselines/md_ortools_residual_oracle.py::solve_residual_forced_batch`
   is the CP-SAT twin of
   `baselines/gurobi_md_residual_oracle.solve_residual_forced_batch`.  It
   reuses the pure-Python helpers (`ResidualMDState`,
   `ForcedAssignmentBatch`, `ResidualOracleResult`, `ResidualOracleStatus`,
   `residual_domain`, `residual_task_map`, `_latest_return_robot`,
   `enumerate_forced_batches`) from the Gurobi module so nothing on the
   caller side has to change.  Semantics preserved: forced assignments must
   appear as each robot's *first* task, `forbidden_immediate_robot_ids`
   pushes the first task's start by at least one tick (plus transport
   travel), and `first_action` pins both the coalition membership and the
   exact start time for the exact-action oracle.
4. **Runtime call-site migration** — every experiment/data-generation
   entry point that previously imported `solve_gurobi_md_oracle` directly
   now imports `solve_md_oracle` from the dispatch module.  Data-type-only
   imports (`OracleAction`, `OracleScheduleEntry`, `GurobiOracleResult`,
   `GurobiOracleStatus`) still come from `baselines.gurobi_md_oracle`
   because they live there as reference definitions and neither backend
   redefines them.
5. **Test hygiene** — `tests/test_gurobi_md_oracle.py` is untouched (all
   its Gurobi calls run through hand-rolled `SimpleNamespace` fakes that do
   not need a license, so the suite still passes without Gurobi).  The two
   IL tests (`tests/test_md_offline_il.py`,
   `tests/test_md_offline_il_pilot.py`) additionally patch
   `baselines.md_ortools_scheduler.solve_ortools_md_oracle` so their
   "solver-free training" assertions cover the new default backend too.

## Usage

```bash
# Default: CP-SAT (no license needed)
MRTA_MILP_SOLVER=ortools \
  /root/miniconda3/envs/md/bin/uv run python <entry-point>.py ...

# Switch back to Gurobi (requires a live license)
MRTA_MILP_SOLVER=gurobi \
  /root/miniconda3/envs/md/bin/uv run python <entry-point>.py ...
```

The dispatch validates the value against `("ortools", "gurobi")` and raises
early on typos.

## Verified equivalence

* Delta-0 replay smoke test — `reports/md_ortools_feasibility_2026-09-17/
  replay_smoke_test.py` solves four Path A cells with CP-SAT and replays
  the `OracleAction` order through the canonical simulator.  All cases
  produce `replay_makespan == solver_makespan` (delta = 0).
* Residual forced-batch smoke — CP-SAT twin returns delta-0 replay against
  the canonical simulator for LATE_MATERIAL under
  `ForcedAssignmentBatch(((0, 2), (1, 3)))`, and correctly returns
  `ResidualOracleStatus.ERROR` when the batch is skill/type illegal.
* Forbidden-idle semantics — first-task start on a forbidden robot is
  ≥ 1 tick as required by the exact-action complement.
* First-action pinning — `first_action=((1, 3),)` locks robot 1's first
  task and forbids any other robot from joining that task's coalition.
* Existing unit tests unchanged behavior — `test_gurobi_md_oracle`,
  `test_md_offline_il`, `test_md_offline_il_pilot`,
  `test_md_expert_dataset*`, `test_exact_online_action_oracle` all still
  pass under the new default.

## Known limitations / risks

* CP-SAT wall time on tc = 60 dependency-deep hits ~19 s (still well under
  the 300 s budget) — expect noisier wall times than Gurobi on the largest
  Path A cells but no infeasibility.  Path B (bigger scales) has not been
  benchmarked.
* Residual CP-SAT oracle's runtime on saturated tc = 42 residual snapshots
  has not been stress-tested end-to-end; a full residual regeneration
  campaign remains future work.  For Path A data generation, only the base
  oracle is on the critical path, so this does not block us.
* The Gurobi modules remain in `baselines/` and are exercised by
  `test_gurobi_md_oracle.py` via fakes.  If the license is renewed, set
  `MRTA_MILP_SOLVER=gurobi` and the runtime path lights up unchanged.

## Cross-references

* `ReadME/2026-09-17/ORTOOLS_FEASIBILITY.md` — full CP-SAT feasibility
  report (benchmark table, translation notes).
* `reports/md_ortools_feasibility_2026-09-17/` — raw benchmark and replay
  smoke test evidence.
* `baselines/md_oracle_dispatch.py` — dispatch and env-var contract.
* `baselines/md_ortools_scheduler.py` — base MD MILP CP-SAT model.
* `baselines/md_ortools_residual_oracle.py` — residual forced-batch CP-SAT
  twin.
