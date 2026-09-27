# OR-Tools Feasibility Report (2026-09-17)

## Verdict: YES — Recommend OR-Tools CP-SAT for Path A

Gurobi WLS license 2770405 expired; the restricted fallback caps at ~2000
vars, blocking tc>=42. OR-Tools 9.15 CP-SAT (installed via `uv add ortools`)
solves the full Path A grid to **proven optimal in seconds**, well inside the
300 s budget.

## Benchmark (300 s time limit, 8 threads, single seed per cell)

| profile           | tc | tasks | robots | status  | makespan | gap | solve (s) |
|-------------------|----|-------|--------|---------|----------|-----|-----------|
| process_scarce    | 24 | 24    | 8      | optimal | 249      | 0.0 |  0.25     |
| process_scarce    | 42 | 42    | 14     | optimal | 266      | 0.0 |  1.73     |
| process_scarce    | 60 | 60    | 20     | optimal | 330      | 0.0 | 19.22     |
| dependency_deep   | 24 | 24    | 10     | optimal | 261      | 0.0 |  0.38     |
| dependency_deep   | 42 | 42    | 17     | optimal | 422      | 0.0 |  2.07     |
| dependency_deep   | 60 | 60    | 25     | optimal | 451      | 0.0 |  8.86     |

Raw records: `reports/md_ortools_feasibility_2026-09-17/results.json`.

## Solver Choice: CP-SAT (not MPSolver+SCIP)

The MRTA formulation is dominated by disjunctive scheduling (per-robot
no-overlap, transition-time sequencing, coalition minimality). CP-SAT's
native interval / Circuit primitives make these constraints tight without
big-M lifting. MPSolver+SCIP would be a pure-MILP fallback but is expected
to be 10x-100x slower on the same model; CP-SAT already hits optimality at
20 s worst case, so no need to try SCIP.

## Prototype

`baselines/md_ortools_scheduler.py` — `solve_ortools_md_oracle(...)` returns
the *same* `GurobiOracleResult` dataclass used by
`baselines/gurobi_md_oracle.py`, so downstream code (`OracleAction`,
`OracleScheduleEntry`, replay pipeline) is drop-in compatible.

Model translation notes:
- Integer start/end/duration per task, `end = start + duration`.
- Skill coverage: `sum(assign[r,t] for r covering skill) >= 1`.
- Coalition minimality: unique-skill auxiliaries with the same exclusion
  trick as the Gurobi model.
- Per-robot sequencing: `AddCircuit` with optional self-loops (`skip` lit =
  `1 - assign`), depot arcs enforcing initial travel via `OnlyEnforceIf`,
  and pairwise arcs enforcing `start[right] >= end[left] + travel`.
- Return travel: `return_horizon` lower-bounded by `last_assignment[r,t]`
  and `unassigned[r]` implications.
- Objective: `weight * makespan + sum(assignment)` identical to Gurobi.

## Next-Step Plan for Path A Data Generation

To swap `md_c0_pathA_scaled_finetune.py` from Gurobi to OR-Tools:

1. In whatever call currently uses `solve_gurobi_md_oracle` (search under
   `data_generation/md_expert_dataset_generation.py`,
   `data_generation/md_residual_generation.py`, or a shared entry), replace
   the import with `from baselines.md_ortools_scheduler import
   solve_ortools_md_oracle` and swap the call name. Both functions share
   the same signature (`domain`, `exit_location`, `time_limit_seconds`)
   and return `GurobiOracleResult`.
2. Keep `time_limit_seconds=300.0` for the largest tc=60 shards; drop to
   `60.0` for tc<=42 to save wall time.
3. Set `threads=8` (or match host cores) — CP-SAT scales well.
4. No changes needed to `_build_replay_actions` / replay pipeline: same
   dataclass, same schedule structure.
5. Regenerate the two scaled shards (`process_scarce_scaled`,
   `dependency_deep_scaled`) at tc in {24, 42, 60}, 150 instances/shard.
   Expected wall time: ~5-30 min per shard (worst-case tc=60 process_scarce
   at ~20 s each * 150 = 50 min; the other cells are much faster).

## Risks / Caveats

- Only one seed tested per cell. Some seeds in the 150-instance shards may
  be harder; the 300 s budget gives ~15x safety margin over the observed
  worst case (20 s).
- CP-SAT's integer objective differs slightly from Gurobi's LP relaxation
  heuristics but both minimise the same makespan; optimal solutions are
  makespan-equivalent (assignment_count tie-break is identical).
- The transport-task duration model treats service (loading + loaded travel
  + unloading) as the interval body and empty travel as a Circuit
  transition. This matches Gurobi's `_transport_service` semantics.

## Deliverables Written

- `baselines/md_ortools_scheduler.py` (prototype)
- `reports/md_ortools_feasibility_2026-09-17/results.json` (raw benchmark)
- `reports/md_ortools_feasibility_2026-09-17/benchmark.py` (repro script)
- `reports/md_ortools_feasibility_2026-09-17/benchmark.log` (run log)
- `ortools>=9.15.6755` added to `pyproject.toml` via `uv add ortools`
