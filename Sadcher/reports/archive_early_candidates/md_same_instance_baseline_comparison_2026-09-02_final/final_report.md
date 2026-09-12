# Ticket 47 Same-Instance MILP and Greedy Comparison

Diagnostic comparison only; no production or final-model claim.

## Summary

| Method | Evaluated | Success rate | Mean makespan | Mean starvation | Solver status |
|---|---:|---:|---:|---:|---|
| milp_oracle | 30/30 | 1.000 | 195.700 | 17.774 | optimal:30 |
| greedy_distance | 30/30 | 1.000 | 236.667 | 18.222 | none |
| greedy_eta | 30/30 | 1.000 | 240.033 | 18.285 | none |
| greedy_unlock | 30/30 | 1.000 | 235.800 | 19.200 | none |
| material_solo_greedy | 30/30 | 1.000 | 239.567 | 18.552 | none |
| process_md_greedy | 30/30 | 1.000 | 240.033 | 18.285 | none |
| physics_only | 30/30 | 1.000 | 240.367 | 18.463 | none |
| eta_unlock_heuristic | 30/30 | 1.000 | 232.500 | 18.593 | none |
| masked_greedy | 30/30 | 1.000 | 235.400 | 19.374 | none |
| C0_seed3101 | 30/30 | 1.000 | 240.400 | 18.463 | none |
| C0_seed3102 | 30/30 | 1.000 | 234.467 | 18.963 | none |
| C0_seed3103 | 30/30 | 1.000 | 240.867 | 18.600 | none |

## C0 Compared With Baselines

Negative makespan delta means the candidate is faster; positive starvation delta means more waiting.

| Candidate | Reference | Common success | Makespan delta | Improvement | Starvation delta |
|---|---|---:|---:|---:|---:|
| C0_seed3101 | milp_oracle | 30 | 44.700 | -44.700 | 0.689 |
| C0_seed3101 | greedy_distance | 30 | 3.733 | -3.733 | 0.241 |
| C0_seed3101 | greedy_eta | 30 | 0.367 | -0.367 | 0.178 |
| C0_seed3101 | greedy_unlock | 30 | 4.600 | -4.600 | -0.737 |
| C0_seed3101 | material_solo_greedy | 30 | 0.833 | -0.833 | -0.089 |
| C0_seed3101 | process_md_greedy | 30 | 0.367 | -0.367 | 0.178 |
| C0_seed3101 | physics_only | 30 | 0.033 | -0.033 | -0.000 |
| C0_seed3101 | eta_unlock_heuristic | 30 | 7.900 | -7.900 | -0.130 |
| C0_seed3101 | masked_greedy | 30 | 5.000 | -5.000 | -0.911 |
| C0_seed3102 | milp_oracle | 30 | 38.767 | -38.767 | 1.189 |
| C0_seed3102 | greedy_distance | 30 | -2.200 | 2.200 | 0.741 |
| C0_seed3102 | greedy_eta | 30 | -5.567 | 5.567 | 0.678 |
| C0_seed3102 | greedy_unlock | 30 | -1.333 | 1.333 | -0.237 |
| C0_seed3102 | material_solo_greedy | 30 | -5.100 | 5.100 | 0.411 |
| C0_seed3102 | process_md_greedy | 30 | -5.567 | 5.567 | 0.678 |
| C0_seed3102 | physics_only | 30 | -5.900 | 5.900 | 0.500 |
| C0_seed3102 | eta_unlock_heuristic | 30 | 1.967 | -1.967 | 0.370 |
| C0_seed3102 | masked_greedy | 30 | -0.933 | 0.933 | -0.411 |
| C0_seed3103 | milp_oracle | 30 | 45.167 | -45.167 | 0.826 |
| C0_seed3103 | greedy_distance | 30 | 4.200 | -4.200 | 0.378 |
| C0_seed3103 | greedy_eta | 30 | 0.833 | -0.833 | 0.315 |
| C0_seed3103 | greedy_unlock | 30 | 5.067 | -5.067 | -0.600 |
| C0_seed3103 | material_solo_greedy | 30 | 1.300 | -1.300 | 0.048 |
| C0_seed3103 | process_md_greedy | 30 | 0.833 | -0.833 | 0.315 |
| C0_seed3103 | physics_only | 30 | 0.500 | -0.500 | 0.137 |
| C0_seed3103 | eta_unlock_heuristic | 30 | 8.367 | -8.367 | 0.007 |
| C0_seed3103 | masked_greedy | 30 | 5.467 | -5.467 | -0.774 |

## Where Differences Come From

The machine-readable `difference_analysis.json` records transport/process order, material starvation, latest task completion, and terminal return tail for every pair.

### Component Decomposition Against MILP

The makespan delta is decomposed into the latest real-task completion delta and the terminal return-tail delta.

| Candidate | Makespan delta | Latest task completion delta | Return-tail delta | Starvation delta | Same transport order | Same process order |
|---|---:|---:|---:|---:|---:|---:|
| C0_seed3101 | 44.700 | 12.467 | 32.233 | 0.689 | 9/30 | 0/30 |
| C0_seed3102 | 38.767 | 11.333 | 27.433 | 1.189 | 9/30 | 0/30 |
| C0_seed3103 | 45.167 | 11.733 | 33.433 | 0.826 | 8/30 | 0/30 |
- `C0_seed3101_vs_milp_oracle`: mean makespan delta 44.700; signals {'different_transport_order': 3, 'longer_terminal_return_tail': 13, 'more_material_starvation': 13, 'same_makespan': 1}.
  - md-c0-diagnostic-46007: delta 112.000, starvation delta -1.444, signal `longer_terminal_return_tail`.
  - md-c0-diagnostic-46002: delta 96.000, starvation delta 7.222, signal `more_material_starvation`.
- `C0_seed3102_vs_milp_oracle`: mean makespan delta 38.767; signals {'different_transport_order': 2, 'longer_terminal_return_tail': 14, 'more_material_starvation': 14}.
  - md-c0-diagnostic-46002: delta 96.000, starvation delta 7.111, signal `more_material_starvation`.
  - md-c0-diagnostic-46025: delta 89.000, starvation delta 4.444, signal `more_material_starvation`.
- `C0_seed3103_vs_milp_oracle`: mean makespan delta 45.167; signals {'different_transport_order': 2, 'longer_terminal_return_tail': 13, 'more_material_starvation': 14, 'same_makespan': 1}.
  - md-c0-diagnostic-46027: delta 128.000, starvation delta 1.778, signal `more_material_starvation`.
  - md-c0-diagnostic-46007: delta 89.000, starvation delta -0.111, signal `longer_terminal_return_tail`.

## Limitations

- This is a diagnostic comparison and makes no production or final-model claim.
- MILP solver statuses across 30 instances: {'optimal': 30}; rows without an incumbent remain null and are excluded from the evaluated-run denominator.
- Ticket 20/44/45 were not executed by this comparison.
