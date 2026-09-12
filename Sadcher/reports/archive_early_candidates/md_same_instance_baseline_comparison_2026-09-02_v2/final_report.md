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
| C0_seed3101 | 30/30 | 1.000 | 240.400 | 18.463 | none |
| C0_seed3102 | 30/30 | 1.000 | 234.467 | 18.963 | none |
| C0_seed3103 | 30/30 | 1.000 | 240.867 | 18.600 | none |

## C0 Compared With Baselines

Negative makespan delta means the candidate is faster; positive starvation delta means more waiting.

| Candidate | Reference | Common success | Makespan delta | Improvement | Starvation delta |
|---|---|---:|---:|---:|---:|
| C0 | milp_oracle | 30 | 44.700 | -44.700 | 0.689 |
| C0 | process_md_greedy | 30 | 0.367 | -0.367 | 0.178 |
| C0 | milp_oracle | 30 | 38.767 | -38.767 | 1.189 |
| C0 | process_md_greedy | 30 | -5.567 | 5.567 | 0.678 |
| C0 | milp_oracle | 30 | 45.167 | -45.167 | 0.826 |
| C0 | process_md_greedy | 30 | 0.833 | -0.833 | 0.315 |

## Where Differences Come From

The machine-readable `difference_analysis.json` records transport/process order, material starvation, latest task completion, and terminal return tail for every pair.
- `C0_seed3101_vs_milp_oracle`: mean makespan delta 44.700; signals {'different_transport_order': 3, 'longer_terminal_return_tail': 13, 'more_material_starvation': 13, 'same_makespan': 1}.
  - md-c0-diagnostic-46007: delta 112.000, starvation delta -1.444, signal `longer_terminal_return_tail`.
  - md-c0-diagnostic-46002: delta 96.000, starvation delta 7.222, signal `more_material_starvation`.
- `C0_seed3102_vs_milp_oracle`: mean makespan delta 38.767; signals {'different_transport_order': 2, 'longer_terminal_return_tail': 14, 'more_material_starvation': 14}.
  - md-c0-diagnostic-46002: delta 96.000, starvation delta 7.111, signal `more_material_starvation`.
  - md-c0-diagnostic-46025: delta 89.000, starvation delta 4.444, signal `more_material_starvation`.
- `C0_seed3103_vs_milp_oracle`: mean makespan delta 45.167; signals {'different_transport_order': 2, 'longer_terminal_return_tail': 13, 'more_material_starvation': 14, 'same_makespan': 1}.
  - md-c0-diagnostic-46027: delta 128.000, starvation delta 1.778, signal `more_material_starvation`.
  - md-c0-diagnostic-46007: delta 89.000, starvation delta -0.111, signal `longer_terminal_return_tail`.

MILP solver statuses across 30 instances: {'optimal': 30}.
MILP rows with no incumbent or unavailable solver are retained with null makespan and excluded from the success-rate denominator.

Ticket 20/44/45 were not executed by this comparison; this remains a diagnostic result.
