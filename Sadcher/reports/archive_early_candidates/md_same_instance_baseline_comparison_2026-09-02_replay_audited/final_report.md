# Ticket 47 Same-Instance MILP and Greedy Comparison

Diagnostic comparison only; no production or final-model claim.

## Summary

| Method | Evaluated | Success rate | Mean makespan | Mean starvation | Solver status |
|---|---:|---:|---:|---:|---|
| milp_oracle | 30/30 | 1.000 | 199.133 | 17.396 | optimal:30 |
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
| C0_seed3101 | milp_oracle | 30 | 41.267 | -41.267 | 1.067 |
| C0_seed3101 | greedy_distance | 30 | 3.733 | -3.733 | 0.241 |
| C0_seed3101 | greedy_eta | 30 | 0.367 | -0.367 | 0.178 |
| C0_seed3101 | greedy_unlock | 30 | 4.600 | -4.600 | -0.737 |
| C0_seed3101 | material_solo_greedy | 30 | 0.833 | -0.833 | -0.089 |
| C0_seed3101 | process_md_greedy | 30 | 0.367 | -0.367 | 0.178 |
| C0_seed3101 | physics_only | 30 | 0.033 | -0.033 | -0.000 |
| C0_seed3101 | eta_unlock_heuristic | 30 | 7.900 | -7.900 | -0.130 |
| C0_seed3101 | masked_greedy | 30 | 5.000 | -5.000 | -0.911 |
| C0_seed3102 | milp_oracle | 30 | 35.333 | -35.333 | 1.567 |
| C0_seed3102 | greedy_distance | 30 | -2.200 | 2.200 | 0.741 |
| C0_seed3102 | greedy_eta | 30 | -5.567 | 5.567 | 0.678 |
| C0_seed3102 | greedy_unlock | 30 | -1.333 | 1.333 | -0.237 |
| C0_seed3102 | material_solo_greedy | 30 | -5.100 | 5.100 | 0.411 |
| C0_seed3102 | process_md_greedy | 30 | -5.567 | 5.567 | 0.678 |
| C0_seed3102 | physics_only | 30 | -5.900 | 5.900 | 0.500 |
| C0_seed3102 | eta_unlock_heuristic | 30 | 1.967 | -1.967 | 0.370 |
| C0_seed3102 | masked_greedy | 30 | -0.933 | 0.933 | -0.411 |
| C0_seed3103 | milp_oracle | 30 | 41.733 | -41.733 | 1.204 |
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
The MILP chooses transport order, process coalition, robot reuse, and terminal positions jointly; C0 commits to online local assignments.
The common terminal protocol requires every robot to return to (0,0), so a schedule with earlier real-task completion can still have a longer makespan.
The following decomposition is descriptive evidence of where the paired gap appears, not a causal intervention.

### Component Decomposition Against MILP

The makespan delta is decomposed into the latest real-task completion delta and the terminal return-tail delta.

| Candidate | Makespan delta | Latest task completion delta | Return-tail delta | Starvation delta | Same transport order | Same process order |
|---|---:|---:|---:|---:|---:|---:|
| C0_seed3101 | 41.267 | 12.900 | 28.367 | 1.067 | 7/30 | 1/30 |
| C0_seed3102 | 35.333 | 11.767 | 23.567 | 1.567 | 8/30 | 1/30 |
| C0_seed3103 | 41.733 | 12.167 | 29.567 | 1.204 | 7/30 | 0/30 |
Across 90 paired C0/MILP instances, the mean makespan gap is 39.444: latest real-task completion contributes 12.278, terminal return tail contributes 27.167 (68.873% of the mean gap).
- `C0_seed3101_vs_milp_oracle`: mean makespan delta 41.267; signals {'different_transport_order': 2, 'longer_terminal_return_tail': 12, 'more_material_starvation': 15, 'same_makespan': 1}.
  - md-c0-diagnostic-46007: delta 112.000, starvation delta 5.667, signal `more_material_starvation`.
  - md-c0-diagnostic-46002: delta 96.000, starvation delta 6.000, signal `more_material_starvation`.
- `C0_seed3102_vs_milp_oracle`: mean makespan delta 35.333; signals {'different_transport_order': 2, 'longer_terminal_return_tail': 9, 'more_material_starvation': 17, 'process_order_or_coalition_timing': 1, 'same_makespan': 1}.
  - md-c0-diagnostic-46002: delta 96.000, starvation delta 5.889, signal `more_material_starvation`.
  - md-c0-diagnostic-46025: delta 89.000, starvation delta 3.111, signal `more_material_starvation`.
- `C0_seed3103_vs_milp_oracle`: mean makespan delta 41.733; signals {'different_transport_order': 3, 'longer_terminal_return_tail': 9, 'more_material_starvation': 17, 'same_makespan': 1}.
  - md-c0-diagnostic-46027: delta 128.000, starvation delta 1.778, signal `more_material_starvation`.
  - md-c0-diagnostic-46007: delta 89.000, starvation delta 7.000, signal `more_material_starvation`.

### Representative Event Traces

Times are start/complete; return tail is the time from the latest real-task completion to the terminal exit.
- md-c0-diagnostic-46002: MILP makespan 207.000 (latest task 150.000, tail 57.000) versus C0_seed3101 303.000 (latest task 189.000, tail 114.000); delta 96.000.
  - Transport order: 10 -> 11 -> 12 versus 12 -> 10 -> 11; process order: 4 -> 5 -> 9 -> 6 -> 1 -> 3 -> 8 -> 2 -> 7 versus 9 -> 4 -> 1 -> 5 -> 3 -> 6 -> 7 -> 8 -> 2.
  - Transport events (task@assigned->complete): MILP 10@0->67, 11@0->89, 12@67->132; C0_seed3101 12@0->57, 10@1->70, 11@57->171.
  - Latest completion changes: task 8 101.000->183.000 (+82.000); task 11 89.000->171.000 (+82.000); task 2 138.000->189.000 (+51.000).
  - Material-linked delay: 11->8: source 89.000->171.000, task 8 101.000->183.000.
  - Longest terminal return: MILP robot 3 from task 11 at (51,100) takes 57; C0_seed3101 robot 2 from task 2 at (66,92) takes 114.
- md-c0-diagnostic-46027: MILP makespan 154.000 (latest task 106.000, tail 48.000) versus C0_seed3102 215.000 (latest task 94.000, tail 121.000); delta 61.000.
  - Transport order: 10 -> 12 -> 11 versus 10 -> 12 -> 11; process order: 5 -> 8 -> 1 -> 3 -> 7 -> 9 -> 6 -> 2 -> 4 versus 6 -> 5 -> 8 -> 1 -> 3 -> 7 -> 9 -> 4 -> 2.
  - Transport events (task@assigned->complete): MILP 10@0->39, 12@0->60, 11@39->87; C0_seed3102 10@0->39, 12@0->60, 11@39->87.
  - Latest completion changes: task 5 9.000->11.000 (+2.000).
  - Longest terminal return: MILP robot 4 from task 11 at (94,16) takes 48; C0_seed3102 robot 1 from task 1 at (99,68) takes 121.

## Limitations

- This is a diagnostic comparison and makes no production or final-model claim.
- MILP solver statuses across 30 instances: {'optimal': 30}; rows without an incumbent remain null and are excluded from the evaluated-run denominator.
- Ticket 20/44/45 were not executed by this comparison.
