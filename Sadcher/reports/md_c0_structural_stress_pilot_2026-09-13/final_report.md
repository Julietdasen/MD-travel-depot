# C0 + exact MIP fallback structural stress test (task_count=24 fixed)

| variant | n | success | mean solver calls | mean solver time (s) | max single solver call (s) | mean makespan |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3 | 3/3 | 4.33 | 0.107 | 0.029 | 317.6666666666667 |
| max_precedence_density | 3 | 3/3 | 6.33 | 0.166 | 0.079 | 341.6666666666667 |
| min_capacity_slack | 3 | 3/3 | 4.00 | 0.096 | 0.026 | 317.3333333333333 |
| max_transport_ratio | 3 | 3/3 | 2.00 | 0.047 | 0.024 | 390.0 |
| combined_hard | 3 | 3/3 | 3.33 | 0.078 | 0.025 | 396.0 |
