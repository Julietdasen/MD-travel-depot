# C0 + exact MIP fallback scale-up stress test

| task_count | robots | n | success | mean solver calls | mean solver time (s) | max single solver call (s) | mean wall time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 5 | 3 | 3/3 | 1.67 | 0.020 | 0.013 | 0.362 |
| 18 | 7 | 3 | 3/3 | 2.00 | 0.034 | 0.018 | 0.474 |
| 24 | 9 | 3 | 3/3 | 3.67 | 0.082 | 0.024 | 0.830 |
| 30 | 12 | 3 | 3/3 | 5.00 | 0.162 | 0.035 | 1.413 |
| 36 | 14 | 3 | 3/3 | 8.67 | 0.367 | 0.044 | 2.168 |
| 42 | 16 | 3 | 3/3 | 10.67 | 0.592 | 0.106 | 3.162 |
