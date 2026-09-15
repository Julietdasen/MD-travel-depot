# Pure MILP baseline (time_limit=300.0s, threads=4)

## Six named profiles

| profile | n | optimal | feasible-only | timeout | mean solve time (s) | mean makespan |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 4 | 4 | 0 | 0 | 0.076 | 136.75 |
| process_scarce | 4 | 4 | 0 | 0 | 0.022 | 93.5 |
| transport_bottleneck | 4 | 4 | 0 | 0 | 0.096 | 327.25 |
| dependency_deep | 4 | 4 | 0 | 0 | 0.030 | 167.0 |
| mixed_hard | 4 | 4 | 0 | 0 | 0.208 | 206.0 |
| scale_medium | 4 | 4 | 0 | 0 | 0.715 | 165.75 |

## Scale ladder

| task_count | n | optimal | feasible-only | timeout | mean solve time (s) | mean makespan |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 3 | 3 | 0 | 0 | 0.065 | 200.0 |
| 24 | 3 | 3 | 0 | 0 | 1.676 | 159.33333333333334 |
| 42 | 3 | 2 | 1 | 1 | 110.481 | 182.33333333333334 |
| 60 | 3 | 1 | 2 | 2 | 272.986 | 197.0 |
| 90 | 3 | 0 | 3 | 3 | 302.163 | 185.33333333333334 |
| 114 | 3 | 0 | 3 | 3 | 304.560 | 189.0 |
| 150 | 3 | 0 | 3 | 3 | 311.300 | 195.0 |
