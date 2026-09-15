# C0 vs greedy baselines across the six MD instance profiles

| profile | method | n | success | mean makespan |
|---|---|---:|---:|---:|
| balanced | C0 | 30 | 30/30 | 226.66666666666666 |
| balanced | greedy_distance | 10 | 10/10 | 219.2 |
| balanced | greedy_eta | 10 | 10/10 | 228.7 |
| balanced | greedy_unlock | 10 | 10/10 | 218.5 |
| process_scarce | C0 | 30 | 30/30 | 206.23333333333332 |
| process_scarce | greedy_distance | 10 | 10/10 | 205.7 |
| process_scarce | greedy_eta | 10 | 10/10 | 205.7 |
| process_scarce | greedy_unlock | 10 | 10/10 | 204.1 |
| transport_bottleneck | C0 | 30 | 30/30 | 514.3 |
| transport_bottleneck | greedy_distance | 10 | 10/10 | 487.0 |
| transport_bottleneck | greedy_eta | 10 | 10/10 | 502.0 |
| transport_bottleneck | greedy_unlock | 10 | 10/10 | 516.2 |
| dependency_deep | C0 | 30 | 30/30 | 253.7 |
| dependency_deep | greedy_distance | 10 | 10/10 | 254.1 |
| dependency_deep | greedy_eta | 10 | 10/10 | 254.8 |
| dependency_deep | greedy_unlock | 10 | 10/10 | 255.6 |
| mixed_hard | C0 | 30 | 30/30 | 358.6 |
| mixed_hard | greedy_distance | 10 | 10/10 | 335.8 |
| mixed_hard | greedy_eta | 10 | 10/10 | 358.3 |
| mixed_hard | greedy_unlock | 10 | 10/10 | 313.5 |
| scale_medium | C0 | 30 | 30/30 | 317.06666666666666 |
| scale_medium | greedy_distance | 10 | 10/10 | 301.2 |
| scale_medium | greedy_eta | 10 | 10/10 | 305.9 |
| scale_medium | greedy_unlock | 10 | 10/10 | 284.4 |
