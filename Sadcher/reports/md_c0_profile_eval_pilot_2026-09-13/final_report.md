# C0 vs greedy baselines across the six MD instance profiles

| profile | method | n | success | mean makespan |
|---|---|---:|---:|---:|
| balanced | C0 | 12 | 12/12 | 264.8333333333333 |
| balanced | greedy_distance | 4 | 4/4 | 253.75 |
| balanced | greedy_eta | 4 | 4/4 | 263.5 |
| balanced | greedy_unlock | 4 | 4/4 | 261.5 |
| process_scarce | C0 | 12 | 12/12 | 200.58333333333334 |
| process_scarce | greedy_distance | 4 | 4/4 | 212.0 |
| process_scarce | greedy_eta | 4 | 4/4 | 212.0 |
| process_scarce | greedy_unlock | 4 | 4/4 | 212.0 |
| transport_bottleneck | C0 | 12 | 12/12 | 449.8333333333333 |
| transport_bottleneck | greedy_distance | 4 | 4/4 | 437.75 |
| transport_bottleneck | greedy_eta | 4 | 4/4 | 448.25 |
| transport_bottleneck | greedy_unlock | 4 | 4/4 | 465.25 |
| dependency_deep | C0 | 12 | 12/12 | 289.5 |
| dependency_deep | greedy_distance | 4 | 4/4 | 310.0 |
| dependency_deep | greedy_eta | 4 | 4/4 | 304.25 |
| dependency_deep | greedy_unlock | 4 | 4/4 | 276.0 |
| mixed_hard | C0 | 12 | 12/12 | 356.25 |
| mixed_hard | greedy_distance | 4 | 4/4 | 332.0 |
| mixed_hard | greedy_eta | 4 | 4/4 | 341.5 |
| mixed_hard | greedy_unlock | 4 | 4/4 | 305.0 |
| scale_medium | C0 | 12 | 12/12 | 298.25 |
| scale_medium | greedy_distance | 4 | 4/4 | 303.0 |
| scale_medium | greedy_eta | 4 | 4/4 | 296.25 |
| scale_medium | greedy_unlock | 4 | 4/4 | 299.5 |
