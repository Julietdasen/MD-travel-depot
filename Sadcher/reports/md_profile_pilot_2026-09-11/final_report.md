# Profile-stratified MD pilot

Each profile is represented in train, development, and test.

| split | profile | seed | tasks | robots | legal actions | exact labels |
|---|---|---:|---:|---:|---:|---|
| train | balanced | 101 | 12 | 5 | 129 | False |
| train | balanced | 102 | 12 | 5 | 64 | True |
| train | process_scarce | 101 | 12 | 4 | 104 | False |
| train | process_scarce | 102 | 12 | 4 | 48 | True |
| train | transport_bottleneck | 101 | 12 | 4 | 17 | True |
| train | transport_bottleneck | 102 | 12 | 4 | 23 | True |
| train | dependency_deep | 101 | 12 | 5 | 90 | True |
| train | dependency_deep | 102 | 12 | 5 | 25 | True |
| train | mixed_hard | 101 | 15 | 5 | 61 | True |
| train | mixed_hard | 102 | 15 | 5 | 309 | False |
| train | scale_medium | 101 | 18 | 7 | 1602 | False |
| train | scale_medium | 102 | 18 | 7 | 457 | False |
| development | balanced | 201 | 12 | 5 | 38 | True |
| development | process_scarce | 201 | 12 | 4 | 62 | True |
| development | transport_bottleneck | 201 | 12 | 4 | 23 | True |
| development | dependency_deep | 201 | 12 | 5 | 51 | True |
| development | mixed_hard | 201 | 15 | 5 | 30 | True |
| development | scale_medium | 201 | 18 | 7 | 228 | False |
| test | balanced | 301 | 12 | 5 | 129 | False |
| test | process_scarce | 301 | 12 | 4 | 69 | True |
| test | transport_bottleneck | 301 | 12 | 4 | 5 | True |
| test | dependency_deep | 301 | 12 | 5 | 77 | True |
| test | mixed_hard | 301 | 15 | 5 | 123 | False |
| test | scale_medium | 301 | 18 | 7 | 4579 | False |

## Greedy baseline pilot

| profile | baseline | n | success | mean makespan |
|---|---|---:|---:|---:|
| balanced | greedy_distance | 4 | 4/4 | 253.75 |
| balanced | greedy_eta | 4 | 4/4 | 263.5 |
| balanced | greedy_unlock | 4 | 4/4 | 261.5 |
| process_scarce | greedy_distance | 4 | 4/4 | 212.0 |
| process_scarce | greedy_eta | 4 | 4/4 | 212.0 |
| process_scarce | greedy_unlock | 4 | 4/4 | 212.0 |
| transport_bottleneck | greedy_distance | 4 | 4/4 | 437.75 |
| transport_bottleneck | greedy_eta | 4 | 4/4 | 448.25 |
| transport_bottleneck | greedy_unlock | 4 | 4/4 | 465.25 |
| dependency_deep | greedy_distance | 4 | 4/4 | 310.0 |
| dependency_deep | greedy_eta | 4 | 4/4 | 304.25 |
| dependency_deep | greedy_unlock | 4 | 4/4 | 276.0 |
| mixed_hard | greedy_distance | 4 | 4/4 | 332.0 |
| mixed_hard | greedy_eta | 4 | 4/4 | 341.5 |
| mixed_hard | greedy_unlock | 4 | 4/4 | 305.0 |
| scale_medium | greedy_distance | 4 | 4/4 | 303.0 |
| scale_medium | greedy_eta | 4 | 4/4 | 296.25 |
| scale_medium | greedy_unlock | 4 | 4/4 | 299.5 |

## C0 training status

A reduced C0 training attempt was made with the existing `sadcher-md` environment and the existing Ticket 46 entry point. The entry point rejects CPU execution and requires a visible `cuda:0`; no GPU is available in this session. Therefore no new C0 checkpoint or C0 profile score is claimed. The pilot remains valid as a profile-stratified simulator/baseline comparison, but C0 training requires rerunning the same command on the project's existing GPU machine.
