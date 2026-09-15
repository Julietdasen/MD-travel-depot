# MILP-supervised C0 vs synthetic-relational C0 vs MILP optimum (six profiles)

| profile | MILP optimum | old C0 (synthetic) | old C0 gap | **new C0 (MILP-IL)** | **new C0 gap** | best greedy | best greedy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 202.2 | 264.8 | +30.9% | 246.5 | +21.9% | 253.8 | +25.5% |
| process_scarce | 154.8 | 200.6 | +29.6% | 183.0 | +18.3% | 212.0 | +37.0% |
| transport_bottleneck | 389.0 | 449.8 | +15.6% | 438.0 | +12.6% | 437.8 | +12.5% |
| dependency_deep | 249.8 | 289.5 | +15.9% | 265.8 | +6.4% | 276.0 | +10.5% |
| mixed_hard | 262.0 | 356.2 | +36.0% | 297.2 | +13.5% | 305.0 | +16.4% |
| scale_medium | 243.5 | 298.2 | +22.5% | 286.2 | +17.6% | 296.2 | +21.7% |
