# MD Prefix Cost Gate Large Validation

Status: **large_validation_passed**.
Frozen evaluation: 50 new states, 3135 exact actions, 150 decoder decisions.

| Decoder | Mean regret | Mean continuation duration | Top-1 | Max regret | Regret >= 20 |
|---|---:|---:|---:|---:|---:|
| prefix | 6.840 | 207.840 | 0.240 | 43.0 | 21 |
| prefix_gated | 2.560 | 203.560 | 0.140 | 61.0 | 5 |
| prefix_zero | 6.540 | 207.540 | 0.247 | 35.0 | 19 |

Mean paired regret reduction: 4.280.
State-clustered 95% bootstrap interval: [1.753, 6.833].
Improved/equal/worse decisions: 74/40/36.
Mac CPU Beam-16 + gate p95: 15.223 ms.

## Limitation

The average result improved, but this version is not production-ready. Exact Top-1 fell from 0.240 to 0.140, and maximum regret rose from 43 to 61. The worst case was state 77400 with model seed 29: the original Beam choice had regret 0, while the gate selected Beam rank 15 with regret 61. A conservative fallback must be tested before integration.

The gate and threshold were frozen before these labels were generated. Production scheduling remains unchanged.
