# MD Prefix Conservative Cost Gate Validation

Status: **conservative_gate_not_validated**.
Frozen confidence threshold: 3.0.
Validation: 50 new states, 3148 exact actions, 150 decisions.

| Decoder | Mean regret | Mean duration | Top-1 | Max regret | Regret >= 20 |
|---|---:|---:|---:|---:|---:|
| prefix | 8.373 | 210.053 | 0.167 | 64.0 | 22 |
| conservative_gate | 7.893 | 209.573 | 0.140 | 64.0 | 21 |
| unrestricted_gate | 2.867 | 204.547 | 0.060 | 59.0 | 5 |
| prefix_zero | 7.833 | 209.513 | 0.207 | 50.0 | 20 |

Gate switches: 13 of 150.
Improved/equal/worse: 7/138/5.
Mean paired regret reduction: 0.480.
State-clustered 95% bootstrap CI: [0.107, 0.967].
Mac CPU Beam-16 + confidence gate p95: 16.249 ms.

The confidence threshold was locked before validation labels were generated. Production scheduling remains unchanged.
