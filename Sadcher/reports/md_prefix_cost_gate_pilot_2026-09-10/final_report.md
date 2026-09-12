# MD Prefix Complete-Action Quality Gate Pilot

Status: **promising_experimental_gate**.
Development-selected screening margin: 0.0 predicted regret units.

| Decoder | Mean regret | Mean continuation duration | Top-1 | Max regret | Regret >= 20 |
|---|---:|---:|---:|---:|---:|
| prefix | 7.833 | 215.333 | 0.111 | 22.0 | 3 |
| prefix_gated | 2.889 | 210.389 | 0.111 | 19.0 | 0 |
| prefix_zero | 9.556 | 217.056 | 0.056 | 44.0 | 4 |

Mac CPU complete Beam-16 + gate p95: 13.559 ms.
Mean candidates screened: 15.00 of 16.
This remains an offline supervised experiment and is not connected to production scheduling.
