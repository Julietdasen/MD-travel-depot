# MD-SADCHER Sequential Prefix Decoder Pilot

Status: **experimental_prototype_only**.

States: train=32, development=12, test=12.
Exact labels: 3506; test labels were generated after checkpoint lock.

| Metric | prefix-zero | prefix | prefix minus zero |
|---|---:|---:|---:|
| Beam-16 top-1 optimal | 0.1667 | 0.1667 | 0.0000 |
| Beam-16 mean regret | 6.7778 | 7.2778 | 0.5000 |
| Maximum beam-16 p95 CPU time (ms) | — | — | 109.560 |

If any acceptance check is false, this remains an experimental prototype and is not connected to production scheduling.
