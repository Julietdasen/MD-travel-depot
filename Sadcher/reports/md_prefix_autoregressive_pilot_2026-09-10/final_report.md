# MD-SADCHER Sequential Prefix Decoder Pilot

Status: **experimental_prototype_only**.

States: train=32, development=12, test=12. Exact labels: 3506; test labels were generated after checkpoint lock.

| Metric | prefix-zero | prefix | prefix minus zero |
|---|---:|---:|---:|
| Beam-16 top-1 optimal | 0.1944 | 0.2778 | 0.0833 |
| Beam-16 mean regret | 7.7222 | 11.3056 | 3.5833 |
| Maximum beam-16 p95 CPU time (ms, 36 repeated test calls/model) | — | — | 17.557 |

The prefix logits do change under different legal prefixes, all decoded actions were legal, exact-label and mask coverage were 100%, checkpoint/gradient checks passed, and the repeated beam-16 latency check was below 20 ms p95. The strict mean-regret improvement threshold was not met, so this prototype is not connected to production scheduling.
