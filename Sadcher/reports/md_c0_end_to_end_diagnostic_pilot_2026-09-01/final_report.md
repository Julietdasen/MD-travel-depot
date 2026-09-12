# Ticket 46 C0 End-to-End Diagnostic Pilot

This is a diagnostic pilot only. It does not select a final or production model.

## Required Answers

1. Real C0 checkpoints completed continuous simulator rollouts for all C0 seeds: true.
2. Normal-path solver calls were zero; explicit fallback solver calls across C0 were 194.
3. C0 seed outcomes:

| Seed | Success rate | Mean success makespan | Mean starvation | Fallback rate | Mean decision latency (s) |
|---|---:|---:|---:|---:|---:|
| 3101 | 1.0000 | 240.4000 | 18.4630 | 0.1115 | 1.0431 |
| 3102 | 1.0000 | 234.4667 | 18.9630 | 0.1114 | 1.3003 |
| 3103 | 1.0000 | 240.8667 | 18.6000 | 0.0994 | 0.9902 |

4. C0 stably improved success versus the frozen strongest fixed non-MIP baseline (process_md_greedy): false.
5. Gate/rollout interpretation: gate_proxy_not_predictive_in_pilot.
6. Potential Gate false negative observed: false.
7. Next step: Modify the model or training objective only under a newly preregistered Gate.
8. Ticket 20 was not executed; Tickets 44 and 45 were not executed; Tickets 17, 20, and 32 were not unfrozen.

Gate correlations are descriptive for n=9 observations and make no positive predictive value claim.

## Visualization

A paired visualization of the most robust local C0 win is available in [ticket46_c0_best_example.md](ticket46_c0_best_example.md), with [PNG](ticket46_c0_best_example.png) and [SVG](ticket46_c0_best_example.svg) outputs. It uses frozen instance `md-c0-diagnostic-46015`, representative C0 seed `3101`, and the paired `process_md_greedy` rollout.
