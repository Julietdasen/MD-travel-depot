# Ticket 46 C0 Event-Trigger Regression

This independent diagnostic reuses the frozen Ticket 46 C0 instances and checkpoints. It does not replace the frozen Ticket 46 report.

## Protocol

- Source package: `/data/ZJZ/3D-Foundation-Model/MRTA/Sadcher/reports/md_c0_end_to_end_diagnostic_pilot_2026-09-01`
- Paired cases: 1 (30 frozen instances x 3 C0 seeds).
- Within each model seed, polling and event-gated runs alternate first-run order by instance.
- The only varied scheduler setting is `skip_non_dispatchable_states`.
- Event-gated mode skips a neural forward only while advancing work exists and no complete assignment can be dispatched.

## Results

| Metric | Polling | Event-gated | Event-gated minus polling |
|---|---:|---:|---:|
| Success rate | 1.0000 | 1.0000 | n/a |
| Mean success makespan | 254.0000 | 254.0000 | 0.0000 |
| Mean material starvation | 24.2222 | 24.2222 | 0.0000 |
| Total scorer calls | 21 | 9 | -12.0000 |
| Total decision latency (s) | 0.2587 | 0.1190 | -0.1397 |
| Total scheduler wall time (s) | 0.3443 | 0.2672 | -0.0771 |
| Fallback calls | 1 | 1 | n/a |

## Semantic Check

- Identical outcome pairs: 1/1.
- All outcomes identical: true.
- Makespan paired mean delta (event-gated minus polling): 0.0000; bootstrap 95% CI [0.0, 0.0].

## Efficiency

- Total scorer-call reduction: 57.1429%.
- Total decision-latency reduction: 53.9994% (paired mean delta -0.1397 s).
- Total scheduler wall-time reduction: 22.3846% (paired mean delta -0.0771 s).

Decision latency and scorer-call counts are the primary efficiency measures. Wall time is retained as a secondary measurement because it also includes simulator and host scheduling overhead.
