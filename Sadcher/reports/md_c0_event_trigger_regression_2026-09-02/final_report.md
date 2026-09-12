# Ticket 46 C0 Event-Trigger Regression

This independent diagnostic reuses the frozen Ticket 46 C0 instances and checkpoints. It does not replace the frozen Ticket 46 report.

## Protocol

- Source package: `/data/ZJZ/3D-Foundation-Model/MRTA/Sadcher/reports/md_c0_end_to_end_diagnostic_pilot_2026-09-01`
- Paired cases: 90
- Within each model seed, polling and event-gated runs alternate first-run order by instance.
- The only varied scheduler setting is `skip_non_dispatchable_states`.
- Event-gated mode skips a neural forward only while advancing work exists and no complete assignment can be dispatched.

## Results

| Metric | Polling | Event-gated | Mean paired delta (gated - polling) |
|---|---:|---:|---:|
| Success rate | 1.0000 | 1.0000 | n/a |
| Mean success makespan | 238.5778 | 238.5778 | 0.0000 |
| Mean material starvation | 18.6753 | 18.6753 | 0.0000 |
| Total scorer calls | 2366 | 918 | -16.0889 |
| Total decision latency (s) | 60.5036 | 63.4529 | 0.0328 |
| Total scheduler wall time (s) | 67.9244 | 76.3800 | 0.0940 |
| Fallback calls | 194 | 194 | n/a |

## Semantic Check

- Identical outcome pairs: 90/90.
- All outcomes identical: true.
- Makespan paired mean delta (event-gated minus polling): 0.0000; bootstrap 95% CI [0.0, 0.0].

## Efficiency

- Total scorer-call reduction: 61.2003%.
- Total decision-latency reduction: -4.8746% (paired mean delta 0.0328 s).
- Total scheduler wall-time reduction: -12.4485% (paired mean delta 0.0940 s).

Decision latency and scorer-call counts are the primary efficiency measures. Wall time is retained as a secondary measurement because it also includes simulator and host scheduling overhead.
