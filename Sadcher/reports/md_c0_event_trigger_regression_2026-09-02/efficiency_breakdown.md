# Ticket 46 C0 Event-Trigger Efficiency Breakdown

This supplement decomposes the independent 90-pair polling versus event-gated regression. It does not modify the frozen Ticket 46 report.

## Semantic Result

- Identical outcomes: 90/90 paired cases.
- Fallback calls: 194 for polling and 194 for event-gated mode.

## Latency Breakdown

| Metric | Polling total | Event-gated total | Total change | Paired mean delta (gated - polling) | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Scorer calls | 2366.0000 | 918.0000 | -61.2003% | -16.0889 | [-19.6444, -12.5778] |
| Model pipeline latency (s) | 27.2478 | 10.6341 | -60.9724% | -0.1846 | [-0.2292, -0.1454] |
| MIP fallback solver time (s) | 32.0779 | 51.9558 | +61.9674% | +0.2209 | [-0.4445, +0.9959] |
| Total decision latency (s) | 60.5036 | 63.4529 | +4.8746% | +0.0328 | [-0.6318, +0.7448] |
| Scheduler wall time (s) | 67.9244 | 76.3800 | +12.4485% | +0.0940 | [-0.5674, +0.8315] |

## Interpretation

- Event gating removes 61.20% of scorer calls and reduces the model pipeline (feature bridge, neural forward, constrained decoder, and repair) by 60.97%; its paired CI is entirely below zero.
- It does not reduce fallback call count. The same 194 explicit MIP calls had materially variable observed solve time, and that variance dominates the measured full decision and wall time in this single interleaved run.
- Therefore this change is a confirmed model-path efficiency improvement with unchanged scheduling behavior, but it is not yet evidence for an end-to-end wall-time speedup. Achieving that needs fewer or faster fallback solves, then a repeated timing study.
