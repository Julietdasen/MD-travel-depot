# Ticket 43 Task And Process Context Ablation

Status: **development_comparison_invalid**.
Selection outcome: **development_comparison_invalid**.
Ticket 44 authorized: **false**.

## Development Metrics

| Variant | Parameters | State agreement | Exact pair | Regret | Utility@0.05 | Residual | Saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_pair_aware | 12459 | 0.7333 | 0.6183 | 0.0301 | 0.8496 | 0.5061 | 0.2305 |
| transport_context_only | 12459 | 0.6185 | 0.5129 | 0.0368 | 0.8277 | 0.4273 | 0.1892 |
| process_context_only | 12459 | 0.5606 | 0.4750 | 0.1124 | 0.6437 | 0.1794 | 0.0000 |
| transport_process_late_fusion | 12459 | 0.7081 | 0.6133 | 0.0472 | 0.8010 | 0.4166 | 0.1157 |
| transport_calibration_control | 12459 | 0.7279 | 0.6179 | 0.0312 | 0.8490 | 0.5216 | 0.2764 |

## Candidate Gains Versus Current Pair-Aware

| Candidate | Pickup exact gain | Pooled pickup state CI95 | High-margin state gain | Overall state gain | Overall exact gain | Saturation | Pass |
|---|---:|---|---:|---:|---:|---:|---:|
| transport_context_only | -0.0833 | [-0.13416666666666666, -0.06916666666666667] | -0.0400 | -0.1148 | -0.1054 | 0.1892 | false |
| process_context_only | -0.1383 | [-0.21083333333333334, -0.14416666666666667] | -0.1258 | -0.1727 | -0.1433 | 0.0000 | false |
| transport_process_late_fusion | 0.0083 | [-0.055, 0.009166666666666667] | 0.0075 | -0.0252 | -0.0050 | 0.1157 | false |
| transport_calibration_control | 0.0150 | [-0.02666666666666667, 0.03] | -0.0042 | -0.0054 | -0.0004 | 0.2764 | false |

## Integrity

Package replay mismatches: **0**.
Training/development template overlap: **0**.
Corrected held-out/development template overlap: **0**.

## Interpretation

The formal comparison is invalid because not all variants met the frozen Ticket 38 training conditions. No candidate or current-architecture development decision is authorized.

## Next Step

Stop: the formal development comparison is invalid.

## Limitations

This is a development-stage scorer comparison. It is not a confirmation result, a relational-generalization claim, or an architecture-gate decision.
