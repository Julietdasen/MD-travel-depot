# Ticket 48 Exact Action Candidate Diagnostic

Final status: **candidate_coverage_supported_for_next_stage**.

Pilot valid: **true**; failures: **0**.

Old forced comparison: 82 first-event mismatches, 44 positive optimism biases, and 0 tolerance-optimal-set disagreements.

## Development candidate coverage

| Model seed | Recall@1 | @2 | @4 | @8 | @16 | Gap@1 | @2 | @4 | @8 | @16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3101 | 0.8333 | 0.9333 | 0.9833 | 1.0000 | 1.0000 | 2.6000 | 0.6833 | 0.2000 | 0.0000 | 0.0000 |
| 3102 | 0.8500 | 0.9333 | 1.0000 | 1.0000 | 1.0000 | 2.1000 | 0.7000 | 0.0000 | 0.0000 | 0.0000 |
| 3103 | 0.8500 | 0.9500 | 0.9833 | 1.0000 | 1.0000 | 1.6000 | 0.3500 | 0.2000 | 0.0000 | 0.0000 |

Full pending-count, action-cardinality, and action-type strata are stored in `summary.json`.

Candidate generation did not read oracle values. Confirmation seeds were not read.
