# Ticket 40 Independent Held-Out Architecture Evaluation

Decision: **relational_generalization_not_supported**.

## Aggregate Evaluation

| Method | State agreement | Exact pair | Flip status | Regret | Utility@0.05 | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| matched_parameter_mlp | 0.6658 | 0.5475 | 0.7642 | 0.0521 | 0.7854 | 0.2643 |
| pair_aware_attention | 0.7446 | 0.6358 | 0.8050 | 0.0329 | 0.8425 | 0.2448 |
| local_eta | 0.5275 | 0.4475 | 0.7700 | 0.1168 | 0.6438 | 0.0000 |
| eta_plus_priority | 0.6613 | 0.5750 | 0.7950 | 0.0946 | 0.7075 | 0.0000 |
| competitor_aware_relational | 0.3325 | 0.2725 | 0.8375 | 0.0290 | 0.7425 | 0.0000 |

## Pair-Aware Minus Matched

State agreement gain: **0.0787**.
Exact-pair gain: **0.0883**.
Paired bootstrap CI95: **[0.053333333333333274, 0.10624999999999994]**.

| Stratum | State-agreement gain |
|---|---:|
| near_tie_lt_0.01 | 0.1433 |
| small_ge_0.01_lt_0.03 | 0.0333 |
| medium_ge_0.03_lt_0.05 | 0.0767 |
| high_ge_0.05 | 0.0617 |

## Criteria

- state_agreement_gain_at_least_0.05: **true**.
- exact_pair_accuracy_gain_at_least_0.05: **true**.
- all_family_exact_pair_accuracy_at_least_0.60: **false**.
- paired_bootstrap_state_agreement_ci_lower_above_zero: **true**.
- mean_evaluation_residual_saturation_at_most_0.25: **true**.

## Next Step

Use this held-out result in the architecture decision; do not generate a production dataset from this evaluation.

## Limitations

This evaluation is limited to the frozen Ticket 39 relational package, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.
