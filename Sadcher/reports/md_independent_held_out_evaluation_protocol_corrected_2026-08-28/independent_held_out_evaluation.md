# Ticket 40 Independent Held-Out Architecture Evaluation

Decision: **relational_generalization_not_supported**.

## Aggregate Evaluation

| Method | State agreement | Exact pair | Flip status | Regret | Utility@0.05 | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| matched_parameter_mlp | 0.6583 | 0.5408 | 0.7683 | 0.0481 | 0.8000 | 0.2649 |
| pair_aware_attention | 0.7275 | 0.6133 | 0.7733 | 0.0345 | 0.8417 | 0.2483 |
| local_eta | 0.5663 | 0.4800 | 0.7925 | 0.1154 | 0.6550 | 0.0000 |
| eta_plus_priority | 0.6687 | 0.5850 | 0.8275 | 0.0933 | 0.7037 | 0.0000 |
| competitor_aware_relational | 0.3013 | 0.2450 | 0.8125 | 0.0309 | 0.6987 | 0.0000 |

## Pair-Aware Minus Matched

State agreement gain: **0.0692**.
Exact-pair gain: **0.0725**.
Paired bootstrap CI95: **[0.04250000000000001, 0.09625000000000002]**.

| Stratum | State-agreement gain |
|---|---:|
| near_tie_lt_0.01 | 0.0467 |
| small_ge_0.01_lt_0.03 | 0.1083 |
| medium_ge_0.03_lt_0.05 | 0.1083 |
| high_ge_0.05 | 0.0133 |

## Criteria

- state_agreement_gain_at_least_0.05: **true**.
- exact_pair_accuracy_gain_at_least_0.05: **true**.
- all_family_exact_pair_accuracy_at_least_0.60: **false**.
- paired_bootstrap_state_agreement_ci_lower_above_zero: **true**.
- mean_evaluation_residual_saturation_at_most_0.25: **true**.

## Next Step

The frozen held-out relational architecture gate remains closed.

## Limitations

This evaluation is limited to the frozen Ticket 39 relational package, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.
