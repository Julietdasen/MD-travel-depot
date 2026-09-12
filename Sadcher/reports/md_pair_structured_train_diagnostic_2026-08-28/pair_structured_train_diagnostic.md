# Ticket 35 Pair-Structured Train Diagnostic

Decision: **optimization_not_feasible**.

Comparison of deterministic family-balanced state-only and pair-structured training.

## Per-Seed Results

| Variant | Method | Seed | Agreement | Saturation | All conditions |
|---|---|---:|---:|---:|---:|
| family_balanced_state_only | matched_parameter_mlp | 3101 | 0.8845 | 0.1176 | false |
| family_balanced_state_only | matched_parameter_mlp | 3102 | 0.8377 | 0.0419 | false |
| family_balanced_state_only | matched_parameter_mlp | 3103 | 0.9135 | 0.2192 | false |
| family_balanced_state_only | pair_aware_attention | 3101 | 0.8770 | 0.1177 | false |
| family_balanced_state_only | pair_aware_attention | 3102 | 0.8988 | 0.1048 | false |
| family_balanced_state_only | pair_aware_attention | 3103 | 0.9565 | 0.2597 | false |
| family_balanced_pair_structured | matched_parameter_mlp | 3101 | 0.6873 | 0.0872 | false |
| family_balanced_pair_structured | matched_parameter_mlp | 3102 | 0.4670 | 0.0003 | false |
| family_balanced_pair_structured | matched_parameter_mlp | 3103 | 0.6850 | 0.1315 | false |
| family_balanced_pair_structured | pair_aware_attention | 3101 | 0.7538 | 0.1815 | false |
| family_balanced_pair_structured | pair_aware_attention | 3102 | 0.7392 | 0.0740 | false |
| family_balanced_pair_structured | pair_aware_attention | 3103 | 0.7578 | 0.0730 | false |

## Candidate Family Flip-Pair Exact

| Method | Seed | Family | Flip pairs | Exact accuracy | Pass |
|---|---:|---|---:|---:|---:|
| matched_parameter_mlp | 3101 | competitor_robot_position | 192 | 0.5938 | false |
| matched_parameter_mlp | 3101 | competitor_robot_speed | 161 | 0.5155 | false |
| matched_parameter_mlp | 3101 | alternative_task_pickup | 211 | 0.3981 | false |
| matched_parameter_mlp | 3101 | alternative_downstream_priority | 121 | 0.4132 | false |
| matched_parameter_mlp | 3102 | competitor_robot_position | 192 | 0.2760 | false |
| matched_parameter_mlp | 3102 | competitor_robot_speed | 161 | 0.1863 | false |
| matched_parameter_mlp | 3102 | alternative_task_pickup | 211 | 0.1706 | false |
| matched_parameter_mlp | 3102 | alternative_downstream_priority | 121 | 0.1405 | false |
| matched_parameter_mlp | 3103 | competitor_robot_position | 192 | 0.6198 | true |
| matched_parameter_mlp | 3103 | competitor_robot_speed | 161 | 0.4410 | false |
| matched_parameter_mlp | 3103 | alternative_task_pickup | 211 | 0.4645 | false |
| matched_parameter_mlp | 3103 | alternative_downstream_priority | 121 | 0.3719 | false |
| pair_aware_attention | 3101 | competitor_robot_position | 192 | 0.6510 | true |
| pair_aware_attention | 3101 | competitor_robot_speed | 161 | 0.6087 | true |
| pair_aware_attention | 3101 | alternative_task_pickup | 211 | 0.5829 | false |
| pair_aware_attention | 3101 | alternative_downstream_priority | 121 | 0.4628 | false |
| pair_aware_attention | 3102 | competitor_robot_position | 192 | 0.7083 | true |
| pair_aware_attention | 3102 | competitor_robot_speed | 161 | 0.5404 | false |
| pair_aware_attention | 3102 | alternative_task_pickup | 211 | 0.5166 | false |
| pair_aware_attention | 3102 | alternative_downstream_priority | 121 | 0.5124 | false |
| pair_aware_attention | 3103 | competitor_robot_position | 192 | 0.6771 | true |
| pair_aware_attention | 3103 | competitor_robot_speed | 161 | 0.5839 | false |
| pair_aware_attention | 3103 | alternative_task_pickup | 211 | 0.5924 | false |
| pair_aware_attention | 3103 | alternative_downstream_priority | 121 | 0.5207 | false |

## Next Step

Stop without generating or viewing held-out evaluation data.

## Limitations

This train-only comparison contains no held-out model result, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.
