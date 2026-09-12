# Ticket 38 Budgeted Saturation Curriculum Diagnostic

Decision: **optimization_feasible**.

Two-stage relational curriculum with a 25% residual saturation budget.

## Selected Checkpoints

| Method | Seed | Fine-tune epoch | Agreement | Saturation | All conditions |
|---|---:|---:|---:|---:|---:|
| matched_parameter_mlp | 3101 | 90 | 0.9788 | 0.2402 | true |
| matched_parameter_mlp | 3102 | 100 | 0.9427 | 0.2363 | true |
| matched_parameter_mlp | 3103 | 100 | 0.9840 | 0.2404 | true |
| pair_aware_attention | 3101 | 90 | 0.8568 | 0.1206 | true |
| pair_aware_attention | 3102 | 90 | 0.9433 | 0.2294 | true |
| pair_aware_attention | 3103 | 70 | 0.9960 | 0.2261 | true |

## Family Flip-Pair Exact

| Method | Seed | Family | Flip pairs | Exact accuracy | Pass |
|---|---:|---|---:|---:|---:|
| matched_parameter_mlp | 3101 | competitor_robot_position | 192 | 0.9635 | true |
| matched_parameter_mlp | 3101 | competitor_robot_speed | 161 | 0.9441 | true |
| matched_parameter_mlp | 3101 | alternative_task_pickup | 211 | 0.9716 | true |
| matched_parameter_mlp | 3101 | alternative_downstream_priority | 121 | 0.9339 | true |
| matched_parameter_mlp | 3102 | competitor_robot_position | 192 | 0.9375 | true |
| matched_parameter_mlp | 3102 | competitor_robot_speed | 161 | 0.8820 | true |
| matched_parameter_mlp | 3102 | alternative_task_pickup | 211 | 0.8768 | true |
| matched_parameter_mlp | 3102 | alternative_downstream_priority | 121 | 0.8595 | true |
| matched_parameter_mlp | 3103 | competitor_robot_position | 192 | 0.9948 | true |
| matched_parameter_mlp | 3103 | competitor_robot_speed | 161 | 0.9503 | true |
| matched_parameter_mlp | 3103 | alternative_task_pickup | 211 | 0.9763 | true |
| matched_parameter_mlp | 3103 | alternative_downstream_priority | 121 | 0.9091 | true |
| pair_aware_attention | 3101 | competitor_robot_position | 192 | 0.7604 | true |
| pair_aware_attention | 3101 | competitor_robot_speed | 161 | 0.7205 | true |
| pair_aware_attention | 3101 | alternative_task_pickup | 211 | 0.7299 | true |
| pair_aware_attention | 3101 | alternative_downstream_priority | 121 | 0.7355 | true |
| pair_aware_attention | 3102 | competitor_robot_position | 192 | 0.9375 | true |
| pair_aware_attention | 3102 | competitor_robot_speed | 161 | 0.8820 | true |
| pair_aware_attention | 3102 | alternative_task_pickup | 211 | 0.8768 | true |
| pair_aware_attention | 3102 | alternative_downstream_priority | 121 | 0.8678 | true |
| pair_aware_attention | 3103 | competitor_robot_position | 192 | 0.9948 | true |
| pair_aware_attention | 3103 | competitor_robot_speed | 161 | 0.9752 | true |
| pair_aware_attention | 3103 | alternative_task_pickup | 211 | 0.9905 | true |
| pair_aware_attention | 3103 | alternative_downstream_priority | 121 | 0.9917 | true |

## Next Step

Freeze a separate held-out relational evaluation protocol.

## Limitations

This train-only diagnostic contains no held-out model result, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.
