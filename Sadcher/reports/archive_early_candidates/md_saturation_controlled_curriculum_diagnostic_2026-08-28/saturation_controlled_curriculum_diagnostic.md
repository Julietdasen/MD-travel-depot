# Ticket 37 Saturation-Controlled Curriculum Diagnostic

Decision: **optimization_not_feasible**.

Two-stage relational curriculum with raw-residual saturation control.

## Selected Checkpoints

| Method | Seed | Fine-tune epoch | Agreement | Saturation | All conditions |
|---|---:|---:|---:|---:|---:|
| matched_parameter_mlp | 3101 | 70 | 0.9762 | 0.0074 | true |
| matched_parameter_mlp | 3102 | 100 | 0.9293 | 0.0129 | true |
| matched_parameter_mlp | 3103 | 100 | 0.9835 | 0.0086 | true |
| pair_aware_attention | 3101 | 100 | 0.8003 | 0.0066 | false |
| pair_aware_attention | 3102 | 80 | 0.9200 | 0.0112 | true |
| pair_aware_attention | 3103 | 80 | 0.9895 | 0.0077 | true |

## Family Flip-Pair Exact

| Method | Seed | Family | Flip pairs | Exact accuracy | Pass |
|---|---:|---|---:|---:|---:|
| matched_parameter_mlp | 3101 | competitor_robot_position | 192 | 0.9844 | true |
| matched_parameter_mlp | 3101 | competitor_robot_speed | 161 | 0.9130 | true |
| matched_parameter_mlp | 3101 | alternative_task_pickup | 211 | 0.9336 | true |
| matched_parameter_mlp | 3101 | alternative_downstream_priority | 121 | 0.8926 | true |
| matched_parameter_mlp | 3102 | competitor_robot_position | 192 | 0.9167 | true |
| matched_parameter_mlp | 3102 | competitor_robot_speed | 161 | 0.8385 | true |
| matched_parameter_mlp | 3102 | alternative_task_pickup | 211 | 0.8294 | true |
| matched_parameter_mlp | 3102 | alternative_downstream_priority | 121 | 0.8182 | true |
| matched_parameter_mlp | 3103 | competitor_robot_position | 192 | 0.9792 | true |
| matched_parameter_mlp | 3103 | competitor_robot_speed | 161 | 0.9627 | true |
| matched_parameter_mlp | 3103 | alternative_task_pickup | 211 | 0.9668 | true |
| matched_parameter_mlp | 3103 | alternative_downstream_priority | 121 | 0.9339 | true |
| pair_aware_attention | 3101 | competitor_robot_position | 192 | 0.6719 | true |
| pair_aware_attention | 3101 | competitor_robot_speed | 161 | 0.5466 | false |
| pair_aware_attention | 3101 | alternative_task_pickup | 211 | 0.6019 | true |
| pair_aware_attention | 3101 | alternative_downstream_priority | 121 | 0.5537 | false |
| pair_aware_attention | 3102 | competitor_robot_position | 192 | 0.8750 | true |
| pair_aware_attention | 3102 | competitor_robot_speed | 161 | 0.8137 | true |
| pair_aware_attention | 3102 | alternative_task_pickup | 211 | 0.8483 | true |
| pair_aware_attention | 3102 | alternative_downstream_priority | 121 | 0.7934 | true |
| pair_aware_attention | 3103 | competitor_robot_position | 192 | 0.9844 | true |
| pair_aware_attention | 3103 | competitor_robot_speed | 161 | 0.9752 | true |
| pair_aware_attention | 3103 | alternative_task_pickup | 211 | 0.9858 | true |
| pair_aware_attention | 3103 | alternative_downstream_priority | 121 | 0.9752 | true |

## Next Step

Stop without generating or viewing held-out evaluation data.

## Limitations

This train-only diagnostic contains no held-out model result, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.
