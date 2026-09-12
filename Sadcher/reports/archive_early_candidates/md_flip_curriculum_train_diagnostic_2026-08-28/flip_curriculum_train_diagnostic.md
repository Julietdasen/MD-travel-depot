# Ticket 36 Flip-Balanced Curriculum Train Diagnostic

Decision: **optimization_not_feasible**.

Two-stage family-balanced pretraining and relational fine-tuning.

## Selected Checkpoints

| Method | Seed | Fine-tune epoch | Agreement | Saturation | All conditions |
|---|---:|---:|---:|---:|---:|
| matched_parameter_mlp | 3101 | 20 | 0.9603 | 0.2012 | true |
| matched_parameter_mlp | 3102 | 70 | 0.9287 | 0.2366 | true |
| matched_parameter_mlp | 3103 | 10 | 0.9720 | 0.2163 | true |
| pair_aware_attention | 3101 | 90 | 0.9353 | 0.1803 | true |
| pair_aware_attention | 3102 | 100 | 0.9467 | 0.2107 | true |
| pair_aware_attention | 3103 | 80 | 0.9952 | 0.4033 | false |

## Family Flip-Pair Exact

| Method | Seed | Family | Flip pairs | Exact accuracy | Pass |
|---|---:|---|---:|---:|---:|
| matched_parameter_mlp | 3101 | competitor_robot_position | 192 | 0.9375 | true |
| matched_parameter_mlp | 3101 | competitor_robot_speed | 161 | 0.8758 | true |
| matched_parameter_mlp | 3101 | alternative_task_pickup | 211 | 0.8863 | true |
| matched_parameter_mlp | 3101 | alternative_downstream_priority | 121 | 0.8512 | true |
| matched_parameter_mlp | 3102 | competitor_robot_position | 192 | 0.9062 | true |
| matched_parameter_mlp | 3102 | competitor_robot_speed | 161 | 0.8323 | true |
| matched_parameter_mlp | 3102 | alternative_task_pickup | 211 | 0.8436 | true |
| matched_parameter_mlp | 3102 | alternative_downstream_priority | 121 | 0.8430 | true |
| matched_parameter_mlp | 3103 | competitor_robot_position | 192 | 0.9583 | true |
| matched_parameter_mlp | 3103 | competitor_robot_speed | 161 | 0.9006 | true |
| matched_parameter_mlp | 3103 | alternative_task_pickup | 211 | 0.9147 | true |
| matched_parameter_mlp | 3103 | alternative_downstream_priority | 121 | 0.8595 | true |
| pair_aware_attention | 3101 | competitor_robot_position | 192 | 0.8958 | true |
| pair_aware_attention | 3101 | competitor_robot_speed | 161 | 0.8820 | true |
| pair_aware_attention | 3101 | alternative_task_pickup | 211 | 0.8626 | true |
| pair_aware_attention | 3101 | alternative_downstream_priority | 121 | 0.8099 | true |
| pair_aware_attention | 3102 | competitor_robot_position | 192 | 0.9427 | true |
| pair_aware_attention | 3102 | competitor_robot_speed | 161 | 0.9130 | true |
| pair_aware_attention | 3102 | alternative_task_pickup | 211 | 0.9052 | true |
| pair_aware_attention | 3102 | alternative_downstream_priority | 121 | 0.8843 | true |
| pair_aware_attention | 3103 | competitor_robot_position | 192 | 0.9948 | true |
| pair_aware_attention | 3103 | competitor_robot_speed | 161 | 0.9814 | true |
| pair_aware_attention | 3103 | alternative_task_pickup | 211 | 1.0000 | true |
| pair_aware_attention | 3103 | alternative_downstream_priority | 121 | 0.9835 | true |

## Next Step

Stop without generating or viewing held-out evaluation data.

## Limitations

This train-only diagnostic contains no held-out model result, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.
