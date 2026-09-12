# Train-Only Optimization Diagnostic

Decision: **optimization_not_feasible**.

This exploratory run measures optimization on the selected training pool only. It contains no held-out model result and does not establish relational generalization.

## Frozen Protocol

- Candidate pool per family: 5000
- First training pairs per family: 500
- Data seed: 3030
- Model seeds: 3101, 3102, 3103
- Bounded-margin objective: 0.1
- Epochs: 200
- Overall agreement threshold: 0.70
- Per-family flip-pair exact threshold: 0.60
- Saturation threshold: 0.25

## Per-Seed Results

| Method | Seed | Overall agreement | Saturation | All conditions |
|---|---:|---:|---:|---:|
| matched_parameter_mlp | 3101 | 0.7143 | 0.0007 | false |
| matched_parameter_mlp | 3102 | 0.4768 | 0.0000 | false |
| matched_parameter_mlp | 3103 | 0.7020 | 0.0004 | false |
| pair_aware_attention | 3101 | 0.7195 | 0.0014 | false |
| pair_aware_attention | 3102 | 0.6863 | 0.0059 | false |
| pair_aware_attention | 3103 | 0.8007 | 0.0353 | false |

## Oracle-Flip Pair Exact

| Method | Seed | Family | Flip pairs | Exact accuracy | Pass |
|---|---:|---|---:|---:|---:|
| matched_parameter_mlp | 3101 | competitor_robot_position | 192 | 0.3802 | false |
| matched_parameter_mlp | 3101 | competitor_robot_speed | 161 | 0.2236 | false |
| matched_parameter_mlp | 3101 | alternative_task_pickup | 211 | 0.2464 | false |
| matched_parameter_mlp | 3101 | alternative_downstream_priority | 121 | 0.2314 | false |
| matched_parameter_mlp | 3102 | competitor_robot_position | 192 | 0.2083 | false |
| matched_parameter_mlp | 3102 | competitor_robot_speed | 161 | 0.0621 | false |
| matched_parameter_mlp | 3102 | alternative_task_pickup | 211 | 0.1185 | false |
| matched_parameter_mlp | 3102 | alternative_downstream_priority | 121 | 0.1074 | false |
| matched_parameter_mlp | 3103 | competitor_robot_position | 192 | 0.4062 | false |
| matched_parameter_mlp | 3103 | competitor_robot_speed | 161 | 0.1180 | false |
| matched_parameter_mlp | 3103 | alternative_task_pickup | 211 | 0.2275 | false |
| matched_parameter_mlp | 3103 | alternative_downstream_priority | 121 | 0.2893 | false |
| pair_aware_attention | 3101 | competitor_robot_position | 192 | 0.4688 | false |
| pair_aware_attention | 3101 | competitor_robot_speed | 161 | 0.2671 | false |
| pair_aware_attention | 3101 | alternative_task_pickup | 211 | 0.3033 | false |
| pair_aware_attention | 3101 | alternative_downstream_priority | 121 | 0.2727 | false |
| pair_aware_attention | 3102 | competitor_robot_position | 192 | 0.4219 | false |
| pair_aware_attention | 3102 | competitor_robot_speed | 161 | 0.2050 | false |
| pair_aware_attention | 3102 | alternative_task_pickup | 211 | 0.2986 | false |
| pair_aware_attention | 3102 | alternative_downstream_priority | 121 | 0.3058 | false |
| pair_aware_attention | 3103 | competitor_robot_position | 192 | 0.5729 | false |
| pair_aware_attention | 3103 | competitor_robot_speed | 161 | 0.3665 | false |
| pair_aware_attention | 3103 | alternative_task_pickup | 211 | 0.4882 | false |
| pair_aware_attention | 3103 | alternative_downstream_priority | 121 | 0.4380 | false |

## Interpretation

At least one fixed training condition failed; stop without viewing a new held-out evaluation.

Ticket 17, Ticket 20, and Ticket 32 remain unchanged. No production expert dataset was generated.
