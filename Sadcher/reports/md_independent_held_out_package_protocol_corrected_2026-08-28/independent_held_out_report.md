# Ticket 39 Independent Held-Out Relational Package

Protocol status: **frozen**.

## Stratum Supply And Selection

| Family | Pair-margin stratum | Available | Selected | Flip after selection |
|---|---|---:|---:|---:|
| competitor_robot_position | near_tie_lt_0.01 | 2495 | 25 | 11 |
| competitor_robot_position | small_ge_0.01_lt_0.03 | 1841 | 25 | 10 |
| competitor_robot_position | medium_ge_0.03_lt_0.05 | 555 | 25 | 3 |
| competitor_robot_position | high_ge_0.05 | 109 | 25 | 4 |
| competitor_robot_speed | near_tie_lt_0.01 | 2435 | 25 | 11 |
| competitor_robot_speed | small_ge_0.01_lt_0.03 | 1895 | 25 | 2 |
| competitor_robot_speed | medium_ge_0.03_lt_0.05 | 562 | 25 | 1 |
| competitor_robot_speed | high_ge_0.05 | 108 | 25 | 3 |
| alternative_task_pickup | near_tie_lt_0.01 | 2551 | 25 | 13 |
| alternative_task_pickup | small_ge_0.01_lt_0.03 | 1881 | 25 | 10 |
| alternative_task_pickup | medium_ge_0.03_lt_0.05 | 463 | 25 | 5 |
| alternative_task_pickup | high_ge_0.05 | 105 | 25 | 3 |
| alternative_downstream_priority | near_tie_lt_0.01 | 2324 | 25 | 8 |
| alternative_downstream_priority | small_ge_0.01_lt_0.03 | 1947 | 25 | 2 |
| alternative_downstream_priority | medium_ge_0.03_lt_0.05 | 593 | 25 | 0 |
| alternative_downstream_priority | high_ge_0.05 | 136 | 25 | 0 |

## Package

Evaluation pairs: **400**.
Evaluation states: **800**.
Train/evaluation template overlap: **0**.
Natural flip rate after selection: **0.215**.

## Next Step

Ticket 40 is authorized only against this committed package.

## Limitations

This ticket freezes data only, contains no model result, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.
