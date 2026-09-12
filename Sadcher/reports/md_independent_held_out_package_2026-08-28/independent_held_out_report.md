# Ticket 39 Independent Held-Out Relational Package

Protocol status: **frozen**.

## Stratum Supply And Selection

| Family | Pair-margin stratum | Available | Selected | Flip after selection |
|---|---|---:|---:|---:|
| competitor_robot_position | near_tie_lt_0.01 | 2467 | 25 | 16 |
| competitor_robot_position | small_ge_0.01_lt_0.03 | 1906 | 25 | 6 |
| competitor_robot_position | medium_ge_0.03_lt_0.05 | 515 | 25 | 2 |
| competitor_robot_position | high_ge_0.05 | 112 | 25 | 1 |
| competitor_robot_speed | near_tie_lt_0.01 | 2399 | 25 | 17 |
| competitor_robot_speed | small_ge_0.01_lt_0.03 | 1918 | 25 | 6 |
| competitor_robot_speed | medium_ge_0.03_lt_0.05 | 562 | 25 | 0 |
| competitor_robot_speed | high_ge_0.05 | 121 | 25 | 1 |
| alternative_task_pickup | near_tie_lt_0.01 | 2545 | 25 | 14 |
| alternative_task_pickup | small_ge_0.01_lt_0.03 | 1913 | 25 | 8 |
| alternative_task_pickup | medium_ge_0.03_lt_0.05 | 463 | 25 | 3 |
| alternative_task_pickup | high_ge_0.05 | 79 | 25 | 1 |
| alternative_downstream_priority | near_tie_lt_0.01 | 2364 | 25 | 8 |
| alternative_downstream_priority | small_ge_0.01_lt_0.03 | 1912 | 25 | 2 |
| alternative_downstream_priority | medium_ge_0.03_lt_0.05 | 576 | 25 | 0 |
| alternative_downstream_priority | high_ge_0.05 | 148 | 25 | 0 |

## Package

Evaluation pairs: **400**.
Evaluation states: **800**.
Train/evaluation template overlap: **0**.
Natural flip rate after selection: **0.2125**.

## Next Step

Run Ticket 40 only against this committed package.

## Limitations

This ticket freezes data only, contains no model result, does not unfreeze Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.
