# Ticket 42 Context Ablation Development Package

Status: **development_package_frozen**.
Protocol status: **frozen**.

## Stratum Supply And Selection

| Family | Pair-margin stratum | Available | Selected | Flip after selection | No flip after selection |
|---|---|---:|---:|---:|---:|
| competitor_robot_position | near_tie_lt_0.01 | 4981 | 50 | 28 | 22 |
| competitor_robot_position | small_ge_0.01_lt_0.03 | 3743 | 50 | 12 | 38 |
| competitor_robot_position | medium_ge_0.03_lt_0.05 | 1068 | 50 | 9 | 41 |
| competitor_robot_position | high_ge_0.05 | 208 | 50 | 1 | 49 |
| competitor_robot_speed | near_tie_lt_0.01 | 4862 | 50 | 26 | 24 |
| competitor_robot_speed | small_ge_0.01_lt_0.03 | 3842 | 50 | 8 | 42 |
| competitor_robot_speed | medium_ge_0.03_lt_0.05 | 1071 | 50 | 3 | 47 |
| competitor_robot_speed | high_ge_0.05 | 225 | 50 | 1 | 49 |
| alternative_task_pickup | near_tie_lt_0.01 | 5058 | 50 | 23 | 27 |
| alternative_task_pickup | small_ge_0.01_lt_0.03 | 3801 | 50 | 14 | 36 |
| alternative_task_pickup | medium_ge_0.03_lt_0.05 | 943 | 50 | 8 | 42 |
| alternative_task_pickup | high_ge_0.05 | 198 | 50 | 4 | 46 |
| alternative_downstream_priority | near_tie_lt_0.01 | 4799 | 50 | 23 | 27 |
| alternative_downstream_priority | small_ge_0.01_lt_0.03 | 3769 | 50 | 6 | 44 |
| alternative_downstream_priority | medium_ge_0.03_lt_0.05 | 1156 | 50 | 2 | 48 |
| alternative_downstream_priority | high_ge_0.05 | 276 | 50 | 0 | 50 |

## Package Checks

Development pairs: **800**.
Development states: **1600**.
Package replay mismatch: **0**.
Training/development template overlap: **0**.
Prior held-out/development template overlap: **0**.

## Authorization

Ticket 43 authorized: **True**.
Ticket 43 is authorized to consume only this committed formal development package.

## Scope

This ticket generated development data only. It ran no model, selected no architecture, generated no production expert dataset, and did not reopen an architecture gate.
