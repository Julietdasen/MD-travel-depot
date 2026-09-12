# Ticket 30 Unconditioned Relational Candidate Audit

Candidates are retained before oracle labels, flips, margins, or model outputs are observed. This audit does not create a Ticket 20 production expert dataset.

Candidate pairs: **20000**.
Natural oracle flip rate: **0.3372**.

| Family | Candidates | Natural flip rate | <0.01 | [0.01,0.05) | [0.05,0.10) | >=0.10 |
|---|---:|---:|---:|---:|---:|---:|
| competitor_robot_position | 5000 | 0.3956 | 2417 | 2460 | 123 | 0 |
| competitor_robot_speed | 5000 | 0.2926 | 2390 | 2482 | 127 | 1 |
| alternative_task_pickup | 5000 | 0.4250 | 2485 | 2427 | 88 | 0 |
| alternative_downstream_priority | 5000 | 0.2356 | 2326 | 2528 | 146 | 0 |

## Ticket 31 Supply Decision

Clear (`pair_margin >=0.10`) counts by family: competitor_robot_position=0, competitor_robot_speed=1, alternative_task_pickup=0, alternative_downstream_priority=0.
Maximum uniform clear-stratum quota: **0**.
Four-strata protocol feasible: **false**.

Ticket 31 must report infeasible rather than lower the frozen >=0.10 boundary when the uniform quota is zero.
