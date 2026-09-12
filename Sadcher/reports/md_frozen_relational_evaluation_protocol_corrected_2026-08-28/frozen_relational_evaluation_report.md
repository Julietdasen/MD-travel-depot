# Ticket 31 Frozen Margin-Stratified Relational Evaluation

## Protocol infeasible

At least one pre-registered family x pair-margin stratum has fewer unconditioned candidates than its fixed evaluation quota.

Frozen quota per family x stratum: **1**.
No evaluation package was written. No model was trained or evaluated, and no Ticket 20 production expert dataset was generated.

| Family | Pair-margin stratum | Available | Required |
|---|---|---:|---:|
| competitor_robot_position | clear_ge_0.10 | 0 | 1 |
| alternative_task_pickup | clear_ge_0.10 | 0 | 1 |
| alternative_downstream_priority | clear_ge_0.10 | 0 | 1 |

## Decision

Do not run Ticket 32. Revisit the completion proxy or pre-register a new margin protocol in a separate ticket; do not lower this protocol's frozen >=0.10 boundary in place.
