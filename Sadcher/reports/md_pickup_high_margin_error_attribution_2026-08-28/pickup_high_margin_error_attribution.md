# Ticket 41 Pickup And High-Margin Error Attribution

Scope: **post_hoc_diagnostic_only**.

## Direct Evidence

| Method | State agreement | Exact pair |
|---|---:|---:|
| matched_parameter_mlp | 0.6583 | 0.5408 |
| pair_aware_attention | 0.7275 | 0.6133 |

Pickup and high-margin attribution classification: **task_context_insufficient**.
State-agreement gain: **0.0692**; exact-pair gain: **0.0725**.

## High-Margin Transitions

| Target | Wrong to correct | Correct to wrong | Unchanged |
|---|---:|---:|---:|
| state | 59 | 51 | 490 |
| exact_pair | 34 | 35 | 231 |

## Family And Margin Evidence

| Family | Matched exact pair | Pair-aware exact pair |
|---|---:|---:|
| competitor_robot_position | 0.5533 | 0.6067 |
| competitor_robot_speed | 0.5100 | 0.6233 |
| alternative_task_pickup | 0.5000 | 0.5633 |
| alternative_downstream_priority | 0.6000 | 0.6600 |

Pair-aware high-margin saturation: **0.3426**.
Pickup changed-task oracle participation: **0.2150**; mean oracle value gap: **0.4496**.

## Interpretation

Pickup exact-pair remains below 0.60 while only 0.2150 of changed tasks participate in the oracle action; the observed pattern is consistent with insufficient alternative-task/process context in the learned residual. High-margin wrong-to-correct and correct-to-wrong counts are near-balanced, while saturation is 0.3426, so the data do not isolate a transport-calibration-only explanation.

The legacy component carries the local backbone score, the fixed physics component carries ETA calibration, and the bounded residual is the learned term that can express relational task context. The observed pickup failure therefore selects task/process context ablation as the next development experiment; high-margin saturation remains a secondary calibration signal.

Next implementation choice: **Use a new development package for task/process context ablation.**

## Limitations

These are post-hoc diagnostic associations, not causal proof. This report does not reopen an architecture gate or authorize held-out model selection. Any later candidate must first use a new development package, freeze its protocol, and then use a completely new confirmation held-out package.
