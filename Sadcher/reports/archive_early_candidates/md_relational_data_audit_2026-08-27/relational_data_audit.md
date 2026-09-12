# Ticket 29 Relational Data Audit

This report audits the controlled relational twins. It is independent from the gate reports and does not create a Ticket 20 production expert dataset.

## Relational Families

| Family | Pairs | Oracle flips | Before margin mean | After margin mean |
|---|---:|---:|---:|---:|
| competitor_robot_position | 25 | 1.000 | 0.018141 | 0.011443 |
| competitor_robot_speed | 25 | 1.000 | 0.011005 | 0.010981 |
| alternative_task_pickup | 25 | 1.000 | 0.017071 | 0.018715 |
| alternative_downstream_priority | 25 | 1.000 | 0.020590 | 0.015560 |

## Train/Evaluation Distributions

Distance is reported separately for robot-to-pickup and pickup-to-delivery legs.

| Feature | Train mean | Evaluation mean | Eval - train |
|---|---:|---:|---:|
| robot_to_pickup_distance | 1.072776 | 1.047440 | -0.025336 |
| loaded_leg_distance | 1.049685 | 1.067181 | +0.017496 |
| robot_speed | 1.090329 | 1.072699 | -0.017629 |
| downstream_priority | 0.624833 | 0.662787 | +0.037954 |
| eta | 1.564697 | 1.601081 | +0.036384 |
| oracle_completion_margin | 0.015185 | 0.016451 | +0.001266 |

## Rule Baselines

| Baseline | State agreement | Oracle pair accuracy | Prediction flip rate |
|---|---:|---:|---:|
| local_eta | 0.275 | 0.000 | 0.150 |
| eta_plus_priority | 0.325 | 0.000 | 0.200 |
| competitor_aware_relational | 0.475 | 0.200 | 0.900 |

## Perturbation Stability

Overall label flip rate: **0.020** (16/800).

| Perturbed feature | Magnitude | Comparisons | Label flip rate |
|---|---:|---:|---:|
| robot_position | 0.010 | 200 | 0.020 |
| robot_speed | 0.010 | 200 | 0.020 |
| task_pickup | 0.010 | 200 | 0.030 |
| downstream_priority | 0.010 | 200 | 0.010 |

## Evaluation Margin Strata

| Stratum | States | Mean margin |
|---|---:|---:|
| near_tie_lt_0.01 | 17 | 0.004686 |
| small_ge_0.01_lt_0.05 | 22 | 0.023924 |
| moderate_ge_0.05_lt_0.10 | 1 | 0.052056 |
| clear_ge_0.10 | 0 | n/a |

## Interpretation

Data contains a relational learning signal: **true**.

Supported by this audit:
- Each accepted twin changes the oracle action after a non-local intervention.
- The explicit completion-aware rule tests whether that dependence is recoverable from competitor context.
- The held-out split has zero exact template overlap with training.

Not supported by this audit:
- Twin acceptance is conditioned on an oracle flip, so the family flip rate is not a natural-frequency estimate.
- A rule-baseline advantage does not establish that the learned pair-aware scorer generalizes.
- This audit does not evaluate end-to-end scheduling quality or authorize Ticket 20 data generation.
