# Ticket 29 Pair-Aware Relational Gate

This controlled 3-transport-robot probe is not Ticket 20 and does not write a production expert dataset.

**Report phase:** `margin_balanced_diagnostic`.
This is a margin-balanced diagnostic subset; it does not alter the formal gate.

The train-only optimization diagnosis is a separate run; this report contains held-out gate results.
Pair-aware relational claim: **diagnostic_only**

| Method | Train | Agreement [95% CI] | Flip [95% CI] | Regret | Utility@0.05 | Saturation | Context gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| physics_only | n/a | 0.417 [0.292, 0.500] | 0.000 [0.000, 0.242] | 0.0826 | 0.583 | 0.000 | 0.000/0.000 |
| local_eta_priority | n/a | 0.458 [0.375, 0.500] | 0.000 [0.000, 0.242] | 0.0980 | 0.500 | 0.000 | 0.000/0.000 |
| matched_parameter_mlp | 0.667 | 0.264 [0.167, 0.347] | 0.000 [0.000, 0.000] | 0.1861 | 0.333 | 0.096 | 0.131/0.231 |
| cross_attention_full | 0.878 | 0.306 [0.194, 0.417] | 0.083 [0.028, 0.167] | 0.2181 | 0.319 | 0.160 | 0.312/0.237 |
| pair_aware_attention | 0.771 | 0.319 [0.208, 0.403] | 0.056 [0.000, 0.139] | 0.1551 | 0.444 | 0.153 | 0.297/0.190 |

Margin-stratified diagnostics (evaluation states):
| Method | Stratum | States | Exact | Regret | Utility@0.05 |
|---|---|---:|---:|---:|---:|
| matched_parameter_mlp | near_tie_lt_0.01 | 0 | n/a | n/a | n/a |
| matched_parameter_mlp | less_ambiguous_ge_0.01 | 21 | 0.254 | 0.204 | 0.333 |
| matched_parameter_mlp | clear_ge_0.05 | 3 | 0.333 | 0.062 | 0.333 |
| cross_attention_full | near_tie_lt_0.01 | 0 | n/a | n/a | n/a |
| cross_attention_full | less_ambiguous_ge_0.01 | 21 | 0.270 | 0.216 | 0.286 |
| cross_attention_full | clear_ge_0.05 | 3 | 0.556 | 0.234 | 0.556 |
| pair_aware_attention | near_tie_lt_0.01 | 0 | n/a | n/a | n/a |
| pair_aware_attention | less_ambiguous_ge_0.01 | 21 | 0.302 | 0.164 | 0.444 |
| pair_aware_attention | clear_ge_0.05 | 3 | 0.444 | 0.096 | 0.444 |

Every intervention changes a competitor excluded from both oracle first actions; labels come from exact 3x4 assignment completion values.
The 0.80 train-agreement prerequisite was pre-registered for every pair-aware and matched-MLP model seed.
Pair-aware minus matched-MLP agreement: 0.056 [95% CI -0.042, 0.139].
Pair-aware axial attention consumes competitor pair metadata while excluding focal ETA from its own learned residual.
The claim is limited to held-out controlled relational templates and does not establish end-to-end schedule quality.
