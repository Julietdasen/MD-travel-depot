# Ticket 29 Pair-Aware Relational Gate

This controlled 3-transport-robot probe is not Ticket 20 and does not write a production expert dataset.

**Report phase:** `bounded_margin_follow_up_held_out_gate`.
This is the bounded-margin follow-up held-out gate.

The train-only optimization diagnosis is a separate run; this report contains held-out gate results.
Pair-aware relational claim: **inconclusive**

| Method | Train | Agreement [95% CI] | Flip [95% CI] | Regret | Utility@0.05 | Saturation | Context gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| physics_only | n/a | 0.275 [0.150, 0.375] | 0.000 [0.000, 0.161] | 0.1637 | 0.450 | 0.000 | 0.000/0.000 |
| local_eta_priority | n/a | 0.325 [0.200, 0.425] | 0.000 [0.000, 0.161] | 0.1531 | 0.475 | 0.000 | 0.000/0.000 |
| matched_parameter_mlp | 0.790 | 0.225 [0.167, 0.283] | 0.000 [0.000, 0.000] | 0.1428 | 0.492 | 0.059 | 0.381/0.431 |
| cross_attention_full | 0.896 | 0.208 [0.142, 0.275] | 0.000 [0.000, 0.000] | 0.1716 | 0.425 | 0.158 | 0.325/0.284 |
| pair_aware_attention | 0.931 | 0.267 [0.200, 0.342] | 0.033 [0.000, 0.083] | 0.1317 | 0.533 | 0.184 | 0.332/0.244 |

Margin-stratified diagnostics (evaluation states):
| Method | Stratum | States | Exact | Regret | Utility@0.05 |
|---|---|---:|---:|---:|---:|
| matched_parameter_mlp | near_tie_lt_0.01 | 17 | 0.294 | 0.132 | 0.608 |
| matched_parameter_mlp | less_ambiguous_ge_0.01 | 22 | 0.182 | 0.148 | 0.424 |
| matched_parameter_mlp | clear_ge_0.05 | 1 | 0.000 | 0.231 | 0.000 |
| cross_attention_full | near_tie_lt_0.01 | 17 | 0.255 | 0.140 | 0.490 |
| cross_attention_full | less_ambiguous_ge_0.01 | 22 | 0.182 | 0.189 | 0.394 |
| cross_attention_full | clear_ge_0.05 | 1 | 0.000 | 0.315 | 0.000 |
| pair_aware_attention | near_tie_lt_0.01 | 17 | 0.294 | 0.095 | 0.725 |
| pair_aware_attention | less_ambiguous_ge_0.01 | 22 | 0.242 | 0.157 | 0.394 |
| pair_aware_attention | clear_ge_0.05 | 1 | 0.333 | 0.195 | 0.333 |

Every intervention changes a competitor excluded from both oracle first actions; labels come from exact 3x4 assignment completion values.
The 0.80 train-agreement prerequisite was pre-registered for every pair-aware and matched-MLP model seed.
Pair-aware minus matched-MLP agreement: 0.042 [95% CI -0.050, 0.125].
Pair-aware axial attention consumes competitor pair metadata while excluding focal ETA from its own learned residual.
The claim is limited to held-out controlled relational templates and does not establish end-to-end schedule quality.
