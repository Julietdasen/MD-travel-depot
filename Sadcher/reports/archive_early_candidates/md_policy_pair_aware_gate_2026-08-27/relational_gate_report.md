# Ticket 29 Pair-Aware Relational Gate

This controlled 3-transport-robot probe is not Ticket 20 and does not write a production expert dataset.

**Report phase:** `initial_cross_entropy_gate`.
This is the initial cross-entropy gate.

The train-only optimization diagnosis is a separate run; this report contains held-out gate results.
Pair-aware relational claim: **inconclusive**

| Method | Train | Agreement [95% CI] | Flip [95% CI] | Regret | Utility@0.05 | Saturation | Context gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| physics_only | n/a | 0.275 [0.150, 0.375] | 0.000 [0.000, 0.161] | 0.1637 | 0.450 | 0.000 | 0.000/0.000 |
| local_eta_priority | n/a | 0.325 [0.200, 0.425] | 0.000 [0.000, 0.161] | 0.1531 | 0.475 | 0.000 | 0.000/0.000 |
| matched_parameter_mlp | 0.427 | 0.242 [0.167, 0.317] | 0.000 [0.000, 0.000] | 0.1649 | 0.533 | 0.769 | 0.228/0.199 |
| cross_attention_full | 0.454 | 0.300 [0.217, 0.400] | 0.033 [0.000, 0.083] | 0.1288 | 0.558 | 0.786 | 0.241/0.185 |
| pair_aware_attention | 0.429 | 0.283 [0.200, 0.358] | 0.000 [0.000, 0.000] | 0.1721 | 0.450 | 0.841 | 0.214/0.181 |

Margin-stratified diagnostics (evaluation states):
| Method | Stratum | States | Exact | Regret | Utility@0.05 |
|---|---|---:|---:|---:|---:|
| matched_parameter_mlp | near_tie_lt_0.01 | 17 | 0.176 | 0.173 | 0.608 |
| matched_parameter_mlp | less_ambiguous_ge_0.01 | 22 | 0.288 | 0.157 | 0.485 |
| matched_parameter_mlp | clear_ge_0.05 | 1 | 0.333 | 0.195 | 0.333 |
| cross_attention_full | near_tie_lt_0.01 | 17 | 0.275 | 0.115 | 0.667 |
| cross_attention_full | less_ambiguous_ge_0.01 | 22 | 0.288 | 0.146 | 0.455 |
| cross_attention_full | clear_ge_0.05 | 1 | 1.000 | 0.000 | 1.000 |
| pair_aware_attention | near_tie_lt_0.01 | 17 | 0.333 | 0.151 | 0.569 |
| pair_aware_attention | less_ambiguous_ge_0.01 | 22 | 0.258 | 0.183 | 0.379 |
| pair_aware_attention | clear_ge_0.05 | 1 | 0.000 | 0.293 | 0.000 |

Every intervention changes a competitor excluded from both oracle first actions; labels come from exact 3x4 assignment completion values.
The 0.80 train-agreement prerequisite was pre-registered for every pair-aware and matched-MLP model seed.
Pair-aware minus matched-MLP agreement: 0.042 [95% CI -0.017, 0.108].
Pair-aware axial attention consumes competitor pair metadata while excluding focal ETA from its own learned residual.
The claim is limited to held-out controlled relational templates and does not establish end-to-end schedule quality.
