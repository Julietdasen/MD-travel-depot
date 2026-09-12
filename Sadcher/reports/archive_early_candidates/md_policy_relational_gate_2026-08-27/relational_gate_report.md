# Ticket 19 Multi-Transport Relational Gate

This controlled 3-transport-robot probe is not Ticket 20 and does not write a production expert dataset.

Cross-Attention relational claim: **inconclusive**

| Method | Train | Agreement [95% CI] | Flip [95% CI] | Regret | Saturation | Context gates |
|---|---:|---:|---:|---:|---:|---:|
| physics_only | n/a | 0.275 [0.150, 0.375] | 0.000 [0.000, 0.161] | 0.1637 | 0.000 | 0.000/0.000 |
| local_eta_priority | n/a | 0.325 [0.200, 0.425] | 0.000 [0.000, 0.161] | 0.1531 | 0.000 | 0.000/0.000 |
| matched_parameter_mlp | 0.400 | 0.275 [0.175, 0.375] | 0.000 [0.000, 0.161] | 0.1204 | 0.623 | 0.162/0.244 |
| cross_attention_full | 0.388 | 0.250 [0.125, 0.350] | 0.000 [0.000, 0.161] | 0.1774 | 0.792 | 0.232/0.138 |

Every intervention changes a competitor excluded from both oracle first actions; labels come from exact 3x4 assignment completion values.
Both trainable models miss the 0.80 train-agreement prerequisite, so this result is inconclusive for incremental Cross-Attention value rather than evidence that attention is harmful.
The present Cross-Attention blocks run before pair metadata enters the scorer and therefore do not directly attend over the complete robot-task ETA matrix used by the oracle proxy.
The claim is limited to held-out controlled relational templates and does not establish end-to-end schedule quality.
