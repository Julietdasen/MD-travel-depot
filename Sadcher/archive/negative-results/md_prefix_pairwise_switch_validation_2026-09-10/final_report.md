# MD Prefix Pairwise Switch Validation

Status: **pairwise_switch_not_validated**.
Frozen switch probability threshold: 0.800.
Validation: 50 new states, 3226 exact actions, 150 decisions.

| Decoder | Mean regret | Mean duration | Top-1 | Max regret | Regret >= 20 |
|---|---:|---:|---:|---:|---:|
| prefix | 6.187 | 205.587 | 0.253 | 56.0 | 15 |
| pairwise_gate | 6.187 | 205.587 | 0.253 | 56.0 | 15 |
| unrestricted_gate | 2.953 | 202.353 | 0.053 | 31.0 | 9 |
| prefix_zero | 5.847 | 205.247 | 0.253 | 56.0 | 13 |

Approved switches: 0.
Improved/equal/worse: 0/150/0.
Mean paired regret reduction: 0.000.
State-clustered 95% bootstrap interval: [0.000, 0.000].

Classifier and threshold were frozen before validation labels were generated. Production scheduling remains unchanged.
