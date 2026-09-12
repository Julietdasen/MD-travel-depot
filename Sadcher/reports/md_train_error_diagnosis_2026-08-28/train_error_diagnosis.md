# Ticket 34 Train Error Diagnosis

Formal reproduction of the frozen Ticket 33 training protocol.

## Observations

- Seed instability observed: **true**.
- Pair-level fit gap observed: **true**.
- Residual saturation is the primary failure: **false**.
- All models respond to relational inputs: **true**.

## Per-Seed Decomposition

| Method | Seed | Agreement | Before | After | Both correct | Flip exact | No-flip exact | Saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matched_parameter_mlp | 3101 | 0.7143 | 0.7075 | 0.7210 | 1142 | 0.2759 | 0.7247 | 0.0007 |
| matched_parameter_mlp | 3102 | 0.4768 | 0.4850 | 0.4685 | 677 | 0.1285 | 0.4479 | 0.0000 |
| matched_parameter_mlp | 3103 | 0.7020 | 0.7025 | 0.7015 | 1137 | 0.2628 | 0.7278 | 0.0004 |
| pair_aware_attention | 3101 | 0.7195 | 0.7280 | 0.7110 | 1171 | 0.3358 | 0.7156 | 0.0014 |
| pair_aware_attention | 3102 | 0.6863 | 0.6920 | 0.6805 | 1118 | 0.3124 | 0.6875 | 0.0059 |
| pair_aware_attention | 3103 | 0.8007 | 0.8070 | 0.7945 | 1373 | 0.4745 | 0.7970 | 0.0353 |

## Next Protocol

Pre-register a train-only comparison that retains the bounded state margin, uses deterministic family-balanced batches, and adds a pair-structured objective without changing model capacity.

## Limitations

This train-only diagnosis contains no held-out model result, does not authorize Ticket 17, Ticket 20, or Ticket 32, and generated no production expert dataset.
