# Prefix-Conditioned Exact-Action Reranker

Status: **prefix_conditioning_not_supported**.

Development-only experiment; confirmation and regression benchmark seeds were not read.

| Seed | Physics top-1 / regret | Prefix-zero top-1 / regret | Prefix top-1 / regret |
|---:|---:|---:|---:|
| 3101 | 0.9167 / 2.3667 | 0.9167 / 1.1167 | 0.9333 / 1.0333 |
| 3102 | 0.8833 / 1.3833 | 0.9500 / 0.5000 | 0.9333 / 1.4167 |
| 3103 | 0.9167 / 0.7833 | 0.9167 / 1.0167 | 0.9000 / 1.7333 |

Mean top-1: physics 0.9056, prefix-zero 0.9278, prefix 0.9222.
Prefix gain over physics: 0.0167.
Mean regret: physics 1.5111, prefix 1.3944.
Maximum within-seed order top-1 range: 0.0333.

## Preregistered checks

- mean_top1_gain: **false**
- mean_regret: **true**
- per_seed_drop: **true**
- prefix_ablation: **false**
- order_sensitivity: **false**

## Interpretation

The prefix model lowers mean regret relative to the retrained physics baseline, but its +0.0167 mean top-1 gain misses the frozen +0.02 threshold. More importantly, prefix-zero has higher mean top-1, so the result does not attribute improvement to prefix information. The maximum within-seed order range also exceeds the 0.03 limit.

The cached candidate records classify all evaluated best proposals as process actions, so this package cannot establish separate transport or mixed-action benefits. No confirmation run or online autoregressive integration is warranted from this result.

No production scheduler was changed. A failed gate is a NO-GO for confirmation and online integration.
