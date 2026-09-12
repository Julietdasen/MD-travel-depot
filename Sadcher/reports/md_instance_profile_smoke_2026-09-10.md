# MD Task-Oriented Instance Profile Smoke Check

Five seeds (81000--81004) were generated for every profile. The check measured
the number of legal complete initial actions and enumeration time on the local
Mac CPU. It did not solve every continuation.

| Profile | Legal actions min / median / max | Enumeration median / max (ms) |
|---|---:|---:|
| balanced | 64 / 129 / 740 | 2.25 / 199.13 |
| process_scarce | 27 / 34 / 90 | 0.45 / 1.13 |
| transport_bottleneck | 5 / 11 / 41 | 0.30 / 0.63 |
| dependency_deep | 25 / 25 / 38 | 0.34 / 0.52 |
| mixed_hard | 30 / 92 / 216 | 2.78 / 10.90 |
| scale_medium | 228 / 457 / 2518 | 373.38 / 14702.52 |

The profiles create meaningfully different decision regimes. The historical
24--96 action filter would systematically exclude many balanced,
transport_bottleneck, and scale_medium cases. Future reports should stratify by
profile and action-count bucket instead of applying that filter globally.

The scale_medium profile should be used for neural inference and latency tests.
Exhaustive continuation labeling should be sampled or time-limited on Mac.

One untrained prefix model was also used as a shape/legality smoke check. It
accepted all six profile shapes (4--7 robots and 12--18 tasks), and all six
Beam-16 outputs passed the oracle-aligned legality check. Single-call times
ranged from 3.84 ms to 44.42 ms. These cold single-call values are not a latency
benchmark; they show that branching can materially change decoder cost.

Existing trained checkpoints have only seen balanced instances. This smoke
check establishes interface compatibility, not cross-profile solution quality.
