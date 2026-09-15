# Prefix Duration Demo

This is an offline one-step comparison. For each test state, the model selects
one complete first action with Beam-16; the exact oracle's continuation cost is
then used as the remaining duration. It is not a multi-step production rollout.

## First three test states

| Method | Mean remaining duration |
|---|---:|
| Exact optimum | 177.33 |
| Prefix-zero, averaged over 3 checkpoints | 191.56 |
| Prefix, averaged over 3 checkpoints | 199.56 |

## All twelve test states

| Method | Mean remaining duration |
|---|---:|
| Exact optimum | 213.42 |
| Prefix-zero, averaged over 3 checkpoints | 221.14 |
| Prefix, averaged over 3 checkpoints | 224.72 |

Prefix is therefore 3.58 time units slower than prefix-zero on this offline
duration metric. The decoder remains legal; the problem is action quality, not
execution validity.
