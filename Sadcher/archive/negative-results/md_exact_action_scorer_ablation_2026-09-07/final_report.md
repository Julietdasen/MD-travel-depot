# Exact Action Scorer Ablation

Status: **scorer_not_supported**.

| Model | Seed | Tolerance top-1 | Mean regret |
|---|---:|---:|---:|
| Additive baseline | 3101 | 0.8333 | 2.6000 |
| Additive baseline | 3102 | 0.8500 | 2.1000 |
| Additive baseline | 3103 | 0.8500 | 1.6000 |
| linear | 3101 | 0.8167 | 3.3667 |
| linear | 3102 | 0.8667 | 1.3667 |
| linear | 3103 | 0.8333 | 1.5167 |
| deepsets | 3101 | 0.8500 | 3.1167 |
| deepsets | 3102 | 0.8667 | 2.0167 |
| deepsets | 3103 | 0.8667 | 1.3667 |

The additive baseline mean top-1 is 0.8444 with mean regret 2.1000. DeepSets
raises mean top-1 to 0.8611 (+0.0167) but raises mean regret to 2.1667, so it
misses both the preregistered +0.02 accuracy gain and lower-regret requirements.
Linear reaches 0.8389 mean top-1 and 2.0833 mean regret and also fails.

All DeepSets checkpoints select epoch 20. By epoch 200, development top-1 falls
to 0.7833/0.7667/0.8000 and regret rises to 4.2667/3.5667/2.6833, showing clear
overfitting on the 150-state training package. The scorer inputs contain action
structure and residual rank but not the full simulator/task/robot physics state.

Training used only train snapshots. Development selected checkpoints. Confirmation was not read and production decoding was not changed.
