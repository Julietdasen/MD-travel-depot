# Terminal Return Protocol Counterfactual

Diagnostic-only analysis using the existing 90 C0/MILP paired execution
records. No assignment or task order was changed.

The canonical simulator starts robot returns only after every real task is
complete. The counterfactual makespan instead lets each robot return immediately
after its own final task. This separates protocol waiting from final-position
and scheduling effects.

## Results

| C0 seed | Canonical gap | Immediate-return gap | Gap removed | C0 protocol wait | MILP protocol wait |
|---:|---:|---:|---:|---:|---:|
| 3101 | 41.267 | 36.367 | 4.900 | 13.833 | 8.933 |
| 3102 | 35.333 | 28.367 | 6.967 | 15.900 | 8.933 |
| 3103 | 41.733 | 32.467 | 9.267 | 18.200 | 8.933 |
| Mean | 39.444 | 32.400 | 7.044 | 15.978 | 8.933 |

The synchronized-return protocol amplifies the C0/MILP gap, but it is not the
main cause. About 7.04 of the 39.44 mean gap is removed by immediate return;
32.40 remains under the counterfactual because task completion and final robot
locations still differ.

The large absolute tail is consistent with the physical scale: locations span
approximately a 100 by 100 map, robot speeds are near one distance unit per
timestep, and the corner-to-exit travel time can approach 142 timesteps. C0
mean terminal tails are 93.23 to 99.23, while the MILP mean is 69.67; observed
C0 tails reach 134.

## Implication For Training Data

Changing only the terminal protocol would not close most of the gap. Training
data must represent both sources: earlier real-task completion and the final
bottleneck robot position. The existing MILP implementation only solves a fresh
domain from time zero, so exact forced-action labels from arbitrary late-stage
simulator snapshots require an explicit residual-state oracle extension before
they can be used as training truth.
