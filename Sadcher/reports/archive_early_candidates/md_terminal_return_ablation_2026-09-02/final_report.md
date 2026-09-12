# Terminal-Return-Aware C0 Ablation

Diagnostic-only paired ablation. The same frozen C0 checkpoints, model seeds, 30 frozen instances, hard mask, constrained decoder, fallback, and terminal simulator protocol were used. The only changed component was a fixed score post-processing penalty of `0.01 * estimated_return_duration` for each `(robot, task)` action.

## Results

| Seed | Baseline makespan | Return-aware makespan | Delta | Baseline tail | Return-aware tail | Tail delta | Real completion delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3101 | 240.400 | 249.433 | +9.033 | 98.033 | 106.800 | +8.767 | +0.267 |
| 3102 | 234.467 | 239.333 | +4.867 | 93.233 | 98.833 | +5.600 | -0.733 |
| 3103 | 240.867 | 242.967 | +2.100 | 99.233 | 102.567 | +3.333 | -1.233 |
| **Mean** | **238.578** | **243.911** | **+5.333** | **96.833** | **102.733** | **+5.900** | **-0.567** |

All 90 paired runs succeeded. The return-aware variant was better on makespan in 20/90 pairs, tied in 23/90, and worse in 47/90. Mean fallback count increased by `0.2` per instance.

## Interpretation

The intervention did not reduce terminal return tail; it increased it by `5.9` on average and increased makespan by `5.33`. Real task completion improved slightly, so the item traded a small amount of work-stage performance for a worse terminal state.

This rejects the tested intervention, not the broader hypothesis that terminal position matters. The implemented penalty is an action-local estimate: it charges the selected task's location-to-exit distance, but does not model which robot will perform the last task, future robot reuse, the maximum over all robot return times, or the opportunity cost of taking a nearby task now. Those omissions explain why “penalize every action by return distance” is not equivalent to optimizing terminal makespan.

The pre-registered continuation condition was not met: return tail did not decrease. No larger-scale run was executed. The next useful experiment is a state-aware terminal objective or a short rollout/lookahead that estimates the final robot-position maximum, with its weight frozen before another paired evaluation.
