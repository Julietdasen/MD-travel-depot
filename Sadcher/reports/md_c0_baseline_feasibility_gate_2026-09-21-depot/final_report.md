# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 24 | 4/4 optimal, 0 timeout | 570.8 | 791.2 | +38.6% | 767.0 (greedy_distance) | +34.4% | +3.2% |
| 42 | 1/4 optimal, 3 timeout | 584.8 | 792.0 | +35.4% | n/a (n/a) | n/a | n/a |

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 24 | 4/4 optimal, 0 timeout | 467.2 | 598.2 | +28.0% | 501.7 (greedy_unlock) | +7.4% | +19.3% |
| 42 | 4/4 optimal, 0 timeout | 435.5 | 594.0 | +36.4% | 559.0 (greedy_distance) | +28.4% | +6.3% |

