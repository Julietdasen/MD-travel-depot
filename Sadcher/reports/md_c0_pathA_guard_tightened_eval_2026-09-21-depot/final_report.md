# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 24 | 4/4 optimal, 0 timeout | 570.8 | 770.2 | +35.0% | 767.0 (greedy_distance) | +34.4% | +0.4% |
| 42 | 0/4 optimal, 4 timeout | 585.5 | 807.8 | +38.0% | n/a (n/a) | n/a | n/a |

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 24 | 4/4 optimal, 0 timeout | 467.2 | 547.2 | +17.1% | 501.7 (greedy_unlock) | +7.4% | +9.1% |
| 42 | 4/4 optimal, 0 timeout | 435.5 | 597.5 | +37.2% | 559.0 (greedy_distance) | +28.4% | +6.9% |

