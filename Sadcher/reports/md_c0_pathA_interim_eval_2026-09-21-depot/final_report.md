# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 24 | 4/4 optimal, 0 timeout | 570.8 | 808.5 | +41.7% | 799.2 (greedy_unlock) | +40.0% | +1.2% |
| 42 | 0/4 optimal, 4 timeout | 593.2 | 1040.2 | +75.3% | 1084.5 (greedy_distance) | +82.8% | -4.1% |

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 24 | 4/4 optimal, 0 timeout | 467.2 | 764.5 | +63.6% | 780.8 (greedy_unlock) | +67.1% | -2.1% |
| 42 | 4/4 optimal, 0 timeout | 435.5 | 1004.0 | +130.5% | 1303.2 (greedy_eta) | +199.3% | -23.0% |

