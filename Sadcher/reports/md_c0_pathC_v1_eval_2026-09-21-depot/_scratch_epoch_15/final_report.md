# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 384.2 | 483.5 | +25.8% | 489.5 (greedy_distance) | +27.4% | -1.2% |
| 24 | 4/4 optimal, 0 timeout | 467.2 | 770.0 | +64.8% | 780.8 (greedy_unlock) | +67.1% | -1.4% |
| 42 | 4/4 optimal, 0 timeout | 435.5 | 1078.5 | +147.6% | 1303.2 (greedy_eta) | +199.3% | -17.2% |
| 60 | 4/4 optimal, 0 timeout | 475.0 | 1591.2 | +235.0% | 1714.8 (greedy_unlock) | +261.0% | -7.2% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 467.0 | 622.0 | +33.2% | 577.5 (greedy_distance) | +23.7% | +7.7% |
| 24 | 4/4 optimal, 0 timeout | 570.8 | 825.8 | +44.7% | 799.2 (greedy_unlock) | +40.0% | +3.3% |
| 42 | 3/4 optimal, 1 timeout | 586.2 | 1022.0 | +74.3% | 1084.5 (greedy_distance) | +85.0% | -5.8% |
| 60 | 0/4 optimal, 4 timeout | 490.5 | 1168.8 | +138.3% | 1221.8 (greedy_distance) | +149.1% | -4.3% |

