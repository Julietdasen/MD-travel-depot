# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 384.2 | 481.8 | +25.4% | 489.5 (greedy_distance) | +27.4% | -1.6% |
| 24 | 4/4 optimal, 0 timeout | 467.2 | 761.0 | +62.9% | 780.8 (greedy_unlock) | +67.1% | -2.5% |
| 42 | 4/4 optimal, 0 timeout | 435.5 | 1089.0 | +150.1% | 1303.2 (greedy_eta) | +199.3% | -16.4% |
| 60 | 4/4 optimal, 0 timeout | 475.0 | 1726.5 | +263.5% | 1714.8 (greedy_unlock) | +261.0% | +0.7% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 467.0 | 632.5 | +35.4% | 577.5 (greedy_distance) | +23.7% | +9.5% |
| 24 | 4/4 optimal, 0 timeout | 570.8 | 844.0 | +47.9% | 799.2 (greedy_unlock) | +40.0% | +5.6% |
| 42 | 3/4 optimal, 1 timeout | 586.2 | 1033.2 | +76.2% | 1084.5 (greedy_distance) | +85.0% | -4.7% |
| 60 | 0/4 optimal, 4 timeout | 490.0 | 1210.5 | +147.0% | 1221.8 (greedy_distance) | +149.3% | -0.9% |

