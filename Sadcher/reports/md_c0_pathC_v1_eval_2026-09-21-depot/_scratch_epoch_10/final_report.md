# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 384.2 | 483.5 | +25.8% | 489.5 (greedy_distance) | +27.4% | -1.2% |
| 24 | 4/4 optimal, 0 timeout | 467.2 | 777.5 | +66.4% | 780.8 (greedy_unlock) | +67.1% | -0.4% |
| 42 | 4/4 optimal, 0 timeout | 435.5 | 1064.2 | +144.4% | 1303.2 (greedy_eta) | +199.3% | -18.3% |
| 60 | 4/4 optimal, 0 timeout | 475.0 | 1655.2 | +248.5% | 1714.8 (greedy_unlock) | +261.0% | -3.5% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 467.0 | 622.0 | +33.2% | 577.5 (greedy_distance) | +23.7% | +7.7% |
| 24 | 4/4 optimal, 0 timeout | 570.8 | 824.8 | +44.5% | 799.2 (greedy_unlock) | +40.0% | +3.2% |
| 42 | 2/4 optimal, 2 timeout | 586.8 | 1048.8 | +78.7% | 1084.5 (greedy_distance) | +84.8% | -3.3% |
| 60 | 0/4 optimal, 4 timeout | 483.5 | 1178.8 | +143.8% | 1221.8 (greedy_distance) | +152.7% | -3.5% |

