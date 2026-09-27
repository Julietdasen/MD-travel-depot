# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 384.2 | 444.2 | +15.6% | 489.5 (greedy_distance) | +27.4% | -9.2% |
| 24 | 4/4 optimal, 0 timeout | 467.2 | 736.0 | +57.5% | 780.8 (greedy_unlock) | +67.1% | -5.7% |
| 42 | 4/4 optimal, 0 timeout | 435.5 | 1042.8 | +139.4% | 1303.2 (greedy_eta) | +199.3% | -20.0% |
| 60 | 4/4 optimal, 0 timeout | 475.0 | 1558.0 | +228.0% | 1714.8 (greedy_unlock) | +261.0% | -9.1% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 467.0 | 617.5 | +32.2% | 577.5 (greedy_distance) | +23.7% | +6.9% |
| 24 | 4/4 optimal, 0 timeout | 570.8 | 807.5 | +41.5% | 799.2 (greedy_unlock) | +40.0% | +1.0% |
| 42 | 3/4 optimal, 1 timeout | 585.5 | 1002.2 | +71.2% | 1084.5 (greedy_distance) | +85.2% | -7.6% |
| 60 | 0/4 optimal, 4 timeout | 491.0 | 1128.8 | +129.9% | 1221.8 (greedy_distance) | +148.8% | -7.6% |

