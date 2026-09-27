# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 226.5 | 245.0 | +8.2% | 245.8 (greedy_unlock) | +8.5% | -0.3% |
| 24 | 4/4 optimal, 0 timeout | 262.2 | 283.0 | +7.9% | 281.5 (greedy_unlock) | +7.3% | +0.5% |
| 42 | 4/4 optimal, 0 timeout | 342.8 | 380.5 | +11.0% | 392.2 (greedy_unlock) | +14.4% | -3.0% |
| 60 | 4/4 optimal, 0 timeout | 444.5 | 482.0 | +8.4% | 492.5 (greedy_unlock) | +10.8% | -2.1% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 164.2 | 175.8 | +7.0% | 207.0 (greedy_distance) | +26.0% | -15.1% |
| 24 | 4/4 optimal, 0 timeout | 250.0 | 275.0 | +10.0% | 304.0 (greedy_unlock) | +21.6% | -9.5% |
| 42 | 4/4 optimal, 0 timeout | 304.0 | 336.0 | +10.5% | 346.5 (greedy_unlock) | +14.0% | -3.0% |
| 60 | 4/4 optimal, 0 timeout | 356.0 | 404.2 | +13.6% | 395.8 (greedy_unlock) | +11.2% | +2.1% |

