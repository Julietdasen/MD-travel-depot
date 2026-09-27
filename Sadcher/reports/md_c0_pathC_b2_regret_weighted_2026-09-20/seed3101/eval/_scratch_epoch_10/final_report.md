# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 226.5 | 245.2 | +8.3% | 245.8 (greedy_unlock) | +8.5% | -0.2% |
| 24 | 4/4 optimal, 0 timeout | 262.2 | 285.0 | +8.7% | 281.5 (greedy_unlock) | +7.3% | +1.2% |
| 42 | 4/4 optimal, 0 timeout | 342.8 | 377.5 | +10.1% | 392.2 (greedy_unlock) | +14.4% | -3.8% |
| 60 | 4/4 optimal, 0 timeout | 444.5 | 474.8 | +6.8% | 492.5 (greedy_unlock) | +10.8% | -3.6% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 164.2 | 175.5 | +6.8% | 207.0 (greedy_distance) | +26.0% | -15.2% |
| 24 | 4/4 optimal, 0 timeout | 250.0 | 276.8 | +10.7% | 304.0 (greedy_unlock) | +21.6% | -9.0% |
| 42 | 4/4 optimal, 0 timeout | 304.0 | 333.0 | +9.5% | 346.5 (greedy_unlock) | +14.0% | -3.9% |
| 60 | 4/4 optimal, 0 timeout | 356.0 | 396.2 | +11.3% | 395.8 (greedy_unlock) | +11.2% | +0.1% |

