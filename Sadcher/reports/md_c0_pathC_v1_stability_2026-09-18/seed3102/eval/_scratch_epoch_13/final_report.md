# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 226.5 | 245.2 | +8.3% | 245.8 (greedy_unlock) | +8.5% | -0.2% |
| 24 | 4/4 optimal, 0 timeout | 262.2 | 288.2 | +9.9% | 281.5 (greedy_unlock) | +7.3% | +2.4% |
| 42 | 4/4 optimal, 0 timeout | 342.8 | 362.8 | +5.8% | 392.2 (greedy_unlock) | +14.4% | -7.5% |
| 60 | 4/4 optimal, 0 timeout | 444.5 | 483.0 | +8.7% | 492.5 (greedy_unlock) | +10.8% | -1.9% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 164.2 | 175.5 | +6.8% | 207.0 (greedy_distance) | +26.0% | -15.2% |
| 24 | 4/4 optimal, 0 timeout | 250.0 | 275.2 | +10.1% | 304.0 (greedy_unlock) | +21.6% | -9.5% |
| 42 | 4/4 optimal, 0 timeout | 304.0 | 338.0 | +11.2% | 346.5 (greedy_unlock) | +14.0% | -2.5% |
| 60 | 4/4 optimal, 0 timeout | 356.0 | 410.5 | +15.3% | 395.8 (greedy_unlock) | +11.2% | +3.7% |

