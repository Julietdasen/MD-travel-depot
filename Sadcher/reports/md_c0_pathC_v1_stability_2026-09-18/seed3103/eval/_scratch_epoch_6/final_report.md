# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 226.5 | 244.0 | +7.7% | 245.8 (greedy_unlock) | +8.5% | -0.7% |
| 24 | 4/4 optimal, 0 timeout | 262.2 | 288.5 | +10.0% | 281.5 (greedy_unlock) | +7.3% | +2.5% |
| 42 | 4/4 optimal, 0 timeout | 342.8 | 372.5 | +8.7% | 392.2 (greedy_unlock) | +14.4% | -5.0% |
| 60 | 4/4 optimal, 0 timeout | 444.5 | 477.8 | +7.5% | 492.5 (greedy_unlock) | +10.8% | -3.0% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 164.2 | 175.8 | +7.0% | 207.0 (greedy_distance) | +26.0% | -15.1% |
| 24 | 4/4 optimal, 0 timeout | 250.0 | 275.8 | +10.3% | 304.0 (greedy_unlock) | +21.6% | -9.3% |
| 42 | 4/4 optimal, 0 timeout | 304.0 | 336.5 | +10.7% | 346.5 (greedy_unlock) | +14.0% | -2.9% |
| 60 | 4/4 optimal, 0 timeout | 356.0 | 396.2 | +11.3% | 395.8 (greedy_unlock) | +11.2% | +0.1% |

