# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 226.5 | 245.5 | +8.4% | 245.8 (greedy_unlock) | +8.5% | -0.1% |
| 24 | 4/4 optimal, 0 timeout | 262.2 | 284.0 | +8.3% | 281.5 (greedy_unlock) | +7.3% | +0.9% |
| 42 | 4/4 optimal, 0 timeout | 342.8 | 383.2 | +11.8% | 392.2 (greedy_unlock) | +14.4% | -2.3% |
| 60 | 4/4 optimal, 0 timeout | 444.5 | 500.5 | +12.6% | 492.5 (greedy_unlock) | +10.8% | +1.6% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 164.2 | 175.8 | +7.0% | 207.0 (greedy_distance) | +26.0% | -15.1% |
| 24 | 4/4 optimal, 0 timeout | 250.0 | 263.5 | +5.4% | 304.0 (greedy_unlock) | +21.6% | -13.3% |
| 42 | 4/4 optimal, 0 timeout | 304.0 | 332.2 | +9.3% | 346.5 (greedy_unlock) | +14.0% | -4.1% |
| 60 | 4/4 optimal, 0 timeout | 356.0 | 398.8 | +12.0% | 395.8 (greedy_unlock) | +11.2% | +0.8% |

