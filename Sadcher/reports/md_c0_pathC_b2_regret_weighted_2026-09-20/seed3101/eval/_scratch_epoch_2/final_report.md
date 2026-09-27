# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 226.5 | 244.2 | +7.8% | 245.8 (greedy_unlock) | +8.5% | -0.6% |
| 24 | 4/4 optimal, 0 timeout | 262.2 | 285.0 | +8.7% | 281.5 (greedy_unlock) | +7.3% | +1.2% |
| 42 | 4/4 optimal, 0 timeout | 342.8 | 373.5 | +9.0% | 392.2 (greedy_unlock) | +14.4% | -4.8% |
| 60 | 4/4 optimal, 0 timeout | 444.5 | 496.2 | +11.6% | 492.5 (greedy_unlock) | +10.8% | +0.8% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 164.2 | 175.8 | +7.0% | 207.0 (greedy_distance) | +26.0% | -15.1% |
| 24 | 4/4 optimal, 0 timeout | 250.0 | 270.2 | +8.1% | 304.0 (greedy_unlock) | +21.6% | -11.1% |
| 42 | 4/4 optimal, 0 timeout | 304.0 | 339.5 | +11.7% | 346.5 (greedy_unlock) | +14.0% | -2.0% |
| 60 | 4/4 optimal, 0 timeout | 356.0 | 400.8 | +12.6% | 395.8 (greedy_unlock) | +11.2% | +1.3% |

