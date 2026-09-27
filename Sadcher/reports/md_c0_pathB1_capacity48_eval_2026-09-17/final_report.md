# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 226.5 | 249.2 | +10.0% | 245.8 (greedy_unlock) | +8.5% | +1.4% |
| 24 | 4/4 optimal, 0 timeout | 262.2 | 314.2 | +19.8% | 281.5 (greedy_unlock) | +7.3% | +11.6% |
| 42 | 4/4 optimal, 0 timeout | 342.8 | 384.8 | +12.3% | 392.2 (greedy_unlock) | +14.4% | -1.9% |
| 60 | 4/4 optimal, 0 timeout | 444.5 | 499.8 | +12.4% | 492.5 (greedy_unlock) | +10.8% | +1.5% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 164.2 | 174.5 | +6.2% | 207.0 (greedy_distance) | +26.0% | -15.7% |
| 24 | 4/4 optimal, 0 timeout | 250.0 | 268.5 | +7.4% | 304.0 (greedy_unlock) | +21.6% | -11.7% |
| 42 | 4/4 optimal, 0 timeout | 304.0 | 339.2 | +11.6% | 346.5 (greedy_unlock) | +14.0% | -2.1% |
| 60 | 4/4 optimal, 0 timeout | 356.0 | 415.0 | +16.6% | 395.8 (greedy_unlock) | +11.2% | +4.9% |

