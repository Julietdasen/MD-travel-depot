# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 8 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 24 | 8/8 optimal, 0 timeout | 615.8 | 810.0 | +31.5% | 648.5 (greedy_distance) | +5.3% | +24.9% |
| 42 | 2/8 optimal, 6 timeout | 596.0 | 799.4 | +34.1% | 707.0 (greedy_distance) | +18.6% | +13.1% |
| 60 | 0/8 optimal, 8 timeout | 632.4 | 893.8 | +41.3% | 817.8 (greedy_distance) | +29.3% | +9.3% |

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 24 | 8/8 optimal, 0 timeout | 424.6 | 524.9 | +23.6% | 470.7 (greedy_eta) | +10.8% | +11.5% |
| 42 | 8/8 optimal, 0 timeout | 494.8 | 697.2 | +40.9% | 648.3 (greedy_distance) | +31.0% | +7.5% |
| 60 | 8/8 optimal, 0 timeout | 478.6 | 570.9 | +19.3% | 591.6 (greedy_unlock) | +23.6% | -3.5% |

