# Scaled dependency_deep + process_scarce evaluation (MILP-IL v2 C0 vs greedy vs MILP, 4 seeds/tier)

**Selection rationale**: dependency_deep = smallest 6-profile gap to MILP (+6.4%); process_scarce = largest advantage over greedy (-13.7pp). Task_count scaled 12 → 60, other structural params preserved. MILP time limit 300s, 4 threads.

## profile: `dependency_deep`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 384.2 | 428.5 | +11.5% | 489.5 (greedy_distance) | +27.4% | -12.5% |
| 24 | 4/4 optimal, 0 timeout | 467.2 | 802.5 | +71.7% | 780.8 (greedy_unlock) | +67.1% | +2.8% |
| 42 | 4/4 optimal, 0 timeout | 435.5 | 1119.0 | +156.9% | 1303.2 (greedy_eta) | +199.3% | -14.1% |
| 60 | 4/4 optimal, 0 timeout | 475.0 | 1692.2 | +256.3% | 1714.8 (greedy_unlock) | +261.0% | -1.3% |

## profile: `process_scarce`

| task_count | MILP status | MILP incumbent | new C0 | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 vs best greedy** |
|---:|---|---:|---:|---:|---|---:|---:|
| 12 | 4/4 optimal, 0 timeout | 467.0 | 619.2 | +32.6% | 577.5 (greedy_distance) | +23.7% | +7.2% |
| 24 | 4/4 optimal, 0 timeout | 570.8 | 837.2 | +46.7% | 799.2 (greedy_unlock) | +40.0% | +4.8% |
| 42 | 2/4 optimal, 2 timeout | 588.8 | 1070.8 | +81.9% | 1084.5 (greedy_distance) | +84.2% | -1.3% |
| 60 | 0/4 optimal, 4 timeout | 488.8 | 1197.8 | +145.1% | 1221.8 (greedy_distance) | +150.0% | -2.0% |

