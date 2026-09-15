# MILP-supervised C0 vs greedy vs MILP on the scale ladder

MILP status: 12/24 all optimal, 42 partial timeout, 60+ mostly timeout. Timed-out MILP rows contribute the best incumbent it found; the true optimum can only be lower, so new-C0 / greedy gaps against MILP are **lower bounds** at large scale.

| task_count | MILP incumbent | new C0 (MILP-IL) | new C0 gap vs MILP | best greedy (name) | greedy gap vs MILP | **new C0 beats greedy by** |
|---:|---:|---:|---:|---|---:|---:|
| 12 | 285.3 | 331.0 | +16.0% | 306.7 (greedy_unlock) | +7.5% | +7.9% |
| 24 | 230.0 | 281.7 | +22.5% | 271.7 (greedy_unlock) | +18.1% | +3.7% |
| 42 | 232.3 | 293.0 | +26.1% | 288.7 (greedy_unlock) | +24.2% | +1.5% |
| 60 | 272.0 | 355.0 | +30.5% | 320.7 (greedy_unlock) | +17.9% | +10.7% |
| 90 | 258.0 | 370.0 | +43.4% | 355.0 (greedy_distance) | +37.6% | +4.2% |
| 114 | 259.3 | 367.3 | +41.6% | 310.7 (greedy_unlock) | +19.8% | +18.2% |
| 150 | 302.7 | 361.0 | +19.3% | 336.0 (greedy_unlock) | +11.0% | +7.4% |
