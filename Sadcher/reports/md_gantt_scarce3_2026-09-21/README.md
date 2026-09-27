# Gantt visualisation: scarce=3 envelope (Path C v1 stability, epoch 5)

Per-robot schedule dumps for C0 MILP-IL v2 rolled out on the same
`process_scarce` problem seed (problem_seed=301) across three checkpoint
seeds (3101/3102/3103) and three scales (tc=24/pr=4, tc=42/pr=7,
tc=60/pr=10), with `scarce_skill_count=3` (every skill owned by one
process robot).

## Makespans

| ckpt seed | tc=24, pr=4 | tc=42, pr=7 | tc=60, pr=10 |
|---:|---:|---:|---:|
| 3101 | 307 | 357 | 597 |
| 3102 | 307 | 357 | 597 |
| 3103 | 307 | 357 | 597 |

All three checkpoint seeds produce **identical schedules** on this
problem seed — the policy is fully deterministic here, so the gantt
figures under `seed3102/` and `seed3103/` visually match `seed3101/`.

## What the gantt shows

- Rows: process robots first (`Px skills=[i,…]`), transport robots
  after (`Tx`).
- Red bar: process task requiring at least one scarce skill.
- Blue bar: process task requiring only common skills.
- Transport rows show `empty travel → loading → loaded travel →
  unloading`, coloured light-grey / peach / tan / orange.

## What jumps out

Look at `seed3101/tc60_pr10/gantt.png`:

- P6 (the only holder of skill index 2) is packed **wall-to-wall red
  from t=0 to t≈580** — a single serial chain of scarce-skill process
  tasks. That's the bottleneck; makespan ≈ its occupancy.
- Other process robots idle for long stretches once their scarce
  skill (0 or 1) tasks drain — nothing left for them to do because
  the remaining precedence chain funnels through skill 2.
- Transport robots finish their entire workload by t≈100, then sit
  idle — process, not transport, is the bottleneck at tc=60/scarce=3.

At `tc=42/pr=7` (seed3101):

- Load spreads across P2/P6/P0 more evenly; scarce skill queues
  are shorter (~5-6 tasks deep vs ~30 at tc=60).
- Transport also drains early (t≈100) but the tail is much smaller.

At `tc=24/pr=4` (seed3101):

- Only 4 process robots. P0 + P3 both carry scarce-skill queues
  around 5 deep, with P2 filling in a mix. Balanced.

## Interpretation vs headline table

| tc/pr | scarce=1 gap | scarce=3 gap | comment |
|---:|---:|---:|---|
| 24/4 | -11.6pp | -6.7pp | balanced load, C0 wins on ordering |
| 42/7 | -4.9pp | -5.9pp | still healthy, gap holds |
| 60/10 | -2.1pp | -6.9pp | scarce=3 restores signal because a full serial chain along one bottleneck robot is exactly the regime where MILP knows to pipeline and greedy doesn't |

The gantt confirms *why* scarce=3 keeps the C0 advantage at tc=60:
the schedule is dominated by one bottleneck robot's queue. Ordering
that queue is the whole game, and that ordering is what the policy
learns. In the default problem (scarce=0), the same tc=60 has no
such forced serialisation, so greedy catches up.

## Files

For each `seed{ckpt}/tc{tc}_pr{pr}/` folder:
- `gantt.png` — figure.
- `schedule.json` — raw `process_execution_records` +
  `transport_execution_records` from the ExperimentResult, plus
  makespan / wall time / seed metadata for replotting.

Top-level `index.json` lists every (seed, tc) entry.

## Repro

```
python -m experiments.md_gantt_scarce3 \
  --output-root reports/md_gantt_scarce3_2026-09-21
```

Optional flags: `--problem-seed`, `--seeds`, `--scales`, `--epoch`.
