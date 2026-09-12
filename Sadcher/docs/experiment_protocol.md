# MD-SADCHER++ Experiment Protocol

Protocol version: `1.0.0`

This contract applies to every legacy and MD training, validation, benchmark,
baseline, ablation, and scaling run. The executable source of truth is
`experiments.protocol`.

## Terminal Semantics

A run succeeds only when every real task is complete, every robot has returned
to `EXIT`, and no illegal assignment occurred. Its makespan is the simulation
time at that joint terminal condition. `START`, `IDLE`, and `EXIT` sentinels
are not real tasks. A deadlock is always a failed terminal state.

A failed run always has `makespan: null` and exactly one stable reason code:

- `static_infeasible`
- `robot_type_mismatch`
- `no_capable_transport_robot`
- `invalid_graph`
- `deadlock`
- `timeout`

Failed runs are never assigned a timeout bound, worst-case estimate, or other
synthetic makespan.

## Reproducibility

Evaluation uses the paired seed list `0..29`. Every compared method must use
the same instance and seed pairs.

Training data is split by task/instance group, never by decision sample. The
split is the first 64 bits of `SHA256("2025:<task_group_id>")`, interpreted on
`[0, 1)`: `[0, 0.8)` is train, `[0.8, 0.9)` is validation, and `[0.9, 1)` is
test. Adding samples to a task group cannot move it or leak its decisions into
another split.

## Statistics

- Report total runs, success count/rate, and failure counts by reason.
- Compute makespan statistics over successful runs only; never impute failures.
- Report arithmetic mean, sample standard deviation (`n - 1`), and a two-sided
  normal-approximation 95% confidence interval (`mean +/- 1.96 * s / sqrt(n)`).
- For method comparisons, pair by `(instance_id, seed)` and report
  `candidate - baseline` over pairs where both runs succeeded. Report matched
  and jointly successful pair counts alongside the difference statistics.
- Standard deviation and confidence bounds are `null` when fewer than two
  observations exist. All statistics are `null` when no observation exists.

## Result Schema

Each JSON result has the following shape. Material starvation is keyed by real
process task ID; robot utilization is keyed by real robot ID. Process and
transport execution records occupy separate arrays so ticket 09 can add the
domain-defined structured records without changing the versioned envelope.
Times use seconds unless the simulator's declared timestep is the experimental
time unit.

```json
{
  "protocol_version": "1.0.0",
  "run_id": "method-instance-seed",
  "method": "method_name",
  "dataset": {
    "instance_id": "instance-id",
    "seed": 0,
    "split": "test"
  },
  "termination": {
    "success": false,
    "failure_reason": "timeout",
    "all_real_tasks_completed": false,
    "all_robots_at_exit": false,
    "illegal_assignment_count": 0
  },
  "metrics": {
    "makespan": null,
    "material_starvation": {},
    "robot_utilization": {},
    "inference_time_seconds": 0.0,
    "wall_time_seconds": 0.0
  },
  "execution_records": {
    "process": [],
    "transport": []
  },
  "metadata": {}
}
```
