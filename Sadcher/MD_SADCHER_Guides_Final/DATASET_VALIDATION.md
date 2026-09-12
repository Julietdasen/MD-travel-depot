# Official Dataset Validation

## Dataset

- Archive: `dataset_optimal_8t3r3s.zip`
- SHA-256: `2aa489976b1150d4a6602ddd06c074c3f3abdb0a645002d69040b0aad3183029`
- Extracted at: `datasets/dataset_optimal_8t3r3s/`
- Archive integrity: `unzip -t` passed
- Problem files: `231,634`
- Solution files: `231,634`

## Validation Results

All problem and solution JSON files parsed successfully. Sorted filenames form a one-to-one mapping:

```text
problem_instance_<p>_<id>.json
optimal_schedule_<p>_<id>.json
```

Every instance has the same legacy shape:

- `Q`: `3 x 3` robot-skill capability matrix
- `R`: `10 x 3` task-skill requirement matrix (8 real tasks plus start and exit rows)
- `T_e`: length `10` (real task durations plus start and exit)
- `T_t`: `10 x 10`
- `task_locations`: `10 x 2`
- `n_tasks` in solutions: `8`
- `n_robots` in solutions: `3`

The filename groups exactly match the number of precedence constraints:

| Group | Instances |
|---|---:|
| `1p` | 34,070 |
| `2p` | 31,996 |
| `3p` | 59,838 |
| `4p` | 37,479 |
| `5p` | 34,552 |
| `6p` | 33,699 |

All precedence edges are valid 1-indexed real-task IDs and are acyclic. All solution robot and task references are valid. `T_t` is exactly the Euclidean distance matrix induced by `task_locations` (maximum absolute validation error: `0.0`).

## Legacy Loader Compatibility

The existing `LazyLoadedSchedulingDataset` loads the extracted directory without a schema adapter:

- indexed decision samples: `2,084,545`
- `n_robots=3`, `robot_dim=7`
- `n_tasks=8`, `task_dim=9`
- first sample tensors returned successfully

No training or GPU job was started during validation.

## MD Boundary

This dataset contains only the legacy SADCHER fields (`Q`, `R`, `T_e`, `T_t`, `task_locations`, and `precedence_constraints`). It has no transport tasks, pickup/delivery locations, material predecessors, robot type tags, or transport phase state. Therefore it is accepted as the official legacy benchmark and regression fixture, but it is **not** converted into MD data.

The future MD loader may directly adapt official data after the dataset semantics are finalized. The runtime domain model and MD transport state machine remain stable; the legacy loader and legacy experiments remain unchanged.
