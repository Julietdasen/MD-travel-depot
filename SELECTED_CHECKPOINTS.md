# Selected MRTA checkpoints

The large generated dataset `Sadcher/dataset_optimal_8t3r3s.zip` is intentionally
not uploaded. It is reproducible from the data-generation scripts and remains
ignored by Git. The complete experiment reports and source code are published;
the following small, important checkpoints are explicitly included:

| Purpose | Seeds | Files |
| --- | --- | --- |
| Formal residual IL policy (scale-10 run) | 3101, 3102, 3103 | `Sadcher/reports/md_residual_scale10_2026-09-04/training/seed310{1,2,3}/C0_residual_tail_seed310{1,2,3}/best_checkpoint.pt` |
| Frozen joint correction companion | 3101, 3102, 3103 | `Sadcher/reports/md_frozen_joint_correction_2026-09-06/seed310{1,2,3}/best_checkpoint.pt` |

These files are ordinary PyTorch checkpoints (about 114 KB and 132 KB each),
so they remain below GitHub's per-file limit. Other run directories, Ray state,
intermediate checkpoints, and the dataset archive stay ignored to keep the
repository usable.
