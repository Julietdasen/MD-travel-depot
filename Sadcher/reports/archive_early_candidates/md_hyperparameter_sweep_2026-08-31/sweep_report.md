# Controlled Sadcher Hyperparameter Sweep

Total runs: **17/17**.
Ticket 44 executed: **false**.
Training/development template overlap: **0**.

## Stage 1 Ranking

| Config | Pickup exact | Overall exact | Train valid | Residual | Selected |
|---|---:|---:|:---:|---:|:---:|
| C2 | 0.585000 | 0.611250 | true | 0.250156 | true |
| C1 | 0.580000 | 0.611250 | true | 0.250521 | true |

Selected candidates: **C2, C1**.

## Stage 2 Paired Comparison

| Candidate | Mean pickup gain | Mean overall exact decline | Accepted |
|---|---:|---:|:---:|
| C2 | -0.023333 | 0.018750 | false |

### Seed Pair Details

- C2 seed 3101: pickup gain -0.005000; overall decline +0.000000; residual 0.250156; training valid true.
- C2 seed 3102: pickup gain +0.015000; overall decline -0.028750; residual 0.156094; training valid true.
- C2 seed 3103: pickup gain -0.080000; overall decline +0.085000; residual 0.000000; training valid false.
| C1 | -0.023333 | -0.000833 | false |

### Seed Pair Details

- C1 seed 3101: pickup gain -0.010000; overall decline +0.000000; residual 0.250521; training valid true.
- C1 seed 3102: pickup gain +0.005000; overall decline -0.030000; residual 0.240313; training valid true.
- C1 seed 3103: pickup gain -0.065000; overall decline +0.027500; residual 0.259948; training valid true.

## Failed Tasks

- `stage1/C3` (config C3, seed 3101, GPU 3): training validity failed.
- `stage1/C4` (config C4, seed 3101, GPU 4): training validity failed.
- `stage1/C7` (config C7, seed 3101, GPU 7): training validity failed.
- `stage2/C2_seed3103` (config C2, seed 3103, GPU 0): training validity failed.

## GPU Assignment

| Stage/run | CUDA_VISIBLE_DEVICES | Visible count |
|---|---:|---:|
| stage1/C0 | 0 | 1 |
| stage1/C1 | 1 | 1 |
| stage1/C2 | 2 | 1 |
| stage1/C3 | 3 | 1 |
| stage1/C4 | 4 | 1 |
| stage1/C5 | 5 | 1 |
| stage1/C6 | 6 | 1 |
| stage1/C7 | 7 | 1 |
| stage2/C0_seed3101 | 0 | 1 |
| stage2/C0_seed3102 | 1 | 1 |
| stage2/C0_seed3103 | 2 | 1 |
| stage2/C1_seed3101 | 3 | 1 |
| stage2/C1_seed3102 | 4 | 1 |
| stage2/C1_seed3103 | 5 | 1 |
| stage2/C2_seed3101 | 6 | 1 |
| stage2/C2_seed3102 | 7 | 1 |
| stage2/C2_seed3103 | 0 | 1 |

All runs wrote complete config, best checkpoint, epoch loss log, metrics, runtime/GPU metadata, JSON and Markdown reports. No held-out artifact was read and no final model is claimed.
