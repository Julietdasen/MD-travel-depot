# Controlled Hyperparameter Run C6

Status: **completed**.
Device: **cuda:0** (CUDA_VISIBLE_DEVICES=6).
Model seed: **3101**.

| Parameter | Value |
|---|---:|
| pretrain_learning_rate | 0.01 |
| fine_tune_learning_rate | 0.002 |
| weight_decay | 0.0 |
| pairs_per_family_per_batch | 250 |
| pretrain_epochs | 200 |
| fine_tune_epochs | 100 |
| scheduler | none |

Best checkpoint: **checkpoints/best_checkpoint.pt**.
Training valid: **true**.
Training/development template overlap: **0**.

## Development Metrics

Pickup exact pair accuracy: **0.610000**.
Overall exact pair accuracy: **0.601250**.
Residual saturation: **0.139948**.

Held-out data was not read; Ticket 44 was not executed.
