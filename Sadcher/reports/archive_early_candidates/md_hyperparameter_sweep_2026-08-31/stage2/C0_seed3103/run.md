# Controlled Hyperparameter Run C0

Status: **completed**.
Device: **cuda:0** (CUDA_VISIBLE_DEVICES=2).
Model seed: **3103**.

| Parameter | Value |
|---|---:|
| pretrain_learning_rate | 0.01 |
| fine_tune_learning_rate | 0.002 |
| weight_decay | 0.0 |
| pairs_per_family_per_batch | 100 |
| pretrain_epochs | 200 |
| fine_tune_epochs | 100 |
| scheduler | none |

Best checkpoint: **checkpoints/best_checkpoint.pt**.
Training valid: **true**.
Training/development template overlap: **0**.

## Development Metrics

Pickup exact pair accuracy: **0.600000**.
Overall exact pair accuracy: **0.605000**.
Residual saturation: **0.287969**.

Held-out data was not read; Ticket 44 was not executed.
