# Controlled Hyperparameter Run C7

Status: **completed**.
Device: **cuda:0** (CUDA_VISIBLE_DEVICES=7).
Model seed: **3101**.

| Parameter | Value |
|---|---:|
| pretrain_learning_rate | 0.01 |
| fine_tune_learning_rate | 0.002 |
| weight_decay | 0.0001 |
| pairs_per_family_per_batch | 100 |
| pretrain_epochs | 300 |
| fine_tune_epochs | 150 |
| scheduler | cosine |

Best checkpoint: **checkpoints/best_checkpoint.pt**.
Training valid: **false**.
Training/development template overlap: **0**.

## Development Metrics

Pickup exact pair accuracy: **0.665000**.
Overall exact pair accuracy: **0.676250**.
Residual saturation: **0.035781**.

Held-out data was not read; Ticket 44 was not executed.
