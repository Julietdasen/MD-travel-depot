# Controlled Hyperparameter Run C2

Status: **completed**.
Device: **cuda:0** (CUDA_VISIBLE_DEVICES=7).
Model seed: **3102**.

| Parameter | Value |
|---|---:|
| pretrain_learning_rate | 0.02 |
| fine_tune_learning_rate | 0.004 |
| weight_decay | 0.0 |
| pairs_per_family_per_batch | 100 |
| pretrain_epochs | 200 |
| fine_tune_epochs | 100 |
| scheduler | none |

Best checkpoint: **checkpoints/best_checkpoint.pt**.
Training valid: **true**.
Training/development template overlap: **0**.

## Development Metrics

Pickup exact pair accuracy: **0.655000**.
Overall exact pair accuracy: **0.647500**.
Residual saturation: **0.156094**.

Held-out data was not read; Ticket 44 was not executed.
