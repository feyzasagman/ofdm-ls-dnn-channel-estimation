# Table 2 — DNN Configuration

| Parameter | Value |
| --- | --- |
| Input dimension | 128 (64 Re + 64 Im LS features) |
| Hidden layer 1 | Dense(256, ReLU) |
| Hidden layer 2 | Dense(256, ReLU) |
| Output dimension | 128 (residual Re/Im) |
| Learning mode | Residual (predict true - LS) |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Maximum epochs | 60 |
| Early stopping patience | 8 (monitor val_loss) |
| Trainable parameters | 131712 |
| Total parameters | 131712 |
