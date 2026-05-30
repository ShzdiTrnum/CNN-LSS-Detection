# Training Configurations

## Dataset size configurations (per paper Section 3.2)

| Size  | Batch | Epochs | Multi-ROI Acc | Single-ROI Acc |
|-------|-------|--------|--------------|----------------|
| 5k    | 64    | 200    | 86.30%       | 93.15%         |
| 10k   | 128   | 34     | 95.83%       | 97.71%         |
| 12.5k | 256   | 34     | 97.01%       | 97.71%         |

## Common hyperparameters

  Optimizer    : Adam  (lr = 0.001)
  Loss         : Categorical cross-entropy
  Dropout rate : 0.14 (after every Conv block and FC layer)
  Train/test   : 80:20 split
  Validation   : 5-fold cross-validation
  Input size   : 240 x 240 x 1
