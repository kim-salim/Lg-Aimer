# Experiment Notes

## Baseline Seq2Seq

Original setup:

- Input length: 28 days
- Forecast horizon: 7 days
- Menu embedding size: 8
- Hidden size: 64
- Batch size: 64
- Optimizer: Adam
- Learning rate: 1e-3
- Loss: MSE

Observed scores:

| Setting | Score |
| --- | ---: |
| 50 epoch | 0.7814005475 |
| 30 epoch | 0.765854557 |

The 30 to 50 epoch range appeared reasonable for the single-model version.

## Ensemble Seq2Seq

Best recorded setup:

- Epochs: 40
- Ensemble size: 5
- Seeds: 42, 43, 44, 45, 46
- Weighting: simple average, no model-specific weights

Observed score:

| Setting | Score |
| --- | ---: |
| 40 epoch, 5-model ensemble | 0.6432123097 |

## Next Experiment Ideas

1. Weighted ensemble by validation score or recent-period validation performance.
2. Reintroduce outlier handling and compare clipping or robust transforms.
3. Feature control experiments:
   - holiday eve / post-holiday flag
   - venue-level embedding
   - rolling mean and rolling std
   - lag features such as 7-day and 14-day lag sales
   - per-menu normalization

