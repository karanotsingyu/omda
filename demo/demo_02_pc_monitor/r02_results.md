# Decision Analysis Results

## Results of ROC

### Object `weight_roc`

| Property      |   Weight_ROC |
|:--------------|-------------:|
| Price         |     0.480000 |
| Brand         |     0.240000 |
| Appearance    |     0.160000 |
| Reasonability |     0.120000 |

### Object `score_roc`

| Model        |   Brand |   Price |   Appearance |   Reasonability |   TotalScore_ROC |
|:-------------|--------:|--------:|-------------:|----------------:|-----------------:|
| AOC U27U2DS  |  0.2400 |  0.3504 |       0.1600 |          0.1200 |           0.8704 |
| DELL U2723QE |  0.2400 |  0.0000 |       0.1600 |          0.1200 |           0.5200 |
| KTC H27P22S  |  0.0000 |  0.4800 |       0.0000 |          0.0000 |           0.4800 |

### Object `rank_roc`

| Model        |   Rank_ROC |
|:-------------|-----------:|
| AOC U27U2DS  |          1 |
| DELL U2723QE |          2 |
| KTC H27P22S  |          3 |

## Results of EWM

### Object `entropy`

| PropertyName   |   Entropy |
|:---------------|----------:|
| Price          |  0.619792 |
| Brand          |  0.630930 |
| Appearance     |  0.630930 |
| Reasonability  |  0.630930 |

### Object `weight_ewm`

| PropertyName   |   Weight_Entropy |
|:---------------|-----------------:|
| Brand          |         0.248128 |
| Price          |         0.255616 |
| Appearance     |         0.248128 |
| Reasonability  |         0.248128 |

### Object `score_ewm`

| Model        |   Brand |   Price |   Appearance |   Reasonability |   TotalScore_EWM |
|:-------------|--------:|--------:|-------------:|----------------:|-----------------:|
| AOC U27U2DS  |  0.2481 |  0.1866 |       0.2481 |          0.2481 |           0.9310 |
| DELL U2723QE |  0.2481 |  0.0000 |       0.2481 |          0.2481 |           0.7444 |
| KTC H27P22S  |  0.0000 |  0.2556 |       0.0000 |          0.0000 |           0.2556 |

### Object `rank_ewm`

| Model        |   Rank_EWM |
|:-------------|-----------:|
| AOC U27U2DS  |          1 |
| DELL U2723QE |          2 |
| KTC H27P22S  |          3 |

## Results Comparison

### Object `weights_comparison`

| PropertyName   |   Weight_ROC |   Weight_Entropy |
|:---------------|-------------:|-----------------:|
| Price          |         0.48 |             0.26 |
| Brand          |         0.24 |             0.25 |
| Appearance     |         0.16 |             0.25 |
| Reasonability  |         0.12 |             0.25 |

### Object `total_scores_comparison`

| Model        |   TotalScore_ROC |   TotalScore_EWM |
|:-------------|-----------------:|-----------------:|
| AOC U27U2DS  |            87.04 |            93.10 |
| DELL U2723QE |            52.00 |            74.44 |
| KTC H27P22S  |            48.00 |            25.56 |

### Object `ranks_comparison`

| Model        |   Rank_ROC |   Rank_EWM |
|:-------------|-----------:|-----------:|
| AOC U27U2DS  |          1 |          1 |
| DELL U2723QE |          2 |          2 |
| KTC H27P22S  |          3 |          3 |