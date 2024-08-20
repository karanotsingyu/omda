# Decision Analysis Results

## Results of ROC

### Object `weight_roc`

| Property   |   Weight_ROC |
|:-----------|-------------:|
| Price      |     0.367937 |
| Brightness |     0.183968 |
| Color      |     0.122646 |
| HDR        |     0.091984 |
| HDMI       |     0.073587 |
| Frequency  |     0.061323 |
| Contrast   |     0.052562 |
| Resolution |     0.045992 |

### Object `score_roc`

| Model                |   Price |   Brightness |   Color |   HDR |   HDMI |   Frequency |   Contrast |   Resolution |   TotalScore_ROC |
|:---------------------|--------:|-------------:|--------:|------:|-------:|------------:|-----------:|-------------:|-----------------:|
| KTC H27P22S          |     0.3 |          0.1 |     0.1 |   0.1 |    0.1 |         0.1 |        0.0 |          0.0 |              0.7 |
| HKC MG27U            |     0.3 |          0.1 |     0.1 |   0.1 |    0.1 |         0.1 |        0.0 |          0.0 |              0.7 |
| HKC VG273U Pro       |     0.3 |          0.1 |     0.1 |   0.1 |    0.1 |         0.1 |        0.0 |          0.0 |              0.7 |
| KTC H27P22Pro        |     0.3 |          0.1 |     0.1 |   0.1 |    0.1 |         0.1 |        0.0 |          0.0 |              0.7 |
| Redmi                |     0.4 |          0.0 |     0.1 |   0.1 |    0.1 |         0.0 |        0.0 |          0.0 |              0.6 |
| LG 27UQ850-W         |     0.3 |          0.1 |     0.1 |   0.1 |    0.0 |         0.0 |        0.1 |          0.0 |              0.6 |
| DELL U2723QE         |     0.3 |          0.1 |     0.1 |   0.1 |    0.0 |         0.0 |        0.1 |          0.0 |              0.6 |
| AOC U27U2DS          |     0.3 |          0.1 |     0.1 |   0.1 |    0.0 |         0.0 |        0.0 |          0.0 |              0.6 |
| AOC U27G3S           |     0.3 |          0.1 |     0.0 |   0.1 |    0.1 |         0.1 |        0.0 |          0.0 |              0.6 |
| HKC P272U Pro        |     0.4 |          0.1 |     0.1 |   0.1 |    0.0 |         0.0 |        0.0 |          0.0 |              0.6 |
| LG 27UP850N          |     0.3 |          0.1 |     0.1 |   0.1 |    0.0 |         0.0 |        0.0 |          0.0 |              0.6 |
| AOC  U27N3R          |     0.4 |          0.1 |     0.0 |   0.1 |    0.0 |         0.0 |        0.0 |          0.0 |              0.5 |
| AOC U27U2DP          |     0.3 |          0.1 |     0.0 |   0.1 |    0.0 |         0.0 |        0.1 |          0.0 |              0.5 |
| LG 27UL650           |     0.3 |          0.0 |     0.0 |   0.1 |    0.0 |         0.0 |        0.0 |          0.0 |              0.5 |
| LG 27UL550           |     0.4 |          0.0 |     0.0 |   0.1 |    0.0 |         0.0 |        0.0 |          0.0 |              0.5 |
| Apple Studio Display |     0.0 |          0.2 |     0.1 |   0.1 |    0.0 |         0.0 |        0.0 |          0.0 |              0.5 |
| HKC P272U            |     0.4 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.0 |              0.4 |
| DELL P2723QE         |     0.3 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.0 |              0.4 |

### Object `rank_roc`

| Model                |   Rank_ROC |
|:---------------------|-----------:|
| KTC H27P22S          |          1 |
| HKC MG27U            |          2 |
| HKC VG273U Pro       |          3 |
| KTC H27P22Pro        |          4 |
| Redmi                |          5 |
| LG 27UQ850-W         |          6 |
| DELL U2723QE         |          7 |
| AOC U27U2DS          |          8 |
| AOC U27G3S           |          9 |
| HKC P272U Pro        |         10 |
| LG 27UP850N          |         11 |
| AOC  U27N3R          |         12 |
| AOC U27U2DP          |         13 |
| LG 27UL650           |         14 |
| LG 27UL550           |         15 |
| Apple Studio Display |         16 |
| HKC P272U            |         17 |
| DELL P2723QE         |         18 |

## Results of EWM

### Object `entropy`

| PropertyName   |   Entropy |
|:---------------|----------:|
| Resolution     |  0.000000 |
| Frequency      |  0.556827 |
| HDMI           |  0.619906 |
| Contrast       |  0.627296 |
| Color          |  0.897215 |
| Brightness     |  0.921931 |
| HDR            |  0.959250 |
| Price          |  0.978364 |

### Object `weight_ewm`

| PropertyName   |   Weight_Entropy |
|:---------------|-----------------:|
| Price          |         0.008870 |
| Brightness     |         0.032006 |
| Color          |         0.042139 |
| HDR            |         0.016706 |
| HDMI           |         0.155827 |
| Frequency      |         0.181687 |
| Contrast       |         0.152797 |
| Resolution     |         0.409969 |

### Object `score_ewm`

| Model                |   Price |   Brightness |   Color |   HDR |   HDMI |   Frequency |   Contrast |   Resolution |   TotalScore_EWM |
|:---------------------|--------:|-------------:|--------:|------:|-------:|------------:|-----------:|-------------:|-----------------:|
| Apple Studio Display |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.4 |              0.5 |
| HKC MG27U            |     0.0 |          0.0 |     0.0 |   0.0 |    0.2 |         0.2 |        0.0 |          0.0 |              0.4 |
| KTC H27P22S          |     0.0 |          0.0 |     0.0 |   0.0 |    0.2 |         0.2 |        0.0 |          0.0 |              0.4 |
| HKC VG273U Pro       |     0.0 |          0.0 |     0.0 |   0.0 |    0.2 |         0.2 |        0.0 |          0.0 |              0.4 |
| KTC H27P22Pro        |     0.0 |          0.0 |     0.0 |   0.0 |    0.2 |         0.2 |        0.0 |          0.0 |              0.4 |
| AOC U27G3S           |     0.0 |          0.0 |     0.0 |   0.0 |    0.2 |         0.2 |        0.0 |          0.0 |              0.4 |
| Redmi                |     0.0 |          0.0 |     0.0 |   0.0 |    0.2 |         0.0 |        0.0 |          0.0 |              0.2 |
| LG 27UQ850-W         |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.2 |          0.0 |              0.2 |
| DELL U2723QE         |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.2 |          0.0 |              0.2 |
| AOC U27U2DP          |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.2 |          0.0 |              0.2 |
| AOC U27U2DS          |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.0 |              0.1 |
| LG 27UP850N          |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.0 |              0.1 |
| AOC  U27N3R          |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.0 |              0.1 |
| HKC P272U Pro        |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.0 |              0.1 |
| LG 27UL650           |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.0 |              0.0 |
| LG 27UL550           |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.0 |              0.0 |
| DELL P2723QE         |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.0 |              0.0 |
| HKC P272U            |     0.0 |          0.0 |     0.0 |   0.0 |    0.0 |         0.0 |        0.0 |          0.0 |              0.0 |

### Object `rank_ewm`

| Model                |   Rank_EWM |
|:---------------------|-----------:|
| Apple Studio Display |          1 |
| HKC MG27U            |          2 |
| KTC H27P22S          |          3 |
| HKC VG273U Pro       |          4 |
| KTC H27P22Pro        |          5 |
| AOC U27G3S           |          6 |
| Redmi                |          7 |
| LG 27UQ850-W         |          8 |
| DELL U2723QE         |          9 |
| AOC U27U2DP          |         10 |
| AOC U27U2DS          |         11 |
| LG 27UP850N          |         12 |
| AOC  U27N3R          |         13 |
| HKC P272U Pro        |         14 |
| LG 27UL650           |         15 |
| LG 27UL550           |         16 |
| DELL P2723QE         |         17 |
| HKC P272U            |         18 |

## Results Comparison

### Object `weights_comparison`

| PropertyName   |   Weight_ROC |   Weight_Entropy |
|:---------------|-------------:|-----------------:|
| Price          |         0.37 |             0.01 |
| Brightness     |         0.18 |             0.03 |
| Color          |         0.12 |             0.04 |
| HDR            |         0.09 |             0.02 |
| HDMI           |         0.07 |             0.16 |
| Frequency      |         0.06 |             0.18 |
| Contrast       |         0.05 |             0.15 |
| Resolution     |         0.05 |             0.41 |

### Object `total_scores_comparison`

| Model                |   TotalScore_ROC |   TotalScore_EWM |
|:---------------------|-----------------:|-----------------:|
| KTC H27P22S          |            74.53 |            41.51 |
| HKC MG27U            |            70.12 |            41.86 |
| HKC VG273U Pro       |            67.87 |            39.88 |
| KTC H27P22Pro        |            66.88 |            39.37 |
| Redmi                |            64.05 |            24.34 |
| LG 27UQ850-W         |            61.71 |            22.93 |
| DELL U2723QE         |            61.35 |            22.92 |
| AOC U27U2DS          |            61.28 |            12.31 |
| AOC U27G3S           |            59.21 |            37.71 |
| HKC P272U Pro        |            59.07 |             6.24 |
| LG 27UP850N          |            58.68 |             9.26 |
| AOC  U27N3R          |            53.63 |             7.18 |
| AOC U27U2DP          |            49.90 |            19.21 |
| LG 27UL650           |            48.79 |             3.57 |
| LG 27UL550           |            45.81 |             2.55 |
| Apple Studio Display |            45.51 |            53.14 |
| HKC P272U            |            38.33 |             1.41 |
| DELL P2723QE         |            37.06 |             1.84 |

### Object `ranks_comparison`

| Model                |   Rank_ROC |   Rank_EWM |
|:---------------------|-----------:|-----------:|
| KTC H27P22S          |          1 |          3 |
| HKC MG27U            |          2 |          2 |
| HKC VG273U Pro       |          3 |          4 |
| KTC H27P22Pro        |          4 |          5 |
| Redmi                |          5 |          7 |
| LG 27UQ850-W         |          6 |          8 |
| DELL U2723QE         |          7 |          9 |
| AOC U27U2DS          |          8 |         11 |
| AOC U27G3S           |          9 |          6 |
| HKC P272U Pro        |         10 |         14 |
| LG 27UP850N          |         11 |         12 |
| AOC  U27N3R          |         12 |         13 |
| AOC U27U2DP          |         13 |         10 |
| LG 27UL650           |         14 |         15 |
| LG 27UL550           |         15 |         16 |
| Apple Studio Display |         16 |          1 |
| HKC P272U            |         17 |         18 |
| DELL P2723QE         |         18 |         17 |