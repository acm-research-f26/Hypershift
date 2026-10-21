# Deep tests (x1_vs_none)

Root `results/SCREEN_SP500_20261021_x`; folds [2020, 2022, 2023]; parent `none`. Bootstrap: stationary, block 10, B=5000, per fold; pooled = equal-weight fold mean. Wilcoxon: two-sided signed-rank on all (fold, seed) deltas; Holm over candidates. Exploratory.

| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| clique | -0.0061 [-0.0175, +0.0061] | 2020: -0.0187 / 2022: -0.0043 / 2023: +0.0049 | 2020: 1/5 / 2022: 1/5 / 2023: 3/5 | 15 | 0.188 | 0.375 | NO EVIDENCE |
| star | -0.0001 [-0.0102, +0.0107] | 2020: -0.0009 / 2022: +0.0015 / 2023: -0.0009 | 2020: 2/5 / 2022: 3/5 / 2023: 3/5 | 15 | 0.978 | 0.978 | NO EVIDENCE |
| static | -0.0081 [-0.0204, +0.0050] | 2020: -0.0142 / 2022: -0.0132 / 2023: +0.0032 | 2020: 1/5 / 2022: 1/5 / 2023: 2/5 | 15 | 0.048 | 0.192 | NO EVIDENCE |
| random_graph | -0.0049 [-0.0156, +0.0061] | 2020: -0.0130 / 2022: -0.0074 / 2023: +0.0057 | 2020: 0/5 / 2022: 1/5 / 2023: 4/5 | 15 | 0.121 | 0.362 | NO EVIDENCE |

## Absolute RankIC (seed mean per fold)

| Arm | 2020 | 2022 | 2023 | seeds |
|---|---|---|---|---|
| none | +0.0149 | +0.0072 | +0.0150 | 5 |
| clique | -0.0039 | +0.0028 | +0.0199 | 5 |
| star | +0.0140 | +0.0086 | +0.0141 | 5 |
| static | +0.0007 | -0.0061 | +0.0182 | 5 |
| random_graph | +0.0018 | -0.0002 | +0.0206 | 5 |
