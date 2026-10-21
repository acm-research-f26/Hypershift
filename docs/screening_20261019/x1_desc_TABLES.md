# Deep tests (x1_desc)

Root `results/SCREEN_SP500_20261021_x`; folds [2020, 2022, 2023]; parent `static`. Bootstrap: stationary, block 10, B=5000, per fold; pooled = equal-weight fold mean. Wilcoxon: two-sided signed-rank on all (fold, seed) deltas; Holm over candidates. Exploratory.

| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| none | +0.0081 [-0.0050, +0.0204] | 2020: +0.0142 / 2022: +0.0132 / 2023: -0.0032 | 2020: 4/5 / 2022: 4/5 / 2023: 3/5 | 15 | 0.048 | 0.096 | NO EVIDENCE |
| random_graph | +0.0032 [-0.0045, +0.0112] | 2020: +0.0012 / 2022: +0.0058 / 2023: +0.0025 | 2020: 3/5 / 2022: 3/5 / 2023: 4/5 | 15 | 0.389 | 0.389 | NO EVIDENCE |

## Absolute RankIC (seed mean per fold)

| Arm | 2020 | 2022 | 2023 | seeds |
|---|---|---|---|---|
| static | +0.0007 | -0.0061 | +0.0182 | 5 |
| none | +0.0149 | +0.0072 | +0.0150 | 5 |
| random_graph | +0.0018 | -0.0002 | +0.0206 | 5 |
