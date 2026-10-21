# Deep tests (x1_star_vs_clique)

Root `results/SCREEN_SP500_20261021_x`; folds [2020, 2022, 2023]; parent `clique`. Bootstrap: stationary, block 10, B=5000, per fold; pooled = equal-weight fold mean. Wilcoxon: two-sided signed-rank on all (fold, seed) deltas; Holm over candidates. Exploratory.

| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| star | +0.0060 [-0.0030, +0.0153] | 2020: +0.0179 / 2022: +0.0058 / 2023: -0.0058 | 2020: 5/5 / 2022: 4/5 / 2023: 2/5 | 15 | 0.208 | 0.208 | NO EVIDENCE |

## Absolute RankIC (seed mean per fold)

| Arm | 2020 | 2022 | 2023 | seeds |
|---|---|---|---|---|
| clique | -0.0039 | +0.0028 | +0.0199 | 5 |
| star | +0.0140 | +0.0086 | +0.0141 | 5 |
