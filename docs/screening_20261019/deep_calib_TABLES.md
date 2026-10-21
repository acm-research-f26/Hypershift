# Deep tests (deep_calib)

Root `results/SCREEN_SP500_20261019_v3_deep`; folds [2020, 2022, 2023]; parent `static`. Bootstrap: stationary, block 10, B=5000, per fold; pooled = equal-weight fold mean. Wilcoxon: two-sided signed-rank on all (fold, seed) deltas; Holm over candidates. Exploratory.

| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| random_graph | +0.0028 [-0.0036, +0.0095] | 2020: +0.0027 / 2022: +0.0067 / 2023: -0.0010 | 2020: 7/10 / 2022: 8/10 / 2023: 5/10 | 30 | 0.124 | 0.124 | NO EVIDENCE |

## Absolute RankIC (seed mean per fold)

| Arm | 2020 | 2022 | 2023 | seeds |
|---|---|---|---|---|
| static | +0.0047 | -0.0031 | +0.0173 | 10 |
| random_graph | +0.0074 | +0.0036 | +0.0163 | 10 |
