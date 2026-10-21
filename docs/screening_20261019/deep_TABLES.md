# Deep tests (deep)

Root `results/SCREEN_SP500_20261019_v3_deep`; folds [2020, 2022, 2023]; parent `static`. Bootstrap: stationary, block 10, B=5000, per fold; pooled = equal-weight fold mean. Wilcoxon: two-sided signed-rank on all (fold, seed) deltas; Holm over candidates. Exploratory.

| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| two_residual | +0.0065 [-0.0005, +0.0137] | 2020: +0.0132 / 2022: +0.0107 / 2023: -0.0045 | 2020: 8/10 / 2022: 8/10 / 2023: 6/10 | 30 | 0.038 | 0.077 | NO EVIDENCE |
| pretrain | +0.0042 [-0.0061, +0.0149] | 2020: +0.0029 / 2022: -0.0004 / 2023: +0.0102 | 2020: 6/10 / 2022: 5/10 / 2023: 8/10 | 30 | 0.184 | 0.184 | NO EVIDENCE |

## Absolute RankIC (seed mean per fold)

| Arm | 2020 | 2022 | 2023 | seeds |
|---|---|---|---|---|
| static | +0.0047 | -0.0031 | +0.0173 | 10 |
| two_residual | +0.0179 | +0.0076 | +0.0128 | 10 |
| pretrain | +0.0076 | -0.0035 | +0.0274 | 10 |

## Reproducibility vs `results/SCREEN_SP500_20261019_v3`

18 matching runs; identical predictions: 0; RankIC diff mean -0.0017, mean |diff| 0.0021, max |diff| 0.0091; prediction correlation median 0.997 (min 0.813). Per run: `deep_REPRO.csv`.
