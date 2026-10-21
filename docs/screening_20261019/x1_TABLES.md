# Deep tests (x1)

Root `results/SCREEN_SP500_20261021_x`; folds [2020, 2022, 2023]; parent `static`. Bootstrap: stationary, block 10, B=5000, per fold; pooled = equal-weight fold mean. Wilcoxon: two-sided signed-rank on all (fold, seed) deltas; Holm over candidates. Exploratory.

| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| clique | +0.0020 [-0.0010, +0.0054] | 2020: -0.0045 / 2022: +0.0089 / 2023: +0.0017 | 2020: 1/5 / 2022: 3/5 / 2023: 4/5 | 15 | 0.890 | 0.890 | NO EVIDENCE |
| star | +0.0080 [-0.0014, +0.0176] | 2020: +0.0133 / 2022: +0.0147 / 2023: -0.0041 | 2020: 5/5 / 2022: 3/5 / 2023: 2/5 | 15 | 0.073 | 0.146 | NO EVIDENCE |

## Absolute RankIC (seed mean per fold)

| Arm | 2020 | 2022 | 2023 | seeds |
|---|---|---|---|---|
| static | +0.0007 | -0.0061 | +0.0182 | 5 |
| clique | -0.0039 | +0.0028 | +0.0199 | 5 |
| star | +0.0140 | +0.0086 | +0.0141 | 5 |

## Reproducibility vs `results/SCREEN_SP500_20261019_v3_deep`

30 matching runs; identical predictions: 23; RankIC diff mean -0.0002, mean |diff| 0.0002, max |diff| 0.0060; prediction correlation median 1.000 (min 0.973). Per run: `x1_REPRO.csv`.
