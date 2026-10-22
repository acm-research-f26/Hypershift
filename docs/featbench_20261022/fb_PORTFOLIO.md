# Study FB portfolio construction on frozen predictions (descriptive)

h = 5 predictions, seed 7, folds 2019–2023, no retraining. Daily book = mean of the last 5 daily sleeve target weights (no drift); cost 10 bp × turnover; Sharpe annualised (√252), no risk-free rate. Pooled Sharpe is computed on the concatenated daily net returns of the 5 folds. `lgbm:rich:residual` was added post hoc because it had the highest neutral RankIC. Rules: A equal-weight top 10; B long top 20% / short bottom 20% within each sector; C top 10 weighted by 1/vol_60; D long top decile / short bottom decile, short leg scaled to zero beta; E top-10 entry, keep until rank > 20.

| Arm | Rule | Pooled net Sharpe | Fold net Sharpe 2019–2023 | Ann. net return | Turnover / day | Beta to market | Max DD (worst fold) |
|---|---|---|---|---|---|---|---|
| block:reversal16 | A_top10 | +0.62 | +2.33 / +0.19 / +1.13 / +0.03 / +1.12 | +0.241 | 0.241 | +1.34 | -0.70 |
| block:reversal16 | B_sector_ls | -0.20 | +0.48 / +0.05 / -0.40 / -0.27 / -1.34 | -0.030 | 0.358 | +0.18 | -0.32 |
| block:reversal16 | C_volscaled | +0.58 | +2.45 / +0.22 / +1.13 / +0.05 / +0.89 | +0.217 | 0.258 | +1.29 | -0.69 |
| block:reversal16 | D_beta_ls | -0.19 | +0.39 / -0.04 / -0.85 / -0.32 / -0.39 | -0.051 | 0.454 | +0.16 | -0.44 |
| block:reversal16 | E_buffer | +0.48 | +2.34 / -0.01 / +1.14 / -0.16 / +0.87 | +0.189 | 0.221 | +1.33 | -0.70 |
| ridge:rich | A_top10 | +0.64 | +1.31 / +0.45 / +1.45 / -0.37 / +1.43 | +0.266 | 0.240 | +1.59 | -0.62 |
| ridge:rich | B_sector_ls | -0.00 | +0.00 / -0.17 / -0.26 / -0.23 / +0.69 | -0.001 | 0.374 | +0.42 | -0.21 |
| ridge:rich | C_volscaled | +0.64 | +1.33 / +0.48 / +1.61 / -0.41 / +1.39 | +0.257 | 0.257 | +1.56 | -0.61 |
| ridge:rich | D_beta_ls | -0.88 | -1.22 / -1.58 / -1.39 / -0.61 / +0.25 | -0.236 | 0.696 | +0.09 | -0.41 |
| ridge:rich | E_buffer | +0.62 | +1.23 / +0.45 / +1.52 / -0.57 / +1.59 | +0.259 | 0.207 | +1.58 | -0.64 |
| lgbm:rich | A_top10 | +0.73 | +1.98 / +0.75 / +1.62 / -0.21 / +0.73 | +0.310 | 0.235 | +1.49 | -0.64 |
| lgbm:rich | B_sector_ls | -0.36 | -2.28 / +0.39 / -0.14 / -1.80 / -1.29 | -0.043 | 0.399 | +0.19 | -0.30 |
| lgbm:rich | C_volscaled | +0.65 | +1.76 / +0.72 / +1.63 / -0.13 / +0.39 | +0.265 | 0.254 | +1.43 | -0.64 |
| lgbm:rich | D_beta_ls | -0.38 | -1.53 / +0.10 / -0.88 / -2.07 / -0.42 | -0.089 | 0.602 | -0.03 | -0.41 |
| lgbm:rich | E_buffer | +0.83 | +2.56 / +0.70 / +1.94 / -0.25 / +0.90 | +0.352 | 0.208 | +1.49 | -0.64 |
| lgbm:rich:residual | A_top10 | +0.90 | +1.51 / +1.18 / +0.88 / -0.64 / +2.39 | +0.355 | 0.209 | +1.42 | -0.53 |
| lgbm:rich:residual | B_sector_ls | -0.57 | -1.82 / +0.93 / -1.03 / -0.68 / -1.44 | -0.037 | 0.394 | +0.02 | -0.09 |
| lgbm:rich:residual | C_volscaled | +0.88 | +1.47 / +1.20 / +0.98 / -0.64 / +2.22 | +0.311 | 0.239 | +1.34 | -0.50 |
| lgbm:rich:residual | D_beta_ls | -0.03 | -0.56 / +2.62 / -1.56 / -1.44 / +0.21 | -0.003 | 0.426 | +0.09 | -0.17 |
| lgbm:rich:residual | E_buffer | +0.95 | +1.60 / +1.22 / +0.84 / -0.46 / +2.40 | +0.382 | 0.172 | +1.41 | -0.53 |
| hold-all (equal weight, no cost) | — | +0.86 | +2.55 / +0.66 / +2.14 / -0.26 / +1.31 | +0.193 | — | +0.98 | -0.38 |

## Sanity: rule A vs the screening sleeve backtest (`trading()`, net 10 bp)

| Fold | Arm | trading() net Sharpe | rule A net Sharpe |
|---|---|---|---|
| 2019 | block:reversal16 | +2.31 | +2.33 |
| 2019 | ridge:rich | +1.28 | +1.31 |
| 2019 | lgbm:rich | +1.93 | +1.98 |
| 2019 | lgbm:rich:residual | +1.50 | +1.51 |
| 2020 | block:reversal16 | +0.18 | +0.19 |
| 2020 | ridge:rich | +0.45 | +0.45 |
| 2020 | lgbm:rich | +0.74 | +0.75 |
| 2020 | lgbm:rich:residual | +1.17 | +1.18 |
| 2021 | block:reversal16 | +1.08 | +1.13 |
| 2021 | ridge:rich | +1.41 | +1.45 |
| 2021 | lgbm:rich | +1.57 | +1.62 |
| 2021 | lgbm:rich:residual | +0.85 | +0.88 |
| 2022 | block:reversal16 | -0.00 | +0.03 |
| 2022 | ridge:rich | -0.40 | -0.37 |
| 2022 | lgbm:rich | -0.24 | -0.21 |
| 2022 | lgbm:rich:residual | -0.66 | -0.64 |
| 2023 | block:reversal16 | +1.11 | +1.12 |
| 2023 | ridge:rich | +1.47 | +1.43 |
| 2023 | lgbm:rich | +0.77 | +0.73 |
| 2023 | lgbm:rich:residual | +2.41 | +2.39 |
