# Long-history feature models: all seeds retained — October 30, 2026

The fixed **282-account** arm has **0** passing improvement recipes and **0** with SR>1.5 in all five predefined ensembles. All **60 forest fits and four deterministic Ridge fits** locked before financial evaluation.

![Model comparison](model_comparison.png)

Fixed chart/table representatives: retention five, averaged cohort targets, no band, base fees and3%borrow. These are not a substitute for the full grid.

| Model | Original five SR | Second five SR | Third five SR | Ten SR | Fifteen SR |
|---|---:|---:|---:|---:|---:|
| Shared ExtraTrees | 0.425 | 0.272 | 0.412 | 0.356 | 0.380 |
| 50:50 tree / neural rank blend | 0.561 | 0.532 | 0.821 | 0.557 | 0.656 |

## Seed-free controls

| Control | Allocation | Band | Sharpe | CAGR | Max DD |
|---|---|---:|---:|---:|---:|
| Ridge | retain5 | 0.0% | 0.170 | 4.50% | -17.23% |
| Ridge | retain5 | 1.0% | 0.168 | 4.47% | -17.06% |
| Ridge | issuer5 | 0.0% | -0.015 | 1.37% | -19.32% |
| Ridge | issuer5 | 1.0% | -0.004 | 1.55% | -19.14% |

Deterministic controls are reported once, not copied into artificial seed ensembles. Neural blended groups overlap and share the same historical outcomes.

## Design and interpretation

A shared ExtraTrees regressor uses512trees,depth6,minimumleaf64,75%feature sampling and80%row bootstrap per tree. All15seed values are fixed from the earlier protocol. The130features combine26causal price/volatility/liquidity/beta styles, eligible cross-sectional feature ranks, existing train-only daily/AH/PM summaries and QQQ context. There are no stock-ID or ticker features. Labels are within-date training-return zscores clipped[-3,3], with equal date weights. Training-only imputation/scaling precedes both forest and Ridge(alpha100). No validation or financial result tunes these parameters. Predictions accumulate trees serially for reproducibility.

A second fixed model blends50%forest ranks with25%THINK and25%dynamic ranks. The existing neural checkpoints remain untouched. The report includes all240ensemble accounts,30individual-seed base accounts and12Ridge control cases. Stronger seed averaging cannot create a missing predictive signal; financial results must be evaluated separately from validation RankIC.

All account paths match independent dollar-position accounting. Frozen source/input hashes, chronology, gross budgets, eligibility and terminal liquidation remain inherited and checked.

These are disclosed additional development searches after the earlier failed shared/family experiments. No unseen market evaluation or multiple-search-adjusted significance is claimed. Seed consistency alone cannot establish future Sharpe.

[Every ensemble account](METRICS.csv) · [Every recipe/cost summary](CONSISTENCY.csv) · [Seed-free controls](CONTROLS.csv) · [Frozen protocol](PROTOCOL.json) · [Selection](SELECTION.json)

The [current overview](../ndx_seed_robustness_20261030/REPORT.md) adds the separately frozen reserved-seed confirmation, full search history and final model comparison.
