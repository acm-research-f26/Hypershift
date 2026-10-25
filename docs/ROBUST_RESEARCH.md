# Robust losses, portfolio construction and daily forecasting

> Historical scope: earlier robust-loss cohort protocol. Current NASDAQ-100, AH/PM and optimizer/network evidence is in the [master charts](../charts.md). Commands below run from the repository root through the current launcher.

<!-- benchmark-audit-20261010:start -->
NYSX is the passive Global X NYSE 100 ETF. It launched March 25, 2026 and has no tradable 2022–2025 history. Its index covers technology and growth companies across exchanges, not the 100 largest NYSE-listed companies. NYSX performance is unavailable here. NYSE historical charts use the simulated equal-weight cohort basket (scheduled rebalances and modeled costs), not an ETF. NASDAQ charts retain QQQ; SPY supplies broad-market context. [Issuer and inception](https://www.globalxetfs.com/funds/nysx/) and [prospectus](https://assets.globalxetfs.com/funds/documents/nysx/prospectus-regulatory/Summary-Prospectus_NYSX.pdf).

The refreshed CHARTS.md pages are the current visual record. Earlier ETF-overlay descriptions are superseded by this correction. Audited return paths and training selections remain preserved.
<!-- benchmark-audit-20261010:end -->

The separate October 8 [Huber plus ranking follow-up](HUBER_RANKING.md)
is complete under [its own frozen plan](../config/huber_ranking_experiment_plan.json).
Its three new five-session variants and uncertainty tables are in
`runs/huber_ranking_2022_2025` and `docs/`; the original protocol below is retained.

This extension follows [the frozen experiment plan](../config/robust_experiment_plan.json). It uses the existing public-data snapshot for 2022-2025 and preserves every earlier result directory. The paper's actual claims and differences from our pipeline are explained in [the StockMixer review](STOCKMIXER_PAPER_REVIEW.md).

**Status: complete.** Read the [report](REPORT.md),
[findings](FINDINGS.md), [charts](CHARTS.md)
and [SPY/QQQ comparisons](ETF_CONTROLS.md).
Authoritative runs are `runs/robust_2022_2025_final` and `runs/daily_2022_2025_final`.
The original interrupted folders remain verified fit caches, not additional trials.
All 576 configurations finished, and the 288 primary paths passed accounting audits.
QQQ was added as a user-requested control after training began and never affected selection.

## Experiments

| Group | Five-session run | Daily run |
|---|---|---|
| Existing neural models | All nine prior architectures | THINK, THINK+mixer, StockMixer-inspired |
| Huber loss | THINK, THINK+mixer, StockMixer-inspired | Same three |
| Mixer residual multiplied by 0.1 | MSE and Huber | Not tested |
| Multi-scale mixer (patch scales 1/2/4) | MSE and Huber | MSE and Huber |
| Other strategies | Ridge, 126-session momentum, 5-session reversal | Same |
| Benchmarks | Equal-weight buy-and-hold, rebalanced universe, SPY | Same |

Each neural configuration uses seeds 7, 19 and 42. Ensemble scores are equal-weight averages of ranks among eligible securities. All trials use a 16-session feature sequence, AdamW at 0.001, up to 30 epochs and six-epoch patience. Daily batches contain 40 chronological observations; five-session batches contain eight. All available training origins are used. The daily comparison changes the forecast horizon, rebalance interval and batch size; it does not isolate rebalance frequency alone.

Huber targets and predictions are in percentage points. The objective is twice Huber with a fixed one-percentage-point threshold, preserving MSE curvature near zero and reducing sensitivity to extreme errors. For StockMixer/multi-scale models, the original pairwise penalty remains. Validation-MSE early stopping is unchanged, isolating the training-loss change. A Huber early-stopping criterion is not part of this batch.

## Dates and decisions

| Test year | Actual training-signal years | Validation year |
|---|---|---|
| 2022 | 2019-2020 | 2021 |
| 2023 | 2019-2021 | 2022 |
| 2024 | 2019-2022 | 2023 |
| 2025 | 2020-2023 | 2024 |

The snapshot starts in 2018 and feature origins begin after 260 sessions; actual earliest training signals are January 2019. Exact train/validation/test dates and label-maturity boundaries are saved per fold. A label must mature before the next split starts. Features, scalers, correlations and stock eligibility never use subsequent observations. Optimization batches remain chronological.

The [completed chronology audit](DATA_AND_SPLITS.md)
reconstructs all 16 folds and verifies saved scalers, mature labels, validation-only
selection and invariance of earlier inputs to changed future prices.
These checks do not remove vendor revisions, survivor-cohort bias or prior test reuse.
The [uncertainty tables](CONFIDENCE_INTERVALS.md) distinguish
95% confidence intervals for net Sharpe from paired intervals for Sharpe differences.

Portfolio parameters are chosen on prior validation data: median half-year/seed Sharpe minus 0.25 times its standard deviation. Architecture selection uses only that year's and previous years' validation scores. The joint frequency/model rule uses the same recorded validation scores. Future validation cannot retroactively pick an earlier year's model.

## Portfolio strategies

Every ranking model is evaluated at K=5, 10 and 20 using equal weights, confidence weights, inverse volatility, a rank-retention buffer with equal weights, and a buffer with inverse volatility. Long-short portfolios remain diagnostic and ineligible for selection without historical borrow data.

Inverse volatility uses 60 signal-time return observations, a 10% annual volatility floor and a maximum weight of 2/K. The buffer retains previous target names while they remain inside the top 2K and fills remaining slots from the strongest scores. It is a **target-retention** buffer: a target that failed to fill can still receive retention priority. Tie handling remains fractional and deterministic.

Accounting is identical to the original evaluator for original portfolio methods, verified by exact daily-return regression checks. Each fold begins with $1m. Minimum lagged dollar ADV is $1m, and transactions are limited to 1% of lagged ADV. Scores generated after close t execute at close t+1; daily returns are then measured to t+2 and five-session returns to t+6. Costs include commissions, slippage and a participation-dependent impact term. Terminal liquidation incurs estimated costs and is explicitly flagged when it exceeds the participation cap.

## What counts as evidence?

- Primary results follow validation-selected policies and architectures, including failed choices.
- Equal-weight K=10 ablations compare losses and architecture components while holding the portfolio rule fixed.
- Portfolio-rule ablations use the exact same scores and K.
- Frequency comparisons report both each pipeline's own matched-SPY calendar and a shared-calendar diagnostic.
- Every model, K, seed and cost stress remains visible in the metric artifacts. All model kinds appear in line charts.
- Fifty random equal-weight top-10 paths per cohort/year/horizon provide a ranking control. They are never selection candidates.
- Four-year statistics pool the saved daily returns of annual accounts. They retain cash resets, fixed annual sizing and omitted boundary sessions; they are not a continuously held account.
- Paired 20-session block bootstrap intervals resample separately inside years. They are exploratory and unadjusted for multiple testing or prior research.

The stop rule is to finish this predeclared batch and report every outcome. There is no success-only search until an attractive test curve appears. The 2024-2025 tests have already been examined. The additional earlier years broaden market conditions, but retrospective selection of the period cannot establish a pristine confirmation test.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m hypershift test -v test_research test_robust_research
.\.venv\Scripts\python.exe -m hypershift run run_robust_research --device cpu --workers 3 --out runs/robust_new
.\.venv\Scripts\python.exe -m hypershift run run_robust_research --device cpu --workers 3 --horizon 1 --batch 40 --variants think think_huber think_mix think_mix_huber stockmixer stockmixer_huber multiscale_mse multiscale_huber --out runs/daily_new
.\.venv\Scripts\python.exe -m hypershift run report_robust_research --runs runs/robust_new runs/daily_new --out runs/comparison_new
.\.venv\Scripts\python.exe -m hypershift run plot_robust_diagnostics --run runs/comparison_new
.\.venv\Scripts\python.exe -m hypershift run summarize_robust_findings --run runs/comparison_new
.\.venv\Scripts\python.exe -m hypershift run add_etf_controls --run runs/comparison_new
.\.venv\Scripts\python.exe -m hypershift run audit_robust_regimes --run runs/comparison_new
.\.venv\Scripts\python.exe -m hypershift run plot_combined_research --run runs/comparison_new
```

The actual five-session execution reuses the 108 previously fitted 2024-2025 baseline checkpoints after verifying exact predictions, inputs, splits, architecture hashes and settings. The complete batch comprises 576 neural fits, of which 468 are newly trained; fresh reproduction retrains all 576. Run folders contain source snapshots, checkpoint histories, predicted scores, date splits, portfolio selections and input hashes.

The cohorts retain missing/delisted-history limitations, revised adjusted-price approximations and substantial survivorship bias. Neither successful ranking nor a high Sharpe removes those limitations. A stronger result here would justify further verification, not a claim of established live performance.

For the authoritative completed run folders, refresh the additional audit and
documentation artifacts with these commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m hypershift run audit_pooled_chronology
.\.venv\Scripts\python.exe -m hypershift run report_pooled_uncertainty
.\.venv\Scripts\python.exe -m hypershift run plot_combined_research
.\.venv\Scripts\python.exe -m hypershift run publish_research_docs
```

These supplemental commands inspect the saved final runs and do not refit models.
The presentation in `docs/slides/` is maintained separately by
`presentations/build/update_pooled_datasets.mjs`.

The completed batch found no consistent all-window winner. Daily trading reduced
fixed-equal-K10 Sharpe in 77 of 88 comparisons. Huber improved the five-session
mixer hybrid in five of eight windows, while other Huber variants usually worsened.
The NASDAQ hybrid's pooled net Sharpe was 0.675, compared with 0.621 for SPY and
0.626 for QQQ, but its exploratory paired interval for the Sharpe advantage includes
zero. The same hybrid scored only 0.194 on NYSE. Validation-selected architecture
procedures did not beat either ETF over the pooled period.
