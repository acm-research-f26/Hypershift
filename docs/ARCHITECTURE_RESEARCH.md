# Previous 2024–2025 architecture evaluation

The expanded **completed 2022–2025 comparison** is described in
[ROBUST_RESEARCH.md](ROBUST_RESEARCH.md), with [results](REPORT.md)
and [all-model graphs](CHARTS.md). It includes daily forecasts,
Huber ablations and both SPY and QQQ. The protocol below documents the earlier batch.

The earlier run is `runs/recent_2024_2025`. It tests all nine existing neural
variants plus ridge, momentum and reversal on two modern annual test windows.
SPY is the common index-fund benchmark for both historical exchange cohorts.
The previous `runs/architecture_research` experiment remains a separate result set.

## Splits and selection

| Test | Training | Validation |
|---|---|---|
| 2024 | 2019-2022 | 2023 |
| 2025 | 2020-2023 | 2024 |

These are nominal year ranges. Exact boundaries in `fold.json` exclude labels
that have not matured before the next split. Training batches stay chronological.
Scaling and correlation-derived graph groups use training data only. Five-session
labels begin at the next session's close, after scores have been generated.

Each checkpoint minimizes validation MSE. K and weighting use a validation
stability score: median half-year/seed Sharpe minus 0.25 times its standard
deviation. Each test year has its own architecture selection, based only on
that year's validation and prior validation folds. Later validation cannot
retroactively select an earlier test architecture. All choices are saved before
test scoring; predictions do not cause further tuning.

The fixed models are THINK; THINK with ranking loss, feature gating, stock
mixing or volatility prediction; MLP; LSTM; StockMixer-inspired MLP; and
MASTER-inspired Transformer. They use width 16, lookback 16, five causal features,
AdamW at 0.001, up to 30 epochs and six-epoch patience. Seeds 7, 19 and 42 are
ensembled without selecting a best seed. Each seed is ranked only among
signal-time eligible securities. These are local architecture adaptations,
not official-paper reproductions or research decisions actually made in 2024.

## Modern public-data snapshot

`fetch_recent_cohorts.py` freezes the same 100 SHA256-selected names from each
published 2017 cohort before downloading 2018-2025 adjusted prices and raw
close/volume. It does not replace failed names with current winners.

Only 68 NYSE-cohort and 67 NASDAQ-cohort original-issuer histories are recoverable
in this snapshot. AVX and CAMP now resolve to unrelated issuers;
`config/recent_identity_exclusions.json` documents their rejection with
corporate/SEC sources. Their original histories remain unavailable. All original
names, failures and identity rejections are recorded in `universe_coverage.csv`.

This preserves a pre-test candidate list but **does not eliminate survivorship
bias**. Current vendor availability still determines which historical securities
can be modeled. Cohorts include funds/preferred securities, and their exchange
labels do not reconstruct current or point-in-time membership. Corporate-action
vintages and delisting payouts are not fully verified. Results are
availability-conditioned research comparisons.

## Trading and evaluation

Each fold begins with $1m cash. Signal-time eligibility requires 40 valid price
observations and $1m trailing dollar ADV. Rebalance every five sessions with
next-close execution; limit orders to 1% of lagged ADV. Costs per side are 2bp
commission, 5bp slippage/spread, and 3bp times the square root of participation
divided by 1%. Sharpe uses a fixed 3% annual hurdle; cash earns zero. Adjusted
price accounting and next-close fills are approximations.

Long-only equal/confidence top K={5,10,20} policies are validation candidates.
Long-short portfolios are diagnostics only because historical borrow is missing.
SPY, equal-weight buy-and-hold and equal-weight rebalanced universes use the same
dates as model portfolios. Annual folds omit incomplete holding periods and are
not presented as a continuous two-year live account.

Reports include gross/net Sharpe and returns, volatility, maximum drawdown,
turnover, profitable periods, full-cross-section IC/RankIC, unannualized
ICIR/RankICIR, and NDCG@5/10/20. NDCG uses linear percentile return relevance.
Additional outputs include matched-policy component ablations, all seeds,
cost stress, SPY beta/OLS alpha/information ratio, and 50 random top-K controls
per cohort/year/K. Exploratory paired Sharpe intervals use 1,000 circular
20-session block bootstrap resamples and are not adjusted for multiple testing.

## Reproduce and read

Commands are in [README.md](../README.md). Start with
[the modern report](../runs/recent_2024_2025/REPORT.md) and
[graphs](../runs/recent_2024_2025/CHARTS.md). `all_metrics.csv` retains every scenario;
`yearly_comparison.csv` contains validation-selected portfolios for every model;
`validation_selected_by_year.csv` contains the architecture choices.

Source snapshots, hashes, data-input locks, fold definitions, checkpoints,
scalers, predictions, masks, ranking metrics and daily returns are retained.
`accounting_audit.csv` independently recomputes saved paths;
`regime_diagnostics.csv` uses signal-time regimes with training-only thresholds.
A rerun on observed test years is reproduction, not new independent evidence.
