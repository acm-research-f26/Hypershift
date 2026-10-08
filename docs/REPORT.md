# Robust-loss and daily-session experiments, 2022-2025

<!-- combined-curves:start -->
## Combined curves and complete pooled ETF tables

[Open the combined 2022–2025 curves and tables](COMBINED_2022_2025.md). All tested models appear alongside SPY and QQQ, with pooled Sharpe in each legend. The NASDAQ five-session hybrid remains 0.675. Existing experiment results and selections are unchanged.

<!-- combined-curves:end -->

<!-- uncertainty:start -->
## Verified pooled confidence intervals

[Updated uncertainty tables](CONFIDENCE_INTERVALS.md) show 95% intervals for net Sharpe and paired Sharpe differences versus both ETFs, plus Huber-versus-MSE ablations. All previously reported pooled benchmark-difference intervals were reproduced. Intervals remain exploratory and unadjusted for model selection or multiple testing.

<!-- uncertainty:end -->

[All graphs](CHARTS.md) | [Paper review](STOCKMIXER_PAPER_REVIEW.md) | [Frozen plan](../config/robust_experiment_plan.json)

## Findings from the completed batch

**No model beat SPY in all eight cohort/year windows.**

That statement uses each model's validation-selected portfolio and is conditional on this dataset. The individual model comparison below includes hindsight discoveries; the selection procedures here are the implementable historical rules.

### Four-year selection-procedure results

| Procedure | Cohort | Gross SR | Net SR | SPY SR | Difference | Years > SPY | Ann. net return | Max DD | Delta SR CI low | Delta SR CI high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Five-session | NYSE_recent | -0.448 | -0.540 | 0.621 | -1.161 | 0.000 | -11.4% | -45.6% | -1.876 | -0.474 |
| Five-session | NASDAQ_recent | 0.479 | 0.379 | 0.621 | -0.242 | 1.000 | 9.3% | -23.4% | -0.817 | 0.288 |
| Daily | NYSE_recent | -0.112 | -0.192 | 0.497 | -0.688 | 0.000 | -4.6% | -39.0% | -1.337 | -0.018 |
| Daily | NASDAQ_recent | -0.422 | -0.616 | 0.497 | -1.113 | 0.000 | -13.6% | -51.3% | -1.739 | -0.524 |
| Joint validation choice | NYSE_recent | -0.492 | -0.584 | 0.571 | -1.156 | 0.000 | -12.4% | -46.7% | -1.817 | -0.466 |
| Joint validation choice | NASDAQ_recent | 0.479 | 0.379 | 0.621 | -0.242 | 1.000 | 9.3% | -23.4% | -0.817 | 0.288 |

Pooled statistics compound separate annual accounts, omit boundary gaps and use a 3% annual hurdle. Each SPY series uses exactly the procedure's dates. Bootstrap intervals are exploratory, not multiplicity-adjusted.

### Did Huber help?

| Horizon | MSE comparator | Huber variant | Median SR change | Positive windows | Windows |
| --- | --- | --- | --- | --- | --- |
| 1.000 | multiscale_mse | multiscale_huber | -0.482 | 0.000 | 8.000 |
| 1.000 | stockmixer | stockmixer_huber | -0.089 | 1.000 | 8.000 |
| 1.000 | think | think_huber | -0.154 | 2.000 | 8.000 |
| 1.000 | think_mix | think_mix_huber | -0.058 | 3.000 | 8.000 |
| 5.000 | multiscale_mse | multiscale_huber | -0.182 | 1.000 | 8.000 |
| 5.000 | stockmixer | stockmixer_huber | -0.406 | 2.000 | 8.000 |
| 5.000 | think | think_huber | 0.036 | 4.000 | 8.000 |
| 5.000 | think_mix | think_mix_huber | 0.083 | 5.000 | 8.000 |
| 5.000 | think_mix_shrink | think_mix_shrink_huber | -0.020 | 4.000 | 8.000 |

These comparisons fix equal weights and K=10. A lower training loss is not counted as success; the table measures realized net Sharpe. Huber and MSE use identical early-stopping criteria.

### Did daily trading help?

At fixed equal K=10, daily forecasts improved net Sharpe in **11 of 88** matched model/cohort/year comparisons. The median change was **-0.567**. This includes the simple ranking strategies as well as neural models; it is not a count of independent trials.

| Horizon | Median gross-minus-net SR | Median annual turnover |
| --- | --- | --- |
| 1.000 | 0.305 | 77.029 |
| 5.000 | 0.090 | 22.342 |

Changing the frequency also changes target horizon and batch size. A daily gain should therefore be attributed to the whole tested pipeline.

### Exploratory candidates, not newly selected winners

The following order uses median annual Sharpe difference from SPY across both cohorts. It describes stability after seeing the tests; it must not replace the validation-selected choices above.

| Horizon | Model | Median SR-SPY | Worst SR-SPY | Wins / 8 | Median IC | Median RankIC |
| --- | --- | --- | --- | --- | --- | --- |
| 5.000 | momentum126 | -0.253 | -1.789 | 4.000 | -0.006 | 0.005 |
| 5.000 | think_mix_huber | -0.491 | -1.017 | 2.000 | 0.004 | 0.006 |
| 1.000 | momentum126 | -0.547 | -2.516 | 2.000 | 0.003 | 0.011 |
| 5.000 | reversal5 | -0.579 | -3.089 | 3.000 | 0.024 | 0.017 |
| 5.000 | multiscale_mse | -0.592 | -1.944 | 2.000 | -0.004 | -0.009 |
| 1.000 | think_mix | -0.628 | -1.396 | 1.000 | 0.004 | 0.009 |
| 5.000 | multiscale_huber | -0.637 | -2.053 | 3.000 | -0.003 | -0.005 |
| 1.000 | multiscale_huber | -0.637 | -2.264 | 1.000 | 0.007 | 0.002 |

Even a positive four-year Sharpe difference or confidence interval would remain conditional on fixed survivor cohorts, retrospective period choice and the full history of experiments. No further variants were added after this batch's results.

### Paper and historical comparison

StockMixer's AAAI paper reports a Sharpe of 1.586 on its S&P500 stock dataset, but provides no matched SPY/index buy-and-hold row. It therefore does not demonstrate outperformance of the S&P 500 itself. See the [paper review](STOCKMIXER_PAPER_REVIEW.md). Our older THINK+mixer NASDAQ 2017 result was net Sharpe 1.600 versus QQQ 2.188; that hybrid is not the complete paper architecture.

## What was tested

Two fixed public-data cohorts, four test years and three initialization seeds. Five-session forecasts: 16 neural variants; daily forecasts: eight variants covering THINK, THINK+mixer, StockMixer-inspired and multi-scale mixer, each paired across MSE/Huber. Ridge, momentum, reversal, equal-weight buy-and-hold, rebalanced universe and SPY are retained. Daily training includes all daily origins; its batch size is 40 versus 8 for five-session training. The frequency comparison therefore changes prediction horizon, rebalance interval and batch size.

Huber delta is fixed at one percentage point, with a factor of two matching MSE curvature near zero. Early stopping uses validation MSE in all variants. Mixer shrinkage fixes the residual multiplier at 0.1. The new multi-scale mixer uses patch scales 1/2/4, causal temporal layers and an eight-state stock-market bottleneck. These remain local adaptations, not official paper replications.

Portfolio search: K=5/10/20; equal, confidence, inverse-volatility, target-retention buffer, or buffer plus inverse-volatility. Inverse volatility uses 60 past sessions, a 10% annualized volatility floor and 2/K maximum weight. Buffer retains previous target names while inside the top 2K; targets may include unfilled orders. Long-short is diagnostic only because historical borrow availability is absent.

## Validation-selected results

Architecture and portfolio choices are fixed from available prior validation data. The table below follows that selection procedure instead of replacing a disappointing selection with a test-period winner.

| Horizon | Cohort | Year | Selected model | Weights | K | Net SR | SPY SR | Difference | Net return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.000 | NYSE_recent | 2022 | THINK+mixer x0.1 / Huber | confidence | 5 | -1.285 | -0.853 | -0.432 | -33.6% |
| 5.000 | NYSE_recent | 2023 | THINK + stock mixer residual | inverse_vol | 5 | 0.907 | 1.738 | -0.830 | 19.5% |
| 5.000 | NYSE_recent | 2024 | 5-session reversal | buffer_inverse_vol | 5 | -1.102 | 1.987 | -3.089 | -20.1% |
| 5.000 | NYSE_recent | 2025 | THINK+mixer x0.1 / Huber | buffer_inverse_vol | 20 | -0.154 | 0.926 | -1.080 | -1.3% |
| 5.000 | NASDAQ_recent | 2022 | THINK / Huber | buffer_equal | 5 | -0.668 | -0.853 | 0.184 | -14.1% |
| 5.000 | NASDAQ_recent | 2023 | Shared MLP | confidence | 5 | 0.437 | 1.738 | -1.301 | 9.5% |
| 5.000 | NASDAQ_recent | 2024 | THINK+mixer x0.1 / Huber | buffer_equal | 5 | 1.333 | 1.987 | -0.654 | 34.8% |
| 5.000 | NASDAQ_recent | 2025 | THINK+mixer x0.1 / MSE | buffer_inverse_vol | 5 | 0.473 | 0.926 | -0.453 | 11.3% |
| 1.000 | NYSE_recent | 2022 | THINK + stock mixer residual | buffer_equal | 5 | -1.013 | -0.862 | -0.151 | -33.1% |
| 1.000 | NYSE_recent | 2023 | THINK+mixer / Huber | inverse_vol | 5 | 0.804 | 1.605 | -0.801 | 17.4% |
| 1.000 | NYSE_recent | 2024 | THINK+mixer / Huber | confidence | 10 | 0.544 | 1.710 | -1.165 | 12.1% |
| 1.000 | NYSE_recent | 2025 | THINK+mixer / Huber | buffer_inverse_vol | 20 | -0.395 | 0.735 | -1.130 | -5.7% |
| 1.000 | NASDAQ_recent | 2022 | THINK+mixer / Huber | buffer_inverse_vol | 20 | -0.865 | -0.862 | -0.003 | -20.3% |
| 1.000 | NASDAQ_recent | 2023 | 126-session momentum | inverse_vol | 5 | -0.911 | 1.605 | -2.516 | -18.2% |
| 1.000 | NASDAQ_recent | 2024 | 5-session reversal | buffer_equal | 5 | -0.493 | 1.710 | -2.203 | -9.1% |
| 1.000 | NASDAQ_recent | 2025 | THINK+mixer / Huber | buffer_inverse_vol | 5 | -0.207 | 0.735 | -0.942 | -5.4% |

## Selection across both frequencies

This additional rule compares the same prior validation scores across both runs; it also never uses test results to select.

| Cohort | Year | Horizon | Selected model | Net SR | Matched SPY SR |
| --- | --- | --- | --- | --- | --- |
| NYSE_recent | 2022 | 5.000 | THINK+mixer x0.1 / Huber | -1.285 | -0.853 |
| NYSE_recent | 2023 | 5.000 | THINK + stock mixer residual | 0.907 | 1.738 |
| NYSE_recent | 2024 | 5.000 | 5-session reversal | -1.102 | 1.987 |
| NYSE_recent | 2025 | 1.000 | THINK+mixer / Huber | -0.395 | 0.735 |
| NASDAQ_recent | 2022 | 5.000 | THINK / Huber | -0.668 | -0.853 |
| NASDAQ_recent | 2023 | 5.000 | Shared MLP | 0.437 | 1.738 |
| NASDAQ_recent | 2024 | 5.000 | THINK+mixer x0.1 / Huber | 1.333 | 1.987 |
| NASDAQ_recent | 2025 | 5.000 | THINK+mixer x0.1 / MSE | 0.473 | 0.926 |

## Complete model comparison

Each pooled Sharpe is recomputed from daily returns across four separate annual folds, not an average of annual Sharpes. Annual cash resets, fixed $1m sizing and excluded boundary sessions mean the pooled series is not a continuous live account. The confidence interval is an exploratory paired 20-session block bootstrap, stratified by year, with 1,000 repeats. It is not adjusted for this search or earlier searches.

### NYSE_recent: 5-session forecasts

| Model | Gross SR | Net SR | SR-SPY | Years > SPY | Ann. return | Cum. return | Ann. vol | Max DD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| THINK reconstruction | 0.127 | 0.017 | -0.605 | 2.000 | 1.5% | 5.9% | 19.0% | -25.8% |
| THINK + ranking loss | -0.026 | -0.185 | -0.806 | 1.000 | -2.1% | -8.1% | 18.5% | -33.2% |
| THINK + market feature gate | 0.237 | 0.096 | -0.525 | 2.000 | 3.1% | 12.6% | 18.0% | -24.2% |
| THINK + stock mixer residual | 0.341 | 0.301 | -0.321 | 1.000 | 7.4% | 31.6% | 21.4% | -22.4% |
| THINK + volatility auxiliary task | -0.164 | -0.246 | -0.867 | 1.000 | -4.2% | -15.3% | 20.7% | -38.1% |
| Shared MLP | -0.130 | -0.237 | -0.859 | 0.000 | -5.7% | -20.2% | 24.4% | -41.4% |
| LSTM | 0.128 | -0.011 | -0.632 | 1.000 | 0.8% | 3.0% | 19.9% | -21.3% |
| StockMixer-inspired MLP | -0.004 | -0.118 | -0.739 | 0.000 | -2.4% | -9.0% | 23.1% | -34.8% |
| MASTER-inspired Transformer | -0.185 | -0.328 | -0.949 | 1.000 | -7.0% | -24.5% | 23.1% | -38.8% |
| THINK / Huber | 0.131 | -0.054 | -0.675 | 1.000 | 0.6% | 2.4% | 16.9% | -24.4% |
| THINK+mixer / Huber | 0.218 | 0.194 | -0.427 | 1.000 | 5.0% | 20.6% | 18.2% | -20.7% |
| StockMixer / Huber | -0.029 | -0.157 | -0.778 | 1.000 | -1.8% | -6.7% | 18.9% | -19.3% |
| THINK+mixer x0.1 / MSE | 0.139 | 0.075 | -0.546 | 0.000 | 2.2% | 8.9% | 22.0% | -24.9% |
| THINK+mixer x0.1 / Huber | -0.136 | -0.176 | -0.797 | 0.000 | -3.6% | -13.2% | 22.8% | -35.8% |
| Multi-scale mixer / MSE | 0.003 | -0.114 | -0.735 | 1.000 | -1.7% | -6.4% | 21.2% | -23.7% |
| Multi-scale mixer / Huber | -0.014 | -0.113 | -0.734 | 1.000 | -1.4% | -5.5% | 20.5% | -25.3% |
| Ridge regression | 0.185 | -0.030 | -0.651 | 1.000 | -0.5% | -1.8% | 23.4% | -28.5% |
| 126-session momentum | 0.447 | 0.393 | -0.228 | 2.000 | 8.9% | 39.2% | 18.7% | -21.4% |
| 5-session reversal | 0.257 | -0.007 | -0.628 | 1.000 | 0.8% | 3.2% | 20.0% | -33.9% |
| Equal-weight buy-and-hold | 0.222 | 0.213 | -0.408 | 1.000 | 5.3% | 22.1% | 17.6% | -19.8% |
| Equal-weight rebalanced universe | 0.196 | 0.179 | -0.443 | 1.000 | 4.7% | 19.3% | 18.4% | -19.9% |
| SPY | 0.629 | 0.621 | 0.000 | 0.000 | 13.4% | 62.7% | 18.1% | -23.4% |

| Model | IC | RankIC | ICIR | RankICIR | NDCG@5 | NDCG@10 | NDCG@20 | Turnover/year | Win rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| THINK reconstruction | 0.005 | -0.001 | 0.037 | -0.010 | 0.516 | 0.528 | 0.573 | 24.504 | 54.9% |
| THINK + ranking loss | 0.004 | -0.001 | 0.028 | -0.004 | 0.518 | 0.533 | 0.574 | 33.616 | 51.8% |
| THINK + market feature gate | 0.006 | -0.001 | 0.039 | -0.009 | 0.514 | 0.531 | 0.574 | 29.207 | 54.9% |
| THINK + stock mixer residual | 0.012 | 0.021 | 0.070 | 0.125 | 0.514 | 0.533 | 0.576 | 10.248 | 55.9% |
| THINK + volatility auxiliary task | -0.007 | -0.010 | -0.047 | -0.067 | 0.505 | 0.527 | 0.570 | 19.460 | 53.3% |
| Shared MLP | -0.008 | -0.015 | -0.040 | -0.074 | 0.519 | 0.532 | 0.569 | 30.778 | 51.8% |
| LSTM | 0.004 | -0.001 | 0.022 | -0.008 | 0.516 | 0.531 | 0.570 | 33.181 | 50.8% |
| StockMixer-inspired MLP | -0.013 | -0.013 | -0.074 | -0.076 | 0.516 | 0.536 | 0.569 | 31.591 | 50.8% |
| MASTER-inspired Transformer | 0.003 | 0.006 | 0.018 | 0.034 | 0.499 | 0.522 | 0.569 | 38.358 | 48.7% |
| THINK / Huber | 0.013 | 0.005 | 0.090 | 0.037 | 0.512 | 0.533 | 0.574 | 35.827 | 54.9% |
| THINK+mixer / Huber | 0.010 | 0.017 | 0.057 | 0.099 | 0.516 | 0.534 | 0.575 | 5.378 | 57.4% |
| StockMixer / Huber | -0.019 | -0.018 | -0.119 | -0.113 | 0.506 | 0.517 | 0.562 | 28.733 | 51.8% |
| THINK+mixer x0.1 / MSE | 0.003 | 0.010 | 0.021 | 0.059 | 0.511 | 0.532 | 0.570 | 16.420 | 53.8% |
| THINK+mixer x0.1 / Huber | 0.001 | 0.007 | 0.005 | 0.041 | 0.508 | 0.527 | 0.569 | 10.591 | 52.3% |
| Multi-scale mixer / MSE | -0.009 | -0.019 | -0.053 | -0.111 | 0.512 | 0.526 | 0.562 | 29.735 | 49.2% |
| Multi-scale mixer / Huber | -0.015 | -0.019 | -0.088 | -0.107 | 0.507 | 0.529 | 0.568 | 24.408 | 51.3% |
| Ridge regression | 0.010 | 0.002 | 0.053 | 0.013 | 0.532 | 0.541 | 0.576 | 58.520 | 49.7% |
| 126-session momentum | 0.005 | 0.009 | 0.022 | 0.044 | 0.514 | 0.531 | 0.575 | 11.939 | 56.4% |
| 5-session reversal | 0.026 | 0.030 | 0.139 | 0.170 | 0.518 | 0.538 | 0.582 | 61.954 | 50.8% |
| Equal-weight buy-and-hold | — | — | — | — | — | — | — | 2.027 | 57.4% |
| Equal-weight rebalanced universe | — | — | — | — | — | — | — | 4.096 | 55.9% |
| SPY | — | — | — | — | — | — | — | 2.068 | 62.1% |

### NASDAQ_recent: 5-session forecasts

| Model | Gross SR | Net SR | SR-SPY | Years > SPY | Ann. return | Cum. return | Ann. vol | Max DD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| THINK reconstruction | 0.372 | 0.265 | -0.356 | 1.000 | 6.6% | 28.0% | 22.1% | -24.1% |
| THINK + ranking loss | 0.343 | 0.222 | -0.399 | 1.000 | 5.6% | 23.4% | 20.7% | -21.4% |
| THINK + market feature gate | 0.454 | 0.361 | -0.261 | 1.000 | 8.7% | 38.1% | 21.0% | -20.7% |
| THINK + stock mixer residual | 0.386 | 0.356 | -0.265 | 1.000 | 9.0% | 39.6% | 24.0% | -25.8% |
| THINK + volatility auxiliary task | 0.366 | 0.259 | -0.362 | 1.000 | 6.4% | 27.3% | 21.7% | -24.7% |
| Shared MLP | 0.403 | 0.291 | -0.331 | 1.000 | 7.3% | 31.3% | 23.4% | -24.2% |
| LSTM | 0.688 | 0.599 | -0.022 | 2.000 | 15.1% | 72.2% | 22.8% | -21.7% |
| StockMixer-inspired MLP | 0.508 | 0.381 | -0.241 | 1.000 | 9.6% | 42.7% | 23.7% | -26.3% |
| MASTER-inspired Transformer | 0.194 | 0.059 | -0.563 | 1.000 | 2.0% | 7.8% | 21.3% | -26.9% |
| THINK / Huber | 0.323 | 0.228 | -0.393 | 1.000 | 5.7% | 23.8% | 20.2% | -22.7% |
| THINK+mixer / Huber | 0.690 | 0.675 | 0.053 | 1.000 | 19.5% | 99.5% | 27.8% | -30.0% |
| StockMixer / Huber | 0.497 | 0.364 | -0.257 | 1.000 | 9.2% | 40.6% | 23.8% | -26.5% |
| THINK+mixer x0.1 / MSE | 0.498 | 0.452 | -0.170 | 1.000 | 11.4% | 51.8% | 23.4% | -23.3% |
| THINK+mixer x0.1 / Huber | 0.643 | 0.620 | -0.002 | 1.000 | 16.4% | 79.8% | 24.5% | -26.0% |
| Multi-scale mixer / MSE | 0.542 | 0.448 | -0.173 | 1.000 | 11.1% | 50.2% | 22.5% | -24.1% |
| Multi-scale mixer / Huber | 0.513 | 0.419 | -0.202 | 2.000 | 10.0% | 44.6% | 20.8% | -23.8% |
| Ridge regression | 0.354 | 0.194 | -0.427 | 1.000 | 4.9% | 20.5% | 21.9% | -26.7% |
| 126-session momentum | 0.417 | 0.381 | -0.240 | 2.000 | 9.3% | 40.9% | 21.6% | -22.7% |
| 5-session reversal | 0.348 | 0.204 | -0.417 | 2.000 | 5.2% | 21.6% | 21.5% | -27.7% |
| Equal-weight buy-and-hold | 0.349 | 0.341 | -0.281 | 1.000 | 8.0% | 34.9% | 19.7% | -23.0% |
| Equal-weight rebalanced universe | 0.381 | 0.364 | -0.258 | 1.000 | 8.7% | 38.1% | 20.7% | -22.2% |
| SPY | 0.629 | 0.621 | 0.000 | 0.000 | 13.4% | 62.7% | 18.1% | -23.4% |

| Model | IC | RankIC | ICIR | RankICIR | NDCG@5 | NDCG@10 | NDCG@20 | Turnover/year | Win rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| THINK reconstruction | 0.002 | 0.013 | 0.013 | 0.068 | 0.522 | 0.543 | 0.583 | 27.771 | 51.8% |
| THINK + ranking loss | 0.000 | 0.010 | 0.000 | 0.055 | 0.516 | 0.541 | 0.582 | 29.342 | 52.3% |
| THINK + market feature gate | 0.002 | 0.011 | 0.009 | 0.057 | 0.518 | 0.540 | 0.581 | 22.917 | 52.3% |
| THINK + stock mixer residual | 0.005 | -0.000 | 0.027 | -0.001 | 0.525 | 0.537 | 0.579 | 8.672 | 54.4% |
| THINK + volatility auxiliary task | -0.002 | 0.006 | -0.011 | 0.032 | 0.517 | 0.537 | 0.580 | 26.670 | 52.8% |
| Shared MLP | 0.005 | 0.007 | 0.025 | 0.037 | 0.538 | 0.546 | 0.588 | 30.282 | 52.3% |
| LSTM | 0.003 | 0.010 | 0.018 | 0.051 | 0.525 | 0.539 | 0.582 | 23.750 | 55.9% |
| StockMixer-inspired MLP | 0.008 | 0.004 | 0.044 | 0.024 | 0.514 | 0.537 | 0.578 | 34.676 | 52.3% |
| MASTER-inspired Transformer | -0.009 | -0.003 | -0.050 | -0.016 | 0.503 | 0.528 | 0.573 | 33.580 | 52.3% |
| THINK / Huber | -0.009 | -0.004 | -0.057 | -0.025 | 0.512 | 0.532 | 0.578 | 21.905 | 53.8% |
| THINK+mixer / Huber | 0.012 | 0.011 | 0.064 | 0.059 | 0.529 | 0.547 | 0.585 | 5.212 | 56.4% |
| StockMixer / Huber | 0.002 | 0.004 | 0.008 | 0.020 | 0.515 | 0.532 | 0.579 | 36.851 | 55.4% |
| THINK+mixer x0.1 / MSE | 0.007 | 0.004 | 0.042 | 0.024 | 0.513 | 0.536 | 0.577 | 12.296 | 53.8% |
| THINK+mixer x0.1 / Huber | 0.013 | 0.010 | 0.069 | 0.052 | 0.518 | 0.539 | 0.582 | 6.722 | 55.4% |
| Multi-scale mixer / MSE | 0.008 | 0.011 | 0.048 | 0.066 | 0.522 | 0.542 | 0.586 | 25.102 | 51.8% |
| Multi-scale mixer / Huber | -0.003 | -0.003 | -0.019 | -0.017 | 0.517 | 0.534 | 0.579 | 22.854 | 53.3% |
| Ridge regression | 0.012 | 0.001 | 0.057 | 0.006 | 0.523 | 0.539 | 0.582 | 41.521 | 56.4% |
| 126-session momentum | -0.002 | 0.005 | -0.007 | 0.026 | 0.503 | 0.527 | 0.576 | 8.996 | 58.5% |
| 5-session reversal | 0.005 | 0.009 | 0.022 | 0.044 | 0.506 | 0.532 | 0.576 | 36.133 | 52.3% |
| Equal-weight buy-and-hold | — | — | — | — | — | — | — | 2.014 | 54.4% |
| Equal-weight rebalanced universe | — | — | — | — | — | — | — | 4.298 | 54.4% |
| SPY | — | — | — | — | — | — | — | 2.068 | 62.1% |

### NYSE_recent: 1-session forecasts

| Model | Gross SR | Net SR | SR-SPY | Years > SPY | Ann. return | Cum. return | Ann. vol | Max DD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| THINK reconstruction | 0.314 | -0.262 | -0.759 | 1.000 | -4.2% | -15.7% | 20.1% | -36.9% |
| THINK / Huber | 0.100 | -0.597 | -1.094 | 1.000 | -9.3% | -31.9% | 18.4% | -42.3% |
| THINK + stock mixer residual | -0.203 | -0.229 | -0.726 | 0.000 | -5.9% | -21.2% | 25.3% | -39.0% |
| THINK+mixer / Huber | -0.092 | -0.180 | -0.676 | 0.000 | -3.4% | -12.7% | 22.1% | -35.9% |
| StockMixer-inspired MLP | 0.132 | -0.217 | -0.714 | 1.000 | -5.0% | -18.5% | 24.1% | -33.0% |
| StockMixer / Huber | -0.047 | -0.482 | -0.979 | 1.000 | -8.9% | -30.7% | 20.9% | -40.1% |
| Multi-scale mixer / MSE | -0.176 | -0.397 | -0.893 | 1.000 | -8.4% | -29.1% | 22.9% | -44.3% |
| Multi-scale mixer / Huber | 0.100 | -0.329 | -0.825 | 0.000 | -6.3% | -22.5% | 21.6% | -40.9% |
| Ridge regression | -0.066 | -0.939 | -1.436 | 0.000 | -16.9% | -52.0% | 20.7% | -58.2% |
| 126-session momentum | 0.163 | 0.079 | -0.417 | 1.000 | 2.4% | 10.0% | 21.0% | -27.4% |
| 5-session reversal | 0.203 | -0.226 | -0.723 | 1.000 | -4.0% | -15.0% | 21.3% | -34.1% |
| Equal-weight buy-and-hold | 0.148 | 0.139 | -0.358 | 1.000 | 3.9% | 16.4% | 17.6% | -20.3% |
| Equal-weight rebalanced universe | 0.113 | 0.086 | -0.411 | 1.000 | 2.9% | 11.9% | 18.3% | -21.0% |
| SPY | 0.505 | 0.497 | 0.000 | 0.000 | 10.8% | 50.2% | 18.1% | -24.5% |

| Model | IC | RankIC | ICIR | RankICIR | NDCG@5 | NDCG@10 | NDCG@20 | Turnover/year | Win rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| THINK reconstruction | 0.002 | -0.004 | 0.013 | -0.021 | 0.519 | 0.534 | 0.573 | 134.818 | 50.3% |
| THINK / Huber | 0.004 | -0.002 | 0.024 | -0.012 | 0.515 | 0.532 | 0.572 | 149.463 | 48.9% |
| THINK + stock mixer residual | 0.001 | 0.007 | 0.003 | 0.040 | 0.511 | 0.532 | 0.572 | 8.355 | 49.7% |
| THINK+mixer / Huber | 0.001 | 0.006 | 0.006 | 0.038 | 0.512 | 0.531 | 0.570 | 23.241 | 49.9% |
| StockMixer-inspired MLP | -0.004 | -0.004 | -0.022 | -0.022 | 0.513 | 0.531 | 0.571 | 97.991 | 48.2% |
| StockMixer / Huber | -0.002 | -0.002 | -0.012 | -0.009 | 0.512 | 0.531 | 0.572 | 106.260 | 49.1% |
| Multi-scale mixer / MSE | -0.007 | -0.009 | -0.033 | -0.042 | 0.515 | 0.530 | 0.570 | 60.109 | 48.5% |
| Multi-scale mixer / Huber | 0.000 | 0.001 | 0.001 | 0.007 | 0.516 | 0.531 | 0.574 | 107.100 | 49.1% |
| Ridge regression | 0.001 | -0.002 | 0.007 | -0.011 | 0.509 | 0.529 | 0.570 | 212.248 | 48.2% |
| 126-session momentum | 0.004 | 0.007 | 0.016 | 0.034 | 0.514 | 0.534 | 0.576 | 20.671 | 50.7% |
| 5-session reversal | 0.012 | 0.014 | 0.057 | 0.073 | 0.517 | 0.534 | 0.576 | 106.181 | 50.6% |
| Equal-weight buy-and-hold | — | — | — | — | — | — | — | 1.998 | 50.9% |
| Equal-weight rebalanced universe | — | — | — | — | — | — | — | 6.352 | 50.5% |
| SPY | — | — | — | — | — | — | — | 2.026 | 54.0% |

### NASDAQ_recent: 1-session forecasts

| Model | Gross SR | Net SR | SR-SPY | Years > SPY | Ann. return | Cum. return | Ann. vol | Max DD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| THINK reconstruction | 0.300 | -0.097 | -0.594 | 0.000 | -1.4% | -5.4% | 21.4% | -27.6% |
| THINK / Huber | 0.336 | -0.182 | -0.678 | 0.000 | -2.5% | -9.7% | 19.7% | -26.7% |
| THINK + stock mixer residual | 0.526 | 0.383 | -0.113 | 1.000 | 10.0% | 45.6% | 25.8% | -32.6% |
| THINK+mixer / Huber | 0.336 | 0.291 | -0.206 | 0.000 | 7.3% | 32.2% | 23.8% | -28.4% |
| StockMixer-inspired MLP | 0.396 | 0.055 | -0.442 | 1.000 | 1.4% | 5.7% | 24.1% | -26.9% |
| StockMixer / Huber | 0.524 | 0.251 | -0.246 | 1.000 | 6.3% | 27.2% | 23.0% | -25.8% |
| Multi-scale mixer / MSE | 0.423 | 0.161 | -0.335 | 1.000 | 4.1% | 17.0% | 23.9% | -27.1% |
| Multi-scale mixer / Huber | 0.416 | 0.088 | -0.409 | 1.000 | 2.5% | 10.1% | 22.2% | -26.0% |
| Ridge regression | 0.017 | -0.842 | -1.339 | 0.000 | -15.7% | -49.0% | 21.1% | -60.3% |
| 126-session momentum | 0.080 | -0.004 | -0.501 | 1.000 | 0.5% | 1.8% | 22.0% | -33.7% |
| 5-session reversal | 0.171 | -0.092 | -0.588 | 1.000 | -1.7% | -6.6% | 22.8% | -28.2% |
| Equal-weight buy-and-hold | 0.259 | 0.251 | -0.246 | 1.000 | 6.2% | 26.6% | 19.9% | -24.5% |
| Equal-weight rebalanced universe | 0.281 | 0.255 | -0.241 | 1.000 | 6.3% | 27.2% | 20.7% | -23.6% |
| SPY | 0.505 | 0.497 | 0.000 | 0.000 | 10.8% | 50.2% | 18.1% | -24.5% |

| Model | IC | RankIC | ICIR | RankICIR | NDCG@5 | NDCG@10 | NDCG@20 | Turnover/year | Win rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| THINK reconstruction | 0.002 | -0.001 | 0.011 | -0.007 | 0.506 | 0.530 | 0.576 | 101.201 | 49.1% |
| THINK / Huber | -0.002 | -0.000 | -0.010 | -0.001 | 0.510 | 0.530 | 0.576 | 118.549 | 47.5% |
| THINK + stock mixer residual | 0.007 | 0.007 | 0.046 | 0.042 | 0.519 | 0.537 | 0.581 | 44.574 | 50.8% |
| THINK+mixer / Huber | 0.004 | 0.005 | 0.024 | 0.030 | 0.517 | 0.535 | 0.580 | 13.040 | 50.5% |
| StockMixer-inspired MLP | 0.000 | -0.001 | 0.003 | -0.007 | 0.520 | 0.536 | 0.579 | 94.704 | 49.9% |
| StockMixer / Huber | 0.004 | 0.003 | 0.023 | 0.016 | 0.515 | 0.532 | 0.579 | 75.315 | 49.8% |
| Multi-scale mixer / MSE | 0.004 | 0.002 | 0.020 | 0.010 | 0.519 | 0.537 | 0.580 | 73.642 | 49.3% |
| Multi-scale mixer / Huber | 0.005 | 0.004 | 0.030 | 0.020 | 0.518 | 0.536 | 0.581 | 87.100 | 49.1% |
| Ridge regression | 0.002 | -0.001 | 0.009 | -0.003 | 0.509 | 0.530 | 0.576 | 211.973 | 46.3% |
| 126-session momentum | 0.005 | 0.013 | 0.022 | 0.056 | 0.517 | 0.537 | 0.583 | 21.592 | 50.4% |
| 5-session reversal | 0.008 | 0.005 | 0.036 | 0.025 | 0.514 | 0.536 | 0.578 | 70.228 | 50.2% |
| Equal-weight buy-and-hold | — | — | — | — | — | — | — | 1.993 | 49.6% |
| Equal-weight rebalanced universe | — | — | — | — | — | — | — | 6.585 | 49.8% |
| SPY | — | — | — | — | — | — | — | 2.026 | 54.0% |

## What helped in paired ablations?

All comparisons here use fixed equal-weight K=10, independent of the validation-selected portfolio. A positive difference favors the second variant. Cohort/year windows overlap in market exposure and are not independent trials.

| Horizon | Before | After | Median SR change | Positive windows | Windows | Mean IC change | Mean RankIC change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | multiscale_mse | multiscale_huber | -0.482 | 0.000 | 8.000 | 0.004 | 0.006 |
| 1.000 | stockmixer | multiscale_mse | 0.188 | 6.000 | 8.000 | 0.000 | -0.001 |
| 1.000 | stockmixer | stockmixer_huber | -0.089 | 1.000 | 8.000 | 0.003 | 0.003 |
| 1.000 | think | think_huber | -0.154 | 2.000 | 8.000 | -0.001 | 0.001 |
| 1.000 | think_mix | think_mix_huber | -0.058 | 3.000 | 8.000 | -0.001 | -0.001 |
| 5.000 | multiscale_mse | multiscale_huber | -0.182 | 1.000 | 8.000 | -0.008 | -0.007 |
| 5.000 | stockmixer | multiscale_mse | -0.131 | 4.000 | 8.000 | 0.002 | 0.001 |
| 5.000 | stockmixer | stockmixer_huber | -0.406 | 2.000 | 8.000 | -0.006 | -0.003 |
| 5.000 | think | think_huber | 0.036 | 4.000 | 8.000 | -0.002 | -0.005 |
| 5.000 | think_mix | think_mix_huber | 0.083 | 5.000 | 8.000 | 0.003 | 0.004 |
| 5.000 | think_mix | think_mix_shrink | -0.011 | 4.000 | 8.000 | -0.003 | -0.003 |
| 5.000 | think_mix_huber | think_mix_shrink_huber | -0.073 | 3.000 | 8.000 | -0.004 | -0.006 |
| 5.000 | think_mix_shrink | think_mix_shrink_huber | -0.020 | 4.000 | 8.000 | 0.001 | 0.001 |

## Portfolio-rule effects at fixed K=10

Each rule is compared with equal weights using exactly the same scores. Lower turnover need not imply better Sharpe.

| Horizon | Rule | Median SR change | Median turnover change | Positive cases | Cases |
| --- | --- | --- | --- | --- | --- |
| 1.000 | buffer_equal | 0.224 | -82.987 | 72.000 | 88.000 |
| 1.000 | buffer_inverse_vol | 0.178 | -77.354 | 62.000 | 88.000 |
| 1.000 | inverse_vol | -0.108 | 10.366 | 20.000 | 88.000 |
| 5.000 | buffer_equal | -0.007 | -15.038 | 74.000 | 152.000 |
| 5.000 | buffer_inverse_vol | 0.004 | -14.411 | 77.000 | 152.000 |
| 5.000 | inverse_vol | -0.025 | 0.819 | 70.000 | 152.000 |

## Daily versus five-session forecasts

Same model name and equal-weight K=10; each uses its own matched-date SPY and forecast schedule. The common-calendar CSV additionally compares only shared dates without resetting existing positions. This is a pipeline comparison, not an isolated rebalance-only experiment.

| Model | Daily minus five SR | Turnover change | Daily better windows | Windows |
| --- | --- | --- | --- | --- |
| momentum126 | -0.098 | 21.989 | 2.000 | 8.000 |
| multiscale_huber | -0.741 | 228.171 | 0.000 | 8.000 |
| multiscale_mse | -0.803 | 161.747 | 1.000 | 8.000 |
| reversal5 | -0.180 | 88.832 | 3.000 | 8.000 |
| ridge | -1.007 | 290.964 | 0.000 | 8.000 |
| stockmixer | -0.776 | 186.793 | 0.000 | 8.000 |
| stockmixer_huber | -0.568 | 203.131 | 1.000 | 8.000 |
| think | -0.756 | 228.784 | 1.000 | 8.000 |
| think_huber | -0.910 | 219.921 | 0.000 | 8.000 |
| think_mix | -0.098 | 36.343 | 2.000 | 8.000 |
| think_mix_huber | -0.242 | 8.300 | 1.000 | 8.000 |

## Trading, data and statistical limits

- Annual train/validation/test folds are chronological and purge labels that cross boundaries. Training uses available history within the preceding four training years; 2022 has less history because the source begins in 2018 and needs a feature warmup. Exact dates are in each fold.json.
- Scores use information through close t; orders execute at close t+1 plus modeled costs. Daily targets run from t+1 to t+2; five-session targets from t+1 to t+6. No same-signal-close fill is assumed.
- $1m initial NAV; minimum lagged dollar ADV $1m; trades capped at 1% signal-time ADV. Costs per side: 2bp commission, 5bp spread/slippage, and 3bp times sqrt(participation/1%). Stress tests at 0x/2x/4x costs are retained. Cash earns zero; Sharpe uses a fixed 3% hurdle.
- Adjusted-price returns approximate total returns and auction fills. Terminal liquidation is charged but can exceed the participation cap; stale marks and execution constraints are recorded. Missing future labels are reported rather than used to remove stocks at signal time.
- The universes are fixed survivor cohorts, not all NYSE/NASDAQ stocks or point-in-time S&P 500 members. Public-history coverage and delisting gaps materially limit conclusions. SPY has a different investment universe.
- 2024-2025 were inspected before this batch. Extending to 2022-2023 broadens regimes, but choosing that extension retrospectively does not make it untouched confirmation data. No additional candidates were introduced after this batch's outcomes.
- IC/RankIC use the full eligible cross-section with observed labels. NDCG uses realized-return percentile relevance, including negative-return periods. ICIR/RankICIR are unannualized period-mean divided by period standard deviation. They should not be compared across horizons as if their sampling frequency were identical.
- Seed stability and K sensitivity are diagnostics. Confidence intervals are not corrected for multiple comparisons; neither an isolated high Sharpe nor its unadjusted interval establishes a repeatable edge.

## Reproduction and detailed artifacts

Training commands (use fresh output directories):

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_research test_robust_research
.\.venv\Scripts\python.exe run_robust_research.py --workers 3 --out runs/robust_new
.\.venv\Scripts\python.exe run_robust_research.py --workers 3 --horizon 1 --batch 40 --variants think think_huber think_mix think_mix_huber stockmixer stockmixer_huber multiscale_mse multiscale_huber --out runs/daily_new
.\.venv\Scripts\python.exe report_robust_research.py --runs runs/robust_new runs/daily_new --out runs/comparison_new
```

The actual five-session run reused verified old 2024-2025 baseline fits; exact predictions, preprocessing, fold definitions, training settings and model/source hashes were checked. FIT_REUSE_AUDIT.json records them. Fresh reproduction retrains all fits.

- [All metrics, policies, K, seeds and cost stresses](all_metrics.csv)
- [Annual comparison and SPY-relative metrics](annual_comparison.csv)
- [Pooled comparison including intervals](pooled_comparison.csv)
- [Fixed-K10 architecture/loss ablations](fixed_k10_ablations.csv)
- [Fixed-K10 portfolio-rule effects](fixed_k10_portfolio_effects.csv)
- [Daily/five-session comparison](horizon_comparison.csv) and [shared-date comparison](horizon_common_calendar.csv)
- [Seed stability](seed_stability.csv) and [50 random top-10 controls per window](random_top10_controls.csv)
- [Verified path accounting](accounting_audit.csv), [combined validation lock](../runs/robust_comparison/COMBINED_SELECTION_LOCK.json), and [input hashes](../runs/robust_comparison/comparison_input_hashes.json)

The complete model specifications, training dates, eligible universe, chosen policies, checkpoints, predictions and histories remain in the two source run directories.

## SPY and QQQ controls

QQQ was added at the user's request after the training batch began. It is a comparison control, never a model-selection candidate. Both ETFs use frozen adjusted prices, the exact model evaluation dates, and 7bp entry plus 7bp exit costs. ETF expenses are already reflected in market prices. Sharpe uses the same fixed 3% hurdle.

### Four-year selection procedures

| Procedure | Cohort | Model net SR | SPY SR | QQQ SR | Minus SPY | Minus QQQ | Years > both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Five-session | NYSE_recent | -0.540 | 0.621 | 0.626 | -1.161 | -1.165 | 0.000 |
| Five-session | NASDAQ_recent | 0.379 | 0.621 | 0.626 | -0.242 | -0.247 | 1.000 |
| Daily | NYSE_recent | -0.192 | 0.497 | 0.499 | -0.688 | -0.690 | 0.000 |
| Daily | NASDAQ_recent | -0.616 | 0.497 | 0.499 | -1.113 | -1.115 | 0.000 |
| Joint validation choice | NYSE_recent | -0.584 | 0.571 | 0.581 | -1.156 | -1.165 | 0.000 |
| Joint validation choice | NASDAQ_recent | 0.379 | 0.621 | 0.626 | -0.242 | -0.247 | 1.000 |

### ETF controls for every annual window

| Horizon | Cohort | Year | ETF | Net SR | Net return | Ann. vol | Max DD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5.000 | NYSE_recent | 2022 | SPY | -0.853 | -18.3% | 24.4% | -23.4% |
| 5.000 | NYSE_recent | 2022 | QQQ | -1.153 | -31.9% | 32.3% | -32.4% |
| 5.000 | NYSE_recent | 2023 | SPY | 1.738 | 27.3% | 13.1% | -10.0% |
| 5.000 | NYSE_recent | 2023 | QQQ | 2.559 | 58.2% | 17.9% | -10.8% |
| 5.000 | NYSE_recent | 2024 | SPY | 1.987 | 30.1% | 12.6% | -8.4% |
| 5.000 | NYSE_recent | 2024 | QQQ | 1.616 | 34.3% | 18.0% | -13.6% |
| 5.000 | NYSE_recent | 2025 | SPY | 0.926 | 20.2% | 19.7% | -18.8% |
| 5.000 | NYSE_recent | 2025 | QQQ | 0.922 | 23.5% | 23.8% | -22.8% |
| 5.000 | NASDAQ_recent | 2022 | SPY | -0.853 | -18.3% | 24.4% | -23.4% |
| 5.000 | NASDAQ_recent | 2022 | QQQ | -1.153 | -31.9% | 32.3% | -32.4% |
| 5.000 | NASDAQ_recent | 2023 | SPY | 1.738 | 27.3% | 13.1% | -10.0% |
| 5.000 | NASDAQ_recent | 2023 | QQQ | 2.559 | 58.2% | 17.9% | -10.8% |
| 5.000 | NASDAQ_recent | 2024 | SPY | 1.987 | 30.1% | 12.6% | -8.4% |
| 5.000 | NASDAQ_recent | 2024 | QQQ | 1.616 | 34.3% | 18.0% | -13.6% |
| 5.000 | NASDAQ_recent | 2025 | SPY | 0.926 | 20.2% | 19.7% | -18.8% |
| 5.000 | NASDAQ_recent | 2025 | QQQ | 0.922 | 23.5% | 23.8% | -22.8% |
| 1.000 | NYSE_recent | 2022 | SPY | -0.862 | -18.7% | 24.3% | -24.5% |
| 1.000 | NYSE_recent | 2022 | QQQ | -1.161 | -32.4% | 32.2% | -34.0% |
| 1.000 | NYSE_recent | 2023 | SPY | 1.605 | 25.6% | 13.1% | -10.0% |
| 1.000 | NYSE_recent | 2023 | QQQ | 2.409 | 55.0% | 17.9% | -10.8% |
| 1.000 | NYSE_recent | 2024 | SPY | 1.710 | 26.4% | 12.6% | -8.4% |
| 1.000 | NYSE_recent | 2024 | QQQ | 1.352 | 28.9% | 18.0% | -13.6% |
| 1.000 | NYSE_recent | 2025 | SPY | 0.735 | 16.4% | 19.5% | -18.8% |
| 1.000 | NYSE_recent | 2025 | QQQ | 0.735 | 18.9% | 23.6% | -22.8% |
| 1.000 | NASDAQ_recent | 2022 | SPY | -0.862 | -18.7% | 24.3% | -24.5% |
| 1.000 | NASDAQ_recent | 2022 | QQQ | -1.161 | -32.4% | 32.2% | -34.0% |
| 1.000 | NASDAQ_recent | 2023 | SPY | 1.605 | 25.6% | 13.1% | -10.0% |
| 1.000 | NASDAQ_recent | 2023 | QQQ | 2.409 | 55.0% | 17.9% | -10.8% |
| 1.000 | NASDAQ_recent | 2024 | SPY | 1.710 | 26.4% | 12.6% | -8.4% |
| 1.000 | NASDAQ_recent | 2024 | QQQ | 1.352 | 28.9% | 18.0% | -13.6% |
| 1.000 | NASDAQ_recent | 2025 | SPY | 0.735 | 16.4% | 19.5% | -18.8% |
| 1.000 | NASDAQ_recent | 2025 | QQQ | 0.735 | 18.9% | 23.6% | -22.8% |

[Every model/year versus both ETFs](model_vs_spy_qqq_by_year.csv) | [Every pooled model versus both ETFs, including QQQ intervals](pooled_model_vs_spy_qqq.csv) | [Selection procedures](selection_procedures_vs_spy_qqq.csv)

QQQ tracks a different, more concentrated investment universe than SPY. Neither ETF is a matched-security benchmark for our small historical cohorts. Pooled intervals are exploratory and unadjusted for repeated research.


## Regime and execution-quality checks

Regimes use information available at each signal: the sign of SPY's trailing 126-session return and its trailing 20-session volatility relative to a training-only median. [Detailed results](regime_diagnostics.csv) compare every model with both ETFs within each regime. Conditional Sharpe values use noncontiguous subsets and small samples; they are not tradable regime-switching portfolios or additional model-selection evidence.

The table records worst-case counts across the separately tested model accounts, not unique market events. Stale position marks indicate missing contemporaneous prices for held securities. Terminal liquidation is costed but may exceed the daily participation cap; that is a simulator limitation. Unobserved labels are excluded only from ranking-metric calculation, not retrospectively from signal-time eligibility.

| Horizon | Cohort | Year | Min eligible | Max eligible | Max stale marks | Missing labels | Terminal orders > cap | Constrained orders |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | NASDAQ_recent | 2022 | 52.000 | 58.000 | 0.000 | 0.000 | 6.000 | 984.000 |
| 1.000 | NASDAQ_recent | 2023 | 50.000 | 57.000 | 0.000 | 0.000 | 9.000 | 1040.000 |
| 1.000 | NASDAQ_recent | 2024 | 51.000 | 57.000 | 0.000 | 0.000 | 9.000 | 1232.000 |
| 1.000 | NASDAQ_recent | 2025 | 53.000 | 59.000 | 0.000 | 0.000 | 10.000 | 893.000 |
| 1.000 | NYSE_recent | 2022 | 56.000 | 63.000 | 0.000 | 0.000 | 7.000 | 1174.000 |
| 1.000 | NYSE_recent | 2023 | 56.000 | 63.000 | 0.000 | 0.000 | 7.000 | 963.000 |
| 1.000 | NYSE_recent | 2024 | 56.000 | 61.000 | 0.000 | 0.000 | 9.000 | 1015.000 |
| 1.000 | NYSE_recent | 2025 | 56.000 | 61.000 | 0.000 | 0.000 | 7.000 | 1255.000 |
| 5.000 | NASDAQ_recent | 2022 | 52.000 | 57.000 | 0.000 | 0.000 | 7.000 | 272.000 |
| 5.000 | NASDAQ_recent | 2023 | 50.000 | 56.000 | 0.000 | 0.000 | 9.000 | 321.000 |
| 5.000 | NASDAQ_recent | 2024 | 51.000 | 57.000 | 0.000 | 0.000 | 11.000 | 290.000 |
| 5.000 | NASDAQ_recent | 2025 | 53.000 | 59.000 | 0.000 | 0.000 | 7.000 | 240.000 |
| 5.000 | NYSE_recent | 2022 | 56.000 | 62.000 | 0.000 | 0.000 | 8.000 | 346.000 |
| 5.000 | NYSE_recent | 2023 | 56.000 | 63.000 | 0.000 | 0.000 | 10.000 | 306.000 |
| 5.000 | NYSE_recent | 2024 | 56.000 | 61.000 | 0.000 | 0.000 | 9.000 | 230.000 |
| 5.000 | NYSE_recent | 2025 | 56.000 | 61.000 | 0.000 | 0.000 | 6.000 | 312.000 |

## Completed portfolio and candidate interpretation

These additional aggregate descriptions were written after the tests. They are not new selection rules. The comparisons are correlated and should not be treated as independent statistical trials.

### Does increasing K help?

| Equal-weight comparison | Positive windows | Total windows | Median net Sharpe change |
|---|---:|---:|---:|
| K=10 minus K=5 | 119 | 240 | -0.004 |
| K=20 minus K=5 | 138 | 240 | +0.063 |

This pools the declared ranking models across both horizons and cohorts. Top-10 alone did not solve inconsistency. Top-20 produced a modest descriptive improvement, but it cannot be selected using these test results.

### Do portfolio changes help?

| Horizon | K=10 policy versus equal weights | Positive windows | Windows | Median Sharpe change | Median annual turnover change |
|---|---|---:|---:|---:|---:|
| 1 sessions | buffer_equal | 72 | 88 | +0.224 | -82.99 |
| 1 sessions | buffer_inverse_vol | 62 | 88 | +0.178 | -77.35 |
| 1 sessions | inverse_vol | 20 | 88 | -0.108 | +10.37 |
| 5 sessions | buffer_equal | 74 | 152 | -0.007 | -15.04 |
| 5 sessions | buffer_inverse_vol | 77 | 152 | +0.004 | -14.41 |
| 5 sessions | inverse_vol | 70 | 152 | -0.025 | +0.82 |

The equal-weight retention buffer reduced turnover and helped daily net Sharpe most clearly. That improvement did not make the daily validation-selected procedure beat either ETF. Inverse-volatility weights alone did not provide a consistent gain.

### Promising hybrid and its limits

The five-session NASDAQ THINK+mixer+Huber variant achieved pooled net Sharpe **0.675**, versus **0.621 SPY** and **0.626 QQQ**, using each year’s validation-selected portfolio. Its net annualized return was **19.5%**, cumulative return **99.5%**, annualized volatility **27.8%**, and maximum drawdown **30.0%**. Mean IC was **0.0118**, RankIC **0.0110**, and NDCG@10 **0.5475**. The same model’s NYSE pooled Sharpe was only **0.194**. It beat both ETFs in only **one of four NASDAQ years**.

The exploratory 95% paired block-bootstrap interval for the NASDAQ Sharpe advantage is **[-0.396, 0.524] versus SPY** and **[-0.445, 0.596] versus QQQ**. Both include zero. The model was identified after inspecting this batch, and its architecture was not the validation-selected choice each year. Its result supports further study, not an established economic or statistical edge. Public-data survivorship bias, repeated historical evaluation and unadjusted multiple comparisons remain material.

Four-year statistics concatenate annual accounts and omit boundary sessions. ETF values are matched-window controls, not published full-calendar-year fund returns. See [all gross/net and ranking metrics](pooled_model_vs_spy_qqq.csv).
