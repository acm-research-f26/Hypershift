## Findings from the completed batch

<!-- uncertainty:start -->
## Verified pooled confidence intervals

[Updated uncertainty tables](CONFIDENCE_INTERVALS.md) show 95% intervals for net Sharpe and paired Sharpe differences versus both ETFs, plus Huber-versus-MSE ablations. All previously reported pooled benchmark-difference intervals were reproduced. Intervals remain exploratory and unadjusted for model selection or multiple testing.

<!-- uncertainty:end -->

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
