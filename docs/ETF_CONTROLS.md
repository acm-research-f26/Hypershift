## SPY and QQQ controls

<!-- uncertainty:start -->
## Verified pooled confidence intervals

[Updated uncertainty tables](CONFIDENCE_INTERVALS.md) show 95% intervals for net Sharpe and paired Sharpe differences versus both ETFs, plus Huber-versus-MSE ablations. All previously reported pooled benchmark-difference intervals were reproduced. Intervals remain exploratory and unadjusted for model selection or multiple testing.

<!-- uncertainty:end -->

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
