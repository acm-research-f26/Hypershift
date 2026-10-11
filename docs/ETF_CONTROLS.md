## SPY and cohort benchmark controls

<!-- benchmark-audit-20261010:start -->
NYSX is the passive Global X NYSE 100 ETF. It launched March 25, 2026 and has no tradable 2022–2025 history. Its index covers technology and growth companies across exchanges, not the 100 largest NYSE-listed companies. NYSX performance is unavailable here. NYSE historical charts use the simulated equal-weight cohort basket (scheduled rebalances and modeled costs), not an ETF. NASDAQ charts retain QQQ; SPY supplies broad-market context. [Issuer and inception](https://www.globalxetfs.com/funds/nysx/) and [prospectus](https://assets.globalxetfs.com/funds/documents/nysx/prospectus-regulatory/Summary-Prospectus_NYSX.pdf).

The refreshed CHARTS.md pages are the current visual record. Earlier ETF-overlay descriptions are superseded by this correction. Audited return paths and training selections remain preserved.
<!-- benchmark-audit-20261010:end -->

<!-- nyse-benchmark:start -->
**Comparison benchmark:** NYSE tables use **NYSE basket — passive benchmark** (saved `equal_universe` portfolio: equal weights across all stocks eligible at each scheduled rebalance, with modeled execution constraints and costs). Earlier tables use their original NYSE cohort; top-100 tables use the new NYSE cohort. This is a simulated portfolio, not an ETF. Nasdaq retains QQQ; SPY remains a secondary reference. A mixed-cohort “benchmark” column means NYSE basket for NYSE rows and QQQ for Nasdaq rows. Paired confidence intervals were recalculated on identical dates using 1,000 circular 20-session block resamples within each annual fold (seed 20261005). Saved model returns and selection choices are unchanged. Embedded charts retain their previous benchmark overlays; this update applies to tables.

<!-- nyse-benchmark:end -->

<!-- huber-ranking:start -->
## Huber plus ranking: updated results and confidence intervals

| Model / procedure | NASDAQ net SR | NYSE net SR |
| --- | ---: | ---: |
| THINK+mixer+Huber | 0.675 | 0.194 |
| Huber+ranking, weight 0.1 | 0.646 | 0.135 |
| Huber+ranking, weight 0.3 | 0.711 | 0.032 |
| Huber+ranking, weight 1.0 | 0.671 | 0.239 |
| Validation choice of ranking weight | 0.638 | 0.189 |
| SPY | 0.621 | 0.621 |
| Comparison benchmark | 0.626 | 0.179 |

Each model uses its own prior-validation-selected portfolio. The separate ranking-weight choice uses only current-and-prior validation and includes zero weight.

**Interpretation:** ranking weight 0.3 reaches NASDAQ Sharpe 0.711 with its selected portfolio, but its advantage over Huber is +0.037, CI [-0.178, 0.252]. NYSE weight 1.0 reaches 0.239 versus Huber 0.194, with gain CI [-0.433, 0.473]. Neither establishes an improvement. The primary fixed-K10 validation-selected ranking procedure changes NASDAQ 0.532→0.488 (gain CI [-0.129, 0.053]) and NYSE 0.089→0.148 (gain CI [-0.137, 0.260]). Secondary validation-chosen ranking weights yield NASDAQ 0.638 and NYSE 0.189. No best test-performing weight or K is promoted to the validation-selected procedure.

[Full ranking report](HUBER_RANKING.md) | [Ranking metrics and intervals CSV](huber_ranking_pooled.csv) | [Consolidated 100-series metrics](combined_2022_2025_metrics_current.csv) | [Consolidated interval table](pooled_confidence_intervals_current.csv)

[Full 95% uncertainty tables](CONFIDENCE_INTERVALS.md). The original study below is retained as a reference.

<!-- huber-ranking:end -->

<!-- uncertainty:start -->
## Verified pooled confidence intervals

[Updated uncertainty tables](CONFIDENCE_INTERVALS.md) show 95% intervals for net Sharpe and paired Sharpe differences versus SPY and the cohort comparison benchmark, plus Huber-versus-MSE ablations. All previously reported pooled benchmark-difference intervals were reproduced. Intervals remain exploratory and unadjusted for model selection or multiple testing.

<!-- uncertainty:end -->

QQQ was added at the user's request after the training batch began. It is a comparison control, never a model-selection candidate. Both ETFs use frozen adjusted prices, the exact model evaluation dates, and 7bp entry plus 7bp exit costs. ETF expenses are already reflected in market prices. Sharpe uses the same fixed 3% hurdle.

### Four-year selection procedures

| Procedure | Cohort | Model net SR | SPY SR | Benchmark SR | Minus SPY | Minus benchmark | Years > both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Five-session | NYSE_recent | -0.540 | 0.621 | 0.179 | -1.161 | -0.718 | 0 |
| Five-session | NASDAQ_recent | 0.379 | 0.621 | 0.626 | -0.242 | -0.247 | 1.000 |
| Daily | NYSE_recent | -0.192 | 0.497 | 0.086 | -0.688 | -0.277 | 0 |
| Daily | NASDAQ_recent | -0.616 | 0.497 | 0.499 | -1.113 | -1.115 | 0.000 |
| Joint validation choice | NYSE_recent | -0.584 | 0.571 | 0.125 | -1.156 | -0.710 | 0 |
| Joint validation choice | NASDAQ_recent | 0.379 | 0.621 | 0.626 | -0.242 | -0.247 | 1.000 |

### Controls for every annual window

| Horizon | Cohort | Year | ETF | Net SR | Net return | Ann. vol | Max DD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5.000 | NYSE_recent | 2022 | SPY | -0.853 | -18.3% | 24.4% | -23.4% |
| 5.000 | NYSE_recent | 2022 | NYSE basket — passive benchmark | -0.599 | -12.6% | 23.5% | -19.9% |
| 5.000 | NYSE_recent | 2023 | SPY | 1.738 | 27.3% | 13.1% | -10.0% |
| 5.000 | NYSE_recent | 2023 | NYSE basket — passive benchmark | 0.704 | 13.5% | 16.2% | -14.6% |
| 5.000 | NYSE_recent | 2024 | SPY | 1.987 | 30.1% | 12.6% | -8.4% |
| 5.000 | NYSE_recent | 2024 | NYSE basket — passive benchmark | 0.350 | 6.9% | 14.0% | -7.2% |
| 5.000 | NYSE_recent | 2025 | SPY | 0.926 | 20.2% | 19.7% | -18.8% |
| 5.000 | NYSE_recent | 2025 | NYSE basket — passive benchmark | 0.600 | 12.5% | 18.5% | -18.1% |
| 5.000 | NASDAQ_recent | 2022 | SPY | -0.853 | -18.3% | 24.4% | -23.4% |
| 5.000 | NASDAQ_recent | 2022 | QQQ | -1.153 | -31.9% | 32.3% | -32.4% |
| 5.000 | NASDAQ_recent | 2023 | SPY | 1.738 | 27.3% | 13.1% | -10.0% |
| 5.000 | NASDAQ_recent | 2023 | QQQ | 2.559 | 58.2% | 17.9% | -10.8% |
| 5.000 | NASDAQ_recent | 2024 | SPY | 1.987 | 30.1% | 12.6% | -8.4% |
| 5.000 | NASDAQ_recent | 2024 | QQQ | 1.616 | 34.3% | 18.0% | -13.6% |
| 5.000 | NASDAQ_recent | 2025 | SPY | 0.926 | 20.2% | 19.7% | -18.8% |
| 5.000 | NASDAQ_recent | 2025 | QQQ | 0.922 | 23.5% | 23.8% | -22.8% |
| 1.000 | NYSE_recent | 2022 | SPY | -0.862 | -18.7% | 24.3% | -24.5% |
| 1.000 | NYSE_recent | 2022 | NYSE basket — passive benchmark | -0.591 | -12.7% | 23.5% | -21.0% |
| 1.000 | NYSE_recent | 2023 | SPY | 1.605 | 25.6% | 13.1% | -10.0% |
| 1.000 | NYSE_recent | 2023 | NYSE basket — passive benchmark | 0.621 | 12.2% | 16.1% | -14.5% |
| 1.000 | NYSE_recent | 2024 | SPY | 1.710 | 26.4% | 12.6% | -8.4% |
| 1.000 | NYSE_recent | 2024 | NYSE basket — passive benchmark | 0.232 | 5.3% | 14.0% | -7.2% |
| 1.000 | NYSE_recent | 2025 | SPY | 0.735 | 16.4% | 19.5% | -18.8% |
| 1.000 | NYSE_recent | 2025 | NYSE basket — passive benchmark | 0.378 | 8.4% | 18.4% | -18.4% |
| 1.000 | NASDAQ_recent | 2022 | SPY | -0.862 | -18.7% | 24.3% | -24.5% |
| 1.000 | NASDAQ_recent | 2022 | QQQ | -1.161 | -32.4% | 32.2% | -34.0% |
| 1.000 | NASDAQ_recent | 2023 | SPY | 1.605 | 25.6% | 13.1% | -10.0% |
| 1.000 | NASDAQ_recent | 2023 | QQQ | 2.409 | 55.0% | 17.9% | -10.8% |
| 1.000 | NASDAQ_recent | 2024 | SPY | 1.710 | 26.4% | 12.6% | -8.4% |
| 1.000 | NASDAQ_recent | 2024 | QQQ | 1.352 | 28.9% | 18.0% | -13.6% |
| 1.000 | NASDAQ_recent | 2025 | SPY | 0.735 | 16.4% | 19.5% | -18.8% |
| 1.000 | NASDAQ_recent | 2025 | QQQ | 0.735 | 18.9% | 23.6% | -22.8% |

[Every model/year versus SPY and the cohort comparison benchmark](model_vs_spy_qqq_by_year.csv) | [Every pooled model versus SPY and the cohort comparison benchmark, including QQQ intervals](pooled_model_vs_spy_qqq.csv) | [Selection procedures](selection_procedures_vs_spy_qqq.csv)

QQQ tracks a different, more concentrated investment universe than SPY. Neither ETF is a matched-security benchmark for our small historical cohorts. Pooled intervals are exploratory and unadjusted for repeated research.
