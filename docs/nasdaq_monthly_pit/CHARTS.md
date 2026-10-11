# Historical monthly NASDAQ-100 charts, 2022–2025

**Original batch — Completed: 120 neural fits, 120 continuous portfolio paths; independent audit passed.**
QQQ returned 11.9% CAGR; SPY returned 10.9%. No neural ensemble beat QQQ in every calendar year. No neural variant had all five individual seeds exceed QQQ’s full-period CAGR.

Original top-10 comparison: THINK + mixer has the highest primary net CAGR (19.6%) and Sharpe (0.629), but is **not a definitive best model**. Its maximum drawdown is -39.4%. Its Sharpe lead over THINK has an exploratory 95% interval [-0.331, 0.758], which includes zero; only three of five individual seeds beat QQQ, and the ensemble trails QQQ in 2024. Higher return in this inspected sample does not establish reliable future superiority.

This uses the user-authorized **official NASDAQ-100 fallback**, selected at each
historical month start, rather than the unavailable custom 100-largest-NASDAQ-company
ranking. Financial companies are excluded. Every included share class is retained
(101–104 securities); the 2026-selected NASDAQ models are retired and deleted.


## Top-5/top-10/top-20 and pseudo-Huber follow-up

**Completed: 20 additional pseudo-Huber fits; 204 compared portfolio paths independently audited.**
The user substituted the already-available top 20 for top 15. All 120 parent
paths are unchanged; 84 additional paths cover the new loss and the old neural
models' individual seeds at top 5 and top 20. No old model was refitted.

The new **THINK + mixer pseudo-Huber** uses exactly the existing mixer
architecture, features, purged train/validation folds, five seeds and training
settings. Pseudo-Huber is a smooth robust loss: near small errors it behaves
like squared error; large errors have bounded gradient influence. Its fixed
scale is **one percentage point** of return, matching the previous Huber scale.
We use twice the usual pseudo-Huber loss to match MSE curvature near zero:
`2 × δ² × (sqrt(1 + (error/δ)²) − 1)`. Epoch selection still uses prior-year
validation MSE. All 20 fits were locked before these new financial metrics.

Top 5 targets about **20% per name**, top 10 **10%**, and top 20 **5%**. Rebalancing
remains every **five sessions**, with frozen signal-time orders filling at the
next close, the same cash account, costs and liquidity limits. ETF and monthly
basket control policies stay the same at every K.

At the unchanged top-10 policy, pseudo-Huber has net CAGR 11.4%, Sharpe 0.412, and maximum drawdown -40.7%. MSE THINK + mixer has 19.6% CAGR and 0.629 Sharpe; Huber THINK + mixer has 6.0% CAGR and 0.245 Sharpe. Changing K changes concentration and turnover, not the forecasts or fitted coefficients. All three sizes are reported; none is selected as a new primary strategy from these results.

Highest observed CAGR at each size: top 5: StockMixer (22.5% CAGR); top 10: THINK + mixer (19.6% CAGR); top 20: THINK + mixer (14.1% CAGR). These are descriptive leaders, not statistically established winners.

Pseudo-Huber top 5 has 22.4% CAGR but a -43.6% maximum drawdown. Individual seed CAGRs range from 3.0% to 22.9%, with median 6.1%; 2/5 seeds beat QQQ. Its Sharpe advantage over QQQ has exploratory 95% interval [-0.281, 0.704]. An ensemble combines rankings and its return need not equal the mean or median individual-seed return. The larger ensemble return is not evidence of reliable performance across training seeds.

Cells below show **net CAGR / Sharpe / maximum drawdown**:

| Model | Top 5 | Top 10 | Top 20 |
| --- | --- | --- | --- |
| THINK | 13.6% / 0.461 / -34.1% | 12.3% / 0.435 / -29.0% | 9.5% / 0.354 / -33.9% |
| THINK + mixer | 11.9% / 0.415 / -46.2% | 19.6% / 0.629 / -39.4% | 14.1% / 0.503 / -32.5% |
| THINK Huber | 13.1% / 0.453 / -38.5% | 13.7% / 0.475 / -36.7% | 12.3% / 0.445 / -34.0% |
| THINK + mixer Huber | 15.8% / 0.527 / -44.3% | 6.0% / 0.245 / -41.7% | 9.0% / 0.350 / -33.5% |
| MLP | -6.8% / -0.046 / -40.3% | -10.1% / -0.208 / -44.5% | -1.8% / 0.012 / -37.8% |
| StockMixer | 22.5% / 0.659 / -36.1% | 8.6% / 0.326 / -34.0% | 9.2% / 0.347 / -27.7% |
| Ridge | -13.7% / -0.260 / -56.5% | -5.0% / -0.058 / -39.1% | 5.1% / 0.216 / -34.4% |
| Momentum 126 | 8.6% / 0.327 / -40.9% | 2.9% / 0.145 / -38.4% | 4.7% / 0.191 / -31.1% |
| Reversal 5 | 5.9% / 0.259 / -43.4% | 5.8% / 0.246 / -35.8% | 5.9% / 0.240 / -35.7% |
| THINK + mixer pseudo-Huber | 22.4% / 0.689 / -43.6% | 11.4% / 0.412 / -40.7% | 9.3% / 0.358 / -36.5% |
| Monthly equal basket | 6.9% / 0.279 / -30.5% | 6.9% / 0.279 / -30.5% | 6.9% / 0.279 / -30.5% |
| SPY | 10.9% / 0.498 / -24.6% | 10.9% / 0.498 / -24.6% | 10.9% / 0.498 / -24.6% |
| QQQ | 11.9% / 0.469 / -34.9% | 11.9% / 0.469 / -34.9% | 11.9% / 0.469 / -34.9% |

All-model wealth charts share one linear vertical scale across K. The annual
charts rebase the continuous accounts to $1 for display, with one vertical
scale across all years and sizes. They do not restart trading accounts.

### Equal top 5

![All models: equal top 5](topk_pseudohuber/figures/wealth_top5.png)

![Year-by-year: equal top 5](topk_pseudohuber/figures/years_top5.png)

| Model | 2022 net return | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| THINK | -15.1% | 73.8% | -4.3% | 17.7% |
| THINK + mixer | -35.1% | 40.0% | 0.8% | 70.7% |
| THINK Huber | -25.5% | 87.8% | -9.3% | 28.8% |
| THINK + mixer Huber | -33.4% | 41.7% | 28.3% | 47.8% |
| MLP | -21.1% | 18.9% | -6.7% | -13.7% |
| StockMixer | -6.0% | 34.3% | 67.0% | 6.4% |
| Ridge | -43.4% | 23.7% | -1.6% | -19.2% |
| Momentum 126 | -21.4% | 29.8% | -11.6% | 53.9% |
| Reversal 5 | -34.2% | 19.8% | -11.7% | 80.6% |
| THINK + mixer pseudo-Huber | -30.9% | 41.5% | 42.6% | 60.3% |
| Monthly equal basket | -23.8% | 36.3% | 7.8% | 16.5% |
| SPY | -18.7% | 26.2% | 24.9% | 17.6% |
| QQQ | -33.3% | 54.9% | 25.6% | 20.7% |

### Equal top 10

![All models: equal top 10](topk_pseudohuber/figures/wealth_top10.png)

![Year-by-year: equal top 10](topk_pseudohuber/figures/years_top10.png)

| Model | 2022 net return | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| THINK | -9.2% | 56.5% | -11.7% | 26.5% |
| THINK + mixer | -28.0% | 58.6% | 10.6% | 61.3% |
| THINK Huber | -19.6% | 63.4% | -5.2% | 33.7% |
| THINK + mixer Huber | -31.1% | 49.7% | 2.8% | 19.2% |
| MLP | -24.6% | 6.0% | -10.0% | -9.1% |
| StockMixer | -17.0% | 30.0% | 12.5% | 14.7% |
| Ridge | -30.1% | 27.0% | -4.1% | -4.3% |
| Momentum 126 | -26.2% | 19.9% | 0.4% | 26.4% |
| Reversal 5 | -28.1% | 41.0% | -13.0% | 41.9% |
| THINK + mixer pseudo-Huber | -29.1% | 51.8% | 8.6% | 31.5% |
| Monthly equal basket | -23.8% | 36.3% | 7.8% | 16.5% |
| SPY | -18.7% | 26.2% | 24.9% | 17.6% |
| QQQ | -33.3% | 54.9% | 25.6% | 20.7% |

### Equal top 20

![All models: equal top 20](topk_pseudohuber/figures/wealth_top20.png)

![Year-by-year: equal top 20](topk_pseudohuber/figures/years_top20.png)

| Model | 2022 net return | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| THINK | -18.0% | 62.6% | -1.3% | 8.9% |
| THINK + mixer | -20.3% | 44.6% | 8.8% | 34.8% |
| THINK Huber | -15.8% | 59.0% | -2.5% | 21.5% |
| THINK + mixer Huber | -23.8% | 40.8% | 13.5% | 15.8% |
| MLP | -32.4% | 32.9% | -7.0% | 11.4% |
| StockMixer | -15.3% | 31.6% | 5.9% | 20.3% |
| Ridge | -25.1% | 43.6% | 6.9% | 5.8% |
| Momentum 126 | -26.0% | 25.2% | 5.0% | 23.5% |
| Reversal 5 | -25.7% | 37.1% | -12.1% | 40.1% |
| THINK + mixer pseudo-Huber | -26.6% | 42.1% | 13.8% | 20.0% |
| Monthly equal basket | -23.8% | 36.3% | 7.8% | 16.5% |
| SPY | -18.7% | 26.2% | 24.9% | 17.6% |
| QQQ | -33.3% | 54.9% | 25.6% | 20.7% |

### Training-seed stability across portfolio sizes

![Every seed at every size](topk_pseudohuber/figures/seed_stability.png)

| Model | K | Seed CAGR min | Median | Max | Seeds CAGR > QQQ |
| --- | --- | --- | --- | --- | --- |
| THINK | 5 | -1.2% | 3.4% | 11.0% | 0/5 |
| THINK + mixer | 5 | 5.3% | 14.7% | 21.7% | 3/5 |
| THINK Huber | 5 | 0.3% | 7.8% | 9.1% | 0/5 |
| THINK + mixer Huber | 5 | 1.4% | 6.6% | 13.7% | 1/5 |
| MLP | 5 | -8.9% | -6.8% | -2.1% | 0/5 |
| StockMixer | 5 | 1.6% | 7.1% | 11.1% | 0/5 |
| THINK + mixer pseudo-Huber | 5 | 3.0% | 6.1% | 22.9% | 2/5 |
| THINK | 10 | 7.5% | 8.0% | 13.3% | 1/5 |
| THINK + mixer | 10 | 8.5% | 12.4% | 21.2% | 3/5 |
| THINK Huber | 10 | 5.3% | 9.1% | 13.3% | 1/5 |
| THINK + mixer Huber | 10 | 1.6% | 9.8% | 20.8% | 1/5 |
| MLP | 10 | -8.5% | -5.6% | -4.9% | 0/5 |
| StockMixer | 10 | 1.2% | 6.5% | 9.6% | 0/5 |
| THINK + mixer pseudo-Huber | 10 | 3.2% | 10.4% | 21.6% | 2/5 |
| THINK | 20 | 6.8% | 8.4% | 10.4% | 0/5 |
| THINK + mixer | 20 | 2.8% | 8.1% | 13.8% | 2/5 |
| THINK Huber | 20 | 8.8% | 10.4% | 12.6% | 1/5 |
| THINK + mixer Huber | 20 | 3.8% | 6.9% | 14.3% | 1/5 |
| MLP | 20 | -3.8% | -1.9% | 0.9% | 0/5 |
| StockMixer | 20 | 6.1% | 8.5% | 10.8% | 0/5 |
| THINK + mixer pseudo-Huber | 20 | 3.4% | 8.3% | 16.1% | 1/5 |

Five seeds share the same market history; their range measures sensitivity to
training initialization, not five independent market trials. Full 2×/4× cost
stress results remain visible in the linked metrics file.

Pseudo-Huber ensemble net CAGR under the declared cost stresses:

| K | Base costs | 2× costs | 4× costs |
| --- | --- | --- | --- |
| 5 | 22.4% | 19.6% | 14.3% |
| 10 | 11.4% | 9.3% | 5.2% |
| 20 | 9.3% | 7.7% | 4.6% |

### Pseudo-Huber forecast quality and uncertainty

| Model | IC | RankIC | ICIR | RankICIR | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| THINK + mixer | 0.0125 | 0.0074 | 0.0966 | 0.0569 | 0.5316 | 0.5392 | 0.5553 |
| THINK + mixer Huber | 0.0062 | 0.0038 | 0.0511 | 0.0311 | 0.5310 | 0.5300 | 0.5494 |
| THINK + mixer pseudo-Huber | 0.0077 | 0.0045 | 0.0630 | 0.0365 | 0.5403 | 0.5377 | 0.5523 |

The same forecasts drive every K, so IC and RankIC do not change merely because
we buy a different number of names. These diagnostics average 200 full
five-session periods. Pseudo-Huber's NDCG@10 is 0.5377, versus
0.5176 expected for random ordering of the same observed stocks. NDCG is
relative ranking quality, not percentage accuracy or net portfolio return.

Exploratory **pseudo-Huber minus comparator** Sharpe differences:

| K | Comparator | Sharpe difference | 95% paired interval |
| --- | --- | --- | --- |
| 5 | THINK + mixer | 0.275 | [-0.122, 0.695] |
| 5 | THINK + mixer Huber | 0.162 | [-0.017, 0.341] |
| 5 | QQQ | 0.220 | [-0.281, 0.704] |
| 10 | THINK + mixer | -0.217 | [-0.582, 0.122] |
| 10 | THINK + mixer Huber | 0.167 | [0.057, 0.270] |
| 10 | QQQ | -0.057 | [-0.524, 0.323] |
| 20 | THINK + mixer | -0.145 | [-0.378, 0.091] |
| 20 | THINK + mixer Huber | 0.009 | [-0.077, 0.108] |
| 20 | QQQ | -0.111 | [-0.419, 0.189] |

Intervals use the same 1,000 paired circular 20-session block resamples as the
parent study and are not adjusted for multiple comparisons. An interval that
includes zero does not establish an advantage. The 2022–2025 years were already
inspected before this user-requested follow-up: this is exploratory research,
not fresh confirmation. Original public-data, membership and settlement limits
remain. No delta, epoch, seed or K was selected by these test returns.

The extension audit reproduced all 20 checkpoints and validation epochs,
independently checked ranking metrics for ten signals, and replayed all 204
share/cash paths. It confirmed all 120 parent paths unchanged; the largest NAV
difference was $0.00000006.

The audit caught a validation-export issue: the originally saved validation
arrays were from the final epoch, while the checkpoint and test forecasts were
from the selected best-validation epoch. Selected-checkpoint validation arrays
were regenerated separately, matched to the recorded best validation MSE,
and all test forecasts reproduced unchanged. Original fit files and hashes
are preserved. No model was retrained or selected by test returns to make this
correction. [Validation export correction](topk_pseudohuber/VALIDATION_EXPORT_CORRECTION.json).

[Extension protocol](topk_pseudohuber/PROTOCOL.json) · [Independent audit](topk_pseudohuber/AUDIT.json) ·
[All sizes, seeds and cost stresses](topk_pseudohuber/metrics.csv) · [All dated paths](topk_pseudohuber/curves.csv) ·
[Annual returns](topk_pseudohuber/annual.csv) · [Monthly returns](topk_pseudohuber/monthly.csv) ·
[All forecast diagnostics](topk_pseudohuber/ranking.csv) · [Forecast diagnostics by year](topk_pseudohuber/ranking_by_year.csv) ·
[Every forecast period](topk_pseudohuber/ranking_periods.csv) · [Paired intervals](topk_pseudohuber/paired_bootstrap.csv) ·
[Twenty new fits](topk_pseudohuber/training_records.csv) · [Input/code lock](topk_pseudohuber/INPUT_CODE_LOCK.json) ·
[Parent artifact lock](topk_pseudohuber/SOURCE_ARTIFACT_LOCK.json) · [Fit-completion lock](topk_pseudohuber/FITS_COMPLETE_LOCK.json) ·
[Completion record](topk_pseudohuber/completion.json)



## NASDAQ-100 historical monthly universe, 5-session forecasts

![All models: net wealth per initial $1](figures/continuous_accounts.png)

All six neural variants, three simple signals, the monthly basket, SPY and QQQ
share one linear wealth axis. Net wealth per initial $1 is the cumulative product
of one plus each daily net return, including entry and exit fees. The legend
shows daily net Sharpe for each curve.

The primary portfolio is fixed equal top 10, long only, rebalanced every five
sessions. Signals freeze at the prior close; share orders fill at the next close.
Accounts start with $1 million on 2022-01-03 and end with liquidation on
2025-12-31. Holdings and cash carry across annual model changes. ETF controls use
the same dates and entry/exit cost schedule. Neural primary results average
eligible-stock ranks from all five seeds; no best seed or best portfolio size is chosen.

**Fixed equal top 10** means the portfolio targets the ten highest-scoring
eligible securities at each rebalance, with 10% of signal-time account value
allocated to each, long only and without leverage. For a $1 million account,
that is a $100,000 target per security before fees and execution constraints.
"Fixed" refers to the predeclared count of ten, not ten permanently held names;
the selected names can change every five trading sessions. "Equal" means score
size does not change the target allocation. The strategy still buys the highest
scores even when all predicted returns are negative. Ties at the tenth-place
boundary split the remaining allocation fairly and can produce more than ten
positions. Prices move between the signal and fill; fees, liquidity caps and
cash limits can leave actual weights unequal or leave residual cash. Weights
also drift between rebalances. Top 5 and top 20 are sensitivity checks, not
choices made after seeing which returned the most.

| Model | Net CAGR | Sharpe | Max drawdown | Worst year | Total return |
| --- | --- | --- | --- | --- | --- |
| THINK | 12.3% | 0.435 | -29.0% | -11.7% | 58.7% |
| THINK + mixer | 19.6% | 0.629 | -39.4% | -28.0% | 103.9% |
| THINK Huber | 13.7% | 0.475 | -36.7% | -19.6% | 66.5% |
| THINK + mixer Huber | 6.0% | 0.245 | -41.7% | -31.1% | 26.2% |
| MLP | -10.1% | -0.208 | -44.5% | -24.6% | -34.7% |
| StockMixer | 8.6% | 0.326 | -34.0% | -17.0% | 39.1% |
| Ridge | -5.0% | -0.058 | -39.1% | -30.1% | -18.4% |
| Momentum 126 | 2.9% | 0.145 | -38.4% | -26.2% | 12.2% |
| Reversal 5 | 5.8% | 0.246 | -35.8% | -28.1% | 25.2% |
| Monthly equal basket | 6.9% | 0.279 | -30.5% | -23.8% | 30.4% |
| SPY | 10.9% | 0.498 | -24.6% | -18.7% | 50.7% |
| QQQ | 11.9% | 0.469 | -34.9% | -33.3% | 56.6% |

Sharpe uses daily net returns, 252 sessions/year and a fixed 3% annual hurdle;
CAGR uses compounded net returns. Drawdowns include initial cash. These are
research backtests, not verified executable returns or a future-return forecast.

## Forecast ranking: IC, RankIC and NDCG

| Model | IC | RankIC | ICIR | RankICIR | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| THINK | 0.0250 | 0.0206 | 0.1503 | 0.1164 | 0.5150 | 0.5257 | 0.5471 |
| THINK + mixer | 0.0125 | 0.0074 | 0.0966 | 0.0569 | 0.5316 | 0.5392 | 0.5553 |
| THINK Huber | 0.0234 | 0.0197 | 0.1488 | 0.1170 | 0.5179 | 0.5305 | 0.5520 |
| THINK + mixer Huber | 0.0062 | 0.0038 | 0.0511 | 0.0311 | 0.5310 | 0.5300 | 0.5494 |
| MLP | -0.0079 | -0.0108 | -0.0341 | -0.0451 | 0.4820 | 0.4944 | 0.5248 |
| StockMixer | 0.0228 | 0.0186 | 0.1379 | 0.1101 | 0.5206 | 0.5208 | 0.5446 |
| Ridge | -0.0005 | -0.0009 | -0.0027 | -0.0041 | 0.4811 | 0.5015 | 0.5321 |
| Momentum 126 | 0.0098 | 0.0139 | 0.0400 | 0.0563 | 0.5254 | 0.5275 | 0.5465 |
| Reversal 5 | 0.0118 | 0.0183 | 0.0567 | 0.0851 | 0.5044 | 0.5174 | 0.5446 |

These are equally weighted averages across **200 complete five-session forecast
periods** in 2022–2025, calculated across signal-time eligible stocks with
observed next-close-to-five-sessions-later returns. They are gross forecast
diagnostics; transaction costs enter the wealth chart, not these metrics.
Neural rows use the same five-seed average rank scores that drive the primary
portfolio, so their IC measures ensemble score/return correlation rather than
calibration of a predicted return magnitude. The final partial holding period
has no complete target and is excluded; 101 eligible stock/period targets in
total are unobserved, including that final period. SPY, QQQ and the monthly
equal basket do not produce cross-sectional stock forecasts, so these metrics
are not applicable to them.

- **IC (information coefficient):** Pearson correlation between the model's
  scores and subsequent realized returns within each forecast period.
- **RankIC:** Spearman correlation, using average ranks for ties. Both
  correlations range from -1 to +1; zero indicates no linear or rank association
  respectively. They describe the whole eligible stock ranking.
- **ICIR / RankICIR:** mean period correlation divided by its sample standard
  deviation. These are not annualized and are not significance tests.
- **NDCG@k:** normalized discounted cumulative gain for the first k predicted
  positions. Realized-return percentile ranks supply nonnegative linear gains;
  positions receive a 1/log2(position + 1) discount, with positions numbered
  from one. Prediction ties average their possible position discounts. One
  equals the hindsight ideal ranking for those outcomes. **0.539 does not mean
  53.9% correct predictions**: random orderings also receive positive gain.
  Even an all-negative-return period can have high NDCG because this metric
  judges relative ordering, not whether buying those stocks makes money.

THINK + mixer: IC 0.0125, RankIC 0.0074, NDCG@10 0.5392; its matched random-ranking expectation is 0.5176. Regular THINK has higher IC (0.0250) and RankIC (0.0206); THINK + mixer has the highest mean NDCG@10 among these nine signals. Its ranking advantage is small in absolute terms and has not been established as statistically significant. These metrics do not identify a universal winner.

The ranking summaries were independently recomputed from frozen predictions
and ridge coefficients using separate rank, correlation and discount
calculations. [Ranking audit](RANKING_AUDIT.json) ·
[Pooled ranking metrics](ranking.csv) · [Ranking metrics by year](ranking_by_year.csv) ·
[Every forecast period and matched random NDCG expectation](ranking_periods.csv).

## Year-by-year wealth charts

Each chart shows all twelve primary model and control paths for one calendar
year, with the same colors, line styles and vertical scale. Curves are rebased
to $1 at the start of each year so that within-year gains and losses can be
compared; this is display normalization of the existing continuous account.
Holdings and cash still carry across years. Each legend reports that year's
daily net Sharpe, and each curve's final value minus one equals the net annual
return in the table below. The four annual growth factors compound to the
full-period return shown above.

### 2022

![2022: all models, net wealth per $1 at year start](figures/accounts_2022.png)
### 2023

![2023: all models, net wealth per $1 at year start](figures/accounts_2023.png)
### 2024

![2024: all models, net wealth per $1 at year start](figures/accounts_2024.png)
### 2025

![2025: all models, net wealth per $1 at year start](figures/accounts_2025.png)

## Consistency by year, month and training seed

![Calendar-year returns](figures/annual_returns.png)

| Model | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| THINK | -9.2% | 56.5% | -11.7% | 26.5% |
| THINK + mixer | -28.0% | 58.6% | 10.6% | 61.3% |
| THINK Huber | -19.6% | 63.4% | -5.2% | 33.7% |
| THINK + mixer Huber | -31.1% | 49.7% | 2.8% | 19.2% |
| MLP | -24.6% | 6.0% | -10.0% | -9.1% |
| StockMixer | -17.0% | 30.0% | 12.5% | 14.7% |
| Ridge | -30.1% | 27.0% | -4.1% | -4.3% |
| Momentum 126 | -26.2% | 19.9% | 0.4% | 26.4% |
| Reversal 5 | -28.1% | 41.0% | -13.0% | 41.9% |
| Monthly equal basket | -23.8% | 36.3% | 7.8% | 16.5% |
| SPY | -18.7% | 26.2% | 24.9% | 17.6% |
| QQQ | -33.3% | 54.9% | 25.6% | 20.7% |

![Every month versus QQQ](figures/monthly_excess_qqq.png)

| Model | Months > QQQ | Years > QQQ | Months > SPY | Years > SPY | Positive years | Monthly QQQ excess SD |
| --- | --- | --- | --- | --- | --- | --- |
| THINK | 27/48 | 3/4 | 25/48 | 3/4 | 2/4 | 4.2% |
| THINK + mixer | 28/48 | 3/4 | 24/48 | 2/4 | 3/4 | 3.3% |
| THINK Huber | 27/48 | 3/4 | 23/48 | 2/4 | 2/4 | 3.7% |
| THINK + mixer Huber | 23/48 | 1/4 | 25/48 | 2/4 | 3/4 | 3.1% |
| MLP | 19/48 | 1/4 | 19/48 | 0/4 | 1/4 | 4.5% |
| StockMixer | 18/48 | 1/4 | 26/48 | 2/4 | 3/4 | 4.4% |
| Ridge | 20/48 | 1/4 | 18/48 | 1/4 | 1/4 | 5.0% |
| Momentum 126 | 19/48 | 2/4 | 17/48 | 1/4 | 3/4 | 4.5% |
| Reversal 5 | 20/48 | 2/4 | 24/48 | 2/4 | 2/4 | 5.1% |
| Monthly equal basket | 21/48 | 1/4 | 19/48 | 1/4 | 3/4 | 2.0% |
| SPY | 24/48 | 1/4 | 0/48 | 0/4 | 3/4 | 2.4% |
| QQQ | 0/48 | 0/4 | 24/48 | 3/4 | 3/4 | 0.0% |

Equal returns count as ties rather than wins. Monthly excess SD is measured in
percentage points; a low spread alone is not success if returns remain poor.
The basket is equal weighted at the first scheduled fill of each month; QQQ
uses index weights and actual fund returns, so they are different controls.

![All five individual training seeds](figures/seed_stability.png)

| Model | Seed CAGR: min / median / max | Seed Sharpe: min / median / max | Seeds CAGR > QQQ | Worst seed/year | Seed/year wins vs QQQ |
| --- | --- | --- | --- | --- | --- |
| THINK | 7.5% / 8.0% / 13.3% | 0.293 / 0.307 / 0.463 | 1/5 | -18.3% | 9/20 |
| THINK + mixer | 8.5% / 12.4% / 21.2% | 0.329 / 0.430 / 0.676 | 3/5 | -27.3% | 10/20 |
| THINK Huber | 5.3% / 9.1% / 13.3% | 0.228 / 0.340 / 0.465 | 1/5 | -16.2% | 7/20 |
| THINK + mixer Huber | 1.6% / 9.8% / 20.8% | 0.080 / 0.362 / 0.697 | 1/5 | -26.7% | 7/20 |
| MLP | -8.5% / -5.6% / -4.9% | -0.146 / -0.066 / -0.046 | 0/5 | -30.5% | 5/20 |
| StockMixer | 1.2% / 6.5% / 9.6% | 0.112 / 0.263 / 0.358 | 0/5 | -34.7% | 6/20 |

Seeds 7, 19, 42, 61 and 97 vary initialization within the same dated universe.
They are not five independent market histories. The earlier three random draws
tested cohort sensitivity separately and had substantial dispersion; see
[all earlier random-sample charts](../random_trials_20261010/CHARTS.md).
Those draws requested 100 names but recovered only 54–70 price histories, so
they are not complete historical random samples of 100. Neither averaging
seeds nor averaging draws removes a biased universe. Historical
membership addresses that selection problem; unseen forward data remain necessary.

## Fixed sensitivity checks and uncertainty

![All portfolio sizes and cost multipliers](figures/sensitivity.png)

Base costs are 7 basis points plus 3 basis points times the square root of
participation divided by 1%. Trades are capped at 1% of signal-time 20-session
average dollar volume. Cash-short buy orders resize proportionally; missing
fills cancel without reranking. Both costs and impact are multiplied in the
2×/4× stress paths. This remains a fractional-unit, adjusted-total-return account
approximation; auction fill quality, integer lots, taxes and cash interest are
not modeled. No alternative size or cost scenario becomes the primary result.

| Model | Comparison | Sharpe difference | Exploratory 95% interval | Zero check |
| --- | --- | --- | --- | --- |
| THINK | QQQ | -0.034 | [-0.585, 0.453] | includes zero |
| THINK + mixer | QQQ | 0.160 | [-0.262, 0.559] | includes zero |
| THINK Huber | QQQ | 0.006 | [-0.425, 0.420] | includes zero |
| THINK + mixer Huber | QQQ | -0.225 | [-0.663, 0.162] | includes zero |
| MLP | QQQ | -0.678 | [-1.309, -0.142] | excludes zero |
| StockMixer | QQQ | -0.144 | [-0.716, 0.344] | includes zero |
| Ridge | QQQ | -0.527 | [-1.111, -0.009] | excludes zero |
| Momentum 126 | QQQ | -0.325 | [-0.860, 0.229] | includes zero |
| Reversal 5 | QQQ | -0.223 | [-0.810, 0.353] | includes zero |
| Monthly equal basket | QQQ | -0.191 | [-0.492, 0.104] | includes zero |
| THINK + mixer | THINK | 0.194 | [-0.331, 0.758] | includes zero |
| THINK + mixer Huber | THINK Huber | -0.231 | [-0.697, 0.172] | includes zero |

The intervals use 1,000 paired circular 20-session block resamples of daily
returns. They preserve same-date pairing but are not adjusted for multiple
comparisons, data revisions or repeated use of these test years.

THINK + mixer versus THINK has a Sharpe difference of 0.194 with exploratory 95% interval [-0.331, 0.758]. The interval includes zero, so this study does not establish a consistent mixer advantage. THINK + mixer Huber versus THINK Huber has a Sharpe difference of -0.231 with exploratory 95% interval [-0.697, 0.172]. The interval includes zero, so this study does not establish a consistent mixer advantage.

Adding a mixer changes the learned cross-stock interactions and adds parameters;
it need not improve returns. Training minimizes forecast loss, while the traded
portfolio depends on the ranking near the top-10 boundary. Better forecast loss
does not automatically mean a better portfolio. The earlier NASDAQ/NYSE
difference cannot establish an exchange-specific advantage from one sample.

## What the models do

**126-session momentum** scores a stock as adjusted close today divided by
adjusted close 126 trading sessions ago, minus one: roughly six months of
total-return momentum. It has no fitted coefficients and does not skip the
latest month. Here it buys the ten highest available scores every five sessions.
Large gains can occur when recent winners keep winning; concentrated exposure
also loses when leadership reverses. The earlier spectacular result used
2026-selected survivors and a different annual-reset experiment, so it is not
directly comparable to this historical, continuous account.

**Ridge** is one shared linear return forecast fitted across the eligible
historical stocks and dates. Each stock contributes 16 sessions of five
features (80 numbers), plus an intercept. It forecasts the next five-session
return with squared-error loss and an L2 coefficient penalty of 10; the
intercept is unpenalized. Shrinkage controls large fitted weights. It does not
directly optimize Sharpe, CAGR or portfolio returns.

**THINK** is this repository's local hyperbolic temporal/hypergraph adaptation.
Its graph links stocks using positive correlations computed only on training
history and historical membership. **THINK + mixer** adds a learned cross-stock
residual to its scores. The **Huber** variants cap the influence of large
forecast errors beyond one percentage point; they still select epochs using
prior-year validation MSE. **MLP** applies a small shared nonlinear network to
each stock's feature window. **StockMixer** combines temporal, feature and
cross-stock mixing and includes a pairwise ranking penalty. These implementations
are controlled local comparisons, not certified reproductions of published papers.
**Reversal 5** ranks by negative recent five-session return and is unfitted.

All learned forecasts use daily return, distance from 5/20-session moving
averages, 20-session return volatility and dollar-volume relative to its
20-session average. Training-only normalization, monthly masks and inactive-node
sanitization prevent future members from entering active graph/mixer calculations.
Targets are next-close to five sessions later; outcomes crossing train/validation
boundaries are purged. Architecture and seed comparisons were declared before
these replacement test metrics; no live candidate is selected from this table.

## Data and audit limits

Training begins July 2019, a declared free-source amendment made before these
metrics. The four folds have 1.5, 2.5, 3.5 and four training years, respectively,
and the preceding calendar year for validation. The first fold has materially
less training history. Earlier prices only warm up features.

| Test year | Train forecasts | Validation forecasts | Test forecasts | Unobserved train / validation targets |
| --- | --- | --- | --- | --- |
| 2022 | 75 | 49 | 51 | 1 / 3 |
| 2023 | 125 | 49 | 50 | 4 / 0 |
| 2024 | 176 | 49 | 50 | 4 / 0 |
| 2025 | 200 | 49 | 50 | 3 / 0 |

Four unique supervised targets spanning older unpriced mergers are unobserved;
fold rows overlap, so those counts should not be summed into distinct targets.
They are excluded from loss without removing their signal-time members. The
last partial test holding period is valued and liquidated but has no full
five-session supervised target.

Membership is a pinned historical event reconstruction with annual closure
checks and Nasdaq annual-announcement cross-checks. It is not a fully licensed
as-published constituent database. The exact 2025 SOLS off-cycle dates conflict
across sources; both produce the same November monthly snapshot, and IPO warmup
keeps SOLS outside the forecasting portfolios. Monthly membership is intentionally
frozen even when the actual index changes within the month.

Tiingo supplies recovered former-member histories; Yahoo supplies continuing
prices. Reused/unidentified histories were rejected. No unexplained selected-stock
trading-price gaps remain. Vendor rows after acquisition were trimmed. XLNX,
SGEN, SPLK and ANSS use verified cash/share acquisition consideration; old
holdings become cash rather than being carried indefinitely at their last quote.
Stock consideration is sold at completion close with fees and cash is available
then: actual broker settlement can be later, and no settlement-delay stress was
run. Retrospectively adjusted public prices can still contain revisions/errors.

The audit reproduced **120 checkpoints**,
training-only scaling/graphs and ridge coefficients, independently replayed
**120 share/cash paths** and recomputed their
metrics. The largest NAV difference was **$0.00000006**.
Accounting reproduction checks arithmetic and timing; it does not certify every
provider fact. The 2022–2025 years were already inspected in earlier experiments,
so these results remain exploratory. Consistently good future performance still
requires an unseen period or paper trading with the frozen strategy.

[Data sources and acquisition terms](DATA_REQUIREMENTS.md) ·
[Frozen protocol](PROTOCOL.json) · [Independent audit](AUDIT.json) ·
[120-fit record](training_records.csv) · [All metrics/sensitivities](metrics.csv) ·
[Annual returns](annual.csv) · [Monthly returns](monthly.csv) ·
[Primary daily curves](primary_curves.csv) · [Paired intervals](paired_bootstrap.csv) ·
[Input/code fingerprints](INPUT_CODE_LOCK.json) · [Imported dependency fingerprints](DEPENDENCY_CODE_LOCK.json) ·
[2026 NASDAQ retirement record](../audit/nasdaq_2026_retirement.json)
