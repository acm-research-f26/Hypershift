# Moving toward hourly and live direction forecasts

We successfully downloaded a real hourly research snapshot for the existing 12 stocks. This gives us data for an hourly model, but it does not turn the previous monthly classifier into one or establish that hourly prices are predictable.

## What “immediately” can mean

There are three separate clocks:

1. **Observation latency:** how long it takes a trade or quote to reach you.
2. **Inference latency:** how long an already-trained model takes to produce a score from those observations.
3. **Forecast horizon:** how far into the future that score refers to.

A model can be trained offline, then called every time new data arrives. It does not need to retrain before every prediction. But it always predicts an uncertain future outcome: it cannot immediately know whether a stock will rise.

For a first intraday model, a precise question is: **using information available at 10:30 New York time, will the price at 11:30 be higher?** “Is the price going up right now?” instead asks about an already observed movement. “Will the next trade be up?” is another, much shorter-horizon problem requiring trade/quote data rather than hourly summaries.

## What was actually downloaded

The snapshot request ran on 2026-09-28 at 16:42:47 UTC. It requested one year of regular-session, one-hour OHLCV observations from Yahoo Finance through yfinance, for AAPL, MSFT, GOOGL, NVDA, JPM, BAC, GS, MS, XOM, CVX, COP and SLB.

- Provider timestamps span 2025-09-29 13:30 UTC through 2026-09-28 16:30 UTC. The final provider bar was still forming when retrieved.
- There are 1,737 distinct provider timestamps across the downloaded panel; one timestamp has incomplete or invalid observations.
- The usable close panel contains 1,485 full-hour timestamps after filtering.
- There are 1,234 pairs of consecutive eligible hours, giving **14,808 stock/hour outcome labels**.
- We retained the original provider OHLCV file, saved the filtered close panel separately, and recorded SHA256 file hashes. Missing observations are not forward-filled.

See `data/hourly/metadata.json`, `provider_ohlcv.csv`, `full_hour_closes.csv`, and `next_hour_outcomes.csv`. The outcomes file contains actual observed answers and prices, **not model forecasts**. No hourly classifier has been trained in this step.

The [yfinance source](https://github.com/ranaroussi/yfinance/blob/main/yfinance/scrapers/history.py) supports hourly history and includes a roughly 730-day hourly lookback. Available history can vary with provider behavior; the successful request above establishes what we actually obtained. Yahoo/yfinance is useful for research snapshots; this experiment makes no claim that it supplies a complete, guaranteed low-latency trading feed.

Here are actual AAPL outcomes from the saved snapshot, rounded for display. Times below are New York time; the files use UTC. The first price becomes known when the feature hour ends. The second is the answer an hourly model would try to predict.

| Date | Feature hour ends | Target hour ends | First close | Next close | Actual move |
|---|---|---|---:|---:|---:|
| 2026-09-25 | 13:30 | 14:30 | $340.1100 | $339.8599 | -0.0735% |
| 2026-09-25 | 14:30 | 15:30 | $339.8599 | $340.0200 | +0.0471% |
| 2026-09-28 | 10:30 | 11:30 | $340.8700 | $341.2700 | +0.1173% |

These small moves also explain why execution costs matter. A correct direction prediction can still produce a losing trade after costs.

## The timestamp trap

An hourly bar aggregates open, high, low, close and volume over an interval. Its timestamp normally names the **start** of that interval. A bar starting at 09:30 cannot be used in full at 09:30: its eventual high, low, volume and closing price are future information at that point.

Our file names both `feature_bar_start_utc` and `signal_available_utc`. For a full hour starting at 09:30, the latter is 10:30. The next-hour target closes at 11:30. This availability timestamp is a theoretical earliest time, not a measurement of when a provider delivered the data.

The downloader uses a conservative extra 20-minute completion buffer for the recent tail; that is our filtering choice, not an advertised Yahoo delay. The raw snapshot is kept so the decision is inspectable. It excludes the ordinary 15:30–16:00 half-hour bar, excludes each day's final supplied bar to avoid early-close partial bars, and only creates labels when consecutive retained bar starts are exactly one hour apart. Overnight, weekend and holiday transitions never become “one-hour” labels. A production pipeline should use the exchange calendar explicitly and track feed arrival timestamps and revisions.

We used `auto_adjust=False` and saved the supplied Close along with other fields. Corporate-action handling still needs a deliberate policy before training: split discontinuities must not be mistaken for predictive returns. The snapshot preserves the provider's conventions rather than asserting that these are an independently reconstructed tape of trades.

## What a live version would need

The intended pipeline would be:

`live trades/quotes -> timestamped features -> trained model -> P(up over chosen horizon) -> execution and risk rules`

For genuine streaming observations, a provider such as Alpaca exposes authenticated WebSocket channels for trades, quotes and minute bars, with distinct IEX, SIP and delayed-SIP feeds. Coverage and entitlements differ; streaming a limited feed is not the same as observing the consolidated market. Minute bars can also receive late-trade corrections. These distinctions are documented in [Alpaca's real-time stock-data API](https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data).

No broker connection or streaming subscription was configured here. A live-feed choice and credentials would be required for that step. A historically weak monthly model should not be presented as a live hourly predictor simply because its inference function runs quickly.

## How I would design the hourly experiment

1. **Fix an executable target.** Predict either the next full hour's direction or whether the expected move clears a declared cost band. Keep neutral/no-trade decisions explicit. If entering after a bar closes, evaluate using a subsequent executable price, not retrospectively filling at that already-known close.
2. **Rebuild the features for intraday behavior.** Include recent full-hour returns, high–low range, volume relative to the same hour on prior days, volatility, market/sector movement and time of day. Distinguish overnight gaps from hourly returns. Do not just relabel daily feature names as hourly ones.
3. **Fit groups using past data only.** Compare industry/corporate groups with trailing correlations. Freeze each fitted graph during its evaluation window, or rebuild it using only observations available at that time. Changing the time resolution can change both correlations and hyperbolicity.
4. **Use chronological, rolling tests.** Train on earlier months, validate on later months, then test on new months. Purge boundaries based on actual target timestamps. Never randomly split overlapping hourly windows across train and test.
5. **Keep the same baselines.** Compare up-rate, recent-direction persistence, logistic regression and a graph-free model with the GNN. More observations do not necessarily mean more independent information.
6. **Evaluate trading separately.** Include spread, fees, slippage, latency, turnover and a no-trade alternative. A 0.10% cost can be much more consequential over one hour than over one month. Classification accuracy alone cannot establish a profitable strategy.

For less-than-hourly decisions, minute bars could support a model updated each minute while still predicting the next hour. That is a different dataset and validation problem from updating once per completed hour. Sub-second forecasting calls for trade/quote microstructure, considerably better timing and execution modeling, and a different feature design.

## Reproduce the download

From the project folder:

```powershell
.\.venv\Scripts\python.exe download_hourly.py --period 1y --out data\hourly_new_snapshot
.\.venv\Scripts\python.exe -m unittest test_audit -v
```

The output directory must be new or empty. Request times change future snapshots, so use the saved snapshot and hashes when reproducing this analysis. The separate `THINK_AUDIT.md` explains the independent geometry investigation; those dataset diagnostics do not require training any forecasting network.
