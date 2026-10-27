# Alpaca 15-minute historical stock data

The separate [October 24 extended-hours experiment](ndx_extended_hours_20261024/REPORT.md)
tests daily inputs, daily plus after-hours, and daily plus after-hours plus
premarket information. Its versioned features are in
`data/nasdaq_monthly_pit/extended_hours_features_20261024/features.npz`.
All three views use a 09:20 Eastern decision, with premarket bars ending by
09:15, and fill at that entry session's regular close. Earlier regular-session
feature snapshots and model results remain unchanged. Sparse trading is
represented explicitly; no member is dropped for missing extended-hours bars.

The full 2018–2025 SIP snapshot and matched 2022–2025 model evaluation are
complete. See the [audited report, charts, all metrics and confidence intervals](nasdaq_alpaca_20261019/REPORT.md).
Raw responses retain their price-only source status; the separate feature
snapshot passed the documented input gate before training.

The downloader is [fetch_alpaca_15min.py](../research/data/fetch_alpaca_15min.py).
Run it through the [workspace launcher](PROJECT_LAYOUT.md). It uses the existing `requests`,
`numpy`, and `pandas` dependencies. It only sends market-data GET requests.

Alpaca's paper trading URL is `https://paper-api.alpaca.markets/v2`. Historical
stock bars use a separate host:
`https://data.alpaca.markets/v2/stocks/bars`, with `timeframe=15Min`.
See the [official bars reference](https://docs.alpaca.markets/us/reference/stockbars).

## Credentials

A local `.env` file has been created in the project root. Paste both values:

```dotenv
APCA_API_KEY_ID=your_paper_account_key_id
APCA_API_SECRET_KEY=your_paper_account_secret_key
APCA_API_BASE_URL=https://paper-api.alpaca.markets/v2
ALPACA_DATA_FEED=sip
```

`.env` and `.env.*` are Git-ignored; `.env.example` is the reusable placeholder.
Keep actual credentials in the local file. The loader reads literal values without
executing shell substitutions; process environment settings take precedence.
Optional matching quotes are supported. Put comments on their own lines.

The downloader always sends credentials to the fixed HTTPS market-data endpoint;
`APCA_API_BASE_URL` is retained as paper-account configuration and is not used to
request data or place orders. HTTP response bodies and exception details are
suppressed on errors, and credentials are excluded from saved metadata.

## Run from the project root

Inspect the full default plan without credentials, network calls, or dataset writes:

```powershell
.\.venv\Scripts\python.exe -m hypershift run fetch_alpaca_15min --dry-run
```

After filling `.env`, first check a small period and the benchmark feeds:

```powershell
.\.venv\Scripts\python.exe -m hypershift run fetch_alpaca_15min --symbols AAPL MSFT SPY QQQ --start 2025-01-02 --end 2025-01-09
```

Download the full historical membership ticker union and SPY/QQQ:

```powershell
.\.venv\Scripts\python.exe -m hypershift run fetch_alpaca_15min
```

Default dates are January 1, 2018 inclusive through January 1, 2026 exclusive,
matching the daily study's price-history years. Dates are interpreted at New York
midnight with daylight saving time handled before conversion to UTC. The API's
inclusive end is set one microsecond before the requested exclusive boundary.

Default output:

```text
data/nasdaq_monthly_pit/alpaca_15min/sip_raw_20180101_20260101/
  REQUEST.json
  manifest.json
  coverage.csv
  bars/
    AAPL/
      2018-01-01_2019-01-01.csv.gz
      2018-01-01_2019-01-01.json
      ...
```

The small sample creates its own date-specific output folder. `--out PATH` selects
another directory. A populated directory with unrelated files is rejected. A
matching download resumes completed symbol/year chunks after verifying hashes;
interrupted chunks are fetched again. Every page token is followed, including
empty pages with continuation tokens. Repeated tokens fail rather than silently
truncating a history. Rate-limit and temporary server failures retry with bounded
backoff. `--requests-per-minute 150` is the default local pacing setting.

Chunks record bars, retrieval timestamps, request settings, row counts, and hashes.
The final coverage file includes zero-row chunks. Missing or failed data is not
replaced with present-day survivors. Fatal authentication, permission, API, or
schema errors stop the run; already completed chunks can be resumed after correction.
The manifest records an interrupted run rather than declaring it complete.

## Feed, symbols, and price adjustments

- **SIP is explicit by default:** consolidated US exchange data. IEX covers one
  exchange and differs substantially in volume and sometimes prices. The
  downloader never silently substitutes IEX. Use `--feed iex` only for an explicit
  separate experiment. Alpaca's FAQ states that historical SIP queries without
  a subscription must end at least 15 minutes in the past; actual account access
  is checked by the API. The script requires an end date at least 16 minutes old.
  See the [market-data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq).
- **Historical symbols:** by default, the union of original ticker spellings from
  `assembled/monthly_members.csv`, plus SPY/QQQ. `asof=-` disables Alpaca's automatic
  current-symbol remapping, so FB and META histories remain separately labeled.
  Membership remains monthly and is not inferred from data availability. Data
  acquisition can include pre-membership history for warmup; eligibility still
  needs the reviewed historical masks.
- **Identity remains a review gate:** disabling remapping does not itself certify
  issuer continuity, reused tickers, former-member coverage, or acquisitions.
  Use the study's reviewed security IDs and alias map before joining files.
- **Raw prices are the default:** preserve unadjusted OHLC, volume, VWAP and trade
  count for inspection. `--adjustment split` or `--adjustment all` creates a
  separately locked snapshot. Raw prices have split/dividend discontinuities;
  vendor-adjusted prices are retrospective. Neither is automatically a complete
  corporate-action/total-return account.

## Time semantics and model integration

Each CSV row contains `symbol`, `timestamp_utc`, `bar_end_utc`, `open`, `high`,
`low`, `close`, `volume`, `trade_count`, and `vwap`. Alpaca timestamps identify
the **start** of the bar. Its OHLCV values are usable only after the bar ends.
The explicit `bar_end_utc` is 15 minutes later. Invalid OHLC, nonfinite values,
duplicate timestamps, off-grid intervals and bars outside the request are rejected.

All returned sessions, including extended hours, are retained. There is no
forward fill or assumption of exactly 26 bars every trading date. A subsequent
regular-hours model must use an exchange calendar that handles holidays and
early closes. A normal full regular session has 26 fifteen-minute intervals.
Absent bars remain absent and require coverage review.

These outputs are labeled `price_only_unreviewed_not_training_inputs`.
The existing daily model loader and frozen backtests remain unchanged. Promoting
the dataset requires checking historical member coverage, issuer identity,
corporate actions and sessions, then building a separate intraday feature/label
pipeline with appropriate maturity purging and execution timing. Replacing a
daily file directly would change the meaning of every lookback and horizon.

Fifteen-minute observations can add intraday information and execution detail.
They do not by themselves guarantee better forecasts, narrower valid confidence
intervals, or more independent market years. Compare daily and intraday inputs
on the same held-out dates with consistent costs and portfolio rules.

## Verification

```powershell
.\.venv\Scripts\python.exe -m hypershift test test_alpaca_15min -v
```

Tests use mocked HTTP responses and synthetic bars. They exercise pagination,
rate limits, credential handling, exclusive date boundaries, DST, malformed data,
empty coverage, resume hashes, and a complete synthetic download. A live download
still requires your key ID and secret in `.env`.

## Authorized matched model experiment

The October 19 experiment is frozen in
`config/nasdaq100_alpaca_20261019.json`. It compares seven neural variants, five
seeds and four annual folds for each of two capacity-matched input views (280
fits), plus ridge and portfolio/ETF controls. Fifteen-minute regular-session
information augments the daily feature windows; the five-session prediction
target and reviewed continuous daily accounting remain fixed.

The actual inputs and training records are located here:

| Artifact | Location | Frequency / role |
| --- | --- | --- |
| Raw Alpaca bars | `data/nasdaq_monthly_pit/alpaca_15min/sip_raw_20180101_20260101/bars/` | 15-minute OHLCV/VWAP/trade counts, 2018–2025; compressed CSV per symbol/year |
| Reviewed intraday features | `data/nasdaq_monthly_pit/alpaca_features_20261019/daily_features.npz` | Eight regular-session channels per security/day, derived from those bars |
| Session review | `data/nasdaq_monthly_pit/alpaca_features_20261019/REVIEW.json` | Coverage, identity rules, raw-data and feature hashes |
| Quarantined identities | `data/nasdaq_monthly_pit/alpaca_features_20261019/rejected_symbol_sessions.csv` | Source sessions outside verified issuer/ticker date windows |
| Daily economic prices and universe | `data/nasdaq_monthly_pit/assembled/` | Daily adjusted prices/volumes, monthly historical membership, corporate-action inputs and ETF controls |
| Per-fold preprocessing | `runs/nasdaq100_alpaca_20261019/{view}/{year}/preprocessing.npz` | Exact train/validation/test origin indices and training-only normalization |
| Fitted models and forecasts | The same per-view/year run folders | `.pt` checkpoints, history JSON, validation/test predictions and integrity records |

Each model sees a 16-session window and predicts a five-session return. The
augmented view adds information from 15-minute bars to the daily channels;
daily economic labels and portfolio accounting still use the assembled panel.
Annual test folds cover 2022–2025, with prior training and validation periods
purged at label boundaries. The final report includes exact fold dates.

Literal tickers need date boundaries even with automatic remapping disabled.
The reviewed rules in `config/alpaca_symbol_identity_20261019.json` prevent
pre-June-2022 META ETF records and later reused FB records from being joined
to Facebook/Meta. Raw records remain intact, and the source citations and
exclusion inventory are saved with the feature review.

The full download can resume with a shared 190-requests/minute limit:

```powershell
.\.venv\Scripts\python.exe -m hypershift run fetch_alpaca_batch
```

Completed symbol/year chunks and model fits are hash-verified before reuse.
The independently resumable stages are:

```powershell
.\.venv\Scripts\python.exe -m hypershift run run_ndx_alpaca_training --view daily_control --workers 2
.\.venv\Scripts\python.exe -m hypershift run prepare_ndx_alpaca
.\.venv\Scripts\python.exe -m hypershift run run_ndx_alpaca_training --view intraday_augmented --workers 2
.\.venv\Scripts\python.exe -m hypershift run report_ndx_alpaca
.\.venv\Scripts\python.exe -m hypershift run audit_ndx_alpaca
.\.venv\Scripts\python.exe -m hypershift run publish_ndx_alpaca
```

`prepare_ndx_alpaca.py --cache-available` precomputes verified regular-session
summaries for already downloaded chunks without promoting incomplete data.
The complete download, coverage review, and source hashes are still required
before augmented fitting. Missing intraday sessions never remove members from
daily eligibility.

`run_ndx_alpaca_pipeline.py` waits for the separately launched download and daily
trainer, then executes the remaining stages. Its current state is recorded in
`runs/nasdaq100_alpaca_20261019/PIPELINE_PROGRESS.json`. It is a finite local
pipeline, not a recurring automation. Do not launch duplicate trainers or edit
frozen model code while it is running. Final publication requires reproduction
of all checkpoints, independent replay of every account, and raw-data integrity
checks. Publication is written to `docs/nasdaq_alpaca_20261019/REPORT.md`.
