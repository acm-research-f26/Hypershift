# Hypershift

The 250-stock forecasting experiment is implemented in `experiments/market_ablation`.
Run `uv run python -m experiments.market_ablation --help` for its phase commands.
Use a fresh `--output local-experiments/runs/<name>` directory for each new run.
The compact [main ablation notebook](notebooks/main_ablation.ipynb) reads saved
results without repeating training.

Local experiment code, prompts, diagnostic reports, forecasts and checkpoints
belong in `local-experiments/`, which is excluded from Git. Shared source data
stays in `local-data/`. Keep reusable library code and tests in the source tree.

The bounded intraday strategy runner is in `experiments/strategy_search`:

```powershell
uv run --extra strategies python -m experiments.strategy_search all `
  --study-root <audited-frozen-forecast-study> `
  --market-root <prepared-market-dataset> `
  --output <private-run-directory> --workers 2
```

The catalog has ten hypothesis families and 66 configurations, tested at ten
one-way execution fees from zero to 5 bp. Normalizers, PCA, same-clock volume
baselines and forecast horizon calibration fit only the first 60 earlier
sessions. Those sessions are excluded from evaluation. Five development blocks
select fixed candidates before the later replay. `c_SR1` denotes the highest
tested fee with net daily Sharpe at least 1; development qualification also
requires median block Sharpe at least 1 and positive returns in both cohorts.
The runner records risk shadows and every completed trial, verifies resume
hashes, and produces JSON/CSV results and an HTML report under `--output`.

These are exploratory results: the later history was previously examined, the
universe and sector taxonomy are retrospective, and fills assume fractional
shares at recorded minute opens. There are no historical quotes, partial fills,
queue/participation constraints, market impact, or dollar-capacity estimates.
Cash earns zero interest; long/short runs assume available borrow. Stops use
completed closes and delayed execution. Full details are frozen in
`experiments/strategy_search/assumptions.json`; no strategy performance is
promised. Neural checkpoints stay frozen and sealed future outcomes are unused.

Push reusable source, tests, dependency metadata and this documentation. Keep
raw data, private paths, credentials, model checkpoints, forecast archives,
trial ledgers, reports, caches and synchronization state out of Git. Put run
outputs in the existing Google Drive sync folder; verify desktop sync separately
before claiming cloud delivery.

Closing-momentum execution validation lives in `experiments/strategy_validation`.
It preserves the frozen S10 baseline and separately evaluates one whole-share
implementation: 0.25% NAV net tolerance, 0.5% NAV minimum discretionary order,
unchanged signal rules, mandatory risk/closing orders, quote depth and 1% of the
previous completed minute's volume. The fee schedule is a versioned current-cost
scenario applied consistently to historical dates, with an explicit $0.003/share
taker-route assumption. Historical locates and live market impact remain unknown.

```powershell
uv run --extra validation python -m experiments.strategy_validation audit `
  --reference <frozen-strategy-search-run> --output <new-private-validation-run> `
  --archive <existing-Alpaca-1min-archive>
uv run --extra validation python -m experiments.strategy_validation replay `
  --reference <frozen-strategy-search-run> --output <same-validation-run> --download-quotes
uv run --extra validation python -m experiments.strategy_validation report `
  --reference <frozen-strategy-search-run> --output <same-validation-run>
uv run --extra validation python -m experiments.strategy_validation shadow `
  --reference <frozen-strategy-search-run> --output <same-validation-run> --session YYYY-MM-DD
```

The quote adapter uses only Alpaca market-data endpoints and existing environment
credentials; it never submits brokerage orders. Snapshots are stored locally,
hashed and reused. Historical quote sizes before November 3, 2025 are converted
from round lots to shares. Nanosecond timestamps remain integers. A missing quote
never falls back to a minute-open execution. `--mode bar_proxy` is a separately
labeled diagnostic, not an execution-cost validation.

`shadow` without a date initializes a prospective freeze. Supply each consecutive
XNYS session after close plus 20 minutes; the first 20 sessions are risk warmup,
followed by 60 evaluation sessions. This is delayed-SIP forward replay, not a
live execution or broker paper-fill test. Corporate-action days are recorded and
affected symbols excluded. Use `--env-file` to locate credentials without putting
keys on the command line. No scheduler is installed by these commands.
