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
