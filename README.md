# Hypershift stock ranking research

The active study is NASDAQ only, using the user-authorized historical NASDAQ-100
fallback at each month start, with every official share class. It excludes
financial companies. The original custom universe of the 100 largest
NASDAQ-listed companies remains deferred because complete historical caps and
publication dates could not be established with the free sources.

- [Historical monthly NASDAQ study and current status](docs/nasdaq_monthly_pit/CHARTS.md)
- [Historical data requirements and free-source findings](docs/nasdaq_monthly_pit/DATA_REQUIREMENTS.md)
- [Frozen active fallback protocol](config/nasdaq100_historical_monthly.json)
- [Deferred custom ranking protocol](config/nasdaq_monthly_pit.json)
- [Earlier random samples and every training seed](docs/random_trials_20261010/CHARTS.md)
- [Original sample charts](docs/CHARTS.md)

Historical monthly membership, Tiingo former-member prices and Yahoo continuing
prices have passed the declared research-grade input review. Dynamic model
masking, verified test-period acquisition consideration and continuous account
checks are implemented. All 120 neural fits and 120 continuous portfolio paths
are complete; checkpoint reproduction and independent accounting audits passed.
The linked charts page contains every declared model, seed and sensitivity.
The user-requested top-5/top-10/top-20 follow-up adds 20 THINK+mixer pseudo-Huber
fits. Its audit reproduced the new checkpoints and all 204 compared portfolio
paths, including 120 unchanged parent paths. The same charts page includes
all-model wealth charts, yearly breakdowns, seed ranges and ranking diagnostics.
The long/short follow-up reuses those forecasts to compare fixed 100% long,
75% long / 25% short and 50% long / 50% short allocations, with matched
cash controls, calendar-day borrow fees, cost stresses and seed sensitivity.
All 570 new signed accounts passed independent replay; the 30 parent
long-only ensemble paths are unchanged, giving 600 disclosed paths in total.
Its signed economic-price accounts assume short availability; they do not
establish that every historical borrow or broker execution was possible.

All 324 neural checkpoints and eight ridge models from the 2026-selected
NASDAQ study, plus their eight annual output folders, were deleted at the
user's request. [Retirement record](docs/audit/nasdaq_2026_retirement.json).
NYSE research is deferred; its prior outputs and aggregate historical audit
evidence are preserved. The 2026-selected NASDAQ results are no longer an
active benchmark or basis for architecture selection.

Consistency will be assessed alongside net return and downside: every seed,
monthly/annual excess return over SPY and QQQ, worst year, drawdown, turnover
and cost stress. Fixed equal top-10 allocation is primary. Predictable losses
are not a useful investment outcome. No outperformance is guaranteed.

Alpha Vantage supplied listings but its share response lacked publication
timestamps, and full/adjusted price requests were premium gated. Tiingo's free
API recovered old stocks; reused or unidentified symbols were rejected.
The price providers do not establish the custom largest-company rankings.

```powershell
.\.venv\Scripts\python.exe -m unittest test_pit_models test_pit_research_engine test_historical_ndx_audit test_historical_ndx_membership test_historical_nasdaq_universe test_research test_robust_research test_largecap_retraining test_huber_ranking
.\.venv\Scripts\python.exe run_historical_ndx.py
.\.venv\Scripts\python.exe audit_historical_ndx.py
.\.venv\Scripts\python.exe publish_historical_ndx.py
```

The run is frozen to its input/code hashes and resumes only matching fits.
The auditor and publisher require the complete batch. The old retraining launcher and
dataset name are blocked to prevent accidentally repeating the retired study.
The historical audit reproduced forecast timing and accounting before
retirement; it did not remove universe selection bias. Previously viewed
2022-2025 years remain exploratory even after better data are obtained.
