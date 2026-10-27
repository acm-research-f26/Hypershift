# Hypershift stock ranking research

Hypershift compares stock-ranking models on historical monthly NASDAQ-100
membership, retaining official share classes and former members. Daily economic
price histories support portfolio accounting; Alpaca SIP 15-minute bars supply
regular-session, after-hours and premarket features. The custom historical
100-largest-NASDAQ-company universe remains deferred.

The current development comparison covers **2022–2025, 1,003 sessions**. These dates
have been inspected repeatedly; none of the historical leaders is a confirmed
outperformance result on a genuinely untouched period.

| Start here | Contents |
| --- | --- |
| [Latest dynamic-hypergraph study](latest/README.md) | Direct model code, training, protocols, tests and sealed summary results |
| [Master charts](charts.md) | Current historical leaders, uncertainty and the complete strategy gallery |
| [Comparison workbook](docs/NASDAQ100_strategy_comparison.xlsx) | Thirteen tabs, 472 main strategy accounts, comparison charts and seed/cost/borrow diagnostics |
| [Documentation index](docs/README.md) | Current guides, dated study reports and historical material |
| [Source map](research/README.md) | Data, models, training, accounting, analysis, audits and publication code |
| [Project layout](docs/PROJECT_LAYOUT.md) | Commands, compatibility handling and frozen-source provenance |
| [Protocol index](config/README.md) | Dataset identities, experiment settings and the unactivated confirmation gate |

The [own-model hedge study](docs/ndx_model_hedge_20261027/REPORT.md) tests 495 accounts
using raw and prior-year-calibrated return forecasts. Raw hedge triggers never fire;
calibrated signals hedge during 2023 and underperform their unhedged baselines.
A continuous partial-QQQ-hedge control has the highest observed Sharpe: 1.454 for
the original five seeds (27.79% CAGR / -14.56% drawdown), and 1.325 for ten seeds
(24.83% / -14.78%). These controls do not establish successful model timing or
prospective superiority. The latest charts and workbook preserve the weaker cases.

The October 26 [drawdown-control study](docs/ndx_drawdown_20261026/REPORT.md) keeps the
THINK+mixer AH+PM / dynamic-hypergraph rank blend and adjusts allocation/exposure.
Within that earlier grid, the highest original-five Sharpe is 1.199 with a 20% volatility target
(28.96% CAGR / -17.78% drawdown). The highest ten-seed Sharpe is 1.112 with
five-issuer selection and a 20% target (26.81% / -21.69%). A 25% target plus
issuer/trend control retains 28.70% ten-seed CAGR with 1.085 Sharpe and -22.98%
drawdown. The chart-first master comparison shows SPY/QQQ and separates the seed
ensembles. All 305 tested recipes, weaker seed results and uncertainty remain
disclosed; these are selected historical development findings.

## Working with the repository

The latest study is grouped under `latest/`; shared NASDAQ-100 code remains in
`research/`, with shared tests and dependencies in `tests/` and `requirements/`.
Run commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m hypershift list
.\.venv\Scripts\python.exe -m hypershift list latest
.\.venv\Scripts\python.exe -m hypershift run run_ndx_reference_networks --help
.\.venv\Scripts\python.exe -m hypershift test -v
```

The launcher temporarily supplies the original flat source paths for historical
imports and actual source-hash checks, then removes unchanged copies. Edit the
organized sources and stage changes after commands finish. Completed experiment
sources and locks remain byte-preserved; use new declared runs for new research.

The [saved-artifact guide](docs/ndx_artifacts_20261027/README.md) indexes the
committed reference-network, extended-hours and drawdown snapshots, including
their checkpoints, forecasts and reviewed inputs. Other datasets and run outputs
remain local in ignored `data/` and `runs/`; credentials remain in ignored `.env`.
See [Alpaca setup](docs/ALPACA_15MIN.md),
[dependency sets](requirements/README.md), [hardware](docs/TRAINING_HARDWARE.md)
and [evaluation commands](research/evaluation/README.md). Cloning the repository
supplies these published snapshots, but not all historical raw data or parent runs.

Unrelated cohort/NYSE studies, retired-universe reports, decks and scratch work
are preserved in [archive/](archive/README.md). Archived sources are excluded from
normal listings and tests; use `python -m hypershift --archived list` to opt in.
The master charts and workbook retain the active NASDAQ-100 comparison history.
