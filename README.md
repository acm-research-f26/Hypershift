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
| [Comparison workbook](docs/NASDAQ100_strategy_comparison.xlsx) | Eight tabs, 346 distinct main strategies, SPY/QQQ and seed/cost diagnostics |
| [Documentation index](docs/README.md) | Current guides, dated study reports and historical material |
| [Source map](research/README.md) | Data, models, training, accounting, analysis, audits and publication code |
| [Project layout](docs/PROJECT_LAYOUT.md) | Commands, compatibility handling and frozen-source provenance |
| [Protocol index](config/README.md) | Dataset identities, experiment settings and the unactivated confirmation gate |

The strongest observed historical candidate is THINK+mixer AH+PM blended with a
restricted dynamic hypergraph network using fixed 50:50 ensemble percentile
ranks. At five holdings, the ten-seed version has 0.896 Sharpe / 31.87% CAGR;
the original five-seed half has 0.919 / 32.94%. These are different seed ensembles
on the same history. The master charts disclose the weaker results, seed halves,
matched benchmarks and confidence limits alongside the leaders.

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

Datasets, checkpoints and credentials remain local in ignored `data/`, `runs/`
and `.env`. See [Alpaca setup](docs/ALPACA_15MIN.md),
[dependency sets](requirements/README.md), [hardware](docs/TRAINING_HARDWARE.md)
and [evaluation commands](research/evaluation/README.md). Cloning the repository
does not supply the full market data, reference-source cache or trained checkpoints.

Unrelated cohort/NYSE studies, retired-universe reports, decks and scratch work
are preserved in [archive/](archive/README.md). Archived sources are excluded from
normal listings and tests; use `python -m hypershift --archived list` to opt in.
The master charts and workbook retain the active NASDAQ-100 comparison history.
