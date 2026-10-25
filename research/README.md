# Research source map

This directory contains shared NASDAQ-100 research code and earlier active
NASDAQ-100 experiments. Direct sources for the latest dynamic-hypergraph study
are grouped under [latest](../latest/README.md). The active experiments use
historical monthly NASDAQ-100 membership, with daily prices for economic portfolio
accounting and Alpaca 15-minute bars for regular-session and extended-hours features.

| Directory | Responsibility |
| --- | --- |
| [data](data/README.md) | Acquire, identify, review and prepare data |
| [models](models/README.md) | Architectures, prediction adapters and losses |
| [core](core/README.md) | Chronological panels, execution, accounting and device support |
| [training](training/README.md) | Protocol locks, annual training and saved forecasts |
| [analysis](analysis/README.md) | Strategy comparisons, rank depth and sensitivity diagnostics |
| [evaluation](evaluation/README.md) | Checkpoint reproduction, accounting audits and publication seals |
| [reporting](reporting/README.md) | Publish charts, tables and the comparison workbook |

Run commands from the repository root through `python -m hypershift run <module>`.
Use `python -m hypershift list` to find a source. Executing nested historical
scripts directly changes their `__file__`-based data paths. The launcher preserves
their original imports and exact source hashes; see the
[layout guide](../docs/PROJECT_LAYOUT.md).

The [master charts](../charts.md) and
[comparison workbook](../docs/NASDAQ100_strategy_comparison.xlsx) are the current
cross-study result entry points. [Study reports](../docs/README.md) retain their
own dated protocols and evidence. Unrelated cohort and retired-universe sources
are now in [archive](../archive/README.md). Some shared modules, including the
original training helpers, remain here because active models import them.
