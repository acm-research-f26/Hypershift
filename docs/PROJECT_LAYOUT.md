# Project layout: latest NASDAQ-100 study first

The latest completed dynamic-hypergraph/rank-blend study is grouped in
[latest/](../latest/README.md). Shared NASDAQ infrastructure and earlier active
NASDAQ-100 experiments remain under `research/`. Unrelated studies are archived.

```text
latest/
  models/               Reference-network adapters, losses and optimizer recipes
  training/             Latest annual fits, freezes and workers
  data/                 Pinned reference implementation/factor acquisition
  evaluation/           Frozen study seals
  reporting/            Latest reports and master-comparison integration
  tests/                Latest study parity, loss and ranking tests
  config/               Exact latest protocols and confirmation gate
  results/              Sealed summary tables, metadata and export hashes
  docs/                 Model explanation and study boundaries
research/               Shared NASDAQ code and earlier active comparisons
config/                 Shared NASDAQ protocols and layout manifests
tests/                  Shared tests and workspace compatibility tests
docs/                   Active NASDAQ reports, gallery and comparison workbook
outputs/                Cross-study tables and workbook-build inputs
hypershift/             Standard-library command launcher
requirements/           Dependency sets
archive/
  studies_20261025/     Unrelated source, protocols, reports and decks
  local_20261025/       Ignored old datasets, runs and scratch artifacts
  legacy/              Earlier archived tutorials and experiments
```

The root [charts.md](../charts.md) and
[docs/NASDAQ100_strategy_comparison.xlsx](NASDAQ100_strategy_comparison.xlsx)
retain their exact bytes and canonical locations. Active market data, checkpoints
and large ledgers stay in ignored root `data/`, `runs/` and `.cache/` locations.
These paths are part of frozen scientific provenance. Credentials stay in `.env`.

## Commands from the repository root

```powershell
python -m hypershift list latest
python -m hypershift run run_ndx_reference_networks --help
python -m hypershift test -v
python -m hypershift test test_ndx_reference_models test_ndx_reference_evaluation -v
python -m hypershift run seal_ndx_comparison ndx_reference_networks_20261024
python -m hypershift python -c "from update_ndx_new_findings_20261024 import audit; audit(write_outputs=False)"
```

Use the existing virtual environment, for example prefix commands with
`.\.venv\Scripts\python.exe`. The active suite discovers both shared and latest
study tests. Sources in the unrelated archive are excluded unless explicitly
selected:

```powershell
python -m hypershift --archived list archive
python -m hypershift --archived test test_huber_ranking test_largecap_retraining -v
```

Archive mode restores archived source imports, old protocols and root document
aliases. It does not restore large old datasets/runs or every old publication
path. See the [archive guide](../archive/studies_20261025/README.md) before a full
historical rerun. Normal commands remain focused on active NASDAQ work.

## Why the compatibility launcher remains

Historical sources import flat module names, derive root paths from `__file__`,
and verify actual source/config hashes. They were moved without rewriting their
bytes or scientific locks. During a command, the launcher supplies real temporary
root source copies and original `config/` filenames for relocated protocols.
It removes unchanged owned copies when the child exits, including nonzero exits.

Edit the canonical files in `latest/`, `research/` and `tests/`, rather than the
temporary copies. Stage Git changes after commands finish. Basenames must remain
unique. New source scripts in these locations are discovered automatically.
Ordinary source/protocol changes still invalidate frozen experiments; declare
new runs for new research. Dataset acquisition does not imply broker execution.

The launcher rejects collisions and concurrent commands. If an interrupted child
is still running, its temporary paths are kept. Once it exits, recover unchanged
copies with `python -m hypershift clean`. Edited temporary copies are preserved;
compare and recover their desired edits into canonical sources before removing
those copies and cleaning the remainder.

## Provenance and navigation

[study_layout_20261025.json](../config/study_layout_20261025.json) records current
moves, file hashes and protocol aliases. The
[latest layout receipt](audit/latest_layout_20261025.json) records verification.
The October 24 [manifest](../config/source_layout_20261024.json) and
[receipt](audit/source_layout_20261024.json) describe the previous layout, not a
requirement that relocated paths still exist today.

Frozen reports keep their original bytes and may contain historical relative
links. Use the archive index and manifest to locate moved evidence. Translate
old `python script.py` commands to `python -m hypershift run script` (with
`--archived` before `run` for archived scripts). Current guides, source maps and
indexes point at the organized locations. No model was retrained for this move.
