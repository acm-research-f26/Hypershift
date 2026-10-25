# Project layout

The research sources are grouped by purpose. Completed experiment sources were
moved byte-for-byte, with their original hashes recorded in
[the relocation manifest](../config/source_layout_20261024.json).
The [verification receipt](audit/source_layout_20261024.json) records all 154
moved-file hash checks, 173 passing tests and the unchanged experiment seals.
Each source category has its own README, indexed in the
[research source map](../research/README.md). Published guides and dated reports
are indexed separately in [docs/README.md](README.md).

```text
hypershift/             Workspace launcher and compatibility handling
research/
  core/                 Data panels, portfolio accounting and device helpers
  models/               THINK, mixers, losses and reference-network adapters
  data/                 Acquisition, membership, input review and preprocessing
  training/             Experiment runners, fit workers and selection locks
  analysis/             Ranking, model comparison and sensitivity analyses
  evaluation/           Accounting audits, saved-model checks and publication seals
  reporting/            Reports, plots, publication and workbook updates
tests/                  Existing research tests and launcher tests
requirements/           CUDA, acquisition and tested-environment dependency sets
config/                 Experiment protocols and source-layout provenance
docs/                   Research reports and the strategy comparison workbook
data/                   Ignored local datasets
runs/                   Ignored local checkpoints, predictions and audit artifacts
outputs/                Published comparison tables and artifact builders
archive/                Earlier preserved research
```

`README.md`, `charts.md`, the base `requirements.txt`, and `.env.example` stay at
the root. Credentials remain in the ignored root `.env` file. Existing artifact,
presentation and temporary directories were left in place.
The root [charts.md](../charts.md) and
[docs/NASDAQ100_strategy_comparison.xlsx](NASDAQ100_strategy_comparison.xlsx)
remain the current combined result artifacts. The
[gallery index](strategy_comparison_20261024/README.md) describes figure coverage;
the [output index](../outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/README.md)
describes comparison tables and workbook-build inputs.

## Running research commands

Run commands from the repository root using the existing virtual environment.
Use a module's original basename; `.py` is optional.

```powershell
# Find source files by category or name.
.\.venv\Scripts\python.exe -m hypershift list models
.\.venv\Scripts\python.exe -m hypershift list extended_hours

# Inspect or run an existing entry point.
.\.venv\Scripts\python.exe -m hypershift run run_ndx_reference_networks --help
.\.venv\Scripts\python.exe -m hypershift run seal_ndx_comparison ndx_reference_networks_20261024

# Run the full suite or specific test modules.
.\.venv\Scripts\python.exe -m hypershift test -v
.\.venv\Scripts\python.exe -m hypershift test test_ndx_extended_hours test_ndx_reference_models -v

# Use the original flat imports in a Python command or interactive session.
.\.venv\Scripts\python.exe -m hypershift python -c "from ndx_extended_hours import ExtendedModel; print(ExtendedModel.__name__)"
```

The ordinary test launcher keeps temporary test directories in `.cache/test_tmp`.
The runtime itself uses only Python's standard library. A different interpreter
can invoke it for reporting commands that need a separate dependency environment.

## Why the launcher is necessary

Many historical scripts import modules by their original flat names, derive the
project root from `__file__`, or verify source paths saved in completed experiment
locks. Rewriting these files during a cosmetic cleanup would change their hashes.

For the duration of a command, the launcher copies the organized sources to
their original root filenames and runs the child interpreter from the project
root. These are actual byte-identical files, so ordinary imports, subprocess
workers, `__file__` and strict source-hash checks continue to work. The original
locks, trained models and result files are not rewritten or bypassed.

The temporary copies disappear when the child exits, including nonzero exits.
The organized files in `research/` and `tests/` are authoritative: edit those
files, not temporary root copies. Files briefly appear at the root while a
research command is active; perform Git staging after it finishes.

Only one launcher owns this compatibility view at a time. A stopped command
that still has a running child keeps its files. After an interrupted child has
exited, use:

```powershell
.\.venv\Scripts\python.exe -m hypershift clean
```

Cleanup checks the recorded ownership, path and file hash. It never overwrites
an existing root file or removes an edited compatibility copy. If a temporary
copy was edited, compare it with its organized source, recover the desired
change there, and remove that recovered temporary copy before cleaning the rest.

## Adding or changing code

Add new scripts to the appropriate research folder and new tests to `tests/`.
The launcher discovers them automatically. Keep basenames unique because the
historical scripts use flat import names. Use the launcher for these historical
modules rather than executing their nested paths directly.

Ordinary code changes still trigger the existing frozen-run hash protections.
The reorganization does not authorize changing a completed experiment's source
locks or results. The relocation manifest records the initial move; it is not a
replacement for experiment input/code locks.

## Dated reports and old commands

Hash-sealed reports retain their published bytes and their study-specific dates,
universes and statistics. They can contain original flat-path command examples.
From the repository root, translate `python script.py ...` to
`python -m hypershift run script ...`, and `python -m unittest ...` to
`python -m hypershift test ...`. The launcher also restores the historical
dependency filenames while a command runs. For direct installation, use their
current `requirements/` paths.

Unsealed historical guides now state their scope and use current launcher
commands. New directory READMEs describe only their own sources or artifacts;
they link to the master charts and workbook for the combined current comparison.
Preserved archives and existing slide decks describe earlier work.
