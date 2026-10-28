# Saved THINK+mixer, reference-network and drawdown artifacts

This publication adds existing local research outputs and reviewed inputs to the
`feat-rs` branch. No training, inference, new forecasts or account replay was run.
The original run files are preserved byte for byte. The accompanying inventory,
forecast index and this guide were created on October 27, 2026.

The later raw-data supplement below adds the original daily histories, full
Alpaca bar snapshot and membership sources. It has a separate inventory so the
original publication manifest and saved experiment locks remain unchanged.

## Where to look

| Path | Saved contents |
| --- | --- |
| [`runs/ndx_reference_networks_20261024/`](../../runs/ndx_reference_networks_20261024/) | 120 checkpoints and 120 validation/test forecast exports: MASTER, restricted dynamic hypergraph and PRISM-VQ, four test years and ten seeds each; fit histories, source snapshots, integrity locks, model comparisons and aggregate account ledgers |
| [`runs/ndx_extended_hours_20261024/`](../../runs/ndx_extended_hours_20261024/) | 120 checkpoints and 120 forecast exports: daily, AH, and AH+PM THINK+mixer; annual preprocessing/scalers, histories, timing, rank scores and primary daily ledgers |
| [`runs/ndx_drawdown_20261026/batch1/`](../../runs/ndx_drawdown_20261026/batch1/) | Frozen drawdown recipes, all 915 base-account return paths, metrics, selected curves and selected daily ledgers, cost stress and audit records |
| [`data/nasdaq_monthly_pit/assembled/`](../../data/nasdaq_monthly_pit/assembled/) | Reviewed economic/raw closes, dollar volumes, historical membership, tradability, ETF prices/volumes and input manifest |
| [`data/nasdaq_monthly_pit/alpaca_features_20261019/`](../../data/nasdaq_monthly_pit/alpaca_features_20261019/) | Reviewed regular-session feature arrays, daily session summaries, coverage and rejected-session records |
| [`data/nasdaq_monthly_pit/extended_hours_features_20261024/`](../../data/nasdaq_monthly_pit/extended_hours_features_20261024/) | Reviewed AH/PM feature arrays, session summaries, coverage and rejected sessions |
| [`data/nasdaq_monthly_pit/jkp_themes_20261024/`](../../data/nasdaq_monthly_pit/jkp_themes_20261024/) | Existing PRISM factor input archive and provenance |
| [`data/nasdaq_monthly_pit/alpaca_15min/`](../../data/nasdaq_monthly_pit/alpaca_15min/) | Saved exchange calendar, calendar provenance, and full-period raw SIP snapshot, including all bar chunks and request/coverage metadata |
| [`.cache/architecture_references_20261024/`](../../.cache/architecture_references_20261024/) | Cached upstream architecture sources and revision receipts used by the reference adapters, with original licenses where present |
| [`.cache/reference_dependencies_20261024/`](../../.cache/reference_dependencies_20261024/) | Existing einops wheel and provenance receipt |

The model and preprocessing implementations are already tracked under
[`latest/`](../../latest/) and [`research/`](../../research/). Relevant published
charts and reports are in [`latest/results/`](../../latest/results/),
[`docs/ndx_extended_hours_20261024/`](../ndx_extended_hours_20261024/) and
[`docs/ndx_drawdown_20261026/`](../ndx_drawdown_20261026/).

## Reading forecasts and the blend

[`forecast_index.csv`](forecast_index.csv) lists every forecast file, model/view,
year, seed and matrix dimensions. Files use `<family>/<year>/<seed>_predictions.npz`
and contain `validation` and `test` matrices. Columns follow the 156-ticker order
in the reviewed feature NPZ files and assembled price CSVs. Predictions are model
outputs trained against returns in percentage points; use the eligibility mask
before comparing ranks.

The AH+PM annual `preprocessing.npz` files contain `train`, `val`, `test`, `mean`,
`std`, `aux_mean` and `aux_std`. The split arrays index the common forecast-sample
calendar, rather than raw daily-price rows. The saved `TIMING.csv` provides the
sample order and prior-close, decision, feature-cutoff and fill timestamps. For
example, from the repository root:

```python
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("runs/ndx_extended_hours_20261024")
with np.load(root / "after_hours_premarket/2022/7_predictions.npz") as saved:
    predictions = saved["test"].copy()
with np.load(root / "after_hours_premarket/2022/preprocessing.npz") as saved:
    sample_ids = saved["test"].copy()
with np.load("data/nasdaq_monthly_pit/extended_hours_features_20261024/features.npz") as saved:
    tickers = saved["tickers"].copy()
timing = pd.read_csv(root / "TIMING.csv").iloc[sample_ids]
assert predictions.shape == (len(timing), len(tickers))
```

The reference-network exports use the same annual validation/test folds, defined
by `pit_research_engine.fold`. Their reference adapter and preprocessing are in
`latest/training/ndx_reference_training.py`. THINK checkpoints also save their
historical groups. Reference-network standalone scaler and graph files were not
persisted: preprocessing and the static graph snapshot are reconstructed by the
preserved code from reviewed inputs; learned parameters/buffers remain in the
checkpoints. Per-date inferred dynamic graph matrices were not exported.

The drawdown runner's `saved_scores` function in
[`research/analysis/run_ndx_drawdown_research.py`](../../research/analysis/run_ndx_drawdown_research.py)
is the precise saved-forecast blend specification: rank each seed within the
eligible stocks, average ranks within each seed group, then combine THINK AH+PM
and the restricted dynamic network with fixed 50:50 weights. No new model fit is
needed to inspect those existing inputs and forecast matrices.

## Account-record limits

These are simulated research accounts. `ledgers.csv` in the reference run and
`primary_ledgers.csv` in the extended-hours run record daily aggregate NAV, cash,
holdings value, fees and traded/settled dollars. `holdings` is a dollar total,
not a per-stock positions table. Drawdown `selected_ledgers.csv` contains the
selected accounts' daily NAV, cash, fees, traded dollars and exposures.
`paths.npz` contains dates, account IDs and daily returns for all 915 base accounts.

The three requested folders do **not** contain persisted per-stock position
snapshots or order/fill blotters. Nor does the drawdown batch contain full daily
ledgers for every tested account. Those records cannot be supplied as existing
outputs; generating them would require an instrumented accounting replay. No
such replay was performed for this publication.

## Integrity and scope

[`manifest.csv`](manifest.csv) records the repository-relative path, byte size
and SHA-256 of every published existing artifact.
[`publication_inventory.json`](publication_inventory.json) records file counts,
validation scope and known omissions. The manifest excludes itself and these
new documentation files. All selected fit-completion hashes, per-fit hashes,
included historical input-lock entries, assembled-input hashes and drawdown
completion hashes were checked before staging.

Historical locks retain their original absolute Windows paths and original flat
source names. Use the relative paths in the new manifest to check bytes on another
machine. Older parent-run artifacts named in provenance locks remain outside this
publication; audit entry points that require those ancestors will still require
them. The raw 15-minute chunks are now included in the supplement below. This is not a claim
that every historical end-to-end audit can run from these three folders alone.

`.env`, credential files, Git metadata, Python bytecode and unrelated datasets
and runs are excluded. Selected files and decompressed archive members were
checked for local credential values and common token/private-key patterns.
The narrowly scoped Git attributes preserve artifact bytes across Windows
checkouts. Only the specified artifacts are force-added from otherwise ignored
local directories; other local outputs remain ignored.

The 2022–2025 results remain repeatedly inspected historical development results.
This artifact publication does not add prospective validation or new experiments.

## Raw-data supplement

The branch also preserves the following original local source directories:

| Path under `data/nasdaq_monthly_pit/` | Contents |
| --- | --- |
| `alpaca_15min/sip_raw_20180101_20260101/` | Full 2018–2025 SIP download: 1,288 symbol/year chunks, 10,839,262 bars, sidecars, original manifest, request, coverage and progress records |
| `yahoo_raw/` | Daily adjusted/raw closes, volume, dividends, splits, coverage, original manifest and download artifact |
| `tiingo_raw/` | Original daily price and issuer metadata responses, including former-member histories used by the assembled input manifest |
| `ndx_source/` | Pinned historical membership sources, source tree receipt and upstream license |
| `ndx_preflight/` | Original provisional membership table, event/identity review and source manifest |

[`raw_data_manifest.csv`](raw_data_manifest.csv) records repository-relative
paths, byte sizes and SHA-256 checksums for this supplement.
[`raw_data_inventory.json`](raw_data_inventory.json) records counts, historical
hash checks, credential-scan scope and remaining limits. Existing reviewed daily
and feature snapshots stay at the paths listed above. The small January 2025
Alpaca sample download and unrelated local datasets remain excluded.

From any checkout of this branch, verify the supplement without API credentials
or third-party Python packages:

```powershell
python docs/ndx_artifacts_20261027/verify_raw_data.py
```

The check reads files and compares hashes without executing cached source or
loading the original Yahoo pickle artifact. Use the CSVs to inspect daily data.
Git attributes preserve exact data bytes across Windows and Unix checkouts.
Historical manifests retain their original Windows path strings; use the new
relative-path manifest to verify files on another machine.

No data were downloaded again, transformed, re-reviewed or used for new training
in this supplement. Original raw-data status labels, identity/coverage limits and
missing bars remain as recorded. Historical audits that require omitted parent
runs still need those runs. Newly added files and decompressed gzip members were
checked for local credential values and common token/private-key patterns;
credentials remain excluded.
