# Reproduce the October 31 continuation study

Use the `feat-rs` branch and run the commands from the repository root. See the
[findings](FINDINGS.md) for the reported results, the
[dependency sets](../../requirements/README.md) for setup, and the
[hardware guide](../TRAINING_HARDWARE.md) for CUDA training. The recorded run used
Python 3.12, PyTorch 2.14.0+cu130, NumPy 2.5.3, and pandas 3.0.6. Saved-forecast
account replay runs on CPU and uses the committed data without API credentials.

## Files

The [run directory](../../runs/ndx_consistency_retraining_20261031) contains all
320 continuation fits across both training arms, both model families, four
annual folds and twenty seeds. Under `fits/<arm>/<family>/<year>/<seed>/`,
`RECORD.json` identifies the selected raw and EMA epochs and the source
checkpoint. Each fit includes four epoch checkpoints (0–3), optimizer and RNG
states, validation forecasts, training history, initialization parity and the
selected raw and EMA forecasts. The `resume_checks/` directory retains the two
training resume diagnostics.

`PATHS.npz`, `TARGETS.npz`, and `ledgers/` hold daily returns, effective
targets, and compressed accounting traces keyed by account ID in `METRICS.csv`.
The earlier [beta baseline](../../runs/ndx_beta_robustness_20261030) supplies
the paths and metrics used for parity checks. `PROTOCOL.json` pins the reviewed
point-in-time daily and extended-hours inputs, source checkpoints, original
forecasts, code and inherited evidence by SHA-256. The exact protocol inputs
are included at their original paths, including the pinned `tom-shlom` exports
under `outputs/retention_source_20261029/`.

## Verify the published files

The fit and account locks cover the saved checkpoints, traces and account
outputs. The report manifest pins the protocol and both completion locks.

```powershell
@'
import hashlib
import json

import run_ndx_consistency_retraining as study

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def check(path, expected):
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, path

study.verify()
study.parent.verify()
study.parent.reserve.verify()
run = study.RUN
manifest = read(study.DOC / "MANIFEST.json")
check(run / "PROTOCOL.json", manifest["protocol_sha256"])
check(run / "FITS_COMPLETE.json", manifest["fit_lock_sha256"])
check(run / "ACCOUNTS_COMPLETE.json", manifest["account_lock_sha256"])
for lock, key in (("FITS_COMPLETE.json", "hashes"), ("ACCOUNTS_COMPLETE.json", "output_sha256")):
    for name, expected in read(run / lock)[key].items():
        check(run / name, expected)
reference = read(run / "REFERENCE_INPUTS.json")
baseline = study.ROOT / reference["baseline"]
check(baseline / "COMPLETE.json", reference["baseline_complete_sha256"])
for name, expected in reference["output_sha256"].items():
    check(baseline / name, expected)
print("Frozen inputs, fit and account locks, and reference paths verified.")
'@ | python -m hypershift python -
```

## Replay the accounts from saved forecasts

In a disposable checkout, move the completed run outside the checkout, then
copy back the fit artifacts and three files required by the evaluator. The
evaluator writes a fresh set of account outputs. The saved copy provides the
reference for comparison.

```powershell
$run = 'runs/ndx_consistency_retraining_20261031'
$saved = Join-Path ([System.IO.Path]::GetTempPath()) ("hypershift-sealed-" + [guid]::NewGuid())
Move-Item -LiteralPath $run -Destination $saved -ErrorAction Stop
New-Item -ItemType Directory -Path $run -ErrorAction Stop | Out-Null
foreach ($name in 'fits', 'PROTOCOL.json', 'FREEZE.json', 'FITS_COMPLETE.json') {
    Copy-Item -LiteralPath (Join-Path $saved $name) -Destination $run -Recurse -ErrorAction Stop
}
python -m hypershift run run_ndx_consistency_retraining --stage evaluate
```

Compare all daily returns and targets, plus the main and annual metrics and
validation tables. This comparison uses an absolute tolerance of `1e-12` for
returns and targets and allows for floating-point rounding in the CSV tables.

```powershell
@'
from pathlib import Path
import sys

import numpy as np
import pandas as pd

run, saved = map(Path, sys.argv[1:])
for name in ("PATHS.npz", "TARGETS.npz"):
    with np.load(run / name) as actual, np.load(saved / name) as expected:
        assert actual.files == expected.files
        for key in expected.files:
            if key == "dates":
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
for name in ("METRICS.csv", "PERIOD_METRICS.csv", "ACCOUNT_CHECKS.csv", "VALIDATION.csv"):
    pd.testing.assert_frame_equal(pd.read_csv(run / name), pd.read_csv(saved / name),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
print("All 810 accounts matched the saved paths, targets and tables.")
'@ | python - $run $saved
```

For a fresh three-epoch continuation on CUDA, start with another disposable
checkout and move the sealed run outside it. Run the `preflight`, `freeze`,
`train`, `evaluate` and `summarize` stages in the order shown in the
[study report](REPORT.md). The new run records its own environment and hashes.
