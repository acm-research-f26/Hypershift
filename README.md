# Hypershift: a first stock-market GNN

Train a small graph neural network to predict each stock's total return over the next **21 trading sessions** (approximately one month),
then compare its forecasts with real historical outcomes and four baselines.
The main example uses **12 US stocks, 2018–2025 Yahoo Finance adjusted closes**.

Start with [the monthly model explained](MONTHLY_MODEL.md) or [the detailed tutorial](TUTORIAL.md), then open
[the monthly experiment](runs/monthly/REPORT.md) and
[daily/weekly/monthly comparison](runs/monthly_horizon_comparison/REPORT.md).
The latter includes actual numbers, model comparisons, and charts.

The monthly GNN reduces RMSE by **2.63%** versus predicting no change, but only
**0.116%** versus the network without a graph. It predicts an increase in every
test case. Its small advantage over the no-graph model reverses on the 18
nonoverlapping monthly periods. See the reports for the limits of this evidence.
Original [daily](runs/historical/REPORT.md), [weekly](runs/weekly/REPORT.md), and
[daily-versus-weekly](runs/horizon_comparison/REPORT.md) results remain available.

## Run it on Windows / PowerShell

Python 3.12 and a CPU are sufficient. The `.venv` in this workspace is already
installed. Run from this folder; activation is unnecessary:

```powershell
# Check the forecasting logic.
.\.venv\Scripts\python.exe -m unittest -v test_stock_gnn

# Train on the downloaded real price snapshot. Use a fresh output directory.
.\.venv\Scripts\python.exe stock_gnn.py --csv data\prices.csv --horizon 21 --out runs\my_monthly_run

# Turn its outputs into a readable report and plots.
.\.venv\Scripts\python.exe report_results.py --run runs\my_monthly_run

# Compare all three horizons on matching forecast dates.
.\.venv\Scripts\python.exe compare_horizons.py --monthly runs\my_monthly_run --out runs\my_monthly_comparison

# Apply the saved GNN to the final complete observation in a CSV.
.\.venv\Scripts\python.exe predict.py --checkpoint runs\monthly\gnn.pt --csv data\prices.csv
```

The final command uses the last date in the file, not today's live market.
Its 21-session prediction has no outcome in that input file. Use the report's
`test_predictions.csv` for historical predictions whose outcomes are known.

## Install on another machine

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-data.txt

# Downloads prices; requires an internet connection. The end date is exclusive.
.\.venv\Scripts\python.exe download_prices.py --start 2018-01-01 --end 2026-01-01 --out data\prices.csv
.\.venv\Scripts\python.exe stock_gnn.py --csv data\prices.csv --horizon 21 --out runs\monthly
.\.venv\Scripts\python.exe report_results.py --run runs\monthly
```

On macOS/Linux, use `python3 -m venv .venv` and replace
`.\.venv\Scripts\python.exe` with `.venv/bin/python`.
`requirements-tested.txt` records the exact packages used here. It is an
environment snapshot; the CPU PyTorch build comes from the CPU index above.

Data downloads and experiment folders refuse to overwrite existing outputs.
Choose a new `--out` for each new dataset or experiment.

The command-line default is now `--horizon 21`. Use `--horizon 1` for daily or
`--horizon 5` for weekly in a fresh output directory. The program trains a model for
the specified horizon; changing the prediction script does not convert an existing
daily checkpoint into a monthly one. Old daily and weekly checkpoints retain their original horizons.

Monthly training retains daily inputs, predicts compounded 21-session returns,
and excludes 21 forecast origins at split boundaries to prevent label overlap.
Reports show all daily-origin forecasts and an additional every-21st-origin check.
To regenerate a three-horizon comparison on another machine, first train daily
and weekly runs with `--horizon 1 --out runs/historical` and
`--horizon 5 --out runs/weekly`, respectively, on the same price snapshot.

An optional offline smoke test is also available:

```powershell
.\.venv\Scripts\python.exe stock_gnn.py --out runs\synthetic_demo
.\.venv\Scripts\python.exe report_results.py --run runs\synthetic_demo --ticker SYN00
```

Omitting `--csv` deliberately uses artificial prices, clearly labeled synthetic.
Use `--csv data\prices.csv` for the real historical experiment.

## Files

| File | Purpose |
| --- | --- |
| `TUTORIAL.md` | Step-by-step explanation, math, shapes, examples, and exercises |
| `MONTHLY_MODEL.md` | Monthly results, input factors, training explanation, limitations, and alternative models |
| `stock_gnn.py` | Data preparation, graph, neural network, baselines, training, evaluation |
| `download_prices.py` | Optional real historical adjusted-close downloader |
| `predict.py` | Reload a checkpoint and forecast from a supplied price history |
| `report_results.py` | Tables, actual-versus-predicted charts, and per-stock results |
| `compare_horizons.py` | Compare daily, weekly, and optional monthly skill on common dates |
| `test_stock_gnn.py` | Checks for future-data leakage, date alignment, and model behavior |
| `runs/historical/REPORT.md` | Completed historical experiment and interpretation |
| `runs/historical/test_predictions.csv` | Every held-out prediction, actual return, and implied price |
| `runs/weekly/REPORT.md` | Weekly forecast results and nonoverlapping checks |
| `runs/horizon_comparison/REPORT.md` | Daily-versus-weekly comparison relative to each baseline |
| `runs/monthly/REPORT.md` | Monthly forecasts, actual outcomes, and nonoverlap checks |
| `runs/monthly_horizon_comparison/REPORT.md` | All three horizons compared on the same forecast dates |

The `data/` and `runs/` folders are local generated artifacts, ignored by Git.
If sharing this project, regenerate the data or distribute it only under the
provider's applicable terms. The example stock universe was selected today;
the tutorial explains why this and historical data revisions limit the conclusions.
