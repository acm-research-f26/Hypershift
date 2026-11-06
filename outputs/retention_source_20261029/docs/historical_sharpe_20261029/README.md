# Rishi: current experiments and reproduction

Publication requested by the user October 29, 2026. Branch `tom-shlom`; pull it or merge it into your working branch without discarding local work. Existing reviewed Rishi daily/AH+PM inputs and checkpoints are still required for running account engines.

Start with `charts.md` for the labeled Rishi versus follow-up comparisons, then the protocols/reports under this directory. The best observed original-five retention/QQQ hedge is 1.731 Sharpe, 33.11% CAGR and -13.36% drawdown on spent 2022–2025 history; ten-seed 1.381. Initial individual-seed comparisons are substantially weaker, and deep attribution is ongoing. These are model/account simulations, not prospective confirmation or certified broker execution.

## Restore published experiment outputs

The checked export `data/historical_sharpe_exports_20261029/experiment_outputs.zip` includes the saved forecasts, targets, daily paths, detailed ledgers and the 48 newly trained objective-model fits from the current historical experiments, plus the compact intermediate inputs those scripts consume. It has 692 files and a SHA256 manifest. Use these commands from the repository root:

```powershell
.\.venv\Scripts\python.exe research/analysis/restore_historical_sharpe_exports.py
.\.venv\Scripts\python.exe research/analysis/restore_historical_sharpe_exports.py --restore
```

The first checks every archived payload and existing destination without writing. The second restores missing files; neither replaces a differing local result. Preserve a conflicting destination separately before restoring. Do not load checkpoints from untrusted origins; this bundle contains the locally produced research fits, with hashes. Older absolute paths in training lock records describe the source machine and are provenance, not commands; a different checkout may require rebuilding the local lock for a new experiment rather than altering published records.

## Run experiments

Original runs refuse to overwrite their destinations. After restoring, use a fresh explicitly named destination in a copy of the relevant runner for new experiments; preserve published outputs. Launch only one `hypershift` command at a time because compatibility source copies are temporary.

- `research/analysis/rishi_historical_selection.py`: top-k, confidence and retention targets.
- `research/analysis/rishi_historical_risk.py`: volatility targets/windows and smoothing.
- `research/analysis/rishi_historical_hedge.py`: QQQ hedge caps, costs and borrowing.
- `research/analysis/rishi_historical_seed_investigation.py`: all ten seeds plus three ensembles and six recipe/control accounts.
- `research/analysis/rishi_historical_objective_training.py`: matched 48-fit training experiment; preserve its frozen settings/results.
- `research/analysis/rishi_historical_objective_evaluation.py`: fixed 84-account objective comparison.

Each runner uses `python -m hypershift run MODULE_NAME` from the root. Read its committed protocol before extending it. Current reports: `hedge_batch1/REPORT.md`, `selection_batch1/REPORT.md`, `risk_batch1/REPORT.md`, `objective_accounts/REPORT.md`, `seed_batch1/REPORT.md`. The original GPU trace instructions and received-trace issue are at `../rishi_mechanism_20261027/SOURCE_GPU_HANDOFF.md` and `SOURCE_TRACE_RECEIPT_20261029.md`; exact trace script hash differs, and the numerical gate remains unresolved.
