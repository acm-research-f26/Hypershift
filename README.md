# Hypershift: chronological stock-ranking research

The active comparison evaluates THINK and eight neural variants on **2024 and
2025**, using modern prices for fixed historical NYSE/NASDAQ research cohorts.
Every model is compared with **SPY, an S&P 500 ETF, on exactly matching dates**.

Start with [the modern report](runs/recent_2024_2025/REPORT.md),
[the graph collection](runs/recent_2024_2025/CHARTS.md), and
[all model/year results](runs/recent_2024_2025/yearly_comparison.csv).
The [research guide](ARCHITECTURE_RESEARCH.md) explains the protocol and limitations.

Public data recovered original-issuer histories for **68 NYSE-cohort and
67 NASDAQ-cohort securities** from the fixed 100-name samples. Two reused ticker
symbols were rejected. Unavailable histories leave **material survivorship bias**;
these samples are not full exchanges or verified point-in-time index universes.

## Active workflow

```powershell
# Acquire once; a completed source snapshot is never overwritten.
.\.venv\Scripts\python.exe fetch_recent_cohorts.py

# Verify causal features, labels, rankings, masking and portfolio accounting.
.\.venv\Scripts\python.exe -m unittest -v test_research

# Reproduce into a fresh directory. Never tune against already observed tests.
.\.venv\Scripts\python.exe run_architecture_research.py --datasets NYSE_recent NASDAQ_recent --test-years 2024 2025 --out runs\recent_2024_2025_new
.\.venv\Scripts\python.exe audit_architecture_research.py --run runs\recent_2024_2025_new
.\.venv\Scripts\python.exe analyze_recent_research.py --run runs\recent_2024_2025_new
.\.venv\Scripts\python.exe report_recent_research.py --run runs\recent_2024_2025_new
.\.venv\Scripts\python.exe plot_research_metrics.py --run runs\recent_2024_2025_new
```

The fixed search covers nine neural architectures, three seeds, two cohorts and
two years: 108 neural fits. Ridge, momentum, reversal, equal-weight benchmarks
and SPY are evaluated as well. Random top-K portfolios provide additional
controls. Architecture and portfolio selections use only prior validation data.

## Files and folders

| Location | Purpose |
|---|---|
| `research_models.py`, `think_model.py` | Controlled variants and THINK backbone |
| `research_engine.py` | Causal features, purged folds, ranking and daily accounting |
| `run_architecture_research.py` | Training, year-specific validation selection, locked evaluation |
| `fetch_recent_cohorts.py` | Modern public prices and download coverage audit |
| `config/recent_identity_exclusions.json` | Documented ticker-reuse exclusions |
| `test_research.py` | Leakage, masking, selection, ranking and accounting checks |
| `audit_architecture_research.py` | Saved-path recomputation and regime diagnostics |
| `analyze_recent_research.py` | SPY-relative metrics, uncertainty and random controls |
| `report_recent_research.py` | Modern report and matched-date equity curves |
| `plot_research_metrics.py` | Metric and benchmark comparison graphs |
| `data/recent_cohorts/` | Frozen 2018-2025 prices, hashes and coverage |
| `runs/recent_2024_2025/` | Active results, predictions, checkpoints and audit records |
| `runs/architecture_research/` | Previous Modern12/2015-2017 experiment, retained for provenance |
| `fetch_research_panel.py`, `fetch_research_cohorts.py`, `repair_research_corporate_actions.py` | Previous snapshot reproduction tools |
| `report_architecture_research.py` | Historical generator; also dispatches modern reports |
| `archive/` | Superseded tutorials, audits, models, data and runs |

The existing `.venv` supports CPU training. Fresh environments need
`requirements.txt` and `requirements-data.txt`; `requirements-tested.txt` records
the existing environment. Generated data and runs are Git-ignored: preserve
their snapshots alongside code when sharing results.

The earlier [report](runs/architecture_research/REPORT.md) and
[graphs](runs/architecture_research/CHARTS.md) remain historical records.
Their old NYSE/NASDAQ numbers must not be interpreted as 2024-2025 results.
