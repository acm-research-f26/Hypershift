# Hypershift: chronological stock-ranking research

**Current documentation:** [docs/README.md](docs/README.md) collects the updated
slides, pooled 2022–2025 study, training-date audit and confidence-interval tables.

The completed comparison evaluates THINK, mixer variants, robust losses and simple
strategies on **2022–2025**, using daily and five-session forecasts for fixed
historical NYSE/NASDAQ cohorts. **SPY and QQQ use matching evaluation dates.**

Start with [the completed report](runs/robust_comparison/REPORT.md),
[all graphs](runs/robust_comparison/CHARTS.md),
[findings](runs/robust_comparison/FINDINGS.md), and
[both ETF controls](runs/robust_comparison/ETF_CONTROLS.md).
The [combined 2022–2025 curves and tables](runs/robust_comparison/COMBINED_2022_2025.md)
show all models and both ETFs with pooled net Sharpe in each legend.
The [research guide](ROBUST_RESEARCH.md) explains the frozen protocol.
The [dataset slides](docs/slides/Hypershift_Pooled_Datasets_2022_2025.pptx)
contain editable tables, source links and presenter notes for the midpoint presentation.

**Result:** no model consistently beat SPY in all eight cohort/year windows.
Daily forecasts improved fixed-top-10 net Sharpe in only 11 of 88 matched
comparisons. Huber helped five-session THINK+mixer in 5 of 8 windows, but was not
a general improvement. The promising NASDAQ THINK+mixer+Huber pooled Sharpe
of 0.675 versus SPY 0.621 and QQQ 0.626 is an exploratory finding with confidence
intervals spanning zero advantage. The validation-selected procedures trailed
both ETFs over the pooled period. No robust trading edge is established.

Public data recovered original-issuer histories for **68 NYSE-cohort and
67 NASDAQ-cohort securities** from fixed 100-name samples. Two reused ticker
symbols were rejected. Missing histories create material survivorship bias.
These are neither full exchanges nor certified historical index universes.

## Completed scope

- 576 model/seed/fold configurations: 384 five-session and 192 daily.
- Three seeds, two cohorts, four annual tests, K=5/10/20 and five long-only policies.
- Equal-weight buy-and-hold, rebalanced-universe, momentum, reversal, ridge and ETF controls.
- 288 primary daily paths audited, 304 annual model/control curves, and 76 combined model/control series.
- Gross/net returns, Sharpe, volatility, drawdown, IC/RankIC/IR, NDCG, turnover,
  profitable periods, seed/regime stability and exploratory block-bootstrap intervals.

Annual accounts reset to cash. Four-year results compound evaluated annual
returns and omit boundary gaps, so they are not one uninterrupted live account.
The 2024–2025 periods were previously examined. This is exploratory research.

## Reproduction and files

See [reproduction commands](ROBUST_RESEARCH.md#reproduction) to run in fresh folders.
Do not overwrite source snapshots or tune against observed test results.

| Location | Purpose |
|---|---|
| `robust_models.py`, `research_models.py`, `think_model.py` | Neural architectures and controlled ablations |
| `robust_engine.py`, `research_engine.py` | Causal features, purged splits, ranking and portfolio accounting |
| `run_robust_research.py` | Frozen training and validation-only selection |
| `config/robust_experiment_plan.json` | Predeclared architectures, losses and portfolios |
| `fetch_recent_cohorts.py`, `data/recent_cohorts/` | Frozen public prices, coverage and hashes |
| `report_robust_research.py`, `plot_robust_diagnostics.py`, `summarize_robust_findings.py` | Metrics, comparisons and charts |
| `add_etf_controls.py`, `audit_robust_regimes.py` | SPY/QQQ controls, regime and execution diagnostics |
| `plot_combined_research.py` | Combined 2022–2025 curves and complete pooled metric tables |
| `verify_robust_outputs.py` | Input hashes, curve inventory and accounting-audit completeness |
| `test_research.py`, `test_robust_research.py` | Leakage, masking, loss, execution and accounting checks |
| `runs/robust_2022_2025_final/`, `runs/daily_2022_2025_final/` | Authoritative fitted models, predictions and annual portfolios |
| `runs/robust_comparison/` | Completed comparison tables and graphs |
| `presentations/output/` | Previous dataset presentation, retained for provenance |
| `docs/` | Current slides, reports, figures, metric tables and chronology/uncertainty audits |
| `audit_pooled_chronology.py`, `report_pooled_uncertainty.py`, `publish_research_docs.py` | Audit chronological splits, reproduce uncertainty and refresh the docs bundle |
| `STOCKMIXER_PAPER_REVIEW.md` | Paper findings and differences from our implementations |
| `runs/recent_2024_2025/` | Previous batch and 2017/2024/2025 side-by-side charts |
| `runs/architecture_research/` | Earlier Modern12/2015–2017 diagnostic, retained for provenance |
| `runs/robust_2022_2025/`, `runs/daily_2022_2025/` | Verified fit caches used to recover the interrupted run |
| `archive/` | Superseded work |

The existing `.venv` supports CPU training. Fresh environments need
`requirements.txt` and `requirements-data.txt`; `requirements-tested.txt` records
the tested environment. Generated runs/data are Git-ignored: preserve snapshots
alongside code when sharing results.
