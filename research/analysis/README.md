# Comparisons and research diagnostics

These sources analyze saved NASDAQ-100 forecasts and portfolio paths. They cover
strategy comparison, rank depth and loss similarity.

- [compare_ndx_strategies_20261020.py](compare_ndx_strategies_20261020.py): base
  cross-study comparison and paired/block-bootstrap Sharpe analysis.
- [analyze_ndx_rank_depth.py](analyze_ndx_rank_depth.py): holding-count and
  ranks-six-to-ten diagnostics.
- [analyze_ndx_loss_similarity.py](analyze_ndx_loss_similarity.py): similarities
  between trained objective variants.
- Earlier cohort, exchange-geometry and pooled sensitivity sources are preserved
  in [archive](../../archive/README.md).

Use `python -m hypershift list analysis` from the repository root to locate sources.
The latest consolidation extends the original comparison in
[update_ndx_new_findings_20261024.py](../../latest/reporting/update_ndx_new_findings_20261024.py).
The [master charts](../../charts.md), [workbook](../../docs/NASDAQ100_strategy_comparison.xlsx)
and [reporting guide](../reporting/README.md) identify the current complete results.

Historical leader tables are retrospective. Absolute Sharpe intervals and paired
differences answer different questions; neither a selected historical CAGR nor
a high point-estimate Sharpe establishes reliable benchmark superiority.
