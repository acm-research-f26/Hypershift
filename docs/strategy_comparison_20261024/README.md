# Strategy comparison gallery

This directory contains the published chart catalog, PNG figures and comparison
audit. The narrative gallery is the repository-root [charts.md](../../charts.md);
the companion [workbook](../NASDAQ100_strategy_comparison.xlsx) stays directly in
`docs/`.

- [chart_catalog.csv](chart_catalog.csv): 634 panel entries covering 346 main
  strategies and 288 additional diagnostic accounts.
- [FIGURE_MANIFEST.json](FIGURE_MANIFEST.json): figure identities and hashes.
- [AUDIT.json](AUDIT.json): workbook, source-study and chart verification.
- `figures/`: 83 figures, including historical leaders, seed comparisons,
  confidence intervals and 79 strategy gallery pages.
- [October 26 master overview](overview_20261026/master_all_strategies.png): all
  346 main strategies together, with the best historical model and SPY/QQQ
  emphasized. This supplements the original 83 figures.
- [Overview verification](overview_20261026/MANIFEST.json): validates all 346 saved
  paths against reported CAGR, Sharpe and drawdown, preserves the original figure
  hashes, and records the revised `charts.md` hash. The original audit remains
  the October 24 publication snapshot.

The comparison covers the same 1,003 sessions in 2022–2025. It preserves weaker
strategies, individual-seed diagnostics and fee stresses; the best observed
strategy is a retrospective selection. Figure captions and the master charts
explain which benchmark paths and holding counts apply.

Publication sources are indexed in [research/reporting](../../research/reporting/README.md).
Metrics and workbook-build inputs are in the
[comparison output directory](../../outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/README.md).
