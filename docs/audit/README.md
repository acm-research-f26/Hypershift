# Audit and provenance evidence

This directory contains published audit receipts and historical explanations.
Executable verifiers live in [research/evaluation](../../research/evaluation/README.md).
Local checkpoint and full portfolio-replay evidence remains in ignored `runs/`.

- [Workbook resave receipt](workbook_resave_20261025.json): the current saved
  workbook preserves all rows, text, formulas and results within 1e-12, with
  reordered New findings rows. Earlier native-layout receipts retain their
  original workbook hashes; this separate check covers the resaved file.
- [Latest/archive layout receipt](latest_layout_20261025.json): current move hashes,
  active and archived tests, portable summary exports and unchanged study seals.
- [Source-layout receipt](source_layout_20261024.json): 149 Python and five
  dependency files moved byte-for-byte on October 24. Its paths and test counts
  describe that earlier layout; the October 25 manifest records subsequent moves.
- [Documentation-scope receipt](documentation_scope_20261024.json): directory
  guides, local-link checks and preserved master-chart/workbook hashes.
- [Current comparison audit](../strategy_comparison_20261024/AUDIT.json): 346 main
  strategies, eight workbook tabs and 634 chart-catalog entries.
- [Historical monthly model audit](nasdaq100_model_audit_20261019/REPORT.md):
  findings specific to that dated model set.
- [Earlier timing/accounting explanation](../../archive/studies_20261025/docs/audit/backtests_20261010/EXPLANATION.md):
  evidence from the earlier cohort pipeline.
- [Retirement record](nasdaq_2026_retirement.json): deletion of the superseded
  2026-selected NASDAQ models, with their selection-bias limitations retained.

Other dated receipts describe their named study or hardware/publication check.
A passing accounting or source-hash audit does not establish future investment
performance. For the current combined interpretation, use the
[master charts](../../charts.md) and [workbook](../NASDAQ100_strategy_comparison.xlsx).

See the [evaluation guide](../../research/evaluation/README.md) for commands that
verify existing study seals and inspect the actual comparison workbook without
rewriting it. Frozen artifacts are preserved rather than regenerated merely to
update navigation.
