# Reports, charts and workbook publication

This directory owns publication code, not the published documents themselves.
Study reports and the workbook live in `docs/`; comparison tables and the workbook
builder live in `outputs/`. The repository-root `charts.md` is the master gallery.

| Sources | Responsibility |
| --- | --- |
| `report_ndx_*`, `report_*` | Study metrics, uncertainty and reports from saved artifacts |
| `publish_ndx_*`, `publish_historical_ndx`, `publish_research_docs` | Study-specific publication |
| `plot_*`, `build_*`, `make_*`, `chart_performance_labels` | Figures and earlier document builders |
| [update_ndx_strategy_comparison_20261024.py](update_ndx_strategy_comparison_20261024.py) | Extended-hours comparison integration |
| [update_ndx_top5_comparison_20261024.py](update_ndx_top5_comparison_20261024.py) | Earlier unblended AH+PM five-holding follow-up |
| [update_ndx_new_findings_20261024.py](update_ndx_new_findings_20261024.py) | Latest sealed loss/optimizer and reference-network integration |

Current user-facing artifacts:

- [Master charts](../../charts.md): every compared strategy and diagnostic gallery.
- [NASDAQ100_strategy_comparison.xlsx](../../docs/NASDAQ100_strategy_comparison.xlsx):
  eight tabs, 346 distinct main strategies, with benchmarks repeated where useful.
- [Chart catalog and figures](../../docs/strategy_comparison_20261024/README.md).
- [Comparison tables and workbook builder](../../outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/README.md).

Models compares baseline portfolio recipes. Defensive applies the fixed QQQ/MA200
exposure rule. Portfolio variants changes holding count or long/cash/short
allocation. Extended-hours and New accounts retain individual-seed and fee-stress
diagnostics; New findings compares fixed seed groups. The Method tab records the
metric and confidence-interval definitions.

Run publishers through the [root launcher](../../docs/PROJECT_LAYOUT.md). Rebuilding
older stages alone will not include later appended studies. Preserve workbook
edits and use the latest consolidation's staged validation before publication.
For verification without rewriting artifacts, use the command in the
[evaluation guide](../evaluation/README.md).
