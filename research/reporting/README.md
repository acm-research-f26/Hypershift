# Reports, charts and workbook publication

This directory owns publication code, not the published documents themselves.
Study reports and the workbook live in `docs/`; comparison tables and the workbook
builder live in `outputs/`. The repository-root `charts.md` is the master gallery.

| Sources | Responsibility |
| --- | --- |
| [publish_ndx_model_guide.py](publish_ndx_model_guide.py) | Current model shortlist, plain-language explanations, and matched SPY/QQQ wealth/drawdown comparisons with benchmark replay |
| [publish_ndx_seed_robustness.py](publish_ndx_seed_robustness.py) | Current confirmed-seed overview,63checked daily paths,all1,578account replays and164fit-lock verification |
| [report_ndx_feature_prior_studies.py](report_ndx_feature_prior_studies.py) | Publish every shared-feature and training-prior comparison, including deterministic controls and failed candidates |
| [report_ndx_portfolio_robustness.py](report_ndx_portfolio_robustness.py), [report_ndx_family_robustness.py](report_ndx_family_robustness.py) | Publish all fixed portfolio/shared-scorer and conditional family experiments, with matched saved wealth/drawdown paths and full failure disclosure. |
| [plot_ndx_model_comparison_20261030.py](plot_ndx_model_comparison_20261030.py) | Replot 32 locked account paths by matched seed cohort, reconstruct metrics, and publish wealth/drawdown figures with provenance; does not retrain or change completed experiments. |
| [publish_ndx_matched_controls.py](publish_ndx_matched_controls.py) | Publish audited QQQ/momentum controls, wealth and drawdown comparisons, and prospective-policy readiness. |
| [update_ndx_drawdown_master.py](update_ndx_drawdown_master.py) | Prepare, audit and publish the chart-first master workbook/narrative integration and preserve existing comparison content. |
| [publish_ndx_drawdown_research.py](publish_ndx_drawdown_research.py) | Publish all 305 audited drawdown-control recipes, matched paths, cost stresses and uncertainty. |
| `report_ndx_*`, `report_*` | Study metrics, uncertainty and reports from saved artifacts |
| `publish_ndx_*`, `publish_historical_ndx` | NASDAQ study-specific publication |
| `chart_performance_labels` | Shared chart labeling |
| [update_ndx_strategy_comparison_20261024.py](update_ndx_strategy_comparison_20261024.py) | Extended-hours comparison integration |
| [update_ndx_top5_comparison_20261024.py](update_ndx_top5_comparison_20261024.py) | Earlier unblended AH+PM five-holding follow-up |
| [update_ndx_new_findings_20261024.py](../../latest/reporting/update_ndx_new_findings_20261024.py) | Latest sealed loss/optimizer and reference-network integration |
| [publish_ndx_master_overview.py](publish_ndx_master_overview.py) | Presentation follow-up: validate all 346 saved paths and plot the full comparison with the leader and SPY/QQQ emphasized. |

Current user-facing artifacts:

- [Master charts](../../charts.md): every compared strategy and diagnostic gallery.
- [NASDAQ100_strategy_comparison.xlsx](../../docs/NASDAQ100_strategy_comparison.xlsx):
  ten tabs, 364 distinct main strategies, with benchmarks repeated where useful.
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

The October 26 master overview is an additional presentation step after the
completed consolidations. Run `python -m hypershift run publish_ndx_master_overview`
to recreate its figure and provenance record from local `analysis_paths.npz`.
The root `charts.md` contains the accompanying model explanation. Original
publication audits remain dated snapshots; the supplementary overview manifest
records the revised narrative hash and verifies the unchanged numerical outputs.

Older pooled/cohort report and presentation builders are preserved in
[archive](../../archive/README.md). Latest-study publishers live in
[latest/reporting](../../latest/reporting/README.md).
