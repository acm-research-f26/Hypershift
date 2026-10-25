# Strategy comparison tables and workbook inputs

This directory holds exported comparison tables, workbook inputs, the native
workbook builder and consolidation audit records. Its date identifies the first
consolidation; it now includes the October 24 AH/PM, loss/optimizer and reference
network additions.

The user-facing workbook is
[docs/NASDAQ100_strategy_comparison.xlsx](../../../docs/NASDAQ100_strategy_comparison.xlsx).
The master narrative and gallery remain in [charts.md](../../../charts.md).
The old workbook path in this output directory has been retired.

| File | Scope |
| --- | --- |
| [all_strategies.csv](all_strategies.csv) | 346 distinct main base-cost strategies, including SPY and QQQ |
| [new_experiment_all_accounts.csv](new_experiment_all_accounts.csv) | 346 accounts from the new loss/optimizer and reference batches, including seeds and fee stresses |
| [new_findings_seed_summary.csv](new_findings_seed_summary.csv) | 26 fixed seed-group comparison rows |
| [sharpe_block_sensitivity.csv](sharpe_block_sensitivity.csv) | Absolute Sharpe intervals at 5/20/60-session block lengths |
| [new_holdings_uncertainty.csv](new_holdings_uncertainty.csv) | Additional paired secondary holding-count uncertainty |
| [extended_hours_all_accounts.csv](extended_hours_all_accounts.csv) | 66 original extended-hours accounts |
| [top5_all_accounts.csv](top5_all_accounts.csv) | Earlier unblended AH+PM holding-count comparison |
| [workbook_data.json](workbook_data.json), [build_workbook.mjs](build_workbook.mjs) | Native workbook data/build inputs, with later additions in `top5_workbook_data.json` and `new_findings_workbook_data.json` |
| [ANALYSIS.json](ANALYSIS.json), [AUDIT.json](AUDIT.json) | Consolidation provenance and latest verification |

`analysis_paths.npz` contains the local daily strategy paths and is Git-ignored.
Other original tables retain their named study scope. The older
[COMPARISON.md](COMPARISON.md) describes its original consolidation snapshot;
use the root master charts for the complete latest interpretation.

The Models sheet has 105 main rows including benchmarks. Defensive adds 31
strategies and Portfolio variants adds 210; benchmarks repeat in those sheets.
The five other tabs explain methods and retain extended-hours, holding-count,
seed-group and individual-account diagnostics. They are not additional distinct
main strategies to add to the 346 count.

Use [reporting sources](../../../research/reporting/README.md) and
[evaluation commands](../../../research/evaluation/README.md). Older build
stages alone do not reproduce all later workbook additions. Market datasets,
checkpoints and reference-source caches must be restored separately for a full
rebuild; committing these tables does not include the local training data.
