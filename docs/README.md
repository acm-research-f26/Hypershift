# Research documentation and published results

This directory contains published research evidence, operational guides and the
current comparison workbook. Executable sources are indexed in
[research](../research/README.md). The repository-root [master charts](../charts.md)
is the current cross-study visual record.

## Current entry points

| Document | Scope |
| --- | --- |
| [Current model guide with SPY/QQQ](ndx_model_guide_20261030/GUIDE.md) | Plain-language shortlist, matched wealth/drawdown charts, benchmark replay, every seed ensemble, and limits of the old1.7comparison |
| [Confirmed seed-robustness improvement, October 30](ndx_seed_robustness_20261030/REPORT.md) | Current overview: four disjoint cohorts and ten/fifteen/twenty ensembles, reserved-five prior-mixture SR1.322 and full-twenty1.367; all164fit locks/1,578accounts checked; SR>1.5 still unmet |
| [Training-prior regularization, October 30](ndx_prior_robustness_20261030/REPORT.md) | All264prior/control cases, including failed source goals and the development choice later frozen before reserved-seed scoring |
| [Long-history shared-feature models, October 30](ndx_tabular_alpha_20261030/REPORT.md) | 60seeded forest models/four deterministic Ridge controls,282accounts; shared causal features do not improve the financial results |
| [Family isolation and validation blends, October 30](ndx_family_robustness_20261030/REPORT.md) | 300 fixed diagnostic accounts; original families alone and validation-only blend weights fail to beat the fixed blend consistently |
| [Portfolio robustness and shared scorer, October 30](ndx_portfolio_robustness_20261030_v2/REPORT.md) | 60 controlled shared-THINK fits and 540 accounts; cohort target averaging improves pooled SR, while the shared-scorer blend fails; full grid and accounting checks |
| [Daily model-path comparison, October 30](ndx_model_comparison_20261030/README.md) | Matched ten/fifteen-seed and three disjoint-cohort wealth/drawdown charts; 32 checked saved accounts, selection disclosures and generalization limits |
| [Rank retraining and initialization robustness, October 29](ndx_rank_retraining_20261029/FINDINGS.md) | 120 fixed fits and 576 accounts, 15 predefined seeds; more consistent cohorts but no SR>1.5 across all five ensembles |
| [Causal factor/control arm, October 29](ndx_factor_controls_20261029/REPORT.md) | Separately frozen 198-account momentum/risk/liquidity comparison; no passing recipe; original accounts preserved through aggregation repair |
| [Seed-gap attribution, October 29](ndx_seed_attribution_20261029/REPORT.md) | Independent stock-level contribution replay, stock-ID initialization diagnostics and issuer-concentration interpretation |
| [Validation ranking stability, October 29](ndx_rank_stability_20261029_v2/FINDINGS.md) | 216 fixed accounts; validation weighting and broader portfolios narrow seed gaps with performance tradeoffs; neither declared goal passes |
| [Retention replication, October 29](ndx_retention_replication_20261029/REPORT.md) | Exact 96-account tom-shlom replay using feat-rs forecasts; original-five SR 1.731, ten-seed SR 1.381, complete path/target/ledger verification |
| [Own-model hedge signals, October 27](ndx_model_hedge_20261027/REPORT.md) | 495 accounts testing raw and prior-year-calibrated model triggers; model timing fails this grid, while continuous hedging leads that study |
| [Weak-market shorting, October 27](ndx_weak_market_short_20261027/REPORT.md) | 39 rules / 531 cost, seed and borrow accounts; partial QQQ hedges improve selected historical risk metrics while switching short reduces returns |
| [Matched controls and candidate freeze, October 27](ndx_matched_controls_20261027/REPORT.md) | 29 predeclared neural, QQQ and momentum strategies at three costs; frozen prospective policy and activation prerequisites |
| [Drawdown-control study, October 26](ndx_drawdown_20261026/REPORT.md) | 305 risk-control recipes on saved blend forecasts; audited return, Sharpe, drawdown, seed and cost comparisons |
| [Latest dynamic-hypergraph study](../latest/README.md) | Direct sources, frozen protocols, tests and portable results for the newest study |
| [Master charts](../charts.md) | Latest retrained-model wealth/drawdown comparisons and seed results; earlier retention and October 27 galleries preserved |
| [NASDAQ100_strategy_comparison.xlsx](NASDAQ100_strategy_comparison.xlsx) | Thirteen tabs. Model hedges opens with the latest comparison; Hedge accounts retains all 495 cases and Hedge paths drives its editable charts. Prior ten tabs remain. |
| [Comparison gallery](strategy_comparison_20261024/README.md) | Figure catalog, audit and chart coverage |
| [Project layout and commands](PROJECT_LAYOUT.md) | Grouped sources, launcher and reproducibility |
| [Alpaca data guide](ALPACA_15MIN.md) | Credentials, historical 15-minute acquisition and feature locations |
| [Training hardware](TRAINING_HARDWARE.md) | Device settings and dependency environments |
| [Audit index](audit/README.md) | Published checks, provenance and current verification commands |

The active universe is historical official NASDAQ-100 membership sampled monthly,
including former members and official share classes. The historical custom
100-largest-NASDAQ-company ranking is deferred. AH/PM bars are prediction inputs;
portfolio execution remains at the regular close, not in extended-hours trading.
The 2022â€“2025 comparison is exploratory and does not establish reliable future
superiority over SPY, QQQ or other models.

## Studies and their own evidence

| Study | Report or evidence |
| --- | --- |
| Original historical monthly study | [Charts](nasdaq_monthly_pit/CHARTS.md), [data contract](nasdaq_monthly_pit/DATA_REQUIREMENTS.md) |
| Matched daily/regular-session Alpaca features | [Report](nasdaq_alpaca_20261019/REPORT.md) |
| Ranking and residual objectives | [Report](ndx_ranking_20261020/REPORT.md) |
| Broader features and model families | [Report](ndx_broad_20261020/REPORT.md) |
| Frozen-core corrections | [Report](ndx_corrections_20261020/REPORT.md) |
| Fixed seed expansion | [Report](ndx_seed_expansion_20261020/REPORT.md) |
| Daily, after-hours and premarket information | [Report](ndx_extended_hours_20261024/REPORT.md) |
| Earlier unblended AH+PM five versus ten holdings | [Report](ndx_extended_hours_top5_20261024/REPORT.md) |
| October 24 loss/optimizer and reference-network tests | [Master findings](../charts.md#new-findings-losses-optimizers-and-complementary-networks), [exported account tables](../outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/README.md) |

The latest loss/optimizer and reference studies have local checkpoints and sealed
run evidence under `runs/ndx_loss_optimizer_20261024` and
`runs/ndx_reference_networks_20261024`. Those ignored directories are not included
in a clone; the committed comparison tables and figures remain available.

Dated reports describe their own study, not every subsequent experiment. The
[October 20 status](NASDAQ_RESEARCH_STATUS_20261020.md) precedes the AH/PM and
October 24 optimizer/network findings. Use the master charts for the combined
latest picture. The [confirmation gate](../latest/config/ndx_confirmation_gate_20261024.json)
is not activated and has no mature confirmation sample.

## Archived unrelated studies

Earlier pooled/cohort documents, random trials, NYSE work, retired 2026-selected
reports, decks and their supporting tables are now in the
[study archive](../archive/studies_20261025/README.md). They are outside the active
historical-monthly NASDAQ-100 comparison. Earlier active NASDAQ-100 experiments
listed above remain here because they are part of the current comparison history.

Hash-sealed reports and charts retain their published bytes. Their old flat-path
commands can be translated with the current [layout guide](PROJECT_LAYOUT.md):
`python example.py ...` becomes `python -m hypershift run example ...`;
`python -m unittest ...` becomes `python -m hypershift test ...`.
