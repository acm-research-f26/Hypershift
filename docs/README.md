# Research documentation and published results

This directory contains published research evidence, operational guides and the
current comparison workbook. Executable sources are indexed in
[research](../research/README.md). The repository-root [master charts](../charts.md)
is the current cross-study visual record.

## Current entry points

| Document | Scope |
| --- | --- |
| [Master charts](../charts.md) | 346 main strategies, seed sensitivity, current historical leaders and 634 chart panels |
| [NASDAQ100_strategy_comparison.xlsx](NASDAQ100_strategy_comparison.xlsx) | Models, Defensive, Portfolio variants, Method, Extended-hours, AH PM holding counts, New findings and New accounts |
| [Comparison gallery](strategy_comparison_20261024/README.md) | Figure catalog, audit and chart coverage |
| [Project layout and commands](PROJECT_LAYOUT.md) | Grouped sources, launcher and reproducibility |
| [Alpaca data guide](ALPACA_15MIN.md) | Credentials, historical 15-minute acquisition and feature locations |
| [Training hardware](TRAINING_HARDWARE.md) | Device settings and dependency environments |
| [Audit index](audit/README.md) | Published checks, provenance and current verification commands |

The active universe is historical official NASDAQ-100 membership sampled monthly,
including former members and official share classes. The historical custom
100-largest-NASDAQ-company ranking is deferred. AH/PM bars are prediction inputs;
portfolio execution remains at the regular close, not in extended-hours trading.
The 2022–2025 comparison is exploratory and does not establish reliable future
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
latest picture. The [confirmation gate](../config/ndx_confirmation_gate_20261024.json)
is not activated and has no mature confirmation sample.

## Historical documents

[Earlier pooled results](REPORT.md), [findings](FINDINGS.md),
[combined comparisons](COMBINED_2022_2025.md), [benchmark controls](ETF_CONTROLS.md),
[robust protocol](ROBUST_RESEARCH.md), [architecture evaluation](ARCHITECTURE_RESEARCH.md),
[Huber/ranking follow-up](HUBER_RANKING.md), [data/splits](DATA_AND_SPLITS.md),
[confidence methodology](CONFIDENCE_INTERVALS.md) and the
[StockMixer paper review](STOCKMIXER_PAPER_REVIEW.md) document earlier cohorts.
Their pooled universes and interval conventions should not be substituted for
those of the latest historical-monthly NASDAQ study.

[Original charts](CHARTS.md), [random trials](random_trials_20261010/CHARTS.md),
[retired 2026-selected study](largecap_retraining_20261010/CHARTS.md) and
[its retirement record](audit/nasdaq_2026_retirement.json) are historical evidence.
NYSE work and existing presentation decks are outside the active comparison.

Hash-sealed reports and charts retain their published bytes. Their old flat-path
commands can be translated with the current [layout guide](PROJECT_LAYOUT.md):
`python example.py ...` becomes `python -m hypershift run example ...`;
`python -m unittest ...` becomes `python -m hypershift test ...`.
