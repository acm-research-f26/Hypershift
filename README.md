# Hypershift stock ranking research

Hypershift compares stock-ranking models on historical monthly NASDAQ-100
membership, retaining official share classes and former members. Daily economic
price histories support portfolio accounting; Alpaca SIP 15-minute bars supply
regular-session, after-hours and premarket features. The custom historical
100-largest-NASDAQ-company universe remains deferred.

The current development comparison covers **2022–2025, 1,003 sessions**. These dates
have been inspected repeatedly; none of the historical leaders is a confirmed
outperformance result on a genuinely untouched period.

| Start here | Contents |
| --- | --- |
| [Plain-language model guide and SPY/QQQ comparisons](docs/ndx_model_guide_20261030/GUIDE.md) | Current high-SR shortlist, every seed ensemble against passive benchmarks, how the models work, and why the old 1.7 result was less consistent |
| [Seed-robustness improvement, October 30](docs/ndx_seed_robustness_20261030/REPORT.md) | 164 additional fits / 1,578 accounts; training-prior mixture improves all seven groups and a reserved cohort (0.877→1.322 SR), while the all-ensemble SR>1.5 target remains unmet |
| [Latest dynamic-hypergraph study](latest/README.md) | Direct model code, training, protocols, tests and sealed summary results |
| [Initialization attribution and retraining](docs/ndx_rank_retraining_20261029/FINDINGS.md) | 120 fits / 774 accounts across retraining and factor-control arms; cohort consistency improves but no recipe meets SR>1.5 across all ensembles |
| [Seed-gap attribution](docs/ndx_seed_attribution_20261029/REPORT.md) | Stock-level P&L, duplicate issuer concentration and near-persistent random mixer biases in the 2023 THINK rankings |
| [Validation ranking stability experiment](docs/ndx_rank_stability_20261029_v2/FINDINGS.md) | 216 fixed accounts; seed consistency improves in some variants, but no recipe preserves approximately 1.7 Sharpe across ensembles |
| [Retention strategy replication](docs/ndx_retention_replication_20261029/REPORT.md) | Exact tom-shlom replay on feat-rs: 96 accounts, original-five SR 1.731 and ten-seed SR 1.381 |
| [Master charts](charts.md) | Current historical leaders, uncertainty and the complete strategy gallery |
| [Comparison workbook](docs/NASDAQ100_strategy_comparison.xlsx) | Thirteen tabs, 472 main strategy accounts, comparison charts and seed/cost/borrow diagnostics |
| [Documentation index](docs/README.md) | Current guides, dated study reports and historical material |
| [Source map](research/README.md) | Data, models, training, accounting, analysis, audits and publication code |
| [Project layout](docs/PROJECT_LAYOUT.md) | Commands, compatibility handling and frozen-source provenance |
| [Protocol index](config/README.md) | Dataset identities, experiment settings and the unactivated confirmation gate |

The completed [rank-aligned retraining study](docs/ndx_rank_retraining_20261029/FINDINGS.md)
keeps all 15 predefined seeds and every epoch, with first-half validation
selection and second-half diagnostics. No recipe reaches SR>1.5 across all five
ensembles. A five-issuer, loss-selected recipe gives **1.171 / 1.147 / 1.200**
for the three disjoint seed groups, and **1.023 / 1.235** for ten/fifteen seeds:
more consistent, with lower selected historical performance. All 576 retraining
accounts and 198 additional causal-factor/control accounts pass independent
accounting checks. The reports retain failed methods, individual seeds and costs.

The October 29 [seed attribution](docs/ndx_seed_attribution_20261029/REPORT.md)
identifies a material concern in the historical leader: in 2023, THINK's mean
stock forecast ordering closely tracks stock-specific mixer biases (rank
correlation about 0.995), which retain approximately 0.999 rank correlation with
their reconstructed initial values. The original account also held both
Alphabet classes. The 1.731 original-five result must be reported with the
0.648 new-five and 1.381 combined-ten results, not as representative robustness.

The October 29 [validation ranking study](docs/ndx_rank_stability_20261029_v2/FINDINGS.md)
tests validation RankIC weighting, quarter-stability penalties, cross-model
agreement and broader allocations across 216 cost/borrow accounts. No recipe
meets the declared performance-and-consistency goals. The highest weaker-half
Sharpe recipe yields 1.230 / 0.958 / 1.250 for original-five/new-five/ten-seed:
a smaller gap, with lower original-five and ten-seed performance. The trained
checkpoints are unchanged; this does not test ranking-based epoch selection.

The October 29 [retention replication](docs/ndx_retention_replication_20261029/REPORT.md)
reproduces all 96 accounts from pinned `tom-shlom` forecasts/targets against the
existing `feat-rs` inputs. The highest original-five Sharpe is **1.731** with
**33.11% CAGR / -13.36% drawdown**, using five-security retention, a 20% basket
volatility target and a continuous QQQ hedge capped at 75%. The ten-seed version
reaches **1.381 / 26.35% / -13.79%**; the new-five half remains weaker at 0.648.
All source daily paths and targets match exactly; independent dollar-position
replay also passes. These are reproduced historical development results, not
untouched-period confirmation. The existing comparison workbook retains its
October 27 snapshot; new accounts and charts are in the replication report.

The earlier [own-model hedge study](docs/ndx_model_hedge_20261027/REPORT.md) tests 495 accounts
using raw and prior-year-calibrated return forecasts. Raw hedge triggers never fire;
calibrated signals hedge during 2023 and underperform their unhedged baselines.
A continuous partial-QQQ-hedge control had that study's highest observed Sharpe: 1.454 for
the original five seeds (27.79% CAGR / -14.56% drawdown), and 1.325 for ten seeds
(24.83% / -14.78%). These controls do not establish successful model timing or
prospective superiority. The latest charts and workbook preserve the weaker cases.

The October 26 [drawdown-control study](docs/ndx_drawdown_20261026/REPORT.md) keeps the
THINK+mixer AH+PM / dynamic-hypergraph rank blend and adjusts allocation/exposure.
Within that earlier grid, the highest original-five Sharpe is 1.199 with a 20% volatility target
(28.96% CAGR / -17.78% drawdown). The highest ten-seed Sharpe is 1.112 with
five-issuer selection and a 20% target (26.81% / -21.69%). A 25% target plus
issuer/trend control retains 28.70% ten-seed CAGR with 1.085 Sharpe and -22.98%
drawdown. The chart-first master comparison shows SPY/QQQ and separates the seed
ensembles. All 305 tested recipes, weaker seed results and uncertainty remain
disclosed; these are selected historical development findings.

## Working with the repository

The latest study is grouped under `latest/`; shared NASDAQ-100 code remains in
`research/`, with shared tests and dependencies in `tests/` and `requirements/`.
Run commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m hypershift list
.\.venv\Scripts\python.exe -m hypershift list latest
.\.venv\Scripts\python.exe -m hypershift run run_ndx_reference_networks --help
.\.venv\Scripts\python.exe -m hypershift test -v
```

The launcher temporarily supplies the original flat source paths for historical
imports and actual source-hash checks, then removes unchanged copies. Edit the
organized sources and stage changes after commands finish. Completed experiment
sources and locks remain byte-preserved; use new declared runs for new research.

The [saved-artifact guide](docs/ndx_artifacts_20261027/README.md) indexes the
committed reference-network, extended-hours and drawdown snapshots, including
their checkpoints, forecasts and reviewed inputs. The full 2018–2025 raw Alpaca
SIP snapshot, daily Yahoo/Tiingo source histories and historical membership
sources are also committed, with a separate raw-data checksum inventory in that
guide. Other datasets and run outputs remain local in ignored `data/` and `runs/`;
credentials remain in ignored `.env`.
See [Alpaca setup](docs/ALPACA_15MIN.md),
[dependency sets](requirements/README.md), [hardware](docs/TRAINING_HARDWARE.md)
and [evaluation commands](research/evaluation/README.md). Cloning the repository
supplies these published snapshots, but not all historical parent runs or unrelated
datasets.

Unrelated cohort/NYSE studies, retired-universe reports, decks and scratch work
are preserved in [archive/](archive/README.md). Archived sources are excluded from
normal listings and tests; use `python -m hypershift --archived list` to opt in.
The master charts and workbook retain the active NASDAQ-100 comparison history.
