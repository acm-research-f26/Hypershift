# Parent information views under the headline risk overlays

The historical Sharpe above one is **not unique to the AH+PM parent**. With the same dynamic family and 20% basket-volatility rule, the ten-seed issuer-limited blend had a Sharpe of **1.112** with the AH+PM parent and **1.082** with the separately trained daily-only parent. Their paired Sharpe difference was **+0.029**, with a 95% block-21 interval of **[−0.169, +0.235]**. This does not establish that the parent views are equivalent or that either result will persist.

Protocol 8f0a87a and code 0013ff4 fixed 36 account replays after the full-exposure parent-view results. All retain the same saved AH+PM dynamic family, source decision/execution rules, $1m and modeled costs. Each portfolio's selected basket supplies its own causal volatility estimate, so the parent swap can change both selections and exposure. Two selected historical headline configurations are shown:

| Seed group / allocation / overlay | Parent view | Sharpe | CAGR | Max drawdown |
|---|---|---:|---:|---:|
| Original five / top 5 / 20% target | Daily | 1.145 | 27.71% | -22.27% |
| Original five / top 5 / 20% target | AH-only | 1.107 | 26.41% | -22.36% |
| Original five / top 5 / 20% target | AH+PM | 1.199 | 28.96% | -17.78% |
| Ten seeds / issuer 5 / 20% target | Daily | 1.082 | 25.95% | -19.70% |
| Ten seeds / issuer 5 / 20% target | AH-only | 1.057 | 25.25% | -18.85% |
| Ten seeds / issuer 5 / 20% target | AH+PM | 1.112 | 26.81% | -21.69% |

For original five/top five, AH+PM minus daily Sharpe is **+0.054 [−0.140, +0.250]**; its annual arithmetic net mean difference is **+0.94 pp [−2.84, +4.82]**. For ten-seed/issuer-five, the corresponding net mean difference is **+0.70 pp [−3.35, +5.05]**. The intervals include both material losses and gains, and do not establish a benefit or economic equivalence. The original 1.199 and 1.112 rows remain selected development-history results, not new independent confirmation.

The **new-five** subgroup is weaker in absolute terms but shows a different point pattern: under issuer-five/20% targeting, AH+PM Sharpe is **0.677** versus daily **0.498**. The paired difference is **+0.179 [0.023, 0.352]** at block21. Across all 24 fixed block21 contrasts in this post-selection batch, four mean-return and three Sharpe intervals exclude zero, all in the new-five group. These are unadjusted pointwise diagnostics among dependent comparisons, not a prespecified discovery or independent market replication; the same market history underlies every seed. Results at block lengths 5 and 63 and the 25% target/trend-cap policy are retained for every cell.

The daily-parent blend is **not a daily-only system**: its dynamic family still consumes AH/PM features. Parent views were separately trained, and the risk overlays can amplify a small selection change through different basket-volatility estimates. Differences cannot identify an AH or PM channel's causal contribution. The prior search, weak subgroup, unverified first-provider publication, issuer/membership identity and execution-capacity limits remain.

All **12** original AH+PM risk paths reproduced exactly and normalized NAV matched within 4.66e-16 before alternative accounts were run. An independent audit rebuilt **3,618** target allocations, all **36** account metrics/NAV and causal target-exposure paths, and all **72** paired mean intervals; maximum respective NAV, exposure and interval errors were 1.59e-14, 1.12e-16 and 9.89e-17. Four headline block-21 Sharpe and CAGR interval pairs were independently recalculated within 9.03e-17. Twelve risk/execution tests passed. `METRICS.csv`, `CONTRASTS.csv`, `CALENDAR.csv`, `OVERLAP.csv` and provenance are here; detailed ledgers and arrays are under ignored `results/RISHI_PARENT_OVERLAY_20261028/`. No financial fitting, checkpoint intervention, protected-universe scoring or original primary-gate change occurred.
