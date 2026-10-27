# First execution batch — frozen October 27, 2026

User authorization: “after you finish the plan, start testing.” This authorizes execution of available bounded tests; the roadmap's conditional dependencies still apply. This protocol refines RESEARCH_PLAN.md before computing the new contrasts. Prior smoke outputs and published account summaries have already been inspected; all inference is exploratory, conditional on selected historical accounts.

## Scope and order

1. Inspect current and three related Rishi/unification checkouts plus bounded Downloads filename search for the existing raw runs/inputs. No broad private-folder crawl or external team messaging.
2. Read the workbook without modifying it; extract its Leader paths sheet's literal daily return columns, excluding the 2021-12-31 initial zero. Validate dates, finite values, unique series, 1,003 observations, metric parity and overlapping CSV paths. Chart wealth/formula caches are corroboration, not substitute raw forecasts.
3. Compare fixed existing daily account paths to isolate portfolio-policy components where the export allows it.
4. Use original reference-study monthly records for a separate diagnostic of parent/dynamic/blend and a fixed synthetic capital-pooling control. Monthly output is not a daily Sharpe or exact half-capital execution replay.
5. Test structural component/identity behavior synthetically on the actual code. Do not replace missing financial checkpoints with newly trained financial models or claim raw reproduction.

## Fixed daily contrasts

For original-five and ten-seed groups separately:

- `equal5/basketvol20_target20_weekly - equal5/full`: same selection rule; full account-policy consequence of volatility targeting, including cash/trading differences.
- `issuer5/basketvol20_target20_weekly - equal5/basketvol20_target20_weekly`: issuer-selection consequence under the same risk rule, including its effect on estimated basket volatility/exposure. Not a pure fixed-exposure stock-picking effect.
- `equal5/ma200off50_basketvol20_target20_weekly - equal5/basketvol20_target20_weekly`: trend-cap increment under the same selection/volatility rule.

Six diagnostic contrasts, no variant search. Report paired annual net arithmetic mean difference, SR difference at source 3% hurdle, CAGR difference and calendar-year return differences. Bootstrap synchronized returns, stationary expected block 21 sessions, 10,000 draws, RNG seed 27102026; fixed 5/63 sensitivity. Report pointwise 95% intervals and Holm-adjusted one-sided centered-null p values across these six for the annual net arithmetic mean. Do not treat conditional selected-history p values as confirmation or pass the separate main-model mechanism gate from these results.

Recompute the original-five 1.199 and ten-seed 1.080 no-issuer 20% target leaders from workbook literals, expanding prior CSV-only coverage. Report realized vol, turnover/fees from source summaries and annual log-growth/return. Omit new issuer5/full inference unless that exact path is recovered. Report how much gain/loss reduction occurs in 2022 versus later calendar years without choosing states or cutoffs.

## Fixed monthly model contrasts

Source: `latest/results/reference_networks/monthly.csv`, paired with `metrics.csv`, only baseline/dynamic_restricted/dynamic_restricted_blend, original-five/new-five/ten-seed, k5/k10, base cost. Require 48 complete unique matching months and reconstruct CAGR via 1,003 source sessions; do not annualize with an assumed exactly four years when validating source CAGR.

Ten-seed contrasts: blend minus parent and blend minus dynamic at k5 and k10 (four). Endpoint is `12*mean(monthly_net_left-monthly_net_right)`, explicitly a monthly arithmetic diagnostic distinct from the roadmap daily mean endpoint. Report paired intervals, expected stationary block 3 months, 10,000 draws, RNG 27102027, 1/6-month sensitivity; Holm across these four if p values are shown. No daily SR reconstructed from monthly data.

For each group/k also build a fixed algebraic 50:50 initial-capital combination of the two *published full-size standalone NAV paths*: `W_pool=.5*W_parent+.5*W_dynamic`, each initialized at 1, not monthly rebalanced. It holds the component return series and their original AUM-dependent fees/caps fixed. This is a non-executable diagnostic because actual half-AUM costs/caps would differ; it cannot replace the plan's half-capital share-account replay. Report pooled CAGR and blend-minus-pool terminal wealth/CAGR without claiming stock-level selection attribution or testing a weight grid.

## Code/data validation

Add meaningful tests before implementation for workbook extraction, malformed/duplicate/missing dates, terminal wealth including initial fee, identical-arm bootstrap degeneracy, paired block synchronization and capital-pooling versus monthly rebalancing. Structural tests distinguish consistent ID reindexing from feature-only swaps and avoid claiming random-init behavior is learned financial skill. Reuse the existing source modules unchanged. No protected returns, new financial fits, orders or automations.

## Outputs and verdict

`ARTIFACT_INVENTORY.json`, `WORKBOOK_PATHS.csv`, `DAILY_CONTRASTS.csv`, `MONTHLY_MODEL_CONTRASTS.csv`, `CAPITAL_POOL_DIAGNOSTIC.csv`, `BATCH_RESULTS.json`, `TESTING_REPORT.md`, test log and independent review. Store under `execution_batch1/`, without rewriting earlier smoke artifacts.

Missing original predictions/checkpoints/positions means **DEFER trained-signal, identity-P&L and raw-account claims**, regardless of successful aggregate tests. Complete all feasible batch tests and report which mechanism questions they resolve and which remain unavailable. No automatic retraining on unavailable inputs. Future financial ablation stages require synthetic learner-power gates and measured/frozen budgets from the roadmap.

## Execution ledger

Ruling: work in the existing tom-shlom checkout because repository branch/coordination instructions specify it; isolate new analysis/tests and preserve unrelated changes. No worktree reset, dependency installation or source-layout rewrite.

Ruling: apply the implementation skill's test-before-code discipline and independent final review to this research batch; use this explicit task ledger rather than its software-plan brief generator, because the authoritative document is a research roadmap with data gates, not pre-generated software task briefs. No production trading/model behavior changes.
