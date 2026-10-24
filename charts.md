# NASDAQ-100 master charts

Updated October 24, 2026. Historical monthly NASDAQ-100 membership, January 3, 2022–December 31, 2025. All charts use the same 1,003 sessions.

## Best historical results

These distinctions rank the completed base-cost strategy results after observing this history. They are not a validation-selected shortlist or evidence of reliable future superiority.

| Historical distinction | Strategy | Sharpe | CAGR | RankIC | Max drawdown | 95% SR interval |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Highest CAGR across base-cost strategies | StockMixer / top 5 | 0.659 | 22.52% | 0.0186 | -36.13% | [-0.129, 1.477] |
| Highest Sharpe across base-cost strategies | StockMixer / top 5 / 75% long / 25% short | 0.738 | 19.52% | 0.0186 | -22.50% | [-0.051, 1.538] |
| Highest CAGR among standard top-ten ensembles | THINK + mixer / Daily + AH + PM / original five seeds | 0.700 | 22.39% | 0.0061 | -35.01% | [-0.139, 1.594] |
| Original daily reference | THINK + mixer (original five seeds) | 0.629 | 19.60% | 0.0074 | -39.44% | [-0.186, 1.505] |
| Highest defensive-policy Sharpe | Original THINK + mixer | 0.654 | 17.01% | 0.0074 | -29.04% | [-0.175, 1.560] |
| New extended-hours primary ten-seed blend | THINK + mixer / Daily + AH + PM / ten seeds | 0.519 | 15.36% | 0.0038 | -38.32% | [-0.345, 1.425] |

StockMixer top five has the highest CAGR across the consolidated base-cost strategies. The 75% long / 25% short StockMixer version has the highest Sharpe, with 3% annual short borrowing and assumed borrow availability. Neither short availability nor concentration risk should be confused with an improvement in stock predictions.

Among standard full-exposure top-ten ensembles, the original-five THINK + mixer with after-hours and premarket leads at 22.39% CAGR. Its new-five counterpart returned 13.69% versus 15.51% for daily inputs. The primary fixed ten-seed comparison was 15.36% versus 14.54%, improving only 5/10 individual seeds. The paired intervals include zero. The original daily five-seed ensemble remains the established reference, although its 25-seed expansion returned 12.86%. No model has established reliable future superiority over SPY/QQQ.

The SR intervals use a fixed 3% annual hurdle and 10,000 circular-block resamples, with 20 sessions primary. They condition on fitted paths, omit the full research search and retraining uncertainty, and do not measure the probability of beating a benchmark. These years have been repeatedly inspected.

![Historical leaders](docs/strategy_comparison_20261024/figures/historical_leaders.png)

## Files and scope

- [Updated comparison workbook](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/NASDAQ100_strategy_comparison.xlsx): 275 distinct base-cost strategies; 72 Models, 31 Defensive and 172 Portfolio variants. Benchmarks repeat in the latter tabs for convenience.
- [Every extended-hours account](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/extended_hours_all_accounts.csv): all 66 accounts, including individual seeds and 2x/4x fees, also in the new workbook tab.
- [Chart catalog](docs/strategy_comparison_20261024/chart_catalog.csv): one chart-panel entry for each of 275 main strategies and 54 additional extended-hours diagnostic accounts. Twelve reference accounts are aliases already represented in the main gallery.
- [All strategy metrics](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/all_strategies.csv) and [5/20/60-session SR sensitivity](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/sharpe_block_sensitivity.csv).

Defensive uses the fixed QQQ/MA200 half-exposure rule. Portfolio variants change holding count or long/cash/short allocation. All main rows are ensemble or declared selection procedures; individual extended-hours seeds are diagnostics. Failed and negative strategies are retained.

For fee-stress charts the red/purple comparators remain the base-cost SPY/QQQ references; the strategy fee multiplier is stated in each panel. No new models were fit for this consolidation. Retired 2026-selected survivor universes and unrelated/deferred NYSE studies are excluded from this active-model ranking.

## Source reports and uncertainty

- [Original monthly study](docs/nasdaq_monthly_pit/CHARTS.md)
- [Original holding-count/loss comparison](docs/nasdaq_monthly_pit/topk_pseudohuber/metrics.csv)
- [Original long/short allocations](docs/nasdaq_monthly_pit/long_short/primary_metrics.csv)
- [Alpaca regular-session features](docs/nasdaq_alpaca_20261019/REPORT.md)
- [Ranking and residual objectives](docs/ndx_ranking_20261020/REPORT.md)
- [Broad features and model families](docs/ndx_broad_20261020/REPORT.md)
- [Frozen-core corrections](docs/ndx_corrections_20261020/REPORT.md)
- [25-seed expansion](docs/ndx_seed_expansion_20261020/REPORT.md)
- [After-hours and premarket experiment](docs/ndx_extended_hours_20261024/REPORT.md)

![Paired information-increment uncertainty](docs/ndx_extended_hours_20261024/figures/information_intervals.png)

## Complete model and strategy gallery

Each panel compares one strategy with SPY and QQQ. Titles report net CAGR, Sharpe and maximum drawdown. All declared main strategies appear, including equal realized paths from distinct selection rules. Numeric results and unique IDs are searchable in the linked catalog.

### Models / Original daily

![Models / Original daily, 9 strategies](docs/strategy_comparison_20261024/figures/models_original_daily_01.png)

![Models / Original daily, 4 strategies](docs/strategy_comparison_20261024/figures/models_original_daily_02.png)

### Models / Alpaca matched features

![Models / Alpaca matched features, 9 strategies](docs/strategy_comparison_20261024/figures/models_alpaca_matched_features_01.png)

![Models / Alpaca matched features, 7 strategies](docs/strategy_comparison_20261024/figures/models_alpaca_matched_features_02.png)

### Models / Ranking objectives

![Models / Ranking objectives, 8 strategies](docs/strategy_comparison_20261024/figures/models_ranking_objectives_01.png)

### Models / Broad features

![Models / Broad features, 9 strategies](docs/strategy_comparison_20261024/figures/models_broad_features_01.png)

![Models / Broad features, 6 strategies](docs/strategy_comparison_20261024/figures/models_broad_features_02.png)

### Models / Frozen-core corrections

![Models / Frozen-core corrections, 9 strategies](docs/strategy_comparison_20261024/figures/models_frozen_core_corrections_01.png)

![Models / Frozen-core corrections, 1 strategies](docs/strategy_comparison_20261024/figures/models_frozen_core_corrections_02.png)

### Models / Seed expansion

![Models / Seed expansion, 2 strategies](docs/strategy_comparison_20261024/figures/models_seed_expansion_01.png)

### Models / Extended-hours information

![Models / Extended-hours information, 8 strategies](docs/strategy_comparison_20261024/figures/models_extended_hours_information_01.png)

### Defensive / Broad features

![Defensive / Broad features, 9 strategies](docs/strategy_comparison_20261024/figures/defensive_broad_features_01.png)

![Defensive / Broad features, 9 strategies](docs/strategy_comparison_20261024/figures/defensive_broad_features_02.png)

![Defensive / Broad features, 1 strategies](docs/strategy_comparison_20261024/figures/defensive_broad_features_03.png)

### Defensive / Frozen-core corrections

![Defensive / Frozen-core corrections, 9 strategies](docs/strategy_comparison_20261024/figures/defensive_frozen_core_corrections_01.png)

![Defensive / Frozen-core corrections, 1 strategies](docs/strategy_comparison_20261024/figures/defensive_frozen_core_corrections_02.png)

### Defensive / Seed expansion

![Defensive / Seed expansion, 2 strategies](docs/strategy_comparison_20261024/figures/defensive_seed_expansion_01.png)

### Portfolio variants / Original daily

![Portfolio variants / Original daily, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_daily_01.png)

![Portfolio variants / Original daily, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_daily_02.png)

![Portfolio variants / Original daily, 2 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_daily_03.png)

### Portfolio variants / Alpaca matched features

![Portfolio variants / Alpaca matched features, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_alpaca_matched_features_01.png)

![Portfolio variants / Alpaca matched features, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_alpaca_matched_features_02.png)

![Portfolio variants / Alpaca matched features, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_alpaca_matched_features_03.png)

![Portfolio variants / Alpaca matched features, 5 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_alpaca_matched_features_04.png)

### Portfolio variants / Original allocations

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_01.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_02.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_03.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_04.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_05.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_06.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_07.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_08.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_09.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_10.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_11.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_12.png)

![Portfolio variants / Original allocations, 9 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_13.png)

![Portfolio variants / Original allocations, 3 strategies](docs/strategy_comparison_20261024/figures/portfolio_variants_original_allocations_14.png)

### Extended-hours / Individual-seed diagnostic

![Extended-hours / Individual-seed diagnostic, 9 strategies](docs/strategy_comparison_20261024/figures/individual_seed_diagnostic_01.png)

![Extended-hours / Individual-seed diagnostic, 9 strategies](docs/strategy_comparison_20261024/figures/individual_seed_diagnostic_02.png)

![Extended-hours / Individual-seed diagnostic, 9 strategies](docs/strategy_comparison_20261024/figures/individual_seed_diagnostic_03.png)

![Extended-hours / Individual-seed diagnostic, 3 strategies](docs/strategy_comparison_20261024/figures/individual_seed_diagnostic_04.png)

### Extended-hours / Cost sensitivity

![Extended-hours / Cost sensitivity, 9 strategies](docs/strategy_comparison_20261024/figures/cost_sensitivity_01.png)

![Extended-hours / Cost sensitivity, 9 strategies](docs/strategy_comparison_20261024/figures/cost_sensitivity_02.png)

![Extended-hours / Cost sensitivity, 6 strategies](docs/strategy_comparison_20261024/figures/cost_sensitivity_03.png)
