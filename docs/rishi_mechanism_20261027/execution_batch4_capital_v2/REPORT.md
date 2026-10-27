# Rank blending versus allocating capital to two standalone accounts

2026-10-27, frozen protocol69da34a; RESEARCH_PLAN §7 initial50:50standalone-capital comparator. Each sleeve starts$500000 and independently trades the canonical policy with actual sleeve NAV, participation caps and costs. Combined returns derive from summed dollar NAV including first fees, with no capital transfers or cross-sleeve order netting. This is not a daily50:50return rebalance or a single netted execution account.

| Seed group | Holdings | Capital-split CAGR | Capital-split SR | Rank-blend CAGR | Rank-blend SR |
|---|---:|---:|---:|---:|---:|
| Original five |5|12.14%|.431|32.94%|.919|
| Original five |10|20.45%|.671|27.79%|.848|
| New five |5|7.92%|.305|20.92%|.633|
| New five |10|11.87%|.431|13.27%|.467|
| Ten seeds |5|18.72%|.606|31.87%|.896|
| Ten seeds |10|15.36%|.530|23.92%|.755|

Half-capital execution improves CAGR over the algebraic half-sum of full-size NAVs by only.025–.058percentagepoints/year across the six cells. It cannot explain the much larger difference between rank blending and portfolio pooling. The ten-seed primary top10 CAGR gap is8.56pp/year. Rank blending changes selected stocks and trade paths rather than just diversifying unchanged standalone books. This is a behavioral decomposition under canonical execution assumptions; it does not establish the causal feature, a learned-graph increment or future alpha. Original/new-five variation remains large. No new hypothesis test was selected for these descriptive cells.

The comparison also changes concentration, drifting family capital weights, turnover, costs and account structure; the gap does not isolate a causal selection-only effect. Participation caps apply per sleeve, not jointly to consolidated orders. These limits match the frozen standalone-account comparator.

Verification: six combined accounts, twelve sleeves and6018daily combined observations. Summed NAV/cash match component ledgers, compounded combined returns match NAV, and all terminal holdings are zero. Two pooling tests pass, including first-fee inclusion and sleeve capital drift. Independent read-only review found no accounting blocker and prompted explicit full-size ledger date checks; the existing artifacts pass those checks without rescoring. First attempt stopped on a metric-key lookup before summary write; its destination and failure note are preserved. V2 changes only the key from cagr to canonical annualized_return and uses fresh destinations. No original files or policies were edited.
