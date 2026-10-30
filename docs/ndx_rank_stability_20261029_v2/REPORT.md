# Validation ranking stability experiment — October 29, 2026

Completed the fixed **24-recipe / 72 base-account / 216 total-account** grid. **0 recipes met the primary goal** (SR >=1.70 in all three ensembles); **0 met the secondary goal** (original >=1.50, new >=1.00, ten >=1.3813, half-gap <=0.50). These are selected historical development results, not prospective or statistical superiority findings.

![Seed consistency comparison](seed_consistency.png)

## All base-cost recipes, ordered by the weaker seed half

| Method | Allocation | QQQ cap | Original SR | New SR | Ten SR | Half SR gap | Ten CAGR | Primary | Secondary |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| stable_consensus | soft10 | 50% | 1.230 | 0.958 | 1.250 | 0.272 | 22.95% | False | False |
| stable_consensus | soft10 | 75% | 1.226 | 0.942 | 1.245 | 0.283 | 22.82% | False | False |
| stable_consensus | retain10 | 50% | 1.286 | 0.877 | 1.181 | 0.409 | 21.51% | False | False |
| equal | retain10 | 50% | 1.244 | 0.859 | 1.240 | 0.385 | 22.90% | False | False |
| stable_consensus | retain10 | 75% | 1.281 | 0.858 | 1.175 | 0.424 | 21.39% | False | False |
| equal | retain10 | 75% | 1.247 | 0.843 | 1.238 | 0.405 | 22.84% | False | False |
| validation_ic | retain5 | 75% | 1.377 | 0.841 | 1.349 | 0.537 | 25.54% | False | False |
| validation_ic | soft10 | 50% | 1.442 | 0.826 | 1.198 | 0.616 | 21.94% | False | False |
| validation_ic | retain5 | 50% | 1.380 | 0.826 | 1.337 | 0.554 | 25.34% | False | False |
| validation_stable | retain10 | 50% | 1.293 | 0.823 | 1.217 | 0.470 | 22.28% | False | False |
| validation_ic | soft10 | 75% | 1.448 | 0.816 | 1.190 | 0.632 | 21.76% | False | False |
| validation_stable | retain10 | 75% | 1.290 | 0.810 | 1.210 | 0.480 | 22.13% | False | False |
| equal | soft10 | 50% | 1.333 | 0.807 | 1.213 | 0.526 | 22.50% | False | False |
| validation_stable | soft10 | 50% | 1.454 | 0.797 | 1.198 | 0.657 | 21.92% | False | False |
| equal | soft10 | 75% | 1.336 | 0.791 | 1.216 | 0.544 | 22.54% | False | False |
| validation_stable | soft10 | 75% | 1.455 | 0.783 | 1.192 | 0.672 | 21.79% | False | False |
| validation_ic | retain10 | 50% | 1.310 | 0.770 | 1.229 | 0.540 | 22.63% | False | False |
| validation_ic | retain10 | 75% | 1.312 | 0.759 | 1.218 | 0.553 | 22.39% | False | False |
| validation_stable | retain5 | 75% | 1.496 | 0.649 | 1.297 | 0.846 | 24.90% | False | False |
| equal | retain5 | 75% | 1.731 | 0.648 | 1.381 | 1.084 | 26.35% | False | False |
| validation_stable | retain5 | 50% | 1.491 | 0.647 | 1.288 | 0.844 | 24.76% | False | False |
| equal | retain5 | 50% | 1.724 | 0.643 | 1.374 | 1.080 | 26.25% | False | False |
| stable_consensus | retain5 | 75% | 1.273 | 0.584 | 1.393 | 0.690 | 27.09% | False | False |
| stable_consensus | retain5 | 50% | 1.268 | 0.573 | 1.391 | 0.695 | 27.07% | False | False |

## Costs for the consistency-ranked candidate

| Cost multiplier | Annual borrow | Original SR | New SR | Ten SR |
|---:|---:|---:|---:|---:|
| 1 | 3% | 1.230 | 0.958 | 1.250 |
| 2 | 3% | 1.114 | 0.822 | 1.135 |
| 2 | 6% | 1.067 | 0.778 | 1.087 |

## What changed

Four fixed scoring methods were tested: equal weights; validation mean RankIC weights; validation stable RankIC weights; and stable weights plus a cross-model disagreement penalty. Stable quality subtracts the standard deviation of four chronological validation-quarter mean RankICs. All quality weights are shrunk 75% toward equal weights; no seed is discarded and the two model families keep equal weight. The consensus method subtracts 0.25 times cross-model percentile-rank standard deviation from the average score. Each annual calibration uses only validation outcomes strictly before the first test decision. NDCG@5/@10, quarterly RankIC and cross-group ranking/selection agreement are retained.

Each method crosses three allocations (five-security retention, ten-security retention, and smoothly weighted top ten) with 50%/75% continuous QQQ caps. The selected basket retains the existing 20-session/20% volatility target. Each of the 72 base accounts was also tested at 2x transaction costs, and at 2x costs with 6% borrow. Share classes count as separate securities. The fixed grid was frozen before the new account results and was not expanded after inspection.

## Limits and verification

These use saved final-checkpoint forecasts. The original models were selected by validation MSE; this experiment changes ensemble postprocessing and allocation, not checkpoint selection or training. The same validation folds already contributed to checkpoint selection, so the new validation weights are not independently validated. The 2022–2025 period has been repeatedly inspected. Ensemble halves share market history and the ten-seed group includes both halves. A smaller gap alone does not establish success if it results from reducing stronger-group returns. No new neural fits, 2026 returns, live trading or prospective activation were performed.

All 18 matching prior retention paths reproduced (max daily error 0); all216 signed-account paths matched independent dollar-position replay (max daily error 8.88e-16). All targets obey current eligibility and gross budgets, annual calibration chronology is checked, and source/input/output hashes are locked. Synthetic tests cover masking, tie handling, seed-permutation invariance, shrinkage, consensus penalties and lack of future dependence.

[All216accounts](METRICS.csv) · [Consistency and cost comparisons](CONSISTENCY.csv) · [Validation metrics](VALIDATION_METRICS.csv) · [Validation seed weights](VALIDATION_WEIGHTS.csv) · [Validation agreement](VALIDATION_AGREEMENT.csv) · [Test selection agreement](TEST_AGREEMENT.csv) · [Audit](AUDIT.json) · [Protocol](PROTOCOL.json)

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m hypershift run run_ndx_rank_stability --stage freeze --name ndx_rank_stability_20261029_v2_fresh
.\.venv\Scripts\python.exe -m hypershift run run_ndx_rank_stability --stage run --name ndx_rank_stability_20261029_v2_fresh
.\.venv\Scripts\python.exe -m hypershift run run_ndx_rank_stability --stage report --name ndx_rank_stability_20261029_v2_fresh
```

Full scores, targets, daily paths and ledgers are retained in `runs/ndx_rank_stability_20261029_v2/`. Existing outputs refuse overwrite.
