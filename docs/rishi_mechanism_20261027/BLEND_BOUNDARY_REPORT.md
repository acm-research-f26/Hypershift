# How averaging ranks changes the portfolio

2026-10-27. Research_Plan Phase C. **Disposition: changed selection beyond ordinary account diversification is demonstrated under the source model; incremental learned changing information remains unresolved.** Available boundary, pooling, identity-substitution, error-covariance and fixed-bundle diagnostics are delivered here. Later [traded-category attribution](execution_batch12_boundary/REPORT.md) partitions cost and P&L under a declared convention; the training-prediction surrogate remains open.

## The blend is a selection operation

Each seed's predictions are percentile-ranked over the original inference-eligible set; ranks are averaged within family; the two family scores are averaged 50:50. There is no extra family reranking or fitted blend weight. Full exported-score accounts reproduce, including boundary ties.

For ten-seed/top10, the blend allocates 29.30% of target weight to names selected by both standalone models, 33.76% to parent-only names, 28.30% to dynamic-only names and 8.63% to names selected by neither standalone top10. At top5, the neither-selected fraction is 16.12%. A moderately high name in both families can outrank a name extreme in only one. This directly explains how rank blending creates a new basket.

The boundary stock table preserves scores, weights, observed outcomes and masks at every original origin. Category label-return summaries are frictionless, not additive traded P&L. The later [traded-category analysis](execution_batch12_boundary/REPORT.md) separately assigns fees and profits under an explicit timing convention; it does not infer them from label averages.

## Ordinary return diversification is insufficient

The executable comparator gives each standalone model $500,000 initially, with actual sleeve sizing, costs and participation caps, no transfers or cross-sleeve netting. Combined daily returns derive from summed dollar NAV, including first fees.

| Ten-seed account | Top5 CAGR / SR | Top10 CAGR / SR |
|---|---|---|
| Initial 50:50 capital split |18.72% / .606|15.36% / .530|
| Rank blend |31.87% / .896|23.92% / .755|

Actual half-capital execution changes the earlier algebraic pool by only .025–.058 pp/year across all six fixed group/holding cells. Thus ordinary pooling does not reproduce the blend's historical account. Selection, concentration, drifting family capital weights, turnover and account structure all differ; the gap is not a causal selection-only effect or proof of neural complementarity.

BLEND_CONTROLS.csv consolidates 79 audited account rows: original fixed controls, executable capital splits and the common-screen sensitivity. Universe and execution columns distinguish them; k=0 labels the all-eligible EW sensitivity. None is a newly selected winner.

## Whole-universe ranking and top-tail returns need different explanations

Mean RankIC is .00377 for parent, .03072 for dynamic and .01889 for blend. Yet the blend has the stronger top-tail/account point results. High whole-universe ranking accuracy alone therefore does not explain the concentrated book. Compared with parent+Hlabel, the blend's top10 label mean and decile spread differences are near zero with wide paired intervals. That observation favors investigating stable upper-tail steering but does not prove it is the whole mechanism.

## Near-one prediction-error correlation is largely a shared-target effect

For mean raw forecasts, pooled parent/dynamic forecast correlation is .113 while error correlation is .9979. Within annual fits, raw forecast correlations vary substantially. The exact covariance decomposition retains the shared target-variance term; for 2023, target variance 22.0698 pp² is nearly the entire error covariance 22.0643 pp², while forecast covariance is .000164 pp².

This is arithmetic, not evidence that two selectors are redundant. Zero forecasts would also share identical errors. Subtracting one common baseline from both forecast and target cannot change an error. Forecast innovations relative to Hlabel are recorded separately; their correlation is not a causal mechanism test. Mean raw forecasts and traded averages of per-seed percentile ranks are distinct quantities.

## The exact seed mix matters

Twenty five-of-ten subsets were fixed before scoring, used identically across families, and all 120 B/D/E accounts were retained. Blend top5 Sharpe ranges .362–1.083 and CAGR 9.97–43.31%; top10 Sharpe .432–.857 and CAGR 12.12–29.10%. These are conditional ranges among overlapping subsets of the existing ten initializations, not fresh financial fits or confidence intervals across independent market histories. No new seed group or overlay is selected from the extrema.

## Remaining boundary requirements

The training-only Hpred replacement, within-ID changing-score controls and compact surrogate depend on regenerated original training predictions. That numerical gate is still unresolved. The exact traded category partition is now complete under its documented fee convention: see execution_batch12_boundary/REPORT.md. Its top10 both-model category contributes 15.42 percentage points of annual arithmetic net return with 29.20% mean exposure; neither-model contributes -0.63 points with 8.62% exposure. This partitions the historical book and is not a removal experiment. The existing daily primary tests and financial-fitting rules are unchanged; selected history and source execution assumptions do not establish future alpha.

Evidence: BLEND_CONTROLS.csv; execution_batch4_capital_v2/REPORT.md; execution_batch5_behavior/REPORT.md; execution_batch9_leader_trace/REPORT.md; execution_batch10_selection/REPORT.md; execution_batch11_availability/REPORT.md. See IDENTITY_REPORT.md and COMPLETION_AUDIT.md for the unchanged open requirements.
