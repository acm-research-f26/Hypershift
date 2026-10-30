# Best current results, with SPY and QQQ — October 30, 2026

- **Best-supported current neural strategy: the prior mixture.** Twenty-seed SR **1.367**, CAGR **24.60%**, worst drawdown **-11.20%**. Four separate five-seed groups score **1.272 / 1.184 / 1.409 / 1.322**. The last group was scored after the candidate was fixed.
- **Strong simple alternative: the historical-prior model alone.** SR **1.306** with five-security retention, or **1.251** with five issuers. It has no neural training randomness. This is one deterministic control, not repeated independent seed evidence.
- **Highest selected historical peak: the old retention recipe.** SR **1.731** on the original five, **0.648** on the second five and **1.381** across ten. Its peak is higher; its initialization consistency is much weaker. The current mixture does not beat that old recipe in every available group.

**No evaluated strategy has achieved SR>1.5 in every predefined ensemble.** The current mixture has stronger cross-seed support than the old peak, not a demonstrated 1.7 Sharpe that generalizes.

![Current shortlist versus passive benchmarks](current_models_vs_benchmarks.png)

| Strategy / reference | Sharpe | CAGR | Annual volatility | Worst drawdown |
|---|---:|---:|---:|---:|
| Prior mixture / twenty seeds | 1.367 | 24.60% | 14.72% | -11.20% |
| Previous neural issuer / twenty seeds | 1.245 | 22.77% | 15.01% | -13.55% |
| Neural retention / cohort targets / twenty seeds | 1.237 | 21.92% | 14.49% | -12.78% |
| Historical prior alone / five securities | 1.306 | 25.39% | 16.04% | -11.27% |
| Historical prior alone / five issuers | 1.251 | 24.19% | 15.97% | -11.35% |
| Old selected retention / original five | 1.731 | 33.11% | 15.52% | -13.36% |
| Old selected retention / second five | 0.648 | 12.31% | 15.14% | -17.11% |
| SPY buy and hold | 0.498 | 10.85% | 18.01% | -24.55% |
| QQQ buy and hold | 0.469 | 11.93% | 23.67% | -34.87% |

Sharpe measures daily return above a 3% annual hurdle relative to how bumpy the path is. CAGR is compound growth per year; worst drawdown is the largest fall from an earlier account peak. The simple prior earns slightly higher CAGR than the twenty-seed mixture but has a bumpier path, so its Sharpe is lower. Different portfolio rules are present: the simple retention control and old strategy select securities, while the preferred mixture counts companies/issuers.

## How the preferred model works, in everyday terms

1. **Make a historical report card.** For each stock, use only the fold's training period and ask how well its returns ranked against other stocks on average. Recent training observations get more weight: an observation 26 decision-weeks older has half the weight. Limited histories are pulled toward a neutral score. The report card is fixed for the annual fold; it does not use that evaluation year's outcomes. For the 2025 fold, this prior ends in 2023, before 2024 validation.
2. **Ask two neural models for their stock orderings.** THINK and the dynamic-hypergraph model use daily, after-hours and premarket inputs to rank stocks. Each seed starts another version of each network. A five-seed group therefore contains five versions of each neural model, not five stocks.
3. **Combine the votes.** The common historical report card receives 50% of the ranking score; mean THINK rank gets 25% and mean dynamic-model rank 25%. For example, report card 85, THINK 90 and dynamic 70 give an overall 82.5. This is an ordering score, not an 82.5% return forecast.
4. **Build and risk-size a portfolio.** About every five trading sessions, each five-seed committee selects five companies. Counting issuers stops GOOG and GOOGL from using two company slots. When the chosen basket has been more volatile, the stock allocation is reduced. A partial QQQ short uses the permitted remaining gross budget. It tends to gain when QQQ falls and lose when QQQ rises. If 70% of the target budget is long stocks, at most 30% can be short QQQ under this rule. The 75% short limit is a ceiling, not a permanent position. The 20% stock-volatility target guides sizing; actual account volatility can differ.
5. **Combine portfolios for larger ensembles.** Twenty seeds form four five-seed committees. Each builds its own five-issuer portfolio; the account averages those signed targets and trades the net order. It can hold more than five issuers. This preserves different reasonable buy lists instead of forcing every model into one top-five vote. All predefined seeds have equal weight.

The simple historical-prior control skips the neural votes. Its stock preference list is mostly fixed within each annual fold; eligibility,retention and risk sizing can still change positions. Its strong result means a simple training-history strategy remains an important competitor. These results do not establish a statistically significant advantage for the more complex ensemble.

## Comparison across every seed ensemble

| Ensemble | New prior mixture SR | Previous neural SR | Old selected recipe SR | SPY SR | QQQ SR |
|---|---:|---:|---:|---:|---:|
| Original five | 1.272 | 1.171 | 1.731 | 0.498 | 0.469 |
| Second five | 1.184 | 1.147 | 0.648 | 0.498 | 0.469 |
| Third five | 1.409 | 1.200 | — | 0.498 | 0.469 |
| Reserved five | 1.322 | 0.877 | — | 0.498 | 0.469 |
| Combined ten | 1.264 | 1.023 | 1.381 | 0.498 | 0.469 |
| Combined fifteen | 1.338 | 1.235 | — | 0.498 | 0.469 |
| Combined twenty | 1.367 | 1.245 | — | 0.498 | 0.469 |

SPY and QQQ have no training seeds. Their repeated columns/curves are the same two buy-and-hold references, not extra experiments. Dashes mean the old recipe was not fitted for that group; no missing old result is extrapolated. The same cost formula, price convention, 1,003 sessions and 3% Sharpe hurdle apply. Benchmarks buy at the first common fill close and liquidate at the last; the active models trade repeatedly and additionally pay short borrowing.

![Four separate seed cohorts and benchmarks](separate_cohorts_vs_benchmarks.png)

![Pooled ensembles and benchmarks](pooled_models_vs_benchmarks.png)

**These are strategy comparisons, not a pure test of neural prediction skill.** SPY/QQQ stay fully invested; the active models select stocks, adjust exposure and partially short QQQ. Their market exposure and volatility differ. Higher historical Sharpe does not establish future superiority.

## Why this is more consistent than the old 1.7 result

The old retention rule gave last week's selected names a small score bonus before choosing five securities. That encourages keeping a near-tied name. It does not correct arbitrary model rankings, and it can count two share classes as separate slots.

The old result had a specific initialization problem. In the inspected 2023 fold, THINK's stock ordering correlated about 0.995 with its stock-specific output biases; those biases retained about 0.999 rank correlation with their random starting values. In plain language, some stocks began with arbitrary head starts that training barely changed. The lucky original group also held a large combined GOOG/GOOGL position. That is evidence for one initialization mechanism, not proof that every year or every part of the network was untrained.

The newer training starts those stock-specific output weights/biases neutrally, gives dynamic stock filters a common start, and trains on relative stock rankings rather than mainly fitting the common market-return level. Validation is separated chronologically for checkpoint selection and diagnostics. These training changes were bundled, so their individual causal contributions are not isolated.

The current winning addition is an anchor built from training data: half of the final score comes from the same historical report card in every seed group. Random starts still affect the neural half. Portfolio averaging further limits the effect of borderline top-five changes.

The reserved cohort is the clearest comparison: the newer neural issuer baseline scores 0.877 and the fixed prior mixture 1.322. Because this group contains only one five-seed committee, portfolio averaging cannot explain that gain; the ranking prior changes the stock choices. Across all four disjoint groups, baseline SR spread falls 0.323 → 0.225. The older 1.731 → 0.648 gap was 1.084 under a different recipe/training generation.

Consistency improved at a cost to the old selected peak. We retained all seeds and exposed the weak cohort instead of choosing the original five. Removing the stock-specific mixer entirely,using longer shared-feature forests,selecting different family weights or applying more beta hedging did not solve the problem; those failed curves and full grids remain below in the master gallery.

## What remains unproven

All cohorts share the repeatedly inspected 2022–2025 market history, and pooled groups overlap their constituent models. The preferred mixture was selected on development results, then frozen before the reserved cohort's financial outcomes. Its secondary seed-confirmation test passes; the original stricter 0.15 spread gate and the all-ensemble 1.5 target remain failed. A deterministic control has no initialization lottery but can still overfit the market history used to choose it. None of this is an untouched-market validation.

[Full experiment history and verification](../ndx_seed_robustness_20261030/REPORT.md) · [Headline metrics](HEADLINE_METRICS.csv) · [Seed comparison](SEED_COMPARISON.csv) · [Benchmark replay](BENCHMARK_REPLAY.csv) · [Provenance](MANIFEST.json)
