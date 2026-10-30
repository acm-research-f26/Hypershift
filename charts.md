# NASDAQ-100 master charts

Updated October 30, 2026. Historical monthly NASDAQ-100 membership, January 3, 2022–December 31, 2025. Every path uses the same 1,003 sessions.

## Best current models, with SPY and QQQ

- **Current neural strategy to focus on: the prior-regularized issuer ensemble.** Twenty-seed SR **1.367**, CAGR **24.60%**, worst drawdown **−11.20%**. Four separate five-seed groups score **1.272 / 1.184 / 1.409 / 1.322**. The last group was evaluated after the candidate was fixed, making this the best-supported current neural strategy for consistency across training starts.
- **Strong simple alternative: the historical-prior model alone.** SR **1.306**, CAGR **25.39%**, worst drawdown **−11.27%**, using five-security retention. It has no neural initialization randomness. The five-issuer version scores **1.251**. These are deterministic controls, not multiple independent seed replications.
- **The old 1.7 recipe remains a selected peak.** Its original-five SR is **1.731**, but the second five score **0.648** and the combined ten **1.381**. It still wins some individual historical comparisons; it is much less consistent across training starts.

**No tested strategy has achieved SR>1.5 in every predefined ensemble.** The newer strategy offers stronger cross-seed evidence at a lower selected peak. The simple control is competitive enough that it should remain a serious baseline; a statistically significant advantage for the more complex ensemble has not been established.

![Current twenty-seed models and the simple control versus SPY and QQQ](docs/ndx_model_guide_20261030/current_models_vs_benchmarks.png)

| Strategy / reference | Sharpe | CAGR | Annual volatility | Worst drawdown |
|---|---:|---:|---:|---:|
| **Prior mixture / twenty seeds** | **1.367** | **24.60%** | **14.72%** | **−11.20%** |
| Previous neural issuer / twenty seeds | 1.245 | 22.77% | 15.01% | −13.55% |
| Neural retention / cohort targets / twenty seeds | 1.237 | 21.92% | 14.49% | −12.78% |
| Historical prior alone / five securities | 1.306 | 25.39% | 16.04% | −11.27% |
| Historical prior alone / five issuers | 1.251 | 24.19% | 15.97% | −11.35% |
| Old selected retention / original five | 1.731 | 33.11% | 15.52% | −13.36% |
| Old selected retention / second five | 0.648 | 12.31% | 15.14% | −17.11% |
| **SPY buy and hold** | **0.498** | **10.85%** | **18.01%** | **−24.55%** |
| **QQQ buy and hold** | **0.469** | **11.93%** | **23.67%** | **−34.87%** |

Sharpe measures return above a 3% annual hurdle relative to how bumpy the daily path is. CAGR measures compounded growth per year. Worst drawdown is the largest decline from an earlier account peak. The simple model's slightly higher CAGR comes with greater volatility, explaining its lower Sharpe than the twenty-seed prior mixture.

### How the models work in plain English

The preferred strategy combines a **historical report card** with **two neural stock-ranking models**. The report card asks how well each stock's returns ranked against other stocks during the training period. Recent training observations count more: a 26-decision-week half-life gives an observation half the weight of one 26 decision-weeks newer. Short histories are pulled toward a neutral score. This report card is fixed for the annual fold; for the 2025 evaluation, its training data end in 2023, before 2024 validation.

THINK and the dynamic-hypergraph model look at daily, after-hours and premarket inputs and produce their own stock orderings. A seed is another starting point for training each network. A five-seed committee contains five versions of each neural model. **Half the final score comes from the historical report card, one quarter from mean THINK rank, and one quarter from mean dynamic-model rank.** A report card of 85, THINK rank of 90 and dynamic rank of 70 produce an overall score of 82.5. That is an ordering score, not a predicted 82.5% return.

About every five trading sessions, each committee selects five **companies**. Counting issuers prevents GOOG and GOOGL from occupying two company slots. The strategy reduces its stock allocation when the selected basket has been more volatile and uses a partial QQQ short within its risk limits. If 70% of the target gross budget is long stocks, at most 30% can be short QQQ. The 75% short ceiling is not a permanent short position, and the 20% stock-volatility target does not guarantee 20% account volatility.

For twenty seeds, four five-seed committees build their own portfolios. We average those signed portfolio targets and execute the net order, paying trading fees and short borrowing. The combined account can own more than five companies. This preserves different reasonable buy lists instead of forcing all models into one fragile top-five vote. Every predefined seed keeps equal weight.

The simple historical-prior model skips the neural votes. Its preference list is mostly fixed within an annual fold, while eligibility, retention and risk sizing can still change positions. Its 13-week half-life differs from the preferred mixture's 26-week half-life. The five-security version can count share classes separately; the five-issuer version consolidates them.

### Every seed ensemble against the same benchmarks

| Ensemble | New prior mixture SR | Previous neural SR | Old selected recipe SR | SPY SR | QQQ SR |
|---|---:|---:|---:|---:|---:|
| Original five | 1.272 | 1.171 | 1.731 | 0.498 | 0.469 |
| Second five | 1.184 | 1.147 | 0.648 | 0.498 | 0.469 |
| Third five | 1.409 | 1.200 | — | 0.498 | 0.469 |
| Reserved five | 1.322 | 0.877 | — | 0.498 | 0.469 |
| Combined ten | 1.264 | 1.023 | 1.381 | 0.498 | 0.469 |
| Combined fifteen | 1.338 | 1.235 | — | 0.498 | 0.469 |
| Combined twenty | 1.367 | 1.245 | — | 0.498 | 0.469 |

SPY and QQQ have no training seeds. Their repeated columns and curves are the same two references, not extra experiments. Dashes mean the old model was not fitted for that group. All paths use the same dates, fee formula, economic-price convention and 3% Sharpe hurdle. Both benchmark paths reproduce their previously published daily returns within 1e-12.

![Separate seed cohorts compared with SPY, QQQ and the available old-model paths](docs/ndx_model_guide_20261030/separate_cohorts_vs_benchmarks.png)

![Ten-, fifteen- and twenty-seed models compared with SPY and QQQ](docs/ndx_model_guide_20261030/pooled_models_vs_benchmarks.png)

**These are comparisons of complete strategies.** SPY/QQQ stay fully invested; the active strategies select stocks, adjust exposure and partially short QQQ. Their market exposure and volatility differ. The higher historical Sharpe cannot be attributed entirely to neural prediction skill, and does not establish future superiority.

### Why the newer strategy is more consistent than the old 1.7 result

The old retention rule gave previously selected stocks a small score bonus, encouraging the strategy to keep near-tied names. That does not fix a model whose stock preferences depend heavily on its random start. In the inspected 2023 fold, THINK's stock ordering correlated about **0.995** with its stock-specific output biases, and those biases retained about **0.999** rank correlation with their random starting values. Some stocks effectively received arbitrary head starts that training barely changed. The original group happened to favor profitable names and also held substantial combined GOOG/GOOGL exposure. This is evidence for one specific initialization problem, not proof that every year or the whole network was untrained.

The newer approach addresses that problem in several ways:

1. **Neutral starting preferences:** stock-specific output weights/biases start neutrally and dynamic stock filters start alike, while remaining trainable.
2. **Train for the actual task:** the objective emphasizes relative stock rankings, and chronological validation separates checkpoint selection from diagnostics. These training changes were bundled, so their separate contributions are not isolated.
3. **Add a common anchor:** half the current score comes from the same training-history report card in every seed group, reducing the direct weight on seed-dependent neural outputs.
4. **Preserve committee portfolios:** average the portfolios of five-seed groups, limiting the effect of borderline top-five changes when more seeds are pooled.
5. **Check new initializations:** choose the candidate using development results, fix it before scoring the reserved cohort, and retain every seed's results.

The reserved group's **0.877 → 1.322** improvement is particularly informative: it has only one five-seed committee, so portfolio averaging cannot explain the gain. The training prior changes the stock ranking within the same issuer/risk/execution framework. Across all four disjoint groups, the newer baseline's SR spread falls **0.323 → 0.225**. The old selected recipe's two reported groups differed by **1.084** under a different model/training generation.

**The remaining limit:** all cohorts share repeatedly inspected 2022–2025 outcomes. The preferred candidate passes its separately frozen secondary seed-confirmation check; the original stricter spread gate and the all-ensemble 1.5 target remain failed. This supports improved initialization robustness, while performance on unseen market periods remains unverified. Failed shared-scorer, forest, family-weight and hedge experiments remain in the dated sections below.

[Full plain-language guide](docs/ndx_model_guide_20261030/GUIDE.md) · [Headline metrics](docs/ndx_model_guide_20261030/HEADLINE_METRICS.csv) · [Seed comparison](docs/ndx_model_guide_20261030/SEED_COMPARISON.csv) · [Benchmark replay and provenance](docs/ndx_model_guide_20261030/MANIFEST.json).

## Earlier study details and full experiment history

## Confirmed improvement across seed cohorts; SR>1.5 remains unmet

**The training-prior mixture improves all seven current reference groups, including a reserved five-seed cohort scored after the candidate was frozen.** This is initialization confirmation on the same market history, not an untouched-market result. The full follow-up retained **164 additional fits and 1,578 account records**, including the failed models, controls and cost stresses.

![Four separate seed cohorts: net wealth and drawdown](docs/ndx_seed_robustness_20261030/disjoint_ensembles.png)

| Ensemble | Previous stable issuer SR | Prior mixture SR | Change | Prior CAGR | Prior max DD |
|---|---:|---:|---:|---:|---:|
| Original five | 1.171 | 1.272 | +0.101 | 24.13% | -11.58% |
| Second five | 1.147 | 1.184 | +0.037 | 22.11% | -12.80% |
| Third five | 1.200 | 1.409 | +0.210 | 27.04% | -11.45% |
| Reserved five | 0.877 | 1.322 | +0.445 | 23.70% | -10.34% |
| Combined ten | 1.023 | 1.264 | +0.240 | 23.31% | -12.16% |
| Combined fifteen | 1.235 | 1.338 | +0.104 | 24.68% | -11.54% |
| Combined twenty | 1.245 | 1.367 | +0.122 | 24.60% | -11.20% |

![Ten-, fifteen- and twenty-seed portfolios compared](docs/ndx_seed_robustness_20261030/pooled_ensembles.png)

The candidate combines **50% training-history prior ranks + 25% THINK ranks + 25% dynamic-hypergraph ranks**. The prior uses observed training-return ranks with a **26-decision-week half-life** and 20 neutral pseudo-observations per stock. It ends before each fold's validation year; it is not a current six-month momentum signal. Every seed retains equal weight. Five-issuer portfolios are constructed within consecutive five-seed cohorts, then their signed targets are averaged and executed as one account. The combined book can hold more than five issuers.

The reserved five improve **0.877 → 1.322**, and the four disjoint groups' SR spread falls **0.323 → 0.225**. A single five-seed cohort has no aggregation change, so its gain comes from prior regularization within the same issuer/risk/execution framework. At **2× fees and 6% borrowing**, reserved-five SR improves **0.700 → 1.166**, and combined-twenty **1.067 → 1.236**. The existing 20% stock-volatility target, QQQ short cap of 75%, gross budget and fee model remain unchanged.

The original stricter source gate failed. This was a separately labeled, development-selected candidate frozen before reserved-seed financial outcomes; it passed its secondary confirmation test. **No strategy reached SR>1.5 across every predefined ensemble.** The source-selected old 1.731 result remains a historical best case, rather than representative seed performance. The reserved cohort is now spent for future model selection.

All 1,578 accounts pass independent dollar-position replay (maximum daily discrepancy **8.88e-16**); 14 new focused tests pass, and all 164 new-fit locks verify. All 63 displayed account/cost metrics reconstruct from saved daily returns. Subsequent beta-hedge and lower-volatility controls also fail; they are not adopted.

[Full findings, failed proposals and limits](docs/ndx_seed_robustness_20261030/REPORT.md) · [All comparison metrics and cost stresses](docs/ndx_seed_robustness_20261030/COMPARISON_METRICS.csv) · [Exact pre-score candidate](docs/ndx_seed_robustness_20261030/PRIOR_CONFIRMATION_PROTOCOL.json) · [Verification](docs/ndx_seed_robustness_20261030/AUDIT.json).

The dated gallery below preserves the experiment sequence; the confirmed comparison above supersedes earlier seed-consistency interpretations.

## Portfolio aggregation and shared-scorer experiments — October 30

![Portfolio target averaging versus the prior stable model and shared-scorer ablation](docs/ndx_portfolio_robustness_20261030_v2/pooled_comparison.png)

The fixed **60-fit / 540-account** study found a useful aggregation improvement: average five-seed cohorts' signed portfolio targets and execute the combined account. With the original retrained models, retention five and no trading band, combined-ten SR rises **0.837 → 1.267**, and combined-fifteen rises **1.057 → 1.330**. The separate five-seed groups remain **1.168 / 1.269 / 1.345**. Portfolio averaging preserves different selections; averaging individual-seed portfolios performed worse. These are development-history improvements, not untouched-market confirmation.

A 1% NAV trading band yields **1.170 / 1.283 / 1.358 / 1.281 / 1.336** across original/second/third/ten/fifteen groups. This row has the highest minimum SR in this fixed study, but **no recipe passes the stricter predefined improvement gate**, and none reaches SR>1.5 in every group. The original cohort is essentially unchanged and the disjoint-cohort gap exceeds the preset 0.15 limit. Bands do not provide a consistent improvement by themselves.

Removing THINK's stock-ID dense mixer while preserving all other initialized tensors, training settings, seeds and the unchanged dynamic family **substantially hurts performance**. The zero-band five-issuer blend's original/second/third/ten/fifteen SR is **0.304 / −0.054 / 0.179 / 0.266 / 0.205**. Removing stock-specific parameters therefore did not establish better generalization. All failed variants remain in the report. The later no-band pooling confirmation fails with reserved-five SR 0.854; the prior mixture above passes its separately frozen confirmation.

[Complete fixed study and separate-cohort plots](docs/ndx_portfolio_robustness_20261030_v2/REPORT.md) · [All 540 accounts](docs/ndx_portfolio_robustness_20261030_v2/METRICS.csv) · [Source/publication hashes](docs/ndx_portfolio_robustness_20261030_v2/MANIFEST.json).

### Family isolation and validation-only blend weights

![Model-family and validation-blend comparison](docs/ndx_family_robustness_20261030/family_comparison.png)

The conditional **300-account** follow-up also yields **zero** improvement-gate passes and no recipe above SR1.5 in every group. With retention/cohort-target averaging/no band, the fixed original 50:50 blend reaches original/second/third/ten/fifteen SR **1.168 / 1.269 / 1.345 / 1.267 / 1.330**. THINK+mixer alone reaches **0.814 / 0.938 / 0.877 / 0.882 / 0.883**; dynamic alone **0.735 / 0.584 / 0.854 / 0.696 / 0.779**. First-half-prior-year validation weights produce **1.148 / 1.103 / 1.270 / 1.154 / 1.219**. Both original families contribute to this nonlinear ranking/portfolio blend; separating them did not recover strong SR. Here "original" means the October 29 neutral-initialization/rank-trained source, not the older selected 1.731 model.

[Every family/control/cost result](docs/ndx_family_robustness_20261030/REPORT.md) · [All 300 accounts](docs/ndx_family_robustness_20261030/METRICS.csv) · [Validation-only weight choices](docs/ndx_family_robustness_20261030/VALIDATION.csv).

### Longer shared features and training-prior controls

![Long-history feature models compared with the existing neural portfolio](docs/ndx_tabular_alpha_20261030/model_comparison.png)

The 64-fit / 282-account shared ExtraTrees/Ridge arm fails. Standalone forest retention/cohort portfolios have original/second/third/ten/fifteen SR **0.425 / 0.272 / 0.412 / 0.356 / 0.380**; the 50:50 tree/neural blend reaches **0.561 / 0.532 / 0.821 / 0.557 / 0.656**. More randomized trees and longer causal price/volatility/liquidity histories did not provide a stronger financial predictor in this fixed test. [All feature-model results and deterministic controls](docs/ndx_tabular_alpha_20261030/REPORT.md).

![All fixed training-prior mixtures compared](docs/ndx_prior_robustness_20261030/model_comparison.png)

The separate 264-account prior arm retains all 13/26/52-week and full-history controls/blends. Its third-five 13-week issuer mixture reaches **1.652**, but the other two disjoint groups reach only **1.136 / 1.179**; that single favorable cohort was not selected as representative. The 26-week mixture's all-group improvement motivated the separately labeled confirmation above. [Every prior and control](docs/ndx_prior_robustness_20261030/REPORT.md).

## New models plotted together: daily wealth and drawdown

These comparisons compound the saved **net daily account returns**, including modeled transaction fees and 3% short borrowing. Each starts at $100. Model families are compared within the same seed cohort; the old selected strategy remains visible where matched forecasts exist. The eight named representatives include failed approaches and a deterministic control, rather than only the best retrained curve. Full grids remain linked below.

### Combined ten seeds

![Ten-seed models compared on net wealth and drawdown](docs/ndx_model_comparison_20261030/ten_seed_paths.png)

The old selected retention account still leads this ten-seed comparison at SR **1.381**. The loss-selected five-issuer model reaches **1.023**, versus **1.250** for the earlier stable-consensus postprocessing recipe and **1.123** for the seed-free training-mean-rank control. Retraining improved cohort consistency, but did not dominate the earlier models or the simple control.

### Three separate five-seed cohorts

![Original, second and third five-seed cohorts plotted separately](docs/ndx_model_comparison_20261030/disjoint_seed_paths.png)

| Model / portfolio representative | Original five SR | Second five SR | Third five SR | Combined ten SR | Combined fifteen SR |
|---|---:|---:|---:|---:|---:|
| Old selected retention | 1.731 | 0.648 | — | 1.381 | — |
| Old forecasts: stable consensus / soft ten | 1.230 | 0.958 | — | 1.250 | — |
| Retrained: loss checkpoint / five issuers | 1.171 | 1.147 | 1.200 | 1.023 | 1.235 |
| Retrained: loss checkpoint / retention five | 1.168 | 1.269 | 1.345 | 0.837 | 1.057 |
| Retrained: rank checkpoint / retention five | 0.902 | 0.690 | 0.827 | 0.863 | 0.872 |
| Retrained: three snapshots / five issuers | 1.005 | 0.977 | 0.991 | 0.894 | 1.057 |
| Rank checkpoint + liquid-consensus factor, 50:50 | 0.751 | 0.837 | 0.911 | 0.762 | 0.891 |

The training-mean-rank retention control is seed-free and has SR **1.123**, not five independent results. Dashes mean the old study did not fit those seeds. All displayed accounts use the 20% stock-volatility target and continuous QQQ short ceiling of 75%, except stable consensus / soft ten, which uses 50%. Issuer selection, retention, weighting and hedge ceilings differ between representatives; these are strategy comparisons, not isolated architecture effects. Several representatives were selected after inspecting their study outcomes; [selection details and full-grid links](docs/ndx_model_comparison_20261030/README.md) disclose those choices.

### Combined fifteen seeds

![Fifteen-seed retrained models, factor hybrid and deterministic control](docs/ndx_model_comparison_20261030/fifteen_seed_paths.png)

The loss-selected five-issuer recipe reaches SR **1.235** here. Pooling seed rankings does not average the individual portfolio returns: it changes top-five selections, retention state and volatility sizing. More seeds can therefore worsen a strategy, as the retention-five row illustrates.

### What improved, and the next experiments

The retraining neutralized random stock-ID offsets at initialization, replaced raw-return fitting with date-centered Huber plus pairwise ranking loss, split prior-year validation into checkpoint-selection and diagnostic halves, and retained all 15 predefined seeds. The five-issuer rule also consolidates share classes. The observed disjoint-cohort SR spread falls from **1.084** in the old retention comparison to **0.053** for the new issuer recipe. Initialization, objective, duration and validation changed together, so this does not isolate the cause of that improvement.

**This establishes better observed seed stability, not unseen-market generalization.** Every cohort shares the repeatedly inspected 2022–2025 market history. No trained recipe achieved SR>1.5 across every predefined ensemble; no untouched confirmation is added by these plots.

The next experiments I would prioritize are:

1. **Average portfolio targets instead of seed rankings.** Construct fixed equal-weight subportfolios for the predefined five-seed cohorts, combine their targets, and replay one account with actual net trades, risk limits and fees. Compare against rank pooling at the same risk budget. The strong disjoint retention accounts and weaker pooled accounts motivate this test; averaging their historical return series is not an executable backtest.
2. **Use actual-holdings no-trade bands.** Test a small predefined set of thresholds that keep a position when its new target is close to its drifted current weight, instead of relying only on the previous target's 0.02 score bonus. Evaluate net return, turnover and stressed costs together. The balanced issuer strategy falls to roughly SR 0.85–1.06 under 2× fees and 6% borrowing. Transaction-cost research motivates no-trade regions, but does not establish that they will help this strategy ([primary paper](https://www.nber.org/papers/w18709.pdf)).
3. **Ablate stock-identity shortcuts.** Compare the current trainable stock-specific output parameters with a shared stock-scoring head under the same initialization, objective, epoch budget and seeds. Require incremental validation performance over the training-mean-rank control. Neutral initialization removes the starting lottery; it does not prevent learning persistent stock preferences. This is a proposed test, not a demonstrated improvement.

Freeze a small experiment budget and compare across earlier chronological validation folds before evaluating reserved market data. Repeatedly selecting the best 2022–2025 result would add selection bias rather than prove a higher transferable SR ([Deflated Sharpe Ratio paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)).

[Comparison notes and reproduction](docs/ndx_model_comparison_20261030/README.md) · [32 compared accounts](docs/ndx_model_comparison_20261030/COMPARISON_METRICS.csv) · [Source hashes and verification](docs/ndx_model_comparison_20261030/MANIFEST.json). Every source path and metrics file matches its completed-run hash; every compared account's performance metrics reconstruct from daily returns. No models were retrained and no new accounts were simulated for this chart update. Earlier dated galleries follow unchanged.

## Rank-aligned retraining: more stable, below the SR>1.5 target

![Old and retrained ensemble comparisons](docs/ndx_rank_retraining_20261029/balanced_ensembles.png)

The fixed 120-fit/15-seed study and separate factor-control arm retained **774 accounts**. **No recipe reaches SR>1.5 in every predefined ensemble.** The retrained loss-selected five-issuer recipe has original/new/third-five SR **1.171 / 1.147 / 1.200**, with ten/fifteen SR **1.023 / 1.235**. The three disjoint groups are much closer, but the original-five and ten-seed historical results decline. Under 2× fees plus 6% borrow, those five Sharpes are 0.983 / 0.969 / 1.013 / 0.848 / 1.062.

The experiment neutralizes random stock-ID initialization, uses centered ranking-aligned training, separates epoch-selection validation from diagnostic validation, and reports all seeds/checkpoints. These are bundled changes, not an isolated estimate of each intervention. Momentum/risk/liquidity controls and blends also fail the target. All account paths pass independent accounting replay. The original selected 1.731 result below is preserved with its initialization and concentration caveats; no fresh-market confirmation is claimed.

[Findings and limitations](docs/ndx_rank_retraining_20261029/FINDINGS.md) · [All retraining accounts](docs/ndx_rank_retraining_20261029/METRICS.csv) · [Factor-control results](docs/ndx_factor_controls_20261029/REPORT.md).

## Seed-attribution concern in the historical leader

The 1.731 original-five Sharpe below is a selected historical account, with 0.648 in the new-five group and 1.381 across ten seeds. Stock-level attribution and parameter inspection identify a specific robustness concern: in 2023, THINK's daily stock ordering correlates about 0.995 with its stock-specific mixer biases, which retain about 0.999 rank correlation with reconstructed random initial values. The original account also averaged 32.6% NAV in combined GOOG/GOOGL positions in that year. This does not establish transferable learned stock-ranking ability. [Attribution and caveats](docs/ndx_seed_attribution_20261029/REPORT.md) · [Paper-reporting guidance](docs/ndx_seed_attribution_20261029/PAPER_REPORTING.md).

## Validation ranking stability: consistency improved with a return tradeoff

![All fixed recipes compared across original and new seed ensembles](docs/ndx_rank_stability_20261029_v2/seed_consistency.png)

The fixed 24-recipe / 216-account study tested validation RankIC weights, quarter-stability penalties, cross-model disagreement penalties, and broader allocations. **No recipe preserved approximately 1.7 Sharpe across all three seed ensembles, and none met the secondary performance-preserving consistency goal.** These are historical development results; the current headline leader below is preserved.

| Recipe | Original-five SR | New-five SR | Ten-seed SR | Original/new SR gap |
|---|---:|---:|---:|---:|
| Prior retention-five / 75% hedge cap | 1.731 | 0.648 | 1.381 | 1.084 |
| Validation RankIC weights / retention-five / 75% cap | 1.377 | 0.841 | 1.349 | 0.537 |
| Stable weights + consensus penalty / smoothly weighted ten / 50% cap | 1.230 | 0.958 | 1.250 | 0.272 |

The highest weaker-half result narrows the gap by about 75%, but reduces original-five and ten-seed Sharpe. With 2× costs and 6% borrow, its original/new/ten Sharpes are 1.067 / 0.778 / 1.087. Higher validation ranking quality did not consistently translate into better subsequent financial performance. These tests use the existing final checkpoints; ranking-based checkpoint selection requires a separate fit experiment. All 18 matching baselines reproduce exactly and all 216 accounts pass independent accounting replay.

[Interpretation and limitations](docs/ndx_rank_stability_20261029_v2/FINDINGS.md) · [All recipes and reproduction](docs/ndx_rank_stability_20261029_v2/REPORT.md) · [Validation diagnostics](docs/ndx_rank_stability_20261029_v2/VALIDATION_METRICS.csv) · [All216accounts](docs/ndx_rank_stability_20261029_v2/METRICS.csv).

## Retention strategy reproduced in feat-rs

![Retention and continuous QQQ hedge comparisons across all seed groups](docs/ndx_retention_replication_20261029/retention_comparison.png)

The pinned `tom-shlom` retention strategy reproduces **1.731 Sharpe / 33.11% CAGR / -13.36% maximum drawdown** in the original-five ensemble. The ten-seed version returns **1.381 / 26.35% / -13.79%**; the new-five half returns **0.648 / 12.31% / -17.11%**. These are selected historical development results, with substantial seed sensitivity.

| Original-five account | Sharpe | CAGR | Max drawdown |
|---|---:|---:|---:|
| Previous issuer-five / 50% hedge cap | 1.454 | 27.79% | -14.56% |
| Retention-five / unhedged | 1.383 | 33.71% | -18.99% |
| Retention-five / 50% hedge cap | 1.724 | 32.97% | -12.24% |
| Retention-five / 75% hedge cap | **1.731** | **33.11%** | **-13.36%** |
| Retention-five / 75% cap / 2× costs and 6% borrow | 1.570 | 29.85% | -15.15% |

Retention adds 0.02 to each previous target's score before selecting five securities. Current eligibility still applies, and state continues across years. The unchanged trained forecasts and 20% basket-volatility target supply the long book. QQQ is hedged continuously up to the cap, remaining gross budget and volatility limit; 75% is a ceiling. Securities are separate from issuers, so share classes can occupy separate slots. This comparison changes selection and risk inputs as well as turnover; it does not isolate transaction-cost savings as the cause of improvement.

All **96 accounts** match the published source daily paths and target schedules exactly. Twelve existing `feat-rs` baselines also reproduce; all accounts pass independent dollar-position replay, and all source ledgers and 384 calendar returns match within declared numerical tolerances. No models were retrained or additional parameters searched. [Report and reproduction commands](docs/ndx_retention_replication_20261029/REPORT.md) · [All metrics](docs/ndx_retention_replication_20261029/METRICS.csv) · [Audit](docs/ndx_retention_replication_20261029/AUDIT.json).

The workbook and 472-account gallery below retain the October 27 snapshot. The new 96-account replication and matched comparisons are supplied separately above.

## Previous October 27 comparison

The following dated results are preserved; the retention reproduction above supersedes their headline leader claim.

## Highest observed Sharpe: model hedge experiment

![Original-five and ten-seed model hedge comparisons with SPY and QQQ](docs/ndx_model_hedge_20261027/high_sharpe_comparison.png)

**The new highest observed Sharpe is 1.454 for the original-five stock blend with a continuous partial QQQ hedge: 27.79% CAGR / −14.56% maximum drawdown. The ten-seed version reaches 1.325 Sharpe: 24.83% CAGR / −14.78% drawdown.** This is an always-on hedge control. Its timing does not come from the trained models.

**Our own models did not supply a useful hedge trigger in this test.** All four raw-forecast signals stayed off at all 201 rebalances in every seed group. Their account paths exactly match the unhedged baseline. All four bias-corrected signals fired at the same 50 rebalances in 2023 and stayed off in 2022/2024/2025. They missed 2022's weakness and hedged during 2023's rebound. The best active model-signal results were 1.111 Sharpe for original-five and 1.045 for ten seeds, below their unhedged 1.178/1.112 baselines. Equal-Sharpe calibrated variants are ties, not distinct replications.

|Strategy|Seeds|Sharpe|CAGR|Max drawdown|95% SR interval|
|---|---|---:|---:|---:|---|
|Continuous / hedge up to 50%|Original five|1.454|27.79%|-14.56%|[0.463, 2.441]|
|QQQ MA200 / hedge up to 50%|Original five|1.315|28.10%|-17.05%|[0.325, 2.309]|
|Calibrated family agreement / hedge up to 50%|Original five|1.111|26.27%|-20.56%|[0.154, 2.078]|
|Five issuers + 20% volatility target|Original five|1.178|28.57%|-20.56%|[0.223, 2.137]|
|SPY|Benchmark|0.498|10.85%|-24.55%|[-0.367, 1.419]|
|QQQ|Benchmark|0.469|11.93%|-34.87%|[-0.408, 1.394]|
|Continuous / hedge up to 50%|Ten seeds|1.325|24.83%|-14.78%|[0.388, 2.238]|
|QQQ MA200 / hedge up to 50%|Ten seeds|1.234|26.24%|-14.82%|[0.300, 2.178]|
|Calibrated family agreement / hedge up to 25%|Ten seeds|1.045|24.64%|-21.69%|[0.130, 1.984]|
|Five issuers + 20% volatility target|Ten seeds|1.112|26.81%|-21.69%|[0.190, 2.047]|

All neural rows select **five stock issuers**, regardless of seed count. Hedge rows add a short QQQ position capped at 25% or 50% NAV, available cash budget and a 20% QQQ volatility target. The highest-SR control keeps the 50% cap on continuously; the model rules use only their predictions to turn hedging on. QQQ's trailing volatility still sizes the short. Gross target exposure is capped at 100%; positions can drift between regular-close fills. Cash earns zero. Base modeled trading costs and assumed 3% annual borrow are included.

**495 accounts / 37 recipes:** eight own-model signals crossed with cash exit, 25% hedge, 50% hedge and 50% short switch, plus unhedged, QQQ-MA200 and continuous-hedge controls, across original-five/new-five/ten-seed groups and cost/borrow assumptions. The strongest Sharpe control is selected from inspected historical results. Confidence intervals condition on fitted paths and do not adjust prior model searches. The newer five-seed half remains materially weaker. No claim of prospective superiority follows.

[Complete signal definitions, seed/cost results and audit](docs/ndx_model_hedge_20261027/REPORT.md) · [All 495 accounts](docs/ndx_model_hedge_20261027/metrics.csv) · [All signal decisions](docs/ndx_model_hedge_20261027/signals.csv) · [Comparison workbook](docs/NASDAQ100_strategy_comparison.xlsx)

## Updated master graph: all 472 main strategy accounts

![All 472 main strategy accounts with continuous-hedge leaders and benchmarks emphasized](docs/ndx_model_hedge_20261027/master_all_strategies.png)

The previous 364 accounts are preserved, with 108 new nonbaseline hedge-study accounts added. Different rules can produce identical paths; 472 counts declared accounts, not 472 independent strategies. Full cost/borrow cases remain in the workbook's Hedge accounts sheet. Open Model hedges for the new comparison charts.

## Previous published comparison: October 26 snapshot

The sections below preserve the preceding dated comparison. The October 27 results above supersede its headline leader claims.

Updated October 26, 2026. Historical monthly NASDAQ-100 membership, January 3, 2022–December 31, 2025. Every path uses the same 1,003 sessions.

## Highest-Sharpe models versus SPY and QQQ

![New high-Sharpe models compared with SPY and QQQ, separated by seed ensemble](docs/ndx_drawdown_20261026/master_update/high_sharpe_comparison.png)

**The highest observed Sharpe is now 1.199** for the original-five blend with a 20% basket-volatility target: **28.96% CAGR and -17.78% drawdown**. The **highest ten-seed Sharpe is 1.112**, using five-issuer selection plus a 20% volatility target: **26.81% CAGR and -21.69% drawdown**. The 25% target plus issuer/trend control retains more ten-seed return at **28.70% CAGR / 1.085 Sharpe / -22.98% drawdown**.

| Strategy | Seed ensemble | Sharpe | CAGR | Max drawdown | 95% SR interval |
| --- | --- | ---: | ---: | ---: | --- |
| **20% basket volatility target** | **Original five** | **1.199** | **28.96%** | **-17.78%** | **[0.264, 2.146]** |
| 25% volatility target + trend cap | Original five | 1.190 | 31.50% | -18.99% | [0.273, 2.118] |
| 20% volatility target + trend cap | Original five | 1.190 | 28.08% | -17.16% | [0.254, 2.130] |
| Five issuers + 25% volatility target + trend cap | Original five | 1.169 | 31.20% | -21.68% | [0.232, 2.110] |
| Five issuers + 20% volatility target | Original five | 1.178 | 28.57% | -20.56% | [0.234, 2.147] |
| **Five issuers + 20% volatility target** | **Ten seeds** | **1.112** | **26.81%** | **-21.69%** | **[0.209, 2.045]** |
| Five issuers + 25% volatility target + trend cap | Ten seeds | 1.085 | 28.70% | -22.98% | [0.191, 1.998] |
| Full-exposure blend | Original five | 0.919 | 32.94% | -34.89% | [0.041, 1.833] |
| Full-exposure blend | Ten seeds | 0.896 | 31.87% | -40.94% | [0.001, 1.829] |
| **SPY** | **Benchmark** | **0.498** | **10.85%** | **-24.55%** | **[-0.367, 1.419]** |
| **QQQ** | **Benchmark** | **0.469** | **11.93%** | **-34.87%** | **[-0.408, 1.394]** |

The figure compares four new risk-control recipes, the corresponding full-exposure blend, and SPY/QQQ. Left: original five seeds; right: all ten seeds. Each panel has its own legend with full-period SR and CAGR for every line. Drawdown panels below the wealth charts show the percentage decline from each account's previous peak; 0% means a new or matched high. The original-five leader is a diagnostic ensemble; the ten-seed comparison remains the broader seed check.

**Historical development findings:** six of 305 frozen recipes passed the screen. All 915 base-cost accounts and 54 additional cost stresses are retained in the workbook's Drawdown accounts sheet and the study report. The new-five half remains weaker, and paired SR-improvement intervals include zero. These results are selected from repeatedly inspected 2022-2025 history.

The new rules keep the THINK/dynamic forecasts and 50:50 rank blend. They change issuer selection and/or stock exposure using prior-close basket volatility and an optional QQQ/200-session-average cap. Five-session decisions remain at 09:20 ET with regular-close fills. No leverage; idle cash earns zero. A volatility target is not a guaranteed drawdown limit. [Complete drawdown study and audit](docs/ndx_drawdown_20261026/REPORT.md).

## Master graph: all 364 main strategies

![All 364 main strategies with the newest Sharpe leaders, SPY and QQQ emphasized](docs/ndx_drawdown_20261026/master_update/master_all_strategies.png)

The thick blue and teal lines identify the original-five and ten-seed Sharpe leaders. SPY and QQQ are emphasized in red and purple. The other 360 paths are thin gray lines. This combines the previous 346 main strategies with 18 seed-group accounts from the six qualifying risk-control recipes. The complete 305-recipe search remains separately disclosed, including failures.

## New drawdown-control gallery

![Drawdown-control qualifiers, first three recipes](docs/ndx_drawdown_20261026/master_update/drawdown_leaders_01.png)

![Drawdown-control qualifiers, next three recipes](docs/ndx_drawdown_20261026/master_update/drawdown_leaders_02.png)

## October 24 full-exposure snapshot: THINK+mixer AH+PM + dynamic hypergraph

**The highest-Sharpe full-exposure model in the October 24 comparison was the 50:50 rank blend of THINK+mixer AH+PM and a restricted dynamic hypergraph network, using the original five seeds and five holdings: 0.919 Sharpe / 32.94% CAGR.** The same recipe across all ten seeds returned **0.896 / 31.87%**. The table keeps these seed ensembles separate and emphasizes the leader and matched SPY/QQQ benchmarks.

| Strategy / seed ensemble                                                           |      Holdings |          Sharpe |             CAGR |           RankIC |      Max drawdown | 95% SR interval           |
| ---------------------------------------------------------------------------------- | ------------: | --------------: | ---------------: | ---------------: | ----------------: | ------------------------- |
| **Best model: THINK+mixer AH+PM + dynamic HG / original five / 50:50 ranks** |   **5** | **0.919** | **32.94%** | **0.0180** | **-34.89%** | **[0.041, 1.833]**  |
| Same blend / ten seeds                                                             |             5 |           0.896 |           31.87% |           0.0189 |           -40.94% | [0.001, 1.829]            |
| Same blend / original five                                                         |            10 |           0.848 |           27.79% |           0.0180 |           -35.04% | [-0.006, 1.739]           |
| Same blend / ten seeds / primary portfolio                                         |            10 |           0.755 |           23.92% |           0.0189 |           -34.88% | [-0.095, 1.649]           |
| THINK+mixer AH+PM / tuned momentum SGD / ten seeds                                 |             5 |           0.719 |           25.05% |           0.0238 |           -45.29% | [-0.227, 1.715]           |
| Original daily THINK+mixer / original five                                         |            10 |           0.629 |           19.60% |           0.0074 |           -39.44% | [-0.186, 1.505]           |
| **SPY / buy and hold**                                                       | **ETF** | **0.498** | **10.85%** |               — | **-24.55%** | **[-0.367, 1.419]** |
| **QQQ / buy and hold**                                                       | **ETF** | **0.469** | **11.93%** |               — | **-34.87%** | **[-0.408, 1.394]** |

The leader has both the highest Sharpe and highest CAGR across the 346 base-cost strategies. This is a retrospective development ranking, including diagnostic seed halves and portfolio variants. Top ten was primary; top five was a declared secondary test. The repeatedly inspected history does not supply an untouched confirmation result.

## Earlier master graph: October 24 models

![All 346 strategy paths, with the best historical model, SPY and QQQ shown as thick labeled lines](docs/strategy_comparison_20261024/overview_20261026/master_all_strategies.png)

All 346 main base-cost strategies are included. The **best model is the thick blue line**, **SPY is the thick dashed red line**, and **QQQ is the thick purple line**; the other 343 paths are thin gray lines. Each starts with $100 before the common evaluation period and compounds its saved daily returns after modeled costs. Additional individual-seed and cost-stress diagnostics remain in the complete gallery below. [Figure provenance and verification](docs/strategy_comparison_20261024/overview_20261026/MANIFEST.json).


## What the dynamic hypergraph does, in plain language

Think of each stock as having a recent trading history and several overlapping groups of peers. The network asks two questions: **"What has this stock been doing?"** and **"What are related stocks doing that helps interpret it?"** It learns how to combine the answers into a forecast of that stock's next five-session return.

An ordinary graph connects stocks in pairs. A **hypergraph connects a whole group at once**. A stock can belong to several groups, with different strengths of membership. In this model, there are **64 learned groups**. Their memberships are calculated from the latest stock representations at each forecast date, so a stock can participate differently as its trading pattern changes. Those groups have no supplied industry names; interpreting one as a particular sector would require a separate analysis.

The network first reads each stock's 16-session history, using attention to combine past observations and a learned decay mechanism to represent how historical influences accumulate and fade. It then gathers information from stocks into the learned groups, mixes the group information, and passes it back to the stocks. This lets a stock's forecast reflect a broader pattern that may be hard to see in that stock alone.

Alongside those changing groups, a fixed relationship map is built from **training-period return correlations**. It gives the network another route for sharing information between historically related stocks. Learned weights combine the stock's own history, the dynamic group information, and this fixed-map information. A small prediction head produces one return forecast per stock.

"Restricted" describes the adaptation used here: one training-only correlation snapshot replaces the author pipeline's broader sector and lead/lag snapshots. It is a NASDAQ transfer, with the project's own inputs and training budget. [Dynamic network adapter](latest/models/ndx_reference_models.py).

## How its scores combine with THINK+mixer

**There are two levels of averaging: seeds within each model family, then a fixed 50:50 average between families.** A seed is a separate training initialization of the same model recipe.

1. Each trained THINK+mixer and dynamic network predicts returns for the eligible stocks.
2. Each network's forecasts become cross-sectional percentile ranks: rank the eligible stocks from lowest to highest forecast, then divide by the eligible stock count. Higher is better; tied forecasts receive average ranks.
3. Average those percentile ranks across the fixed seeds within each family. The historical leader uses five THINK+mixer networks and five dynamic networks; the ten-seed version uses ten of each.
4. Calculate **final score = 0.5 × THINK family average + 0.5 × dynamic family average**. The family averages enter this calculation directly, without another ranking step first.
5. Select the five highest final scores and assign equal 20% target weights. Boundary ties receive fractional allocation across tied names; trading constraints can make actual weights differ from targets.

For example, suppose the family averages for three hypothetical stocks are:

| Stock | THINK+mixer average percentile | Dynamic average percentile | Final 50:50 score |
| ----- | -----------------------------: | -------------------------: | ----------------: |
| A     |                           0.96 |                       0.90 |    **0.93** |
| B     |                           1.00 |                       0.50 |    **0.75** |
| C     |                           0.72 |                       0.94 |    **0.83** |

The combined order is A, C, B. Agreement near the top helps A outrank B even though THINK alone prefers B. Converting forecasts to ranks gives both families comparable influence regardless of the numerical scale of their predicted returns. **The average combines stock rankings into one portfolio.** Its 50:50 weights are fixed in advance; they are not fitted to test returns. [Seed averaging](research/training/run_robust_research.py), [family blending](latest/reporting/report_ndx_reference_networks.py).

```mermaid
flowchart TD
  A[16-session daily + after-hours + premarket history] --> B[THINK+mixer AH+PM forecasts per seed]
  A --> C[Dynamic hypergraph forecasts per seed]
  B --> D[Percentile ranks, then average across THINK seeds]
  C --> E[Percentile ranks, then average across dynamic seeds]
  D --> F[Final score = 0.5 THINK + 0.5 dynamic]
  E --> F
  F --> G[Choose top five stocks]
  G --> H[Equal target weights; fill at regular close]
```

## Architecture, inputs and training behind the leader

Both families use **16 sessions and 25 stock-feature channels**. Five daily channels describe return, price relative to its 5- and 20-day averages, 20-day volatility, and relative dollar volume. Ten after-hours and ten premarket channels describe return, range, within-bar volatility, price relative to VWAP, dollar volume, last-hour return, late volume share, coverage, first-trade delay, and last-trade age. Train-only normalization clips inputs at eight standard deviations and scales them by 0.2; missing extended-hours summaries use training means.

The **THINK daily core** maps inputs into a Poincaré ball with curvature -1, compresses 16 sessions to four using a kernel of four, exchanges group information through distance-modulated hypergraph attention, then compresses the remaining four positions into one forecast. Its hidden width is 16. Each training-derived group contains a stock and up to its four most positively correlated peers. A cross-stock mixer adds a learned correction through an N-to-16-to-N network, where N is the fixed security-ID universe. This THINK implementation is an equation-based reconstruction, with documented choices where the paper was ambiguous.

The **AH+PM correction branch** flattens each stock's 16 × 20 extended-hours channels into a 320-to-16-to-1 network with GELU and a final tanh. Its output layer starts at zero. The correction is `0.5 × core score SD × (stock tanh output − eligible-stock mean tanh output)`, with the SD detached from gradient computation. This constrains the correction relative to the core's forecast spread. The core and auxiliary branch train jointly. [THINK core](research/models/think_model.py), [mixer](research/models/research_models.py), [extended-hours branch](research/models/ndx_extended_hours.py).

The **dynamic network** embeds 25 inputs into 32 dimensions, uses four-head causal attention plus a learned intensity/decay function and temporal attention, and combines two wavelet graph-convolution layers with 64 learned dynamic hyperedges. Learned softmax weights fuse the three representations before a 32-to-32-to-1 prediction head. It has 25,445 parameters in the saved fits.

| Training setting                         | THINK+mixer AH+PM                                                | Restricted dynamic hypergraph                   |
| ---------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------- |
| Target                                   | Five-session economic stock return, in percentage points         | Same target                                     |
| Optimizer                                | AdamW                                                            | Adam                                            |
| Learning rate                            | 0.001                                                            | 0.0001                                          |
| Weight decay                             | 0.001                                                            | 0                                               |
| Training loss                            | Masked mean squared error                                        | Per-date sqrt(masked MSE + 1e-8)                |
| Update batch                             | Eight chronological forecast dates                               | One forecast date, across stocks                |
| Hidden width                             | 16; auxiliary width 16                                           | 32; four attention heads; 64 learned hyperedges |
| Dropout                                  | No explicit dropout in this branch                               | 0.1                                             |
| Maximum epochs / early-stopping patience | 30 / 6                                                           | 30 / 6                                          |
| Gradient norm cap                        | 1.0                                                              | 1.0                                             |
| Checkpoint selection                     | Earliest prior-year validation MSE improvement greater than 1e-7 | Same rule                                       |

Each test year receives fresh annual fits. Training/validation/test periods advance chronologically: July 2019–2020 / 2021 / 2022; July 2019–2021 / 2022 / 2023; July 2019–2022 / 2023 / 2024; and 2020–2023 / 2024 / 2025. Outcomes crossing fold boundaries are purged. Normalization and correlation maps use training data only. The leader's original seeds are **7, 19, 42, 61 and 97** in each family. Neither winning branch directly optimizes Sharpe or a pairwise ranking loss; validation MSE selects checkpoints, and the fixed ranking rule converts forecasts into holdings. [Parent protocol](config/ndx_extended_hours_20261024.json), [reference protocol](latest/config/ndx_reference_networks_20261024.json).

Premarket inputs end at **09:15 ET**, the decision is at **09:20**, and execution occurs at that session's regular close. Daily inputs end at the preceding regular close. Rebalancing occurs every five exchange sessions. Accounts include modeled fees and participation-dependent costs, liquidity caps, cash constraints, and corporate-action accounting.

## Why the blend looks useful in this history

At five holdings, the ten-seed THINK+mixer AH+PM branch returned **0.531 Sharpe / 16.73% CAGR**, the dynamic branch returned **0.657 / 20.53%**, and their blend returned **0.896 / 31.87%**. The two families' rankings correlate about **0.41**, and their top-ten selections overlap about **29%**, so they contribute substantially different stock choices. Averaging across seeds reduces dependence on one initialization; averaging across families combines different ways of interpreting the same inputs.

These diagnostics support complementarity as an explanation, without establishing that any one architectural feature causes the gain. Raw forecast errors remain highly correlated (about 0.998). The original-five leader's RankIC is only 0.018, so the full ranking signal is weak even though the selected top-five portfolio did well. Its annual net returns were approximately **-23.1%, +72.9%, +30.7% and +78.8%** for 2022–2025. The wider uncertainty and seed comparisons below remain essential context. [Complementarity](latest/results/reference_networks/complementarity.csv), [annual results](latest/results/reference_networks/annual.csv).

## New findings: losses, optimizers and complementary networks

The fixed THINK+mixer AH+PM + restricted dynamic-hypergraph rank blend is the strongest new historical candidate. Its ten-seed portfolio returned **31.87% CAGR / 0.896 Sharpe at five holdings**, or **23.92% / 0.755 at ten**. Five-holding momentum SGD returned **25.05% / 0.719**, with 0.759 and 0.799 Sharpe in its two fixed seed halves. Top ten was primary; top five was a predeclared secondary portfolio test. No genuinely untouched-period result exists yet.

| Strategy                                     | Holdings | Ten-seed CAGR | Ten-seed SR | Original-five SR | New-five SR | Max drawdown | 95% SR interval |
| -------------------------------------------- | -------: | ------------: | ----------: | ---------------: | ----------: | -----------: | --------------- |
| THINK+mixer AH+PM / MSE / fixed AdamW        |       10 |        15.36% |       0.519 |            0.700 |       0.480 |      -38.32% | [-0.345, 1.425] |
| THINK+mixer AH+PM + dynamic HG / 50:50 ranks |       10 |        23.92% |       0.755 |            0.848 |       0.467 |      -34.88% | [-0.095, 1.649] |
| THINK+mixer AH+PM / MSE / tuned momentum SGD |       10 |        19.22% |       0.626 |            0.634 |       0.466 |      -33.81% | [-0.277, 1.553] |
| THINK+mixer AH+PM / MSE / fixed AdamW        |        5 |        16.73% |       0.531 |            0.485 |       0.106 |      -44.05% | [-0.360, 1.467] |
| THINK+mixer AH+PM + dynamic HG / 50:50 ranks |        5 |        31.87% |       0.896 |            0.919 |       0.633 |      -40.94% | [0.001, 1.829]  |
| THINK+mixer AH+PM / MSE / tuned momentum SGD |        5 |        25.05% |       0.719 |            0.759 |       0.799 |      -45.29% | [-0.227, 1.715] |

SPY: 10.85% CAGR / 0.498 Sharpe. QQQ: 11.93% / 0.469. Huber and pseudo-Huber weakened this AH+PM recipe. Standalone MASTER and PRISM-VQ transfers also underperformed. The restricted dynamic network alone returned 15.32% / 0.530, showing why a complementary blend can be more useful than choosing the strongest standalone CAGR. All negative cells remain in the workbook and gallery.

![New findings overview](docs/strategy_comparison_20261024/figures/new_findings_overview.png)

## Seed robustness and uncertainty

The ten-holding blend’s original-five/new-five Sharpe split is 0.848 versus 0.467; its new-five result does not improve the matched baseline’s 0.480. At five holdings the blend improves on the matched baseline in both halves, while SGD is more consistent across those halves. These are initialization diagnostics on one shared history, not independent market replications. SGD’s low turnover appears partly related to stable stock-specific mixer preferences, which also need testing on new dates.

For the primary ten-holding blend, the paired 95% SR difference versus SPY is approximately [-0.161, 0.649]. Its improvement versus the baseline is not conclusive after the within-batch comparison adjustment. An additional secondary five-holding sensitivity gives a paired SR difference versus SPY of [-0.077, 0.849]; accounting for both holding counts and tested methods also leaves superiority over QQQ inconclusive. These secondary diagnostics do not turn the selected historical leader into a prespecified confirmation result. [Additional secondary sensitivity](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/new_holdings_uncertainty.csv).

Absolute SR intervals use a fixed 3% annual hurdle and 10,000 circular-block draws, with 20 sessions primary and 5/60-session sensitivity. They condition on fitted paths, omit the full prior research search and retraining uncertainty, and do not give the probability of beating a benchmark.

![Seed robustness](docs/strategy_comparison_20261024/figures/new_findings_seeds.png)

![Primary uncertainty](docs/strategy_comparison_20261024/figures/new_findings_intervals.png)

The author networks are NASDAQ transfers with declared feature and training adaptations. The restricted dynamic model replaces the full sector/lead-lag graph inputs with one train-only correlation snapshot. PRISM’s current factor vintage does not establish historical publication availability. These are not reproductions of published paper performance.

## Historical leaders in closer detail

![Historical leaders](docs/strategy_comparison_20261024/figures/historical_leaders.png)

## Earlier unblended AH+PM holding-count test

Before the new optimizer/network tests, the original-five unblended AH+PM ensemble returned 14.51% CAGR / 0.485 SR at five holdings versus 22.39% / 0.700 at ten. Those saved-forecast results are preserved in AH PM holding counts and [the audited report](docs/ndx_extended_hours_top5_20261024/REPORT.md). They are a different strategy from the new dynamic rank blend.

## Files and scope

- [Updated comparison workbook](docs/NASDAQ100_strategy_comparison.xlsx): 364 distinct base-cost strategies; 105 Models, 49 Defensive and 210 Portfolio variants. Benchmarks repeat in the latter tabs for convenience.
- [Every extended-hours account](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/extended_hours_all_accounts.csv): all 66 accounts, including individual seeds and 2x/4x fees, also in Extended-hours.
- [Chart catalog](docs/strategy_comparison_20261024/chart_catalog.csv): 652 chart-panel entries covering 364 main strategies and 288 additional diagnostic accounts. Repeated reference accounts are represented once.
- [All strategy metrics](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/all_strategies.csv) and [5/20/60-session SR sensitivity](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/sharpe_block_sensitivity.csv).

Defensive uses the fixed QQQ/MA200 half-exposure rule. Portfolio variants change holding count or long/cash/short allocation. All main rows are ensemble or declared selection procedures; individual extended-hours seeds are diagnostics. Failed and negative strategies are retained.

For fee-stress charts the red/purple comparators remain the base-cost SPY/QQQ references; the strategy fee multiplier is stated in each panel. No new models were fit for this consolidation. Retired 2026-selected survivor universes and unrelated/deferred NYSE studies are excluded from this active-model ranking.

- [All 346 new experiment accounts](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/new_experiment_all_accounts.csv). New findings compares seed halves; New accounts includes individual seeds and fee stresses.

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

- [Sealed loss/optimizer metrics](runs/ndx_loss_optimizer_20261024/metrics.csv) and [audit](runs/ndx_loss_optimizer_20261024/AUDIT.json)
- [Sealed reference-network metrics](runs/ndx_reference_networks_20261024/metrics.csv) and [audit](runs/ndx_reference_networks_20261024/AUDIT.json)

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

### All additional five-holding follow-up accounts

![Five-holding follow-up](docs/strategy_comparison_20261024/figures/top5_followup_01.png)

![Five-holding follow-up](docs/strategy_comparison_20261024/figures/top5_followup_02.png)

![Five-holding follow-up](docs/strategy_comparison_20261024/figures/top5_followup_03.png)

### All new loss, optimizer, reference and rank-blend accounts

![New experiment accounts 1](docs/strategy_comparison_20261024/figures/new_experiments_01.png)

![New experiment accounts 2](docs/strategy_comparison_20261024/figures/new_experiments_02.png)

![New experiment accounts 3](docs/strategy_comparison_20261024/figures/new_experiments_03.png)

![New experiment accounts 4](docs/strategy_comparison_20261024/figures/new_experiments_04.png)

![New experiment accounts 5](docs/strategy_comparison_20261024/figures/new_experiments_05.png)

![New experiment accounts 6](docs/strategy_comparison_20261024/figures/new_experiments_06.png)

![New experiment accounts 7](docs/strategy_comparison_20261024/figures/new_experiments_07.png)

![New experiment accounts 8](docs/strategy_comparison_20261024/figures/new_experiments_08.png)

![New experiment accounts 9](docs/strategy_comparison_20261024/figures/new_experiments_09.png)

![New experiment accounts 10](docs/strategy_comparison_20261024/figures/new_experiments_10.png)

![New experiment accounts 11](docs/strategy_comparison_20261024/figures/new_experiments_11.png)

![New experiment accounts 12](docs/strategy_comparison_20261024/figures/new_experiments_12.png)

![New experiment accounts 13](docs/strategy_comparison_20261024/figures/new_experiments_13.png)

![New experiment accounts 14](docs/strategy_comparison_20261024/figures/new_experiments_14.png)

![New experiment accounts 15](docs/strategy_comparison_20261024/figures/new_experiments_15.png)

![New experiment accounts 16](docs/strategy_comparison_20261024/figures/new_experiments_16.png)

![New experiment accounts 17](docs/strategy_comparison_20261024/figures/new_experiments_17.png)

![New experiment accounts 18](docs/strategy_comparison_20261024/figures/new_experiments_18.png)

![New experiment accounts 19](docs/strategy_comparison_20261024/figures/new_experiments_19.png)

![New experiment accounts 20](docs/strategy_comparison_20261024/figures/new_experiments_20.png)

![New experiment accounts 21](docs/strategy_comparison_20261024/figures/new_experiments_21.png)

![New experiment accounts 22](docs/strategy_comparison_20261024/figures/new_experiments_22.png)

![New experiment accounts 23](docs/strategy_comparison_20261024/figures/new_experiments_23.png)

![New experiment accounts 24](docs/strategy_comparison_20261024/figures/new_experiments_24.png)

![New experiment accounts 25](docs/strategy_comparison_20261024/figures/new_experiments_25.png)

![New experiment accounts 26](docs/strategy_comparison_20261024/figures/new_experiments_26.png)

![New experiment accounts 27](docs/strategy_comparison_20261024/figures/new_experiments_27.png)

![New experiment accounts 28](docs/strategy_comparison_20261024/figures/new_experiments_28.png)

![New experiment accounts 29](docs/strategy_comparison_20261024/figures/new_experiments_29.png)

![New experiment accounts 30](docs/strategy_comparison_20261024/figures/new_experiments_30.png)

![New experiment accounts 31](docs/strategy_comparison_20261024/figures/new_experiments_31.png)
