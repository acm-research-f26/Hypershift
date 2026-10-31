# Fine-tuning improves the blend, but the strict seed-consistency goal fails

The strongest recipe in this first screen is **three-epoch input-dropout fine-tuning without the extra consistency penalty or EMA**. It improves all seven predefined ensemble accounts, including combined-twenty Sharpe **1.367 → 1.455**. It does not pass the frozen gate: the four disjoint cohort spread increases **0.225 → 0.267**, exceeding the declared 0.25 limit. No method reaches Sharpe >1.5 across every group.

This screen covers consistency regularization and weight averaging through short continuation training. The proposed prior-decomposition architecture, multi-horizon heads and execution-aware training objectives remain untested in this batch. These findings do not replace experiments on those hypotheses or on full retraining from scratch.

## What ran

- 320 scientific continuation fits: two arms × two model families × four annual folds × all twenty existing seeds; three additional epochs each.
- Both arms start from identical saved source checkpoints, use two supervised input-dropout passes and the same lower/reset optimizer settings. The consistency arm adds a symmetric ranking-distribution KL penalty.
- Raw and EMA checkpoints are independently selected by first-half prior-year supervised validation loss, with the unchanged source checkpoint available as epoch zero. The second validation half remains diagnostic.
- 810 accounts cover the blended and single-family portfolios, seven predefined ensemble groups, every individual seed, normal costs and doubled fees plus 6% borrowing. All source data, portfolio rules and seed memberships remain fixed.
- All results use the same previously inspected 1,003 sessions in 2022–2025. This is development evidence, with no untouched market or seed confirmation.

## Primary blended strategy

| Ensemble | Current SR | Fine-tuned control SR | Change |
|---|---:|---:|---:|
| Original five | 1.272 | 1.439 | +0.167 |
| Second five | 1.184 | 1.235 | +0.051 |
| Third five | 1.409 | 1.503 | +0.093 |
| Fourth five | 1.322 | 1.368 | +0.046 |
| Combined ten | 1.264 | 1.369 | +0.105 |
| Combined fifteen | 1.338 | 1.441 | +0.102 |
| Combined twenty | 1.367 | 1.455 | +0.088 |

Combined-twenty CAGR increases **24.60% → 26.20%**, while maximum drawdown moves **−11.20% → −11.37%** and annual turnover rises **17.40 → 18.41**. Under doubled fees and 6% borrowing, combined-twenty Sharpe improves **1.236 → 1.319**. Every ensemble improves under stress; the smallest stressed gain is approximately +0.042 SR.

The combined-twenty Sharpe change has a conditional, pointwise 95% paired block interval of **[−0.019, +0.206]**, which includes zero. This interval does not correct for the model/checkpoint search or earlier experiments. The gain is a useful candidate for further validation rather than established future superiority.

## Consistency and EMA do not improve on the matched control

| Method | Minimum ensemble SR | Four-cohort spread | Combined-twenty SR |
|---|---:|---:|---:|
| Current strategy | 1.184 | 0.225 | 1.367 |
| Input-dropout fine-tuning, raw weights | 1.235 | 0.267 | 1.455 |
| Input-dropout fine-tuning, EMA | 1.149 | 0.372 | 1.433 |
| Consistency penalty, raw weights | 1.114 | 0.534 | 1.387 |
| Consistency penalty, EMA | 1.192 | 0.412 | 1.426 |

The consistency arm's third-cohort 1.648 Sharpe is not representative: its original and second cohorts are only 1.126 and 1.114. All failed and weaker cases remain in the [full report](REPORT.md).

The control comparison bundles additional training, a lower/reset optimizer and input dropout. It does **not** isolate which of those components causes the improvement. The consistency penalty and EMA comparisons are matched within those trajectories. A plain continuation arm would be needed to attribute the control's gain specifically to input dropout.

## Individual seeds and model families qualify the result

For the primary blended strategy, the fine-tuned control improves **11/20 individual seeds**; nine decline. The worst paired change is **−0.254 SR**. Its singleton minimum/median/standard deviation changes from **0.751 / 1.103 / 0.186** to **0.830 / 1.147 / 0.161**. Thus the singleton distribution improves in some respects, while uniform seed-level improvement is absent.

Each five-seed cohort averages model ranks before selecting companies. The combined account then averages risk-sized cohort targets. These nonlinear steps mean better cohort Sharpe need not imply every single seed improves.

The gain also depends on the blend. With the same 50% training prior, the THINK-only neural component's combined-twenty Sharpe falls **1.099 → 1.078**, and the dynamic-only component falls **0.870 → 0.809** under the fine-tuned control. This is not a universal improvement across standalone model portfolios.

Mean pairwise issuer-target Jaccard rises only slightly for the blend, **0.534 → 0.541**. [Agreement diagnostics](STABILITY.md) distinguish seed agreement from financial improvement. [Every paired seed change](SEED_CHANGES.csv) is retained, including stressed costs and single-family cases.

## Verification and interpretation

All 160 matched training pairs have exactly equal starting parameters and identical CPU/CUDA RNG states at every saved epoch. All initial validation forecasts reproduce the source forecasts exactly. Two additional diagnostic runs actually resume epochs 2–3 from epoch-1 states, one per family; their model parameters, EMA parameters and forecasts match with zero discrepancy. These two replays are diagnostics in addition to the 320 scientific fits.

The initial process pool encountered a Windows directory-creation permission error before one worker's first fit. A replacement ran that same job with the frozen code and seed list. All completed fits were preserved, and the resumed controller verified and sealed all 320 records. [Execution record](EXECUTION_NOTE.json) · [Training audit](TRAINING_AUDIT.json).

The focused consistency/EMA tests passed. Every one of the 810 accounts passes independent dollar-position replay, and all fourteen current-strategy reference paths reproduce. The final report retains exact discrepancies and source/input/publication hashes.

The practical result is a promising **fine-tuning recipe for the blended strategy**, with a failed strict spread gate and mixed individual-seed changes. Consistency and EMA have no supported advantage over its matched control in this screen. Full retraining, architecture changes and fresh-market confirmation require their own experiments; none is established by this short continuation result.

[Complete methods and tables](REPORT.md) · [All account metrics](METRICS.csv) · [Annual results](PERIOD_METRICS.csv) · [All paired Sharpe intervals](PAIRED_INTERVALS.csv) · [Validation and chosen epochs](VALIDATION.csv) · [Frozen protocol](PROTOCOL.json)
