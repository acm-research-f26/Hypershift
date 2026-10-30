# Excluding issuers selected from training-period returns

The blend's historical account did **not collapse** when its targets excluded the top one or three issuers selected by a simple training-period return rule. At ten holdings, the original blend's 23.92% CAGR and 0.755 Sharpe became 25.37% / 0.802 under the one-issuer screen and 25.52% / 0.827 under the three-issuer screen. The paired return differences are highly uncertain, so these point estimates do not establish an improvement or justify adopting the screen.

Before examining these results, protocol a2c2927 fixed the exclusion rule: at the first source decision of each annual fold, rank originally eligible issuers by their best eligible share class's **training-only mean five-session label**, requiring at least 30 valid training labels. Exact issuer ties are alphabetical; the selected lists are held for that fold. The selected three were:

| Fold | First-ranked | Second | Third |
|---|---|---|---|
| 2022 | Tesla | Zoom | NVIDIA |
| 2023 | Tesla | Moderna | NVIDIA |
| 2024 | Tesla | Constellation Energy | Moderna |
| 2025 | Tesla | NVIDIA | Constellation Energy |

The first decision can be in early January; the historical-label information is from the earlier training fold. At each subsequent decision, the selected issuers' share classes are excluded from *new targets* and the saved model scores are ranked over the remaining eligible stocks. Existing positions exit only as the source fill rules permit. The neural model, peer inputs and seed ranks were not recomputed with a smaller universe. This is a post-score target-screen sensitivity, not a model trained or inferred without those issuers.

| Blend E, ten seeds | CAGR | Sharpe | Screen minus original annual arithmetic net mean | 95% block-21 interval |
|---|---:|---:|---:|---:|
| Original top 10 | 23.92% | 0.755 | — | — |
| Exclude top 1 training-selected issuer | 25.37% | 0.802 | +0.99 pp | [-3.57, +5.68] pp |
| Exclude top 3 training-selected issuers | 25.52% | 0.827 | +0.75 pp | [-5.44, +7.03] pp |
| Original top 5 | 31.87% | 0.896 | — | — |
| Exclude top 1, top 5 | 29.15% | 0.846 | -2.33 pp | [-10.95, +5.18] pp |
| Exclude top 3, top 5 | 33.39% | 0.988 | +0.16 pp | [-12.07, +12.24] pp |

For top ten, the top-three screen removes **15.02%** of original target weight on average; the one-issuer screen removes **6.02%**. Thus this is not a no-op, but much of the original target book remains. Screening may change subsequent fills, costs, risk, timing and realized holdings. The top-three rule removes NVIDIA in three folds, but excludes a different set in each fold and cannot establish that the strategy is independent of NVIDIA or other winners in general. The same screens applied to the parent-plus-training-label substitute leave E-minus-substitute intervals wide: at top ten, **+0.26 pp [-10.35, +11.05]** for the one-issuer screen and **-1.60 pp [-11.45, +8.40]** for three. Neither the original failed primary gate nor the equivalence decision changes.

This was a fixed exploratory sensitivity motivated by prior concentration findings, on repeatedly inspected 2022–2025 history. Source membership, issuer mapping, actual execution and data publication timing are not independently certified. Selecting a promising screen from these outcomes would add another search; it is not a future test. All 12 account metrics, 36 paired block-5/21/63 intervals, four-year calendar results, target-overlap diagnostics and issuer-selection receipts are retained here; target arrays and detailed ledgers are in ignored `results/RISHI_TRAINING_EXCLUSION_20261028/`.

The four original paths reproduce within 1e-16 daily return and 4.66e-16 normalized NAV. An independent audit recomputed issuer rankings directly from training labels, verified **2,412** target allocations and all 12 account metrics and 36 paired mean intervals; maximum normalized NAV, metric and interval errors were 1.12e-14, 5.56e-17 and 9.72e-17. Seven selection and engine tests passed. No new model fit, trained-checkpoint intervention, protected S&P scoring or gate change occurred.
