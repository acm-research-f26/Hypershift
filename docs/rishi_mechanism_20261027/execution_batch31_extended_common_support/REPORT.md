# Extended-hours associations on the same stocks

The previous comparison used roughly twice as many stocks for coverage as for extended-hours returns. Restricting both measures to the same stocks leaves the dynamic-model coverage association stronger than the return association in the 2023–2025 folds, and reveals that the reported coverage association itself changes considerably with the stock subset. For example, the 2024 dynamic/after-hours coverage correlation rises from **0.253 on all available coverage observations to 0.547 on stocks with an observed after-hours return**. These are descriptive associations, not predictive performance or proof of a learned coverage mechanism.

After inspecting batch30, protocol 3b9ba06 fixed this exploratory follow-up and code e06bd90. Within each origin, model and segment, coverage and return observations are matched on stock ID and reranked on their common support. The table gives equal-origin mean Spearman correlations for the dynamic model; support is the mean number of stocks per origin. Annual means describe dependent weekly origins in selected development history.

| Year | AH support | AH coverage | AH return | PM support | PM coverage | PM return |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 51.1 | -0.040 | -0.012 | 47.2 | -0.023 | 0.032 |
| 2023 | 50.0 | 0.183 | -0.031 | 44.7 | 0.226 | 0.043 |
| 2024 | 56.8 | 0.547 | -0.008 | 47.4 | 0.589 | 0.047 |
| 2025 | 62.4 | 0.415 | -0.001 | 60.2 | 0.375 | -0.033 |

The common subset is not random: the source sets numeric return features missing when too few bars are observed. Conditioning on observed return can change the relationship with coverage, so the 2024 increase is **not evidence of a stronger learned signal**. Common support also does not remove stock identity, industry, liquidity, cross-feature dependence, differing feature noise or retrospective selection. The measures have different economic meanings; a larger correlation does not show which input matters more to the model or trading performance. PM return excludes the prior-close gap. First-provider-publication timing remains uncertified.

`PER_ORIGIN.csv` contains all **1,608** fixed year/origin/model/segment cells and original full-support coverage correlations; `ANNUAL.csv` includes B, D, E and Hlabel and all summary fields. All 1,608 cells were independently reconstructed from the prior rank records using a separate Spearman implementation; maximum numerical difference was 1.12e-16. Four focused tests passed, covering finite support, ties, small samples, duplicate IDs and inconsistent score ordering. No labels, model inference, interventions, account scoring, financial fitting or protected S&P window was used. The original-device checkpoint and financial-retraining gates remain unchanged.
