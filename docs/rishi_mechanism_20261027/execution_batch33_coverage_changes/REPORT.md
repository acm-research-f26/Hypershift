# Do ranking changes track changes in extended-hours coverage?

Across the fixed consecutive-date comparisons, changes in the dynamic model's stock rankings show **small, inconsistent** associations with changes in extended-hours coverage ranks. In 2024, for example, the average after-hours association is **-0.125**, compared with the positive cross-sectional level association reported earlier. These are different statistics: the result suggests the earlier level relationship should not be assumed to describe a stable relationship between changes, but it does not prove the model ignores coverage.

The fixed protocol 8c1521e and code 2bff933 compare each decision with the prior decision in the same annual fold: **197 pairs**, each five exchange sessions apart. For each pair, model and AH/PM segment, we retain only stock IDs eligible and observed at both dates with finite positive 20-day average dollar volume. We correlate the changes in the original cross-sectional score and coverage percentile ranks, then calculate a partial rank correlation controlling for the change in ranked log dollar volume. The table shows equal-pair means for the ten-seed dynamic model:

| Year | Segment | Paired dates | Mean matched stocks | Raw rank-change correlation | Adjusted for liquidity change |
|---|---|---:|---:|---:|---:|
| 2022 | After hours | 50 | 101.7 | -0.093 | -0.091 |
| 2022 | Premarket | 50 | 101.7 | -0.060 | -0.060 |
| 2023 | After hours | 49 | 100.9 | -0.033 | -0.036 |
| 2023 | Premarket | 49 | 100.9 | 0.044 | 0.045 |
| 2024 | After hours | 49 | 100.9 | -0.125 | -0.126 |
| 2024 | Premarket | 49 | 100.9 | -0.029 | -0.030 |
| 2025 | After hours | 49 | 100.9 | 0.025 | 0.026 |
| 2025 | Premarket | 49 | 100.9 | -0.009 | -0.007 |

The Hlabel control has undefined correlations in **368 of its 394** segment/date cells because at least one change vector is constant. Its remaining 26 cells are not representative of all dates and must not be interpreted as a full-period comparator. The B, D and E cells are defined for every pair. Undefined results are retained, not set to zero.

This study does not use future-fold means or financial outcomes. However, rank changes can arise from changes in other stocks or in the eligibility set even when the focal stock is unchanged, and requiring both dates selects a subset. Change in liquidity is an imperfect control, and raw and partial correlations do not establish statistical significance, cause, predictive content or model-feature sensitivity. The data are the same repeatedly examined historical sample, and first-provider publication timing remains unverified. Hpred, trained feature interventions and prospective tests remain gated.

All **1,576** pair/model/segment cells, status and support counts, and **159,328** paired stock rows are preserved in `PER_PAIR.csv` and `PAIRED_STOCKS.csv`; `ANNUAL.csv` contains all models and defined counts. An independent reconstruction using SciPy ranking and QR residualization matched every cell to 1.67e-16. Two focused tests passed, including constant and insufficient-data cases; input hashes are in `RESULTS.json`. No trained-model inference, financial fitting, account scoring, protected-universe scoring or gate change occurred.
