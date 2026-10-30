# Coverage association after liquidity and industry adjustment

The dynamic model's positive ranking-versus-coverage association in 2023–2025 weakens substantially when we adjust for ordinary trading liquidity. For 2024 after-hours coverage, the average within-origin association falls from **+0.241 on the classified sample to -0.112 after adjusting for ranked 20-day dollar volume**, and to **-0.119 after also adjusting for coarse industry groups**. This is consistent with liquidity being an important confounder of the descriptive relationship; it does not prove that liquidity caused the original association or that the trained model ignores coverage.

Protocol b38ee58 and code 3df6ef6 were frozen after the prior coverage results and before this exploratory follow-up. At each original NDX test origin, we rank the saved model score, observed extended-hours coverage and as-of-origin 20-session average dollar volume on the *same* eligible, classified sample. We calculate correlations of the rank residuals after an intercept only, ranked liquidity, industry indicators, and both together. The table shows equal-origin means for the ten-seed dynamic model; all four measures in a row use identical stocks.

| Year | Segment | Mean stocks | Unadjusted | Liquidity adjusted | Industry adjusted | Both adjusted |
|---|---|---:|---:|---:|---:|---:|
| 2022 | After hours | 76.1 | -0.051 | -0.166 | -0.122 | -0.203 |
| 2022 | Premarket | 76.1 | -0.044 | -0.151 | -0.132 | -0.211 |
| 2023 | After hours | 79.0 | 0.280 | -0.043 | 0.210 | -0.053 |
| 2023 | Premarket | 79.0 | 0.310 | 0.014 | 0.244 | -0.002 |
| 2024 | After hours | 97.0 | 0.241 | -0.112 | 0.219 | -0.119 |
| 2024 | Premarket | 97.0 | 0.245 | -0.095 | 0.222 | -0.096 |
| 2025 | After hours | 98.9 | 0.366 | 0.082 | 0.297 | 0.022 |
| 2025 | Premarket | 98.9 | 0.309 | 0.017 | 0.244 | -0.034 |

These values differ from batch30's full-eligible correlations because this analysis excludes unknown industry IDs and missing/nonpositive liquidity. In particular, 2022–2023 retain roughly 76–79 of about 101 eligible names. The SEC-derived SIC/FF12 industry labels are fixed at each year's first origin, may become stale, are not GICS, and do not independently certify the mapping from issuer to NDX price series. Later entrants can remain unclassified. First-publication timing is unverified. No causal interpretation, p-value, future-alpha inference or independence of weekly origins is implied. Controlling for liquidity and industry can also remove genuine signal or create conditioning artifacts; the test is not a checkpoint feature ablation.

All **1,608** origin/model/segment cells and all four views, sample counts, exclusions and status fields are in `PER_ORIGIN.csv`; annual mean, median and defined counts for B, D, E and Hlabel are in `ANNUAL.csv`. A separate reconstruction joined the raw date/ID liquidity and industry inputs and recomputed all **6,432** correlations using QR projections; maximum error 3.89e-16. Two focused tests passed, including under-supported and singular controls. `RESULTS.json` contains source hashes. No model forward intervention, account/outcome scoring, financial fitting, protected-universe scoring, or gate change occurred.
