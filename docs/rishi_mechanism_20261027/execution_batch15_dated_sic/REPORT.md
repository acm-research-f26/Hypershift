# Dated SIC feasibility and historical-label preference

The precommitted ../DATED_SIC_PROTOCOL.md (751d333) uses existing SEC filing-header SIC records and the separate S&P study's dated ticker/CIK segments. It joins them to the first eligible NDX signal in each fold and the already-saved, training-only Hlabel rank. A filing must predate the signal, and the ticker/CIK segment must cover it. This is a retrospective cross-study mapping, not an independently certified point-in-time NDX entity map. One historical SIC per company can become stale; SIC/FF12 is not an interchangeable GICS sector classification.

| Fold | First signal | Eligible | Matched with dated SIC | Between-industry share of covered Hlabel rank variation |
|---|---|---:|---:|---:|
| 2022 | 2021-12-31 | 101 | 77 | 18.5% |
| 2023 | 2023-01-06 | 101 | 80 | 19.1% |
| 2024 | 2024-01-05 | 101 | 0 | unavailable |
| 2025 | 2025-01-03 | 101 | 0 | unavailable |

The 2024–2025 gap occurs because the source mapping's segments end by 2023-12-31; no stale mapping was extended. The fractions describe a squared-deviation decomposition of percentile ranks among covered names only. They are not explained returns, effect sizes for the neural model, or evidence that sector neutrality would preserve or improve performance. There are small and singleton industry groups and substantial coverage selection. Six covered names in 2022 and seven in 2023 have fewer than 30 training labels and receive the original Hlabel training-median fallback. The full group means and counts, including all small groups, are in INDUSTRY_PREFERENCE.csv; the matched and unmatched eligible IDs are in ELIGIBLE_JOIN.csv.

All 157 matched rows passed an independent segment/date/SIC check, and both variance decompositions were independently recomputed. The source S&P mapping marks all 157 associated CIK checks as successful, but those checks do not certify NDX identity or as-published membership. COVERAGE.csv records all failure categories, and PROVENANCE.json plus VALIDATION.json identify inputs and hashes. Three focused tests pass, including future filing, overlapping segments and expired mapping. No returns, new financial models or protected-window scores were used.
