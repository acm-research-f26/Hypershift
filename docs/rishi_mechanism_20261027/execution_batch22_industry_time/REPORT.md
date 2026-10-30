# Industry patterns across all saved decision origins

The frozen protocol ../SAVED_INDUSTRY_TIME_PROTOCOL.md (8a67834) extends the prior four first-signal snapshots to all 201 original test origins and the fixed ten-seed B/D/E/Hlabel saved scores. Each year's SIC/FF12 classifications are frozen from that year's first eligible snapshot; stocks newly appearing later remain unclassified. No outcome returns or new accounts were used.

| Year | Origins | Parent: mean between-industry rank variation | Dynamic: mean | Blend: mean | Origins where dynamic exceeds parent |
|---|---:|---:|---:|---:|---:|
| 2022 | 51 | 16.57% | 22.88% | 19.78% | 46 / 51 |
| 2023 | 50 | 6.51% | 23.62% | 16.33% | 50 / 50 |
| 2024 | 50 | 6.28% | 17.73% | 9.39% | 49 / 50 |
| 2025 | 50 | 10.65% | 16.61% | 13.16% | 45 / 50 |

The statistic partitions covered names' cross-sectional squared deviations of eligible-universe percentile ranks; it is not an estimate of explained returns. The persistent descriptive difference supports investigating whether the dynamic ranking is more aligned with broad industry membership than the parent ranking. It does **not** establish that the network learned industry membership, that peer information caused it, or that industry exposure produces alpha. Consecutive origins, seeds and stocks are dependent; these percentages are not independent trials or significance tests. Mean covered IDs are 76.1, 79.0, 97.0 and 98.9 in successive years, with incomplete and differently sourced classification.

The blend's mean top-ten **target** weight in FF12 group 6 (computers, electronics and software) is 74.6%, 79.2%, 74.4% and 86.4% by year. Mean unclassified target mass is 15.7%, 0.4%, 0.2% and 0.0%. These are intended weights at decisions, not filled holdings or P&L. A comparison for every model, k and group is in ANNUAL_TARGET_MASS.csv; all 804 model-origin rows and all group target partitions are preserved. Hlabel is a training-label-return control, not a neural prediction.

Seven focused tests pass. An independent check verified all 804 unique model-origin rows, 1,608 top-k partitions summing to one, 16 annual summaries and four paired comparisons; all 352 first-origin group target weights reproduce the earlier batch. Original data remains exploratory development history. A filing's SIC and symbol can change after filing, frozen classifications do not follow later changes, and neither NDX price-series identity nor as-published index membership is independently certified. No strict checkpoint gate, financial-training rule or protected window was changed.
