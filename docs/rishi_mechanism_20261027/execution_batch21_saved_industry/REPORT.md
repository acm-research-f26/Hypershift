# Saved model rankings and dated industry snapshots

The fixed protocol ../SAVED_RANK_INDUSTRY_PROTOCOL.md (561ab66) and implementation 7ed876c describe the first test signal of each year for the ten-seed parent B, dynamic D, fixed blend E and training-label control Hlabel. They use saved scores, not regenerated checkpoint predictions. For comparability only, each score is converted to an eligible-universe percentile rank; the original trading blend is unchanged. SIC classifications come from the earlier documented filing evidence, with 77/101, 80/101, 98/101 and 100/101 eligible IDs covered in 2022–2025.

| First signal | Parent B | Dynamic D | Blend E | Hlabel |
|---|---:|---:|---:|---:|
| 2022 | 14.77% | 23.15% | 18.37% | 18.51% |
| 2023 | 6.36% | 27.25% | 20.94% | 19.09% |
| 2024 | 4.40% | 13.58% | 7.13% | 10.22% |
| 2025 | 11.26% | 11.89% | 12.10% | 9.66% |

Cells are the share of covered names' cross-sectional squared rank deviations that lies between FF12 industry groups. For these particular snapshots, dynamic ranks show a larger between-industry component than parent ranks, especially in 2023. The size and composition of industries change, and these four observations are not independent replications or a measure of forecast skill. The table does not establish what a model learned, a peer effect, or the fraction of portfolio returns explained by industry.

The blend's original top-ten target allocates 50%, 80%, 80% and 90% to FF12 group 6 in the four snapshots. This broad SIC group includes computers, electronic equipment and software; it is not an exact GICS technology sector. It contains 34, 33, 43 and 46 covered eligible IDs respectively. The 2022 blend target also has 40% unclassified mass, and the 2024 target has 10%; the other two have none. Target weights describe intended selections before fills and drift, not realized exposure or P&L. All groups, including zero target masses and unmatched names, are retained in TARGET_INDUSTRY_MASS.csv; all industry mean ranks and per-ID scores are retained in INDUSTRY_RANKS.csv and DETAIL.csv.

An independent calculation using SciPy ranks reproduced all 16 rank/variance cells and unclassified top-k masses; Hlabel fractions match the previous studies in all four years. An initial audit using pandas' default CSV float parser failed because tiny saved-score differences were rounded into ties. Reading the same CSV with `float_precision='round_trip'` preserved the original values and passed; results themselves were calculated from the original NPZ arrays, not the rounded audit import. Five focused tests pass. See VALIDATION.json and PROVENANCE.json. SEC filing dates do not guarantee subsequent identity/SIC stability, linkage to the NDX price series or as-published membership. No returns, financial training, model interventions or checkpoint tolerance changes were used.
