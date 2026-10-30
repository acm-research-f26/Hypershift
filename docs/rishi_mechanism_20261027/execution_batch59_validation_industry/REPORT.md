# Validation preferences after broad industry adjustment

The ten-seed validation-preference ranking remains positively associated with dollar volume and QQQ beta after adjusting for the other three measured style ranks **and** broad industry indicators. These are partial rank correlations on the same classified support within each fold:

| Test fold | Dollar volume, adjusted | QQQ beta, adjusted | Classified / common-support securities |
|---|---:|---:|---:|
| 2022 | 0.327 | 0.415 | 76 / 95 |
| 2023 | 0.498 | 0.187 | 76 / 94 |
| 2024 | 0.318 | 0.206 | 93 / 94 |
| 2025 | 0.413 | 0.222 | 96 / 97 |

The smaller classified samples in the first two folds mean these values should not be directly compared with the previous all-stock correlations as though only the controls changed. Classifications are coarse FF12 groups based on the earlier dated SEC/SIC work, and 19, 18, 1 and 1 common-support names respectively remain unclassified. All four characteristics and all three seed groups are retained in `CORRELATIONS.csv` and `SUPPORT.csv`.

This is an unadjusted descriptive association following earlier inspected results. It does not show that liquidity, beta or industry caused the model's scores or returns; correlated characteristics, coarse or stale classifications and unmeasured inputs remain possible explanations. Validation selected checkpoints, and its rankings are not training Hpred. Data-vintage and price-identity limitations remain.

The [protocol](../VALIDATION_INDUSTRY_STYLE_PROTOCOL_20261029.md) and code were committed as `18fdc72` and `8cafd44` before the calculation. An independent QR projection verified all 48 correlations and classified/unclassified support counts, maximum error 3.43e-16. It reused the earlier saved characteristics and industry evidence, rather than independently revalidating the filings. No model was fitted or intervened on, no returns were scored, and the original checkpoint and financial gates remain unchanged.
