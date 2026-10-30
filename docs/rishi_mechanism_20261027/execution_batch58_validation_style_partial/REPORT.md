# Fixed validation preference after joint style adjustment

The ten-seed validation-preference ranking remains positively associated with liquidity and QQQ beta after rank-adjusting for the other three measured characteristics. The partial correlations below use the **same stock support within each fold** and control, in turn, for ranked beta, momentum, volatility and dollar volume other than the characteristic shown.

| Test fold | Beta, adjusted | Momentum, adjusted | Volatility, adjusted | Dollar volume, adjusted | Securities |
|---|---:|---:|---:|---:|---:|
| 2022 | 0.323 | -0.001 | -0.222 | 0.269 | 95 |
| 2023 | 0.145 | 0.094 | -0.080 | 0.444 | 94 |
| 2024 | 0.146 | 0.075 | 0.075 | 0.321 | 94 |
| 2025 | 0.333 | -0.085 | -0.203 | 0.362 | 97 |

For example, the ten-seed dollar-volume association changes from **0.334 to 0.269** in 2022 and from **0.481 to 0.362** in 2025 when the other three ranks are controlled. The volatility association changes sign in three folds. These are descriptive partial rank associations, not evidence that liquidity or beta causes the preference or its returns. Correlated inputs, industry composition, omitted variables, retrospective source data and checkpoint selection limit interpretation. There are no performance outcomes, significance claims or independent market replications here.

The [protocol](../VALIDATION_FIXED_STYLE_PARTIAL_PROTOCOL_20261029.md) and implementation were frozen as `838b03a` and `528b31a` after the earlier univariate results but before these outcomes. A separate QR-projection calculation independently checked all **48** correlations over **1,140** retained common-support stock rows, maximum error 3.33e-16. The check uses batch57's saved characteristic values; it is not an independent vendor-data reconstruction. No model or financial strategy was fitted, and no checkpoint inference, test-label scoring or protected-window scoring occurred. Hpred and all original gates remain open.
