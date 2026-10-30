# What characteristics track fixed validation preferences?

At the first test origin in each fold, the ten-seed validation-preference ranking is positively associated with stock liquidity and QQQ beta in all four years. The relationship with prior six-month momentum is weaker and changes sign. Values are cross-sectional Spearman correlations among eligible securities with sufficient validation and characteristic history:

| Test fold | QQQ beta | 126-session momentum | 20-session volatility | 20-session dollar volume | Securities |
|---|---:|---:|---:|---:|---:|
| 2022 | 0.343 | 0.069 | 0.160 | 0.334 | 95 |
| 2023 | 0.238 | 0.017 | 0.184 | 0.476 | 94 |
| 2024 | 0.356 | 0.161 | 0.192 | 0.431 | 94 |
| 2025 | 0.458 | -0.142 | 0.190 | 0.481 | 97 |

These are descriptive associations, not evidence that liquidity or beta caused the model's scores or returns. Each feature is examined separately, so correlated characteristics and industry composition may confound the relationship; no multivariate adjustment or predictive test is implied. Source prices and membership retain their vintage and identity limitations. Validation selected the checkpoints; its mean forecast rank is not training Hpred or a learned stock intercept. These years and overlapping seed groups are not independent market histories.

The [protocol](../VALIDATION_FIXED_STYLE_PROTOCOL_20261029.md) and implementation were fixed as `e3f9985` and `a8cd93b` before results. `CORRELATIONS.csv` retains all 48 group/fold/feature cells; `STOCKS.csv` contains all 4,560 supporting rows. A separate pandas calculation verified all correlations and support counts to 9.71e-17; that check reuses the saved characteristic rows and is not an independent reconstruction of every source feature. No test labels, later test forecasts, account returns, model inference, financial fitting or protected scoring were used. The numerical and financial gates remain unchanged.

Later verification independently recomputed all **4,560 saved characteristic values** from the assembled daily arrays, using explicit covariance and volatility formulas: maximum absolute error 1.78e-15 (`python -m hypershift run rishi_validation_style_input_verify`). This verifies the characteristic calculations from the same assembled data, not the vendor's original publication timing or security identity.
