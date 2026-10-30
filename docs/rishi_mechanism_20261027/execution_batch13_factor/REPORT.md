# Supportive market and spanning regressions

Frozen protocol: ../FACTOR_NW_PROTOCOL.md. Seven fixed daily regressions use 1,003 observations, an intercept, and Newey–West standard errors with Bartlett lags 5 and 20. Individual legs use the stated 3% annual hurdle; active differences do not. The table shows lag-5 annualized arithmetic intercepts and normal 95% intervals, in percentage points per year; all lag-20 intervals also include zero. These are exploratory, unadjusted for multiple comparisons, and are not evidence of confirmed alpha.

| Regression | Intercept | 95% interval | Slope |
|---|---:|---:|---:|
| Blend top10 vs QQQ | 9.24 | [-0.71, 19.19] | 1.226 |
| Parent + training-label identity vs QQQ | 8.21 | [-5.41, 21.83] | 1.353 |
| Blend minus parent vs QQQ | 6.95 | [-1.32, 15.22] | 0.008 |
| Blend minus parent+identity vs QQQ | 1.03 | [-10.46, 12.51] | -0.127 |
| Blend equal5 volatility target vs full book | 4.93 | [-2.08, 11.94] | 0.569 |
| Blend issuer5 volatility target vs full book | 5.00 | [-1.92, 11.92] | 0.565 |
| QQQ volatility target vs full QQQ | 2.34 | [-1.90, 6.58] | 0.772 |

QQQ is the only market factor *used here*, not a claim that no other factor data exist: the repository contains other research factor archives, but no full style or point-in-time sector adjustment is implemented in this batch. The volatility comparisons are spanning regressions, not pure exposure-only treatment effects. These results do not change the preregistered primary decision or justify financial retraining. Full coefficients, both lags, inputs and code hashes are in FACTOR_ATTRIBUTION.csv and RESULTS.json. SLOPE_INTERVALS.csv supplies the slope normal intervals and two-sided p-values calculated directly from the already-saved estimates and standard errors; no financial account was rerun.
