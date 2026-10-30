# Retrospective multifactor account attribution

The ten-seed/top-ten blend's standalone factor-adjusted intercept is positive on this inspected history, but its increment over the training-label-identity control is near zero with a wide interval. Neither observation establishes future alpha or explains the model's learned signal. The 2022–2025 paths and candidate have been repeatedly examined, and these supportive regressions do not replace the prespecified primary comparison or correct its broader selection history.

Protocol 065b6bc and code be9eda4 were frozen before the run. Each regression uses all 1,003 matched sessions; the factor files identify their vintage as the 202608 CRSP database. FF7 comprises the five Fama–French factors, momentum and short-term reversal; FF7+QQQ adds QQQ excess return. Intercepts and intervals below are annualized arithmetic percentage points, using Bartlett Newey–West covariance and normal intervals. Single-account dependent variables subtract factor RF; differences do not. The saved portfolio returns already reflect modeled costs. Factor returns have their own construction and are not executable hedge returns.

| Fixed design | Account or difference | Intercept | 95% lag-5 interval | 95% lag-20 interval |
|---|---|---:|---:|---:|
| FF7 | E | 12.41 | [1.10, 23.72] | [1.07, 23.75] |
| FF7 | B | 5.84 | [-7.01, 18.70] | [-6.82, 18.50] |
| FF7 | E_Hlabel | 13.36 | [-2.43, 29.14] | [-1.59, 28.30] |
| FF7 | E − B | 6.57 | [-1.61, 14.75] | [-0.86, 13.99] |
| FF7 | E − E_Hlabel | -0.94 | [-12.42, 10.53] | [-11.34, 9.45] |
| FF7 + QQQ | E | 10.50 | [0.54, 20.46] | [0.19, 20.81] |
| FF7 + QQQ | B | 3.83 | [-7.45, 15.11] | [-7.70, 15.36] |
| FF7 + QQQ | E_Hlabel | 10.59 | [-2.42, 23.60] | [-2.19, 23.36] |
| FF7 + QQQ | E − B | 6.67 | [-1.49, 14.83] | [-0.76, 14.10] |
| FF7 + QQQ | E − E_Hlabel | -0.09 | [-11.04, 10.87] | [-10.04, 9.86] |

In the FF7+QQQ design, the E account's QQQ coefficient is 1.335 and HML coefficient -0.133; coefficients are conditional on all included factors, and market/QQQ collinearity makes them unsuitable as standalone causal exposures. The E-versus-identity intercept interval is far wider than the ±2 pp/year equivalence margin. It cannot establish that changing model information adds value, nor that it adds none. Dynamic exposure, sector, issuer concentration, economic regimes, intraday trading conventions and factor-vintage revisions remain possible confounds.

`COEFFICIENTS.csv` gives all 170 coefficient rows for all 20 fitted regressions and both lags; `RESULTS.json` records input hashes. The launcher implementation hash refers to its temporary compatibility copy. An independent reconstruction using a separate parser and explicit two-sided lag summation recovered all 170 coefficients and standard errors with maximum differences 1.21e-13 and 1.68e-15, respectively. Five focused tests passed. These are descriptive selected-history findings; no financial retraining, protected-window scoring, account rerun, or change to the checkpoint gate occurred.
