# Factor attribution of matched coverage-control differences

The blend's historical advantage over the matched six-channel coverage control remains sizable in these fixed linear factor regressions. With FF5, momentum, short-term reversal and QQQ included, the E-minus-coverage annual arithmetic intercept is **+20.83 percentage points**, with a lag-5 95% interval of **[+7.61, +34.06] points**. This is a selected-history attribution, not proof of future alpha or a neural feature's causal contribution.

Protocol 1878e85 and code e2917bb fixed this follow-up after batches 28 and 34–37 had been examined. Each regression uses the same 1,003 saved daily modeled-cost returns. FF7 is the five Fama–French factors plus momentum and short-term reversal; FF7+QQQ adds QQQ excess return. Both factors and the daily risk-free series come from the repository's retrospective 202608 CRSP vintage. Intercepts and normal intervals below are annualized arithmetic percentage points using Bartlett Newey–West covariance.

| Matched difference | Factor design | Intercept | 95% lag-5 interval | 95% lag-20 interval |
|---|---|---:|---:|---:|
| E minus coverage6 | FF7 | +21.00 | [+7.86, +34.15] | [+8.43, +33.58] |
| E minus coverage6 | FF7 + QQQ | +20.83 | [+7.61, +34.06] | [+8.17, +33.50] |
| E_Hlabel minus coverage6 | FF7 | +20.26 | [+5.42, +35.10] | [+6.51, +34.02] |
| E_Hlabel minus coverage6 | FF7 + QQQ | +19.10 | [+4.57, +33.62] | [+5.36, +32.83] |

These are not the same comparison as the original E-minus-E_Hlabel primary test, whose interval remains wide and whose declared Holm family did not pass. The coverage control uses a limited linear model, inherits the original's industry budgets and trades different stocks with different turnover and costs. These regressions cannot resolve stock-specific exposures, changing factor loadings, nonlinear risks, omitted factors, dependence on a few winners, source selection, or historical search. QQQ and market factors overlap, so individual regression slopes are conditional on a correlated design and should not be read as independent causal contributions. Daily factor returns are not costed executable hedges, and contemporaneous revised factors are used only for retrospective explanation.

`COEFFICIENTS.csv` contains all **204** coefficient rows for 24 fixed regressions, including both account levels and paired differences. The two original-account regressions under each design and lag reproduce the earlier batch28 coefficients/standard errors within 4.22e-15; their underlying daily paths agree with the original exported curves within 1e-16. Independent reconstruction using a separate factor parser and two-sided HAC summation recovered all coefficients and standard errors with maximum errors 1.20e-13 and 1.78e-15. Five factor/calculation tests passed. Inputs and hashes are in `RESULTS.json`; no account rerun, financial model training, checkpoint intervention, protected-window scoring or primary gate change occurred.
