# E/F. S&P 500 changer avoidance: the single predeclared 2016–2023 evaluation

Spec `PREREG.md` (frozen before download); checks `PRECHECKS.md`. Generated once by `scripts/lazy_eval.py`. Window 2016-07-01 to 2023-12-29 (7.5 years), 90 monthly formations, next-open execution, weights drift between rebalances, delisted names liquidated at their last close.

## Primary: active return of avoidance (A) over the equal-weight PIT S&P 500 (B)

| Cost (bp per unit traded) | Active ann. return | Tracking error | **Active IR** [95% CI] | Max active drawdown |
|---|---|---|---|---|
| 10 | -0.0001 | 0.0071 | **-0.01** [-0.66, +0.68] | -0.026 |
| 20 | -0.0012 | 0.0072 | **-0.17** [-0.80, +0.49] | -0.029 |
| 30 | -0.0024 | 0.0072 | **-0.34** [-0.96, +0.32] | -0.031 |

- **Active FF5 + MOM + ST_Rev alpha:** -0.0003/yr (t -0.13); exposures Mkt-RF -0.001, SMB -0.009, HML -0.006, RMW +0.003, CMA -0.001, MOM +0.005, STREV +0.002; R² 0.09.
- **Turnover:** A 2.03× NAV/yr, B 0.84× NAV/yr; the avoidance overlay adds 1.19× NAV/yr of trading.
- **Concentration:** changers per formation median 97 (range 68–100); maximum active weight 0.0028; 10-K share of changers median 0.06.
- **Absolute:** A net +0.1289/yr, B net +0.1290/yr at 10 bp.

| Year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|---|---|
| Active @10 bp | +0.0061 | +0.0014 | +0.0064 | -0.0073 | -0.0104 | +0.0040 | +0.0026 | -0.0033 |

## Declared diagnostics (non-gating)

| Portfolio | Ann. net vs B (10 bp) | IR [95% CI] |
|---|---|---|
| Changers (bottom 20% within form, EW) − B | -0.0040 | -0.12 [-0.85, +0.54] |
| Stable reporters (top 20% within form, EW) − B | +0.0119 | +0.34 [-0.43, +1.15] |
| Long-short: stable − changers, 50 bp/yr borrow | +0.0109 | +0.19 [-0.52, +1.00] |

Long-short FF alpha +0.0084 (t +0.42), market beta +0.02. The diagnostics are not industry-neutral and are reported once, for interpretation only.

## Predeclared decision

Net active IR at 10 bp -0.01; active alpha t -0.13 → **STOP (S&P 500 changer avoidance)**.
