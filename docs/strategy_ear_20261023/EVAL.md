# E. EAR overlay: the single predeclared 2016–2023 evaluation

Spec `PREREG.md` (committed before this run). Generated once by `scripts/ear_eval.py`. Window 2016-07-01 to 2023-12-29 (7.5 years), 90 monthly formations, next-open execution, drifting weights, delisted names liquidated at the last close.

## Primary: EAR tilt (A) minus the equal-weight PIT S&P 500 (B)

| Cost | Active ann. | Tracking error | **Active IR** [95% CI] | Max active drawdown | Worst year |
|---|---|---|---|---|---|
| 10 bp | -0.0052 | 0.0111 | **-0.47** [-1.11, +0.14] | -0.043 | -0.0135 |
| 20 bp | -0.0075 | 0.0111 | **-0.67** [-1.32, -0.08] | -0.056 | -0.0156 |
| 30 bp | -0.0098 | 0.0112 | **-0.88** [-1.51, -0.28] | -0.072 | -0.0177 |

- **Active FF5 + MOM + ST_Rev alpha (10 bp):** -0.0050/yr (t -1.44), n = 1887 days; exposures Mkt-RF +0.001, SMB -0.011, HML -0.001, RMW +0.001, CMA -0.009, MOM +0.026, STREV -0.007; R² 0.25.
- **Turnover (traded notional / NAV / yr):** A 3.16, B 0.84, C 3.16; the EAR overlay adds 2.32.
- **Coverage and concentration:** signal coverage 0.984 (min 0.968); bottom names per formation median 100, top 99; industry weights equal B by construction; maximum weight 2× B.
- **Absolute (10 bp):** A +0.1237/yr, B +0.1289/yr.

| Year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|---|---|
| A − B (10 bp) | -0.0110 | +0.0003 | -0.0022 | -0.0135 | +0.0010 | -0.0119 | +0.0097 | -0.0112 |
| C − B (10 bp) | -0.0110 | -0.0029 | -0.0066 | -0.0033 | +0.0049 | -0.0079 | -0.0117 | +0.0065 |

## Matched price-only control (C: same construction on the pre-event window d0−4..d0−2)

| Contrast (10 bp) | Ann. | IR [95% CI] | FF alpha (t) |
|---|---|---|---|
| C − B (control tilt) | -0.0043 | -0.45 [-1.09, +0.14] | -0.0036 (-1.22) |
| **A − C (event information beyond the control)** | -0.0009 | **-0.08** [-0.72, +0.51] | -0.0014 (-0.33) |

## Predeclared decision

Net active IR at 10 bp -0.47; active alpha t -1.44; incremental IR over the control -0.08 ; net active -0.0052/yr (bar 0.0030) → **STOP (S&P 500 EAR overlay)**.
