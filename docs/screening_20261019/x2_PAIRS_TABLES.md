# Study X2 — pairs trading on graph structure (exploratory)

Panel `data/fresh/sp500_daily/screen_sp500_2015_2026.npz`; years [2020, 2022, 2023]; formation 252, trading 126 sessions, monthly overlapping portfolios; random null = mean of 20 random partitions with cluster sizes. Stationary bootstrap block 10, B=5000, years pooled at equal weight. Survivorship-biased panel: absolute returns are an upper bound.

## Primary: wait-one-day execution, top-20, 10 bp per leg per side

| Family | Grouping | Ann. net return (pooled) [95% CI] | Sharpe 2020 / 2022 / 2023 | Holm p (abs, 6) | vs random null, ann. [95% CI] | Holm p (struct, 4) | Label |
|---|---|---|---|---|---|---|---|
| pairwise | cluster | +0.018 [-0.018, +0.060] | +0.95 / +0.42 / -0.39 | 1.000 | -0.009 [-0.046, +0.030] | 1.000 | NO EVIDENCE |
| pairwise | sector | +0.016 [-0.021, +0.059] | +0.81 / +0.51 / -0.35 | 1.000 | -0.010 [-0.048, +0.029] | 1.000 | NO EVIDENCE |
| pairwise | all | +0.017 [-0.021, +0.060] | +0.51 / +1.17 / -0.34 | 1.000 | — | — | NO EVIDENCE |
| pairwise | random | +0.026 [-0.025, +0.082] | +0.76 / +0.10 / +0.39 | — | — | — | calibration |
| star | cluster | -0.002 [-0.042, +0.041] | +0.49 / -0.99 / -0.97 | 1.000 | -0.007 [-0.040, +0.029] | 1.000 | NO EVIDENCE |
| star | sector | -0.019 [-0.062, +0.025] | -0.05 / -1.32 / -1.02 | 1.000 | -0.023 [-0.055, +0.008] | 0.550 | NO EVIDENCE |
| star | all | -0.002 [-0.058, +0.057] | +0.26 / -0.53 / -0.00 | 1.000 | — | — | NO EVIDENCE |
| star | random | +0.004 [-0.048, +0.060] | +0.21 / -0.28 / +0.28 | — | — | — | calibration |

## Star minus pairwise (primary spec, 10 bp; descriptive)

| Grouping | Ann. diff [95% CI] | p |
|---|---|---|
| cluster | -0.020 [-0.050, +0.010] | 0.198 |
| sector | -0.035 [-0.073, +0.001] | 0.064 |
| all | -0.019 [-0.069, +0.033] | 0.467 |
| random | -0.022 [-0.049, +0.004] | 0.106 |

## Sensitivities (pooled ann. net return; Sharpe by year)

| Spec | bp | Family | Grouping | Ann. return [95% CI] | Sharpe 2020 / 2022 / 2023 | legs/day |
|---|---|---|---|---|---|---|
| primary | 0 | pairwise | cluster | +0.031 [-0.006, +0.074] | +1.25 / +0.89 / +0.08 | 0.052 |
| primary | 0 | pairwise | sector | +0.029 [-0.009, +0.073] | +1.09 / +0.99 / +0.12 | 0.051 |
| primary | 0 | pairwise | all | +0.031 [-0.008, +0.074] | +0.77 / +1.65 / +0.15 | 0.052 |
| primary | 0 | pairwise | random | +0.039 [-0.012, +0.096] | +0.94 / +0.42 / +0.86 | 0.051 |
| primary | 0 | star | cluster | +0.010 [-0.031, +0.054] | +0.70 / -0.54 / -0.44 | 0.046 |
| primary | 0 | star | sector | -0.009 [-0.051, +0.035] | +0.10 / -0.90 / -0.64 | 0.040 |
| primary | 0 | star | all | +0.009 [-0.047, +0.067] | +0.39 / -0.35 / +0.30 | 0.041 |
| primary | 0 | star | random | +0.015 [-0.037, +0.071] | +0.34 / -0.07 / +0.64 | 0.043 |
| primary | 25 | pairwise | cluster | -0.002 [-0.037, +0.038] | +0.51 / -0.28 / -1.08 | 0.052 |
| primary | 25 | pairwise | sector | -0.003 [-0.040, +0.038] | +0.38 / -0.20 / -1.04 | 0.051 |
| primary | 25 | pairwise | all | -0.002 [-0.040, +0.039] | +0.13 / +0.44 / -1.05 | 0.052 |
| primary | 25 | pairwise | random | +0.007 [-0.044, +0.062] | +0.50 / -0.38 / -0.30 | 0.051 |
| primary | 25 | star | cluster | -0.020 [-0.059, +0.023] | +0.16 / -1.66 / -1.73 | 0.046 |
| primary | 25 | star | sector | -0.034 [-0.077, +0.010] | -0.27 / -1.95 / -1.58 | 0.040 |
| primary | 25 | star | all | -0.017 [-0.074, +0.041] | +0.06 / -0.80 / -0.46 | 0.041 |
| primary | 25 | star | random | -0.012 [-0.064, +0.043] | +0.00 / -0.60 / -0.27 | 0.043 |
| same_close | 0 | pairwise | cluster | +0.036 [-0.006, +0.086] | +1.40 / +0.84 / +0.11 | 0.052 |
| same_close | 0 | pairwise | sector | +0.035 [-0.009, +0.087] | +1.25 / +0.96 / +0.13 | 0.051 |
| same_close | 0 | pairwise | all | +0.034 [-0.009, +0.084] | +0.90 / +1.55 / +0.17 | 0.053 |
| same_close | 0 | pairwise | random | +0.041 [-0.013, +0.101] | +0.96 / +0.37 / +0.87 | 0.051 |
| same_close | 0 | star | cluster | +0.013 [-0.028, +0.058] | +0.80 / -0.60 / -0.30 | 0.046 |
| same_close | 0 | star | sector | -0.007 [-0.051, +0.038] | +0.15 / -0.88 / -0.64 | 0.040 |
| same_close | 0 | star | all | +0.012 [-0.047, +0.074] | +0.41 / -0.24 / +0.33 | 0.041 |
| same_close | 0 | star | random | +0.016 [-0.040, +0.074] | +0.33 / -0.08 / +0.71 | 0.043 |
| same_close | 10 | pairwise | cluster | +0.023 [-0.018, +0.071] | +1.14 / +0.38 / -0.34 | 0.052 |
| same_close | 10 | pairwise | sector | +0.022 [-0.021, +0.073] | +1.00 / +0.50 / -0.32 | 0.051 |
| same_close | 10 | pairwise | all | +0.021 [-0.022, +0.070] | +0.66 / +1.08 / -0.30 | 0.053 |
| same_close | 10 | pairwise | random | +0.028 [-0.026, +0.088] | +0.79 / +0.06 / +0.41 | 0.051 |
| same_close | 10 | star | cluster | +0.001 [-0.039, +0.046] | +0.59 / -1.04 / -0.83 | 0.046 |
| same_close | 10 | star | sector | -0.017 [-0.062, +0.028] | +0.01 / -1.30 / -1.02 | 0.040 |
| same_close | 10 | star | all | +0.002 [-0.058, +0.064] | +0.29 / -0.42 / +0.03 | 0.041 |
| same_close | 10 | star | random | +0.005 [-0.051, +0.062] | +0.20 / -0.29 / +0.35 | 0.043 |
| same_close | 25 | pairwise | cluster | +0.004 [-0.037, +0.050] | +0.73 / -0.30 / -1.00 | 0.052 |
| same_close | 25 | pairwise | sector | +0.003 [-0.040, +0.052] | +0.62 / -0.19 / -0.98 | 0.051 |
| same_close | 25 | pairwise | all | +0.001 [-0.041, +0.049] | +0.30 / +0.38 / -0.99 | 0.053 |
| same_close | 25 | pairwise | random | +0.009 [-0.046, +0.068] | +0.54 / -0.41 / -0.28 | 0.051 |
| same_close | 25 | star | cluster | -0.017 [-0.056, +0.027] | +0.28 / -1.69 / -1.59 | 0.046 |
| same_close | 25 | star | sector | -0.032 [-0.077, +0.012] | -0.21 / -1.92 / -1.58 | 0.040 |
| same_close | 25 | star | all | -0.014 [-0.074, +0.047] | +0.10 / -0.68 / -0.43 | 0.041 |
| same_close | 25 | star | random | -0.012 [-0.068, +0.046] | -0.00 / -0.60 / -0.20 | 0.043 |
| top50 | 0 | pairwise | cluster | +0.036 [-0.003, +0.082] | +1.51 / +0.53 / +0.30 | 0.049 |
| top50 | 0 | pairwise | sector | +0.034 [-0.009, +0.084] | +1.42 / +0.28 / +0.50 | 0.048 |
| top50 | 0 | pairwise | all | +0.045 [+0.004, +0.090] | +1.36 / +1.17 / +0.74 | 0.052 |
| top50 | 0 | pairwise | random | +0.034 [-0.023, +0.096] | +0.76 / +0.18 / +0.78 | 0.050 |
| top50 | 0 | star | cluster | +0.025 [-0.015, +0.071] | +0.78 / +0.01 / +1.04 | 0.042 |
| top50 | 0 | star | sector | +0.018 [-0.025, +0.063] | +0.37 / +0.54 / +0.61 | 0.038 |
| top50 | 0 | star | all | +0.013 [-0.040, +0.069] | +0.32 / -0.10 / +0.46 | 0.039 |
| top50 | 0 | star | random | +0.014 [-0.038, +0.070] | +0.30 / +0.07 / +0.40 | 0.040 |
| top50 | 10 | pairwise | cluster | +0.023 [-0.015, +0.068] | +1.25 / +0.03 / -0.22 | 0.049 |
| top50 | 10 | pairwise | sector | +0.021 [-0.021, +0.071] | +1.17 / -0.23 / -0.03 | 0.048 |
| top50 | 10 | pairwise | all | +0.031 [-0.009, +0.076] | +1.11 / +0.71 / +0.18 | 0.052 |
| top50 | 10 | pairwise | random | +0.022 [-0.035, +0.083] | +0.61 / -0.08 / +0.38 | 0.050 |
| top50 | 10 | star | cluster | +0.015 [-0.026, +0.060] | +0.60 / -0.43 / +0.52 | 0.042 |
| top50 | 10 | star | sector | +0.008 [-0.034, +0.053] | +0.22 / +0.16 / +0.22 | 0.038 |
| top50 | 10 | star | all | +0.003 [-0.050, +0.059] | +0.20 / -0.29 / +0.12 | 0.039 |
| top50 | 10 | star | random | +0.004 [-0.048, +0.060] | +0.17 / -0.13 / +0.05 | 0.040 |
| top50 | 25 | pairwise | cluster | +0.005 [-0.034, +0.048] | +0.85 / -0.71 / -0.97 | 0.049 |
| top50 | 25 | pairwise | sector | +0.003 [-0.038, +0.051] | +0.78 / -0.97 / -0.80 | 0.048 |
| top50 | 25 | pairwise | all | +0.012 [-0.028, +0.055] | +0.73 / +0.03 / -0.65 | 0.052 |
| top50 | 25 | pairwise | random | +0.003 [-0.054, +0.063] | +0.38 / -0.48 / -0.22 | 0.050 |
| top50 | 25 | star | cluster | -0.001 [-0.042, +0.043] | +0.33 / -1.08 / -0.24 | 0.042 |
| top50 | 25 | star | sector | -0.006 [-0.049, +0.038] | -0.01 / -0.41 / -0.34 | 0.038 |
| top50 | 25 | star | all | -0.012 [-0.065, +0.044] | +0.02 / -0.59 / -0.39 | 0.039 |
| top50 | 25 | star | random | -0.012 [-0.064, +0.044] | -0.02 / -0.44 / -0.47 | 0.040 |
