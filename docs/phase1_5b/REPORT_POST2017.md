# Post-2017 frozen THINK test (Phase 1.5b): report

Date 2026-10-04. Data: Alpaca SIP daily bars, `adjustment=split`, 1,647 of 1,737 RSR nodes (identity-pass), masked after last bar (418 ended before 2023-12-29), 1,509 trading days 2018-01-02..2023-12-29, frozen graph v2. Models: new `R5_f3_alpha0_train` HH and EH, seeds 0-4, frozen at the pre-2017 validation-best epoch (FREEZE_MANIFEST.md, committed before any 2018+ inference). One locked inference pass. Numbers: `post2017_results.json`; scripts `eval_post2017.py`, `post2017_analysis.py`.

## Executive summary

Two separate conclusions.

**1. Sharpe persistence: the 2017 Sharpe did not persist.**
- Predeclared replication check R7 on 2017 passed: new HH mean test Sharpe 2.200, inside [1.771, 2.254] (historical `R5_f2_alpha0_train/HH`, widened by SD). The new EH mean 1.463 is below the historical 2.049 (descriptive).
- Over 2018-2023 the frozen HH models have pooled gross top-5 Sharpe 0.855 / 0.096 / 0.798 / 0.227 / 0.193 (mean 0.43; seed-averaged daily series 0.450, 95% block-bootstrap CI -0.31 to 1.23). EH: 0.267 / 0.144 / 0.204 / 0.113 / 0.075 (mean 0.16; seed-avg 0.183).
- Equal-weight hold-all on the same mask: Sharpe 0.512 (CI -0.35 to 1.35). HH is about level with the market (seed-avg excess-over-hold-all Sharpe +0.15), EH is below it.
- Net of cost at 10 bp per side, HH is 0.52 (seed 0), 0.29 (seed 2) and -0.15 to 0.00 for the other three and negative at 25 bp for four of five seeds.
- Annual HH seed-average: 2018 -1.45, 2019 -0.74, 2020 +1.26, 2021 +0.86, 2022 +1.01, 2023 +0.70 (the pooled figure is computed from concatenated daily returns, not from these).
- Caveat: one 6-year path with wide CIs. "Did not persist" means the 2017 level (about 2) lies far outside every seed's interval; it does not mean the true Sharpe is zero.

**2. Evidence of learned ranking: NO EVIDENCE.**
- Primary formal family (Holm over 5 tests; intersection-union over seeds for F1-F4): HH vs random daily top-5 p 0.740; vs beta-quintile-matched 0.757; vs industry-matched 0.870; vs label permutation 0.731; HH vs EH Wilcoxon 0.125. All Holm-adjusted p >= 0.625.
- Seeds 0 and 2 individually beat the random top-5, beta-matched and label-permutation nulls (per-seed p 0.005-0.011) but not the industry-matched null (0.57, 0.17); the other three seeds beat nothing. The intersection-union rule needs all five seeds, so the family does not reject.
- Mean daily cross-sectional IC for HH is 0.0002-0.0032 (near zero). HH top-5 baskets are high-beta (basket beta about 1.3 vs universe 1.02) and land in the realised top decile (0.13-0.16, 0.10 by chance) about as often as in the bottom decile (0.13-0.18): volatility selection rather than direction skill. HH predictions tie at the 5th/6th rank on 263-784 of 1,509 days per seed (collapsed output; lowest index decides).
- **HH vs EH (F5): INSUFFICIENT SEEDS** (repo rule: 5 seeds cannot reach p < 0.01). Descriptively HH leads on 4 of 5 seeds (differences +0.59, -0.05, +0.59, +0.11, +0.12); the seed-averaged daily contrast is +0.27 (95% CI -0.13 to +0.69, p_boot 0.20), not distinguishable from zero. The lead rests mainly on seeds 0 and 2.

Bottom line: a new validation-selected replication of the 2017 strategy earned a Sharpe near the market's in 2018-2023, with nothing in the nulls to support ranking skill. This tests neither the authors' implementation nor the historical run, and a frozen model says nothing about periodic retraining.

## Pipeline check (2017, not 2018+)

Seed 0 HH weights run on the Alpaca panel for the 2017 test days: Sharpe 1.809 vs 2.199 on the stored RSR-panel run; mean daily cross-sectional Pearson between the two prediction sets 0.949 (1,647 vs 1,731 eligible names per day). The Alpaca panel reproduces the model inputs closely but not exactly; the 0.39 Sharpe gap shows sensitivity to data source and universe.

## Hand checks (raw bars, independent pandas path)

Portfolio day 2020-10-13, HH seed 0: top-5 MPV, VAR, BRFS, BP, PTR. Returns recomputed from Alpaca `split` closes match stored `gt` to 6 decimals, and the equal-weight return -0.00836785 equals the stored `test_daily`. Stock day MPV (input day 2020-10-12): stored MA5/10/20/30/close times the pre-2017 scale equal values recomputed from raw closes (10.7569, 10.7884, 10.9176, 10.9456, 11.02).

## Predeclaration (fixed and committed before the formal statistics, commit 0332d16)

- Pooled Sharpe from concatenated daily top-5 returns (mean/std ddof 0 x sqrt 252), never a mean of annual Sharpes.
- Primary family (Holm over 5): F1 random daily top-5; F2 beta-quintile-matched (beta from RSR 2013-2015 training period only); F3 industry-matched; F4 stock-label permutation (one-sided empirical upper-tail p per seed, intersection-union = max over 5 seeds); F5 HH vs EH (Wilcoxon on per-seed pooled Sharpes, paired by seed; plus stationary block-bootstrap on the date-joined daily contrast).
- B_NULL = 2000, B_PERM = 500, N_BOOT = 5000, mean block 10 days, common day blocks; rng from `forensics.rng_for`.
- F5 verdict word by the repo rule (`aggregate.py`): INSUFFICIENT SEEDS if m * 2^(1-n) >= 0.01.
- Everything else (annual, leave-one-year-out, hold-all, costs, IC, NDCG, decile hits, basket beta, eligible counts) is descriptive.

## Tables

### T1. Pooled 2018-2023 gross Sharpe (concatenated daily returns, top-5, stable ties)

Days: 1509 (2018-01-02 to 2023-12-29). Hold-all (equal weight, same mask): Sharpe 0.512 (95% block-bootstrap CI -0.348 to 1.354), mean daily 4.71 bp.

| arm | seed | Sharpe | 95% CI (block 10) | excess-over-hold-all mean (bp/day) [95% CI] | excess SR | net 5bp | net 10bp | net 25bp | turnover | IC | NDCG@5 | tie days |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HH | 0 | 0.855 | [0.018, 1.618] | 9.20 [-3.70, 24.50] | 0.675 | 0.688 | 0.521 | 0.020 | 0.27 | 0.0032 | 0.541 | 432 |
| HH | 1 | 0.096 | [-0.679, 0.902] | -3.15 [-13.22, 7.08] | -0.243 | -0.028 | -0.152 | -0.525 | 0.20 | 0.0021 | 0.538 | 784 |
| HH | 2 | 0.798 | [0.047, 1.553] | 4.61 [-1.96, 10.92] | 0.526 | 0.545 | 0.293 | -0.465 | 0.30 | 0.0002 | 0.540 | 263 |
| HH | 3 | 0.227 | [-0.487, 0.992] | -0.82 [-10.37, 9.57] | -0.060 | 0.112 | -0.003 | -0.346 | 0.20 | 0.0009 | 0.540 | 616 |
| HH | 4 | 0.193 | [-0.548, 0.962] | -1.66 [-10.66, 7.39] | -0.131 | 0.055 | -0.084 | -0.500 | 0.22 | 0.0011 | 0.539 | 608 |
| HH | avg of 5 seeds | 0.450 | [-0.305, 1.232] | | 0.153 | | | | | | | |
| EH | 0 | 0.267 | [-0.529, 1.094] | -0.99 [-9.58, 7.65] | -0.088 | 0.016 | -0.234 | -0.983 | 0.35 | 0.0014 | 0.539 | 448 |
| EH | 1 | 0.144 | [-0.668, 0.951] | -2.41 [-13.15, 8.76] | -0.190 | -0.248 | -0.640 | -1.814 | 0.63 | -0.0001 | 0.538 | 844 |
| EH | 2 | 0.204 | [-0.559, 0.982] | -1.86 [-10.37, 6.22] | -0.173 | -0.055 | -0.313 | -1.084 | 0.36 | -0.0003 | 0.538 | 617 |
| EH | 3 | 0.113 | [-0.668, 0.946] | -2.74 [-13.04, 7.88] | -0.201 | -0.043 | -0.199 | -0.666 | 0.27 | 0.0011 | 0.538 | 737 |
| EH | 4 | 0.075 | [-0.719, 0.885] | -3.57 [-12.23, 4.78] | -0.302 | -0.066 | -0.208 | -0.632 | 0.21 | 0.0041 | 0.538 | 466 |
| EH | avg of 5 seeds | 0.183 | [-0.605, 0.999] | | -0.256 | | | | | | | |

### T2. Annual Sharpe (days) and leave-one-year-out pooled Sharpe

| series | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | LOYO: -2018 / -2019 / -2020 / -2021 / -2022 / -2023 |
|---|---|---|---|---|---|---|---|
| hold-all | -1.28 (251) | 1.88 (252) | 0.43 (253) | 1.68 (252) | -0.53 (251) | 0.82 (250) | 0.69 / 0.41 / 0.59 / 0.22 / 0.69 / 0.49 |
| HH seed 0 | -1.13 (251) | -0.53 (252) | 2.04 (253) | 1.16 (252) | 1.50 (251) | 0.48 (250) | 1.08 / 1.08 / 0.42 / 0.83 / 0.70 / 0.91 |
| HH seed 1 | -1.09 (251) | -1.40 (252) | 0.97 (253) | 0.47 (252) | 0.66 (251) | -0.03 (250) | 0.23 / 0.38 / -0.25 / 0.05 / -0.03 / 0.11 |
| HH seed 2 | -0.93 (251) | 0.68 (252) | 1.37 (253) | 0.98 (252) | 0.29 (251) | 2.07 (250) | 1.03 / 0.82 / 0.62 / 0.78 / 0.89 / 0.62 |
| HH seed 3 | -1.58 (251) | -0.79 (252) | 0.96 (253) | 0.05 (252) | 0.82 (251) | 0.47 (250) | 0.42 / 0.38 / -0.11 / 0.25 / 0.12 / 0.20 |
| HH seed 4 | -1.80 (251) | -0.86 (252) | 0.51 (253) | 1.31 (252) | 1.01 (251) | 0.39 (250) | 0.43 / 0.38 / 0.08 / 0.06 / 0.02 / 0.17 |
| HH seed-avg | -1.45 (251) | -0.74 (252) | 1.26 (253) | 0.86 (252) | 1.01 (251) | 0.70 (250) | 0.67 / 0.65 / 0.14 / 0.41 / 0.33 / 0.43 |
| EH seed 0 | -0.94 (251) | -0.41 (252) | 1.23 (253) | 0.57 (252) | 0.21 (251) | 0.11 (250) | 0.42 / 0.41 / -0.08 / 0.23 / 0.28 / 0.29 |
| EH seed 1 | -1.53 (251) | 0.02 (252) | 1.36 (253) | 0.24 (252) | -0.99 (251) | 0.63 (250) | 0.35 / 0.16 / -0.31 / 0.13 / 0.38 / 0.07 |
| EH seed 2 | -1.57 (251) | -0.16 (252) | 0.51 (253) | 1.61 (252) | 0.90 (251) | -0.60 (250) | 0.43 / 0.28 / 0.11 / 0.01 / 0.06 / 0.33 |
| EH seed 3 | -0.97 (251) | -0.40 (252) | 0.59 (253) | 0.78 (252) | 0.25 (251) | -0.27 (250) | 0.23 / 0.20 / -0.08 / 0.03 / 0.08 / 0.16 |
| EH seed 4 | -0.59 (251) | -0.19 (252) | 0.65 (253) | 0.77 (252) | -0.93 (251) | 0.23 (250) | 0.15 / 0.12 / -0.17 / -0.02 / 0.26 / 0.05 |
| EH seed-avg | -1.29 (251) | -0.27 (252) | 1.03 (253) | 0.90 (252) | -0.14 (251) | 0.04 (250) | 0.36 / 0.27 / -0.13 / 0.08 / 0.25 / 0.21 |

### T3. Eligible names per year (mask used for every series)

| year | mean | min | max |
|---|---|---|---|
| 2018 | 1595 | 1546 | 1636 |
| 2019 | 1512 | 1469 | 1545 |
| 2020 | 1440 | 1410 | 1468 |
| 2021 | 1368 | 1330 | 1410 |
| 2022 | 1302 | 1274 | 1330 |
| 2023 | 1251 | 1229 | 1274 |

### T4. Ranking-skill nulls (one-sided empirical p on pooled Sharpe; B_NULL=2000, B_PERM=500)

Random daily top-5 null Sharpe 5/50/95 pct: -0.157 / 0.258 / 0.632.

| arm | seed | F1 random top-5 | F2 beta-quintile matched | F3 industry matched | F4 label permutation |
|---|---|---|---|---|---|
| HH | 0 | 0.005 | 0.010 | 0.573 | 0.006 |
| HH | 1 | 0.740 | 0.756 | 0.790 | 0.731 |
| HH | 2 | 0.011 | 0.006 | 0.174 | 0.008 |
| HH | 3 | 0.547 | 0.563 | 0.870 | 0.547 |
| HH | 4 | 0.595 | 0.649 | 0.854 | 0.553 |
| EH | 0 | 0.482 | 0.551 | 0.936 | n/a (HH only) |
| EH | 1 | 0.672 | 0.633 | 0.731 | n/a (HH only) |
| EH | 2 | 0.577 | 0.643 | 0.902 | n/a (HH only) |
| EH | 3 | 0.718 | 0.842 | 0.569 | n/a (HH only) |
| EH | 4 | 0.766 | 0.888 | 0.908 | n/a (HH only) |

### T5. Primary formal family (Holm over 5 tests)

| test | p (IUT over seeds for F1-F4) | Holm-adjusted |
|---|---|---|
| F1 HH vs random daily top-5 | 0.7396 | 1.0000 |
| F2 HH vs beta-matched | 0.7565 | 1.0000 |
| F3 HH vs industry-matched | 0.8703 | 1.0000 |
| F4 HH vs label permutation | 0.7305 | 1.0000 |
| F5 HH vs EH (Wilcoxon, per-seed pooled Sharpe) | 0.1250 | 0.6250 |

HH vs EH: per-seed pooled Sharpe diff (HH-EH) [0.588, -0.047, 0.594, 0.114, 0.118]; seed-averaged daily-series contrast 0.267 (95% block-bootstrap CI -0.133 to 0.687, p_boot 0.203). Smallest attainable two-sided Wilcoxon p with 5 seeds is 0.0625. **Verdict word (repo rule): INSUFFICIENT SEEDS.** repo rule (aggregate.py/_holm): INSUFFICIENT SEEDS when m*2^(1-n)=0.312 >= 0.01 with n=5 seeds, m=5.

### T6. Diagnostics (top-5 baskets)

| arm | seed | hit top-10% | hit top-20% | miss bottom-10% | precision@5 | basket beta | universe beta |
|---|---|---|---|---|---|---|---|
| HH | 0 | 0.147 | 0.254 | 0.149 | 0.0121 | 1.31 | 1.02 |
| HH | 1 | 0.159 | 0.259 | 0.178 | 0.0168 | 1.36 | 1.02 |
| HH | 2 | 0.133 | 0.242 | 0.126 | 0.0070 | 1.29 | 1.02 |
| HH | 3 | 0.158 | 0.261 | 0.168 | 0.0137 | 1.35 | 1.02 |
| HH | 4 | 0.150 | 0.254 | 0.161 | 0.0131 | 1.34 | 1.02 |
| EH | 0 | 0.140 | 0.248 | 0.142 | 0.0101 | 1.31 | 1.02 |
| EH | 1 | 0.181 | 0.277 | 0.188 | 0.0119 | 1.34 | 1.02 |
| EH | 2 | 0.141 | 0.246 | 0.149 | 0.0125 | 1.32 | 1.02 |
| EH | 3 | 0.162 | 0.263 | 0.179 | 0.0159 | 1.36 | 1.02 |
| EH | 4 | 0.156 | 0.253 | 0.163 | 0.0113 | 1.39 | 1.02 |

## Limitations

- **Source:** Alpaca SIP daily bars, not CRSP. No delisting returns: 418 identity-pass names end before 2023-12-29 and are masked after their last bar (terminal return UNKNOWN, never zero-filled). This probably flatters long-only baskets; its size cannot be measured here. History starts 2016-01-04, so identity was verified on a 2-year overlap only; identity is symbol-keyed (`asof=2017-12-08`).
- **Price convention A2 is approximate:** `adjustment=split` prices are rounded to 2 decimals and adjust about 4 of 13 spin-off events that RSR leaves unadjusted. 90 nodes fail identity and are masked; 66 further names would pass under `raw`. Prices are as of the download date.
- **MA fill rule is INFERRED** (forward-fill when a close is missing inside the window; RSR's rule is UNKNOWN, DATA_COMPATIBILITY.md section 4).
- **Replication, not the historical run:** the 2017 weights do not exist; R7 passed on a tolerance, and EH (1.46) differs from the historical EH (2.05).
- **Universe:** frozen in 2017; later listings are excluded by design and the eligible set shrinks from 1,595 to 1,251 names per day (T3). The bias relative to the full market is unmeasured.
- **Inference:** the five seeds are repeated trainings on the same days, not independent market samples. Wilcoxon with 5 seeds cannot go below p = 0.0625. One 6-year path; Sharpe CIs are wide. Nulls use B_NULL 2000 and B_PERM 500 (smallest attainable p about 0.0005 and 0.002).
- **Cost model:** repo convention (2 x bps x fraction of names replaced), closing-price execution, no borrow or impact.
- **Not done:** walk-forward retraining, geometry controls, exposure regressions beyond beta.
