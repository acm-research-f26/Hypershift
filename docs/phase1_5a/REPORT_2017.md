# Phase 1.5a report: 2017 Sharpe near 2 (R5_f2 alpha=0 THINK, HH)

Reproduce: `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/forensic_2017.py --stage all` (CPU, a few minutes plus the nulls). Primary: `R5_f2_alpha0_train/HH` seeds 0-4, validation-selected epoch, top-5 equal weight, daily rebalance, 237 test days (2017-01-03 to 2017-12-08). References: `R5_f_train/{HH,EH}` seeds 0-4. Seeds are repeated runs on the same days, not independent samples. 2017 is an exploratory year. Market = equal-weight hold-all of valid stocks (cap-weighted market: UNKNOWN, not in the data). Batch 2 was trimmed by the predeclared sequential stopping rule (Gate A outcome 2); skipped parts are listed in the section "Skipped by the predeclared sequential stopping rule".

## Executive summary

1. **Why Sharpe near 2:** a concentrated daily top-5 portfolio in a strong market year. Equal-weight hold-all already has Sharpe 1.53 on these 237 days, the basket carries a high-beta tilt (mean basket beta 1.23 to 1.32 vs universe 1.00), and a random daily 5-stock basket has a wide Sharpe distribution (5th-95th percentile -0.52 to 2.26).
2. **The model is not distinguishable from random selection.** The five Sharpes (1.88 to 2.14) sit at the 88-93 percentile (in percent of random baskets beaten) of random daily top-5 baskets (F1 per-seed p 0.067 to 0.120, family p 0.120, Holm 0.36). None of F1-F4 rejects (Holm 0.32 to 0.64). Gate A outcome 2 (exposure sufficient) is implied by the non-rejection of F1: a Sharpe typical of random 5-stock baskets in this year needs no skill to explain it.
3. **Power is low.** Seed-0 block-bootstrap Sharpe 95% CI is 0.28 to 3.48; over seeds the CI bounds range 0.03 to 3.91. "Not distinguishable from random" is not "no signal": hypothesis D (a genuine top-tail signal) is not excluded, only unsupported.
4. **Weak positive signs, no calibrated ranking.** Global IC 0.0050 to 0.0081, NDCG@5 0.5673 to 0.5710 vs random 0.5639; the top-5 are in the realised top decile 0.155 to 0.203 of the time (random 0.10) but also in the realised bottom decile 0.138 to 0.182, so most of that is a volatility effect; top minus bottom predicted-decile return -3.9 to 2.3 bp/day with a top-decile 95% half-width of about 9 to 12 bp. Margin buckets show no monotone pattern.
5. **Mechanisms ruled down:** ties/index order (A) are not the main mechanism (random exact-tie median Sharpe stays above hold-all in every seed); backtest integrity (C) is clean except one stock-day (SYX 2017-03-27, +36.8%, selected by all five seeds) that is about 19 to 25% of each seed's total P&L. Retrospective Sharpe without the day(s) with an abs(return) > 0.2 stock in the basket, seeds 0-4: 1.64, 1.70, 1.61, 2.17, 2.28 (seeds 3-4 also lose the WAIR 2017-08-09 loss day, so they rise).
6. **Costs and training:** at 10 bp per side the net Sharpe is 0.98 to 1.57 and at 25 bp -0.39 to 0.70 (hold-all has no cost modelled, 1.53). Test Sharpe after the first training pass is already 1.82 to 2.54 and its mean over all 100 epochs is 1.61 to 2.00: the level is not something training clearly added.
7. **Verdict:** Sharpe near 2 in this run is **not** evidence of learned stock ranking (NO EVIDENCE; exploratory, 2017 only, seeds not independent). See the decision table and the verdict section.

## Reproduction table by seed (primary run)

Spread = median daily cross-sectional SD of the scores; spread/realised = that divided by the median daily cross-sectional SD of realised next-day returns. Tie rate = share of days with an exact tie at the 5th/6th score. Turnover = mean daily share of names replaced.

| seed | selected epoch | Sharpe | mean (bp/day) | vol (%/day) | max drawdown | IC | NDCG@5 | spread | spread / realised | tie rate | turnover | hold-all Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8 | 1.88 | 15.6 | 1.31 | -0.092 | 0.0081 | 0.5710 | 2.00e-04 | 0.015 | 0.47 | 0.22 | 1.53 |
| 1 | 32 | 1.93 | 15.3 | 1.25 | -0.114 | 0.0054 | 0.5688 | 1.77e-04 | 0.013 | 0.46 | 0.21 | 1.53 |
| 2 | 0 | 1.89 | 12.6 | 1.06 | -0.071 | 0.0058 | 0.5673 | 1.40e-04 | 0.010 | 0.10 | 0.31 | 1.53 |
| 3 | 1 | 2.04 | 14.3 | 1.11 | -0.092 | 0.0068 | 0.5692 | 1.66e-04 | 0.012 | 0.07 | 0.21 | 1.53 |
| 4 | 2 | 2.14 | 16.3 | 1.20 | -0.088 | 0.0050 | 0.5708 | 1.47e-04 | 0.011 | 0.18 | 0.22 | 1.53 |

All 25 run-seeds (primary + four references) recompute from the saved arrays: daily returns to atol 1e-7 and Sharpe to 1e-5 against `metrics.json` and `history.jsonl` (`inventory.json`).

## Gate A (Tasks 1, 2, 2B, 5)

Hold-all (equal-weight market of valid stocks) Sharpe on the same days: 1.53 (cap-weighted market: UNKNOWN, not in data).

### Ties and index order (Task 2)

| seed | stable (saved) | random exact-tie order p5 / p50 / p95 | reverse index | hold-all | constant-score first-5 | tie days (of 237) | outcome-1 |
|---|---|---|---|---|---|---|---|
| 0 | 1.88 | 1.40 / 1.78 / 2.15 | 1.60 | 1.53 | 0.37 | 112 | False |
| 1 | 1.93 | 1.43 / 1.79 / 2.15 | 1.61 | 1.53 | 0.37 | 109 | False |
| 2 | 1.89 | 1.91 / 2.07 / 2.23 | 2.21 | 1.53 | 0.37 | 24 | False |
| 3 | 2.04 | 1.99 / 2.14 / 2.30 | 2.21 | 1.53 | 0.37 | 16 | False |
| 4 | 2.14 | 1.96 / 2.17 / 2.41 | 2.42 | 1.53 | 0.37 | 43 | False |

Random fixed-index basket rule (1000 relabellings, constant scores): Sharpe mean 0.97, p5/p50/p95 -0.56 / 0.95 / 2.52.

### Nulls (Task 5): per-seed upper-tail empirical p of the Sharpe

B = 10000 (random top-5, fixed basket), 2000 (label permutation, matched baskets).

| seed | Sharpe | random top-5 (F1) | beta-matched (F2) | industry-matched (F3) | label perm (F4) | fixed basket (not in family) | Sharpe of excess over hold-all | block-10 bootstrap 95% CI |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.88 | 0.120 | 0.066 | 0.227 | 0.142 | 0.159 | 1.60 | 0.28 to 3.48 |
| 1 | 1.93 | 0.110 | 0.068 | 0.153 | 0.098 | 0.145 | 1.64 | 0.34 to 3.47 |
| 2 | 1.89 | 0.118 | 0.072 | 0.641 | 0.125 | 0.155 | 1.51 | 0.34 to 3.44 |
| 3 | 2.04 | 0.088 | 0.081 | 0.441 | 0.100 | 0.119 | 1.71 | 0.03 to 3.91 |
| 4 | 2.14 | 0.067 | 0.035 | 0.145 | 0.078 | 0.095 | 1.87 | 0.42 to 3.85 |

Formal family (intersection-union = max per-seed p, then Holm over F1-F4): family p {'F1': 0.12038796120387961, 'F2': 0.08095952023988005, 'F3': 0.6406796601699151, 'F4': 0.14192903548225888}, Holm {'F2': 0.3238380809595202, 'F1': 0.3611638836116388, 'F4': 0.3611638836116388, 'F3': 0.6406796601699151}.

### Factor proxy (Task 2B)

| seed | model Sharpe | top-3 daily Spearman(score, feature) | mean per-day R2 of all-feature fit | fitted portfolio Sharpe | residual portfolio Sharpe | proxies with Sharpe >= 0.8 x model | resid inside central 90% of random null |
|---|---|---|---|---|---|---|---|
| 0 | 1.88 | [('ret20', -0.31), ('ma30_rel', 0.3), ('ma20_rel', 0.25)] | 0.21 | 0.93 | 2.04 | ['index'] | True |
| 1 | 1.93 | [('ma20_rel', 0.22), ('ma30_rel', 0.21), ('ret20', -0.2)] | 0.09 | 1.76 | 1.96 | ['fitted'] | True |
| 2 | 1.89 | [('ma30_rel', 0.39), ('ret20', -0.38), ('ma20_rel', 0.32)] | 0.13 | 0.33 | 1.83 | ['degree', 'index'] | True |
| 3 | 2.04 | [('ret20', -0.27), ('ma30_rel', 0.27), ('ma20_rel', 0.21)] | 0.11 | 1.47 | 2.29 | ['index'] | False |
| 4 | 2.14 | [('ma30_rel', 0.37), ('ma20_rel', 0.36), ('ret20', -0.35)] | 0.11 | 1.21 | 1.96 | ['degree', 'index'] | True |

### Predeclared outcomes

1. Tie/index decisive: **False** (per seed [False, False, False, False, False]).
2. Exposure (B) sufficient: **True** (per seed [True, True, True, True, True]).
3. Factor tilt sufficient (literal rule, any proxy incl. fixed-basket 'index' and 'degree'): **False** (per seed [True, True, True, False, True]); restricted to economic proxies and fitted: **False** (per seed [False, True, False, False, False]).
4. Not decisive: False.

### Interpretation (Gate A, exploratory; written after batch 1)

1. Outcome 2 (exposure B sufficient) holds: in every primary seed the Sharpe sits inside the central 90% of the beta-matched or industry-matched null (industry p 0.145 to 0.641; beta p 0.035 to 0.081, so seed 4 is below 0.05 on beta alone). Outcomes 1 and 3 do not hold, so Gate A is not "not decisive" but it is a single-mechanism result, not a full explanation. Outcome 2 is also implied by the non-rejection of F1 (random daily top-5): if the Sharpe is already typical of random 5-stock baskets in this year, market exposure plus chance suffices to account for it.
2. The 2017 equal-weight market (hold-all) already has Sharpe 1.53. The 5 seeds' 1.88 to 2.14 are 0.35 to 0.61 above it; the Sharpe of the daily excess over hold-all is 1.5 to 1.9 but its block-bootstrap CI is wide (lowest lower bound 0.03, highest upper bound 3.9).
3. Against a uniform random daily top-5 (F1, same days, mask, k) the per-seed one-sided p is 0.067 to 0.120. Family p (max over seeds) F1 0.120, F2 0.081, F3 0.641, F4 0.142; Holm-adjusted 0.32 to 0.64. No test rejects. This is "no evidence against the null" on 237 days, not proof of no skill; power is low (see the CI width).
4. Tie/index path (A) is not decisive: with random exact-tie order the median Sharpe stays above hold-all in every seed (1.78, 1.79, 2.07, 2.14, 2.17 vs 1.53); the reverse-index Sharpe is 1.60 to 2.42. For seeds 0 and 1 the saved stable value (1.88, 1.93) sits inside the random-tie 5-95% band (1.40 to 2.15), so the stable tie-break neither inflates nor deflates materially. 112 and 109 of 237 days have an exact tie at the 5th/6th boundary in seeds 0 and 1 (24, 16, 43 in seeds 2 to 4). Constant-score first-5 gives 0.37 and a random fixed-index basket rule has median 0.95 (p95 2.52), so an index-order basket alone does not reproduce 2 as a typical outcome. Not an artifact claim from tie rate alone.
5. Tie composition in the primary run: boundary tie groups are small (mean size 3.4 to 6.5 in seeds 0, 2, 3, 4; seed 1 has a few large groups, max 1296), every day has a different tied value (no repeated saturated value except 4 days in seed 1), 0% graph-isolated, 0% stale window, 0% partly masked. So the ties are not explained by identical inputs under these three tests (mechanism cause UNKNOWN). In the R5_f_train/HH reference, ties are mass ties of about 1700 valid stocks sharing one constant output (collapsed days), a different phenomenon.
6. Model-level equivariance (random-init THINK, 4 temporal/spatial combos, node permutation of inputs and graph) holds to rtol 1e-5, so the model has no index-dependent step; the index only enters through the evaluator's stable tie-break. The trained-model version is not recoverable (no weights saved).
7. Factor proxy (descriptive, not causal): the score is positively associated with ma30_rel and ma20_rel and negatively with ret20 (daily Spearman 0.2 to 0.4 in magnitude): a short-horizon mean-reversion / oversold tilt, shared across seeds. An all-feature linear fit explains a mean per-day R2 of 0.09 to 0.21. The fitted portfolio gets 0.33 to 1.76 (below the model in every seed), while the residual (score minus fit) portfolio keeps 1.83 to 2.29. So the simple proxies do not carry the return; most of it stays in the unexplained part. Literal outcome 3 is not met (seed 3 residual 2.29 lies above the random null p95; also 'index' and 'degree' qualify as "proxies" only as arbitrary fixed baskets, which the economic-proxy reading excludes).

**What Gate A shows:** F1-F4 do not reject; the tie path is not decisive; the score has a mean-reversion tilt that does not carry the return. **What it does not show:** that the model has no skill (power is low), or anything about years other than 2017.

## Integrity (Task 3, uncovered risks only)

Existing Phase 1.5 A/C checks re-run: `pytest -m data tests/test_phase15_eval.py tests/test_phase15_data.py -q` -> 15 passed, 9 deselected in 18.23s (verbatim last line). Target timing, split dates, mask-to-raw-close trace and Sharpe convention (`metrics.py:50-52`: mean / np.std ddof 0 x sqrt(252), no risk-free rate; pinned by `test_sharpe_matches_authors_up_to_annualisation_constant`) are therefore not redone.

- Duplicate stock series: 0 pairs.
- Selected stock-days whose 17-day window touches a fill value: 0 (asserted 0; mask = min over the window).
- Stale close (unchanged for 3+ days ending at t-1) inside selected baskets: 2 to 4 of 1185 stock-days per seed, P&L share -0.34 to -0.29%. Exactly-zero realised return on 29 to 36 selected stock-days (P&L share 0 by construction).
- Selected stock-days with |return| > 0.2: 2 unique (SYX 2017-03-27 +36.8% x5 seeds, WAIR 2017-08-09 -24.2% x2 seeds). Neither is followed by a reversal above 50% of its size within 3 days (0 candidates), so there is no sign of a split/adjustment error; whether SYX really rose that day is not verified against an external source (UNKNOWN). None is deleted.
- SYX 2017-03-27 is selected by every seed and is 19 to 25% of each seed's total P&L (one stock-day, weight 1/5). **Retrospective** Sharpe without the day(s) with abs(ret) > 0.2 in the basket: seed 0 1.64 (hold-all same days 1.54, 1 day(s) removed), seed 1 1.70 (hold-all same days 1.54, 1 day(s) removed), seed 2 1.61 (hold-all same days 1.54, 1 day(s) removed), seed 3 2.17 (hold-all same days 1.62, 2 day(s) removed), seed 4 2.28 (hold-all same days 1.62, 2 day(s) removed).

**What this shows:** no data or backtest defect changes the returns. **What it does not show:** that the result is robust to a single large winner; one stock-day carries a fifth of the P&L, which is itself an example of the concentration in explanation B.

## Decomposition (Task 4, trimmed: persistence, exposure, seed concentration, gross/net)

- Persistence: mean day-to-day basket Jaccard 0.56 to 0.71, median holding spell 2 to 3 days, 56 to 133 distinct stocks ever selected (of 1737). Top-5 most-selected stocks take 18 to 39% of the 1185 selection slots, top-20 take 43 to 90%.
- Exposure: mean training-period beta of the selected stocks 1.23 to 1.32 vs 1.00 for the universe (beta from training days only, against the equal-weight hold-all). The model never selects a stock with no industry label (0% of slots vs 29% of the universe). Over-weighted industries are tiny ones (e.g. Wholesale Distributors: 7-11% of slots, 0.06% of the universe, one stock), which also means the industry-matched null F3 replaces those stocks with themselves and is a weak test (conservative toward not rejecting).
- Gross vs net (cost = 2 x bps x share of names replaced, day 1 full; mean turnover 0.21 to 0.31); hold-all is charged no cost (equal-weight hold, turnover convention 0):

| seed | gross | 5 bp | 10 bp | 25 bp | hold-all (any cost) |
|---|---|---|---|---|---|
| 0 | 1.88 | 1.61 | 1.34 | 0.53 | 1.53 |
| 1 | 1.93 | 1.67 | 1.41 | 0.63 | 1.53 |
| 2 | 1.89 | 1.44 | 0.98 | -0.39 | 1.53 |
| 3 | 2.04 | 1.73 | 1.43 | 0.54 | 1.53 |
| 4 | 2.14 | 1.85 | 1.57 | 0.70 | 1.53 |

- Seed concentration: pairwise daily basket Jaccard between seeds 0.41 (range 0.23 to 0.67), pairwise correlation of daily returns 0.75 (range 0.67 to 0.89); stocks in the top-10 selection frequency of at least 3 seeds: CSS, PSO, R, RPM, SNX, TPC, VAR, WAIR. Agreement across seeds under the same data, ordering and backtester is not independent evidence.

Skipped by the predeclared sequential stopping rule (Gate A outcome 2): per-stock contribution, retrospective exclude-and-reselect and best-day removal (T4 item 3); costs/benchmarks beyond model and hold-all, momentum and daily excess-series CIs (T4 item 4).

Figure: `docs/figures/phase1_5a_persistence.png`.

**What this shows:** a rotating, moderately persistent basket tilted to high-beta names, with a net Sharpe below hold-all once cost is charged. **What it does not show:** which stocks drive the P&L beyond the single SYX day (contribution analysis skipped).

## Tiny score gaps (Task 6, trimmed: margin buckets, top-k, local/global IC, calibration)

Spread: median 5th-minus-6th margin 2.0e-07 to 7.6e-05, median daily score SD 1.4e-04 to 2.0e-04, zero-spread days 0, exact-tie days 16 to 112.

Margin buckets (mean next-day top-5 return in bp, block-bootstrap 95% CI; exact tie first, then margin quintiles q1 smallest to q5 largest):

| seed | exact_tie | q1 | q2 | q3 | q4 | q5 | Spearman(margin, return), non-tie days (95% CI) |
|---|---|---|---|---|---|---|---|
| 0 | 29 (5,53) n=112 | -21 (-68,23) n=25 | -42 (-65,-19) n=25 | 49 (-2,110) n=25 | 8 (-17,33) n=25 | 24 (-8,56) n=25 | 0.09 (-0.07,0.23) |
| 1 | 23 (2,43) n=109 | -1 (-24,22) n=26 | -8 (-44,27) n=25 | 18 (1,33) n=26 | 30 (-11,73) n=25 | 6 (-24,38) n=26 | 0.10 (-0.03,0.24) |
| 2 | 3 (-20,28) n=24 | 15 (-4,34) n=43 | -0 (-26,24) n=42 | 13 (-5,34) n=43 | 4 (-26,37) n=42 | 36 (12,64) n=43 | 0.03 (-0.09,0.15) |
| 3 | -26 (-54,2) n=16 | 4 (-11,19) n=44 | 31 (9,52) n=44 | 11 (-16,36) n=45 | 5 (-40,49) n=43 | 35 (11,61) n=45 | 0.04 (-0.08,0.16) |
| 4 | 29 (-19,77) n=43 | 4 (-17,27) n=39 | 22 (0,45) n=39 | 8 (-14,29) n=38 | 20 (-6,46) n=39 | 14 (-5,32) n=39 | 0.03 (-0.10,0.17) |

Top-k (primary k = 5; the others are diagnostic). Sharpe / excess mean daily return over the random-k null mean (bp/day) / hit rate of the realised top decile / realised bottom decile:

| seed | k=1 | k=5 | k=10 | k=20 | k=50 |
|---|---|---|---|---|---|
| 0 | 1.98 / 38.7 / 0.26 / 0.19 | 1.88 / 11.5 / 0.20 / 0.18 | 1.73 / 8.6 / 0.21 / 0.18 | 1.17 / 4.0 / 0.20 / 0.19 | 1.17 / 2.9 / 0.19 / 0.18 |
| 1 | 0.87 / 14.1 / 0.23 / 0.22 | 1.93 / 11.2 / 0.20 / 0.17 | 1.33 / 5.6 / 0.20 / 0.19 | 1.04 / 3.2 / 0.19 / 0.19 | 1.24 / 3.3 / 0.18 / 0.17 |
| 2 | 0.12 / -2.3 / 0.18 / 0.20 | 1.89 / 8.6 / 0.16 / 0.14 | 2.31 / 9.4 / 0.16 / 0.14 | 2.07 / 7.9 / 0.17 / 0.16 | 1.39 / 4.0 / 0.18 / 0.17 |
| 3 | 1.22 / 19.9 / 0.22 / 0.18 | 2.04 / 10.2 / 0.17 / 0.14 | 3.13 / 13.4 / 0.17 / 0.13 | 1.97 / 7.4 / 0.17 / 0.15 | 1.72 / 5.5 / 0.18 / 0.16 |
| 4 | 1.81 / 32.6 / 0.24 / 0.19 | 2.14 / 12.2 / 0.17 / 0.15 | 3.01 / 14.4 / 0.18 / 0.15 | 1.97 / 7.6 / 0.17 / 0.16 | 1.76 / 5.6 / 0.17 / 0.16 |

Random-k null Sharpe 5th / 50th / 95th percentile: k=1: -1.12 / 0.42 / 1.98; k=5: -0.61 / 0.83 / 2.29; k=10: -0.11 / 1.10 / 2.32; k=20: 0.25 / 1.25 / 2.21; k=50: 0.67 / 1.39 / 2.11.

Ranking diagnostics: global IC 0.0050 to 0.0081; local IC within the predicted top 5% / 10% / 20%: q=0.05: 0.001 to 0.026; q=0.1: -0.000 to 0.010; q=0.2: -0.007 to 0.011. Calibration (mean realised next-day return per predicted decile, bp/day; overall mean 4.2):

| seed | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | top - bottom |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3.4 | 2.2 | 1.8 | 1.8 | 1.6 | 0.7 | 4.9 | 6.1 | 6.0 | 5.4 | 2.0 |
| 1 | 6.3 | 6.1 | 1.8 | 0.2 | 1.8 | 1.7 | 0.3 | 0.0 | 1.6 | 2.4 | -3.9 |
| 2 | 3.6 | 4.1 | 4.0 | 3.2 | 3.4 | 0.2 | 3.7 | 8.7 | 6.1 | 5.9 | 2.3 |
| 3 | 4.2 | 0.1 | 7.4 | 5.6 | -2.9 | 4.5 | -0.2 | 1.2 | 8.3 | 5.0 | 0.9 |
| 4 | 6.6 | 2.4 | 0.9 | -0.0 | 1.3 | -0.5 | 2.6 | 1.1 | 3.9 | 6.5 | -0.1 |

Day-clustered 95% half-width of a single decile mean is about 9 to 12 bp/day, so the decile profiles are not distinguishable from flat. Figures: `docs/figures/phase1_5a_margin.png`, `docs/figures/phase1_5a_calib.png`.

**Reconciliation (T6 item 7).** NDCG@5 is 0.5673 to 0.5710 against 0.5639 for random scores (100 draws, same function, days and mask), a gap of +0.0034 to +0.0071: the shifted-relevance NDCG has a high floor, so a gap of half a percent is tiny. The hit rate of the realised top decile (0.155 to 0.203) is above the random 0.10, but the realised bottom decile is hit 0.138 to 0.182: picking volatile stocks puts more mass in both tails. The directional part is hit minus miss = +0.015 to +0.031 (about 1.5 to 3 percentage points, no confidence interval computed), consistent with the small positive IC and NDCG gap but not with a calibrated top-tail effect: the top predicted decile is not reliably above the bottom (-3.9 to 2.3 bp/day, CI about +/-9 to 12 bp). So top-tail skill (hit_top10) and NDCG@5 near random are compatible: both reflect a small directional lean on top of a large volatility/beta tilt. This is a weak positive sign for D, not support.

Skipped by the predeclared sequential stopping rule (Gate A outcome 2): epsilon tie-group curves and score-jitter curves (T6 items 2-3); the json carries `{"skipped": "gate A outcome 2"}`.

**What this shows:** tiny score gaps carry no consistent information about the next-day return (no monotone margin pattern; Spearman 0.03 to 0.10 with intervals including 0); the exact-tie bucket is positive in some seeds and negative in others. **What it does not show:** that no signal exists; each bucket holds 16 to 112 days.

## Epochs, seeds, controls (Task 7, history only)

Epoch 0 means after the first training pass (about 93 Adam steps, `loop.py:211-243`), not an untrained model.

| seed | selected epoch | test Sharpe epoch 0 | selected | last (epoch 99) | best-test (diagnostic) | mean over 100 epochs | corr over epochs of test Sharpe with pred SD | with test IC |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | 2.54 | 1.88 | 1.70 | 2.62 | 1.61 | -0.26 | -0.01 |
| 1 | 32 | 2.39 | 1.93 | 1.12 | 2.95 | 1.91 | -0.29 | 0.13 |
| 2 | 0 | 1.89 | 1.89 | 1.88 | 3.15 | 1.76 | -0.41 | 0.06 |
| 3 | 1 | 2.04 | 2.04 | 1.93 | 3.20 | 1.81 | -0.38 | -0.00 |
| 4 | 2 | 1.82 | 2.14 | 1.84 | 2.97 | 2.00 | 0.09 | 0.28 |

Between-seed comparison (confounded with seed, not a within-run trajectory): seeds selected at epoch <= 2 ([2, 3, 4]) vs >= 8 ([0, 1]): tie rate 0.12 vs 0.47, median score SD 1.5e-04 vs 1.9e-04, median margin 6.4e-05 vs 1.9e-06, top-5 stock share 0.38 vs 0.20, Sharpe 2.02 vs 1.91; mean daily basket Jaccard within early 0.49, within late 0.61, between 0.34. Seeds selected late have many more exact ties; with three versus two seeds this is a pattern, not a test.

Not answerable from the artifacts (no weights saved; predictions kept only for the last validation-improving epoch):
- basket identities, margins and tie rates at non-selected epochs
- overlap of epoch-0 vs selected baskets
- through-model index permutation
- frozen-model 2018+ evaluation

Seeds: agreement across seeds under the same data, ordering and backtester is not independent evidence.

Weight-decay control (`docs/phase1_5/F_learnability.md` lines 7 and 11-12): same planted signal, data, code path and seeds, only weight decay changed; THINK (HH_hyper) IC/oracle 26% -> 63-84% with wd 0, decay gradient 10-40x the loss gradient, `|z|` shrinks multiplicatively over three stacked layers. Narrow conclusion: wd 5e-4 impaired planted-signal learnability; this does not attribute the real-data Sharpe change to wd (input mode, log_ic and seed count also changed). No new run is recommended. Figure: `docs/figures/phase1_5a_trajectory.png`.

**What this shows:** a test Sharpe of 1.8 to 2.5 is already present after the first training pass in every seed and averages 1.6 to 2.0 across all 100 epochs, so it is not the product of a specific selected epoch. **What it does not show:** anything within-run about baskets or ties.

## Decision table

| Hypothesis | Evidence for | Evidence against | Unresolved |
|---|---|---|---|
| **A. Tie / index-order artifact** | Exact 5th/6th ties on 16 to 112 of 237 days (112 and 109 in seeds 0, 1); stable tie-break gives the lowest-index name. | Random exact-tie order keeps the median Sharpe above hold-all in every seed (1.78 to 2.17 vs 1.53); the saved stable value lies inside the random-tie 5-95% band for seeds 0-1; model-level equivariance holds; tie groups are small, not stale, isolated or masked; a constant-score first-5 basket gets 0.37. | Why ties occur at all (cause UNKNOWN); epoch-wise tie behaviour and the trained-model permutation test (not recoverable). |
| **B. Luck / concentration / market exposure** | Hold-all Sharpe 1.53; basket beta 1.23 to 1.32 vs 1.00; Sharpe at about the 88-93 percentile of random daily top-5, F1 p 0.067 to 0.120, Holm 0.36; F2-F4 also not rejected; 56-133 distinct stocks and one stock-day (SYX) about 19 to 25% of P&L; test Sharpe 1.82 to 2.54 already after the first training pass; net of 10 bp the Sharpe is 0.98 to 1.57, below hold-all. | Beta-matched p is 0.035 to 0.081 (seed 4 0.035 < 0.05 before correction, Holm family 0.32); the industry-matched null is weak because several over-weighted industries hold one stock. | Power (CI 0.03 to 3.91); other years; cap-weighted market UNKNOWN; whether a better exposure control would reject. |
| **C. Data / backtest defect** | One stock-day (SYX 2017-03-27, +36.8%) is selected by all seeds and is about 19 to 25% of P&L (not verified externally); 2 to 4 stale-close stock-days per seed in baskets. | Existing 15 data tests pass; 25/25 run-seeds recompute; 0 duplicate series; 0 selected stock-days with a fill value; no reversal after either abs(ret) > 0.2 day; stale share of P&L about -0.3%; no look-ahead in `norm=train`. | SYX event authenticity (UNKNOWN); the reference `R5_f_paper` arm uses the full-series max (look-ahead) and is secondary only. |
| **D. Genuine top-tail skill** | IC 0.0050 to 0.0081 > 0 in all five seeds; NDCG@5 0.5673 to 0.5710 vs random 0.5639; realised-top-decile hit 0.155 to 0.203 vs bottom-decile 0.138 to 0.182 (directional gap about 1.5 to 3 points); top-5 beats the random-5 mean by 8.6 to 12.2 bp/day. | Top-minus-bottom predicted decile -3.9 to 2.3 bp/day with CI about +/-9 to 12 bp; no monotone margin bucket pattern, Spearman 0.03 to 0.10 with CIs including 0; local IC about 0; F1-F4 not rejected; excess over random-5 is of the size of its own sampling noise. | Not excluded, only unsupported; a small real signal cannot be detected with 237 days. Needs more years or independent seeds (see `PROPOSAL_followups.md`). |

## Verdict

**Sharpe near 2 in the R5_f2 alpha=0 THINK 2017 run is not evidence of learned stock ranking (NO EVIDENCE).** The five per-seed Sharpes (1.88 to 2.14) are at about the 88-93 percentile of random daily top-5 baskets in a year where equal-weight hold-all already gets 1.53; F1-F4 are not rejected (family p F1 0.120, F2 0.081, F3 0.641, F4 0.142; Holm 0.32 to 0.64); the block-bootstrap Sharpe 95% intervals are wide (0.28 to 3.48 in seed 0). A weak genuine signal (hypothesis D) is neither supported nor excluded. This is a statement about this exploratory 2017 sample and these seeds (not independent), not about the authors' model.

## Skipped by the predeclared sequential stopping rule (Gate A outcome 2)

- T4 items 3-4: per-stock contribution, exclude-and-reselect, best-day removal, momentum benchmark, daily excess-series CIs.
- T6 items 2-3: epsilon tie-group curves and score-jitter curves.
Each is written `skipped by the predeclared sequential stopping rule (Gate A outcome 2)` in `decompose.json` / `gaps.json`; the full script run writes `{"skipped": "gate A outcome 2"}` for the T6 curves and for the T4 `contribution` and `costs_benchmarks` keys.

## Limitations

- Not recoverable (`inventory.json`): model weights / checkpoints (none saved; loop.py has no torch.save); per-stock predictions for non-selected epochs (overwritten; only the last val-improving epoch is on disk); through-model index permutation of the trained model; frozen-model evaluation outside 2017.
- 2017 is an exploratory year; the five seeds share the days, the ordering and the evaluator, so they are repeated runs, not independent samples. Formal family: F1-F4 only (intersection-union over seeds, Holm); everything else is exploratory.
- Market = equal-weight hold-all of valid stocks; cap-weighted market UNKNOWN (not in the data).
- Matched nulls resample within a stratum with replacement across draws; beta uses training-period data only; industry strata include singleton industries and a large `n/a` bucket.
- Hit-rate and decile gaps have no formal interval except where stated; the margin-bucket intervals use a stationary block bootstrap over the bucket's days in time order, which only approximates dependence.
- Epoch 0 means after the first training pass (about 93 Adam steps), never untrained.

## Evidence file index

Large per-stock-day exports are in `results/forensics_1_5a/` (git-ignored). Committed files (sha256, first 12 hex):

| file | sha256 |
|---|---|
| `docs/phase1_5a/decompose.json` | 0bd3e6b03111 |
| `docs/phase1_5a/gaps.json` | 151380a0c375 |
| `docs/phase1_5a/gate_a.json` | 859a8888519a |
| `docs/phase1_5a/integrity.json` | 6b7d34410903 |
| `docs/phase1_5a/inventory.json` | 35469af74ea0 |
| `docs/phase1_5a/mechanism.json` | f87e76a1dd38 |
| `docs/phase1_5a/nulls.json` | fa00187c5553 |
| `docs/phase1_5a/proxy.json` | 1828d8cd95a0 |
| `docs/phase1_5a/trajectory.json` | a72fe61ec16f |
| `docs/figures/phase1_5a_calib.png` | 56270e23d4b5 |
| `docs/figures/phase1_5a_margin.png` | b3b6c9a9aa34 |
| `docs/figures/phase1_5a_mechanism.png` | f6cb4fd948ef |
| `docs/figures/phase1_5a_nulls.png` | 21eb506267bc |
| `docs/figures/phase1_5a_persistence.png` | efbc3eb83084 |
| `docs/figures/phase1_5a_proxy.png` | 41a52e8c16b6 |
| `docs/figures/phase1_5a_trajectory.png` | dace1c393524 |

Figures: persistence, margin, calib, trajectory, mechanism, proxy, nulls (`docs/figures/phase1_5a_*.png`).

## Test results

```
$ .venv/Scripts/python.exe -m pytest tests/test_forensics.py tests/test_think_equivariance.py tests/test_metrics.py tests/test_phase15_eval.py tests/test_loop.py -q
.............................................................            [100%]
61 passed in 23.69s

$ .venv/Scripts/python.exe -m pytest -m data tests/test_forensics_artifacts.py tests/test_phase15_eval.py tests/test_phase15_data.py -q
....................                                                     [100%]
20 passed, 9 deselected in 18.98s

$ .venv/Scripts/python.exe -m pytest -q

324 passed, 1 warning in 100.24s (0:01:40)
```
