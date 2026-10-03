# Phase 1.5a report: 2017 Sharpe near 2 (R5_f2 alpha=0 THINK, HH)

Primary: `R5_f2_alpha0_train/HH` seeds 0-4, validation-selected epoch, top-5 equal weight, 237 test days (2017-01-03 to 2017-12-08). Seeds are repeated runs on the same days, not independent samples. 2017 is exploratory.

## Gate A

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

1. Outcome 2 (exposure B sufficient) holds: in every primary seed the Sharpe sits inside the central 90% of the beta-matched or industry-matched null (industry p 0.145 to 0.641; beta p 0.035 to 0.081, so seed 4 is below 0.05 on beta alone). Outcomes 1 and 3 do not hold, so Gate A is not "not decisive" but it is a single-mechanism result, not a full explanation.
2. The 2017 equal-weight market (hold-all) already has Sharpe 1.53. The 5 seeds' 1.88 to 2.14 are 0.35 to 0.61 above it; the Sharpe of the daily excess over hold-all is 1.5 to 1.9 but its block-bootstrap CI is wide (lowest lower bound 0.03, highest upper bound 3.9).
3. Against a uniform random daily top-5 (F1, same days, mask, k) the per-seed one-sided p is 0.067 to 0.120. Family p (max over seeds) F1 0.120, F2 0.081, F3 0.641, F4 0.142; Holm-adjusted 0.32 to 0.64. No test rejects. This is "no evidence against the null" on 237 days, not proof of no skill; power is low (see the CI width).
4. Tie/index path (A) is not decisive: with random exact-tie order the median Sharpe stays above hold-all in every seed (1.78, 1.79, 2.07, 2.14, 2.17 vs 1.53); the reverse-index Sharpe is 1.60 to 2.42. For seeds 0 and 1 the saved stable value (1.88, 1.93) sits inside the random-tie 5-95% band (1.40 to 2.15), so the stable tie-break neither inflates nor deflates materially. 112 and 109 of 237 days have an exact tie at the 5th/6th boundary in seeds 0 and 1 (24, 16, 43 in seeds 2 to 4). Constant-score first-5 gives 0.37 and a random fixed-index basket rule has median 0.95 (p95 2.52), so an index-order basket alone does not reproduce 2 as a typical outcome. Not an artifact claim from tie rate alone.
5. Tie composition in the primary run: boundary tie groups are small (mean size 3.4 to 6.5 in seeds 0, 2, 3, 4; seed 1 has a few large groups, max 1296), every day has a different tied value (no repeated saturated value except 4 days in seed 1), 0% graph-isolated, 0% stale window, 0% partly masked. So the ties are not explained by identical inputs under these three tests (mechanism cause UNKNOWN). In the R5_f_train/HH reference, ties are mass ties of about 1700 valid stocks sharing one constant output (collapsed days), a different phenomenon.
6. Model-level equivariance (random-init THINK, 4 temporal/spatial combos, node permutation of inputs and graph) holds to rtol 1e-5, so the model has no index-dependent step; the index only enters through the evaluator's stable tie-break. The trained-model version is not recoverable (no weights saved).
7. Factor proxy: the score is positively associated with ma30_rel and ma20_rel and negatively with ret20 (daily Spearman 0.2 to 0.4 in magnitude): a short-horizon mean-reversion / oversold tilt, shared across seeds. An all-feature linear fit explains a mean per-day R2 of 0.09 to 0.21. The fitted portfolio gets 0.33 to 1.76 (below the model in every seed), while the residual (score minus fit) portfolio keeps 1.83 to 2.29. So the simple proxies do not carry the return; most of it stays in the unexplained part. Literal outcome 3 is not met (seed 3 residual 2.29 lies above the random null p95; also 'index' and 'degree' qualify as "proxies" only as arbitrary fixed baskets, which the economic-proxy reading excludes).
8. Verdict on whether Sharpe near 2 is evidence of learned ranking: NO EVIDENCE (one-sided, exploratory, 2017 only, seeds not independent). The result is compatible with market exposure plus luck, with a weak mean-reversion tilt, and it is not excluded that a small genuine top-tail signal exists (hypothesis D is not ruled out here; margin buckets and top-k diagnostics are in batch 2 / Task 6).
9. Sequential stopping applies (outcome 2 holds): batch 2 should run Task 3 (integrity), Task 6 items 4-5 (margin buckets, top-k) and Task 8; epsilon/jitter curves and Task 4 items 3-4 are "skipped by sequential stopping rule".

### Limitations and notes

- Matched null (industry / beta quintile) resamples within stratum with replacement across draws and may repeat a stock within one basket; beta uses training-period data only (stocks with fewer than 250 train days form their own stratum, 'n/a' industry is one large stratum).
- Proxy portfolios and the fitted fit exclude stock-days with a non-finite feature (ret20 / vol20 use the close history beyond the 16-day window; days where any of those days is masked are NaN); the universe is therefore slightly smaller for them.
- Seed 3's residual-portfolio check uses the random-null p5/p95 (central 90%) of daily top-5 Sharpe.
- Not recoverable (see `inventory.json`): weights, per-epoch predictions, through-model permutation of the trained model, frozen-model evaluation outside 2017.
- Epoch 0 means after the first training pass (about 93 Adam steps), never "untrained".
