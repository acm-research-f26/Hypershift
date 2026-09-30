# R7: NASDAQ 3-class movement (F1) on the corrected graph (g2)

Runs: `results/E11_clf_g2/{HH,EH,EE}/seed_0..24` (25 seeds per arm, complete). HH = THINK, EH = TCONV+DHHAN (the paper's Euclidean arm), EE = fully Euclidean. Level inputs, `seq` 16, `hidden` 32, `lr` 1e-3, 60 epochs max, patience 20, exact eq.14 attention, corrected NASDAQ graph (1066 hyperedges). The epoch is selected on **validation macro-F1** (`src/hypershift/train/clf.py`).
Scripts: `scripts/r7_stats.py` (table, chance baselines, paired tests), `scripts/r7_diag.py` and `scripts/r7_diag_summary.py` (per-epoch diagnosis, CPU re-runs).

## Paper reference
p. 852 Table II, "Clf" column ("F1(↑)"; caption: mean of 25 runs, Clf = stock movement classification on NASDAQ). Values as read from the page image: THINK 0.49 ± 4e-3, TCONV+DHHAN 0.44 ± 8e-3, STHGCN 0.40 ± 1e-3, RSR-I 0.38 ± 6e-3, EGCN-H 0.37 ± 1e-3, ST-TGCN 0.37 ± 1e-3, DyGrAE 0.36 ± 2e-3, TGCN 0.34 ± 4e-3, DCRNN 0.32 ± 2e-3, EGCN-O 0.28 ± 3e-3, GConvGRU 0.25 ± 2e-4. The paper does not say whether F1 is macro or micro, nor how classes or thresholds are defined (Sec. IV-A and IV-B give none). Three of the baselines (GConvGRU 0.25, EGCN-O 0.28, DCRNN 0.32) are below 0.333 and TGCN is just above; we cannot say why.

## 1. Results (test set, mean ± sample std over 25 seeds)

| Arm | test macro-F1 | test micro-F1 | val macro-F1 | val micro-F1 | best epoch (median, range) |
|---|---|---|---|---|---|
| HH (THINK) | 0.2897 ± 0.0108 | 0.3809 ± 0.0008 | 0.3589 ± 0.0044 | 0.3665 ± 0.0051 | 23 (13-55) |
| EH (TCONV+DHHAN) | 0.2819 ± 0.0169 | 0.3795 ± 0.0043 | 0.3525 ± 0.0087 | 0.3678 ± 0.0035 | 15 (0-49) |
| EE | 0.2946 ± 0.0120 | 0.3824 ± 0.0005 | 0.3549 ± 0.0063 | 0.3686 ± 0.0036 | 12 (6-33) |
| paper THINK | 0.49 (averaging unstated) | | | | |

Micro-F1 of a single-label multiclass problem equals accuracy. **No arm is near 0.49 under either averaging**; the best single seed is macro 0.318 (EE), micro 0.384.

## 2. Chance baselines on the same test labels
Labels: tertile thresholds of pooled **training** returns, `[-0.00480, +0.00575]` (from `tertile_thresholds`), classes 0 = down, 1 = neutral, 2 = up. Label shares:

| split | n stock-days | down | neutral | up |
|---|---|---|---|---|
| train | 753,669 | 0.3333 | 0.3333 | 0.3333 |
| val (2016) | 256,451 | 0.3330 | 0.3177 | 0.3494 |
| test (2017) | 242,400 | 0.3108 | 0.3696 | 0.3195 |

The test neutral class is over-represented (consistent with 2017 being a low-volatility year; volatility was not measured here).

| baseline (200 draws where random) | macro-F1 | micro-F1 |
|---|---|---|
| uniform random | 0.3329 | 0.3334 |
| class-prior sampling, train prior (1/3 each) | 0.3328 | 0.3333 |
| class-prior sampling, test prior (oracle) | 0.3333 | 0.3353 |
| always test-majority (neutral) | 0.1799 | 0.3696 |
| always train-majority (a three-way tie; `argmax` gives class 0, down) | 0.1581 | 0.3108 |
| always up | 0.1614 | 0.3195 |

Train has exactly equal class shares, so "the train-majority class" is a tie, not a real majority. A constant predictor has macro-F1 of at most 0.18 whatever the class. Our arms' micro-F1 (0.380-0.382) is 0.011-0.013 above always-neutral and 0.047 above random, so there is a very small real signal in accuracy. Macro-F1 (0.28-0.29) is **below** the 0.333 random level.

## 3. Paired by seed (25 seeds; Wilcoxon signed-rank on seed differences, bootstrap 95% CI of the mean difference, 20,000 resamples; raw p, no Holm)

| contrast | metric | mean diff | 95% CI | Wilcoxon p | HH wins |
|---|---|---|---|---|---|
| HH − EH (paper's comparison) | test macro | +0.0077 | [0.0000, +0.0154] | 0.080 | 16/25 |
| | test micro | +0.0014 | [-0.0001, +0.0033] | 0.47 | 13/25 |
| | val macro (selection metric) | +0.0065 | [+0.0033, +0.0096] | 0.0006 | 20/25 |
| HH − EE | test macro | -0.0049 | [-0.0121, +0.0021] | 0.25 | 10/25 |
| | test micro | -0.0015 | [-0.0019, -0.0011] | 3.6e-5 | 2/25 |
| | val macro | +0.0040 | [+0.0011, +0.0071] | 0.024 | 19/25 |
| EH − EE | test macro | -0.0126 | [-0.0211, -0.0042] | 0.0088 | 7/25 |

Reading: THINK is nominally ahead of EH on test macro-F1 (+0.008, 16/25 seeds) but p = 0.08 and the CI touches zero: not significant at the paper's p < 0.01 level, and the gap is 0.008 against the paper's 0.05 (0.49 vs 0.44). The val macro-F1 gap is significant but val is the selection metric, so it is a selection-biased number (a maximum over epochs). THINK vs EE: no test macro difference; EE is nominally better on test micro (0.3824 vs 0.3809, p 4e-5), but EE's micro-F1 std is 0.0005, i.e. its predictions are close to a constant "neutral" (see section 4), so a 0.0015 micro difference is not evidence of a better model. EH is lowest on test macro (below EE, p 0.009). Differences of this size on a metric that sits below chance should not be read as a geometry effect.

## 4. Why macro-F1 is below 0.33 (diagnosis)

**Code check (`clf.py`).** Labels are `np.digitize(gt, th)` with `th` = the 1/3 and 2/3 quantiles of `gt` over valid stock-days of the training windows only (no leakage; test tertile shares differ: 0.311 / 0.370 / 0.320). Loss is masked cross-entropy over 3 logits, no class weights. Both F1s are computed once over all valid stock-days pooled across the split (not per day). Epoch = best validation macro-F1 with patience 20. Nothing here is a bug; it is a pooled 3-class problem.

**The 25-seed runs saved no predictions and no per-epoch history** (only `metrics.json` with the selected epoch). The check below therefore re-ran the same training loop on CPU (`scripts/r7_diag.py`, new folder `results/E11_clf_g2_diag/`), logging predicted class shares, per-day agreement and F1 per epoch. CPU and GPU runs are not bit-identical (HH seed 0: selected epoch 11 on CPU vs 18 on GPU; test macro 0.320 vs 0.296), so this diagnoses the mechanism from 4 runs (HH seeds 0 and 1, EH seed 0, EE seed 0; 27-51 epochs each), not the 25-seed numbers. Only 4 runs were affordable on CPU (about 50-140 s per epoch).

Findings (all four runs; `results/E11_clf_g2_diag/*.json`):
1. **The model barely learns.** Training loss goes from 1.0989 to 1.0934-1.0951; ln 3 = 1.0986 is the loss of a uniform predictor. Almost no signal is extracted from the 5 price features for the tertile label.
2. **Test predictions collapse toward the neutral class.** At the selected epoch the test predicted shares (down / neutral / up) are HH s0 0.19 / 0.70 / 0.11, HH s1 0.09 / 0.83 / 0.08, EH s0 0.07 / 0.82 / 0.10, EE s0 0.11 / 0.75 / 0.14, against true 0.31 / 0.37 / 0.32. Late in training (epochs 25-50) neutral is 0.85-0.92 of test predictions. At epoch 0 HH is almost degenerate (98.5% neutral, test macro 0.19).
3. **Predictions are day-level, not stock-level.** The mean over test days of the share of stocks given the day's modal class is 0.70-0.90 (0.70-0.83 at the selected epochs); the model mostly outputs one class for every stock on a day. Independent random predictions would give about 0.34. The dominant class changes from day to day and epoch to epoch (HH s0 flipped its dominant test class from neutral to down in epoch 16, macro 0.209), which points to a market-level effect rather than cross-sectional discrimination. F1 for the rarely predicted down and up classes is then very low (HH s1 test confusion matrix at the selected epoch: down recall 0.10, neutral 0.86, up 0.10), and macro-F1 averages those with the neutral F1.
4. **Below chance in every epoch.** Across the four diagnosis runs (about 140 epochs) test macro-F1 was >= 0.333 in **zero** epochs (range 0.16-0.33, max 0.329), while val macro-F1 was >= 0.333 in 2-7 epochs per run (max 0.358).
5. **Val-selection does not pick a constant-class epoch, but the choice does not transfer.** The selected epoch is one where the *validation* predictions are relatively balanced (val predicted shares e.g. 0.44 / 0.30 / 0.26, 0.25 / 0.50 / 0.24, 0.21 / 0.38 / 0.41 vs true val 0.33 / 0.32 / 0.35), giving val macro 0.354-0.358 (a maximum over epochs, so optimistic). The same epoch on the 2017 test set predicts 0.70-0.83 neutral and gives test macro 0.28-0.32. A constant-class epoch would score val macro <= 0.19 and is never selected. Val and test are different years with different class shares, and the selection metric is a noisy maximum, so the val advantage does not carry over.

**Established vs not.** Established (4 CPU re-runs plus the 25-seed metrics): the result is 0.28-0.29 macro and 0.38 micro; the models output a near-constant class per day, mostly neutral; train loss stays near ln 3. Not established: *why* neutral dominates (it is also the most frequent test class, 0.37, but the tilt is far larger than that; we did not test volatility, input scaling, level vs relative inputs, or the head), and whether the paper's 0.49 uses a different label definition, averaging or protocol. The early macro-F1 of about 0.28 mentioned in the brief matches the final 25-seed numbers (0.28-0.29), so it is the same behaviour; the earlier runs themselves were not re-inspected (only `results/E11_clf_g2` exists).

## 5. Bottom line
- **R7 not reproduced**: THINK 0.290 macro / 0.381 micro vs paper 0.49; neither averaging reaches it. Some of the paper's baselines (0.25-0.32) are also at or below the 0.333 random level, so we do not know what F1 the paper reports.
- THINK does not clearly beat TCONV+DHHAN (HH − EH = +0.008 macro, p 0.08, CI [0.000, 0.015]; micro +0.001, p 0.47) and does not beat EE.
- Macro-F1 is below 0.333 because predictions are near-constant per day and mostly neutral, while accuracy is only 0.011-0.013 above always-neutral. With these inputs and this loss the task is close to unlearnable, so the hyperbolic-vs-Euclidean ordering should not be read from these numbers.
- Untested follow-ups (not run): relative inputs, class-weighted loss, cross-sectionally demeaned labels, and an STHGCN-style setup (lookback 50).
