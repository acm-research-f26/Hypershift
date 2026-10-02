# R8 results: the paper's baselines (RSR-I, STHGCN) vs THINK and EH

*2026-10-01. Script: `scripts/r8_analysis.py` (`CUDA_VISIBLE_DEVICES=-1`, recomputes everything from `metrics.json`, `history.jsonl`, `test_*.npy`; run with `--fig`). Figure: `docs/figures/r8_baselines.png`. Implementation, deviations and run plan: `docs/phase1/R8_baselines.md`. Style, definitions and THINK/EH numbers: `docs/phase1/R5_full_nyse_g2.md`, `docs/phase1/g2_small_results.md`.*

- **Leak-free** = the epoch with the best validation (2016) Sharpe, scored on 2017. **Best-test** = max over epochs of the 2017 Sharpe (the paper-style upper bound; it selects on the test year; C1 shows shuffled labels give the same best-test, so it is not evidence of skill). Sharpe is ours: top-5, mean/std x sqrt(252), R_f = 0, not the paper's formula (p852 Sec. IV-B; k and R_f unstated, U7, PA4), so levels are not like-for-like with the paper's.
- **Paper reference** (`docs/paper/icdm22-think.pdf` p852 Table II, NYSE Sharpe): THINK 1.18, TCONV+DHHAN (EH) 1.14, STHGCN 1.10, RSR-I 1.05.
- **Runs.** Full NYSE (1737 stocks, 4350 edges, `norm paper`, 100 epochs, level inputs, lr 1e-3, alpha 1, `batch_days 8`): RSR-I 10 seeds (laptop 0-3, Kaggle s3 4-9), STHGCN 10 seeds (Kaggle s2), against `R5_g2` THINK 25 (laptop) and EH 25 (Kaggle). Small scale (309 stocks, 558 edges, `norm train`, level inputs, 30 epochs, patience 10, lr 1e-3, alpha 1): RSR-I 10 and STHGCN 10 (`POC_sectors_R8_*_g2`) against `POC_sectors_g2`. Config check (script output): every arm has a single config tuple, all 100 epochs run on full NYSE, identical test masks within each scale; the small R8 runs and `POC_sectors_g2` have identical lr, alpha, input mode, norm, epoch cap, nodes and edges (the R8 small runs are level inputs, as the g2 arms I compare against).
- **Merge of Kaggle s3 (this task):** `fetch.sh s3` status COMPLETE; dry run then real merge: **6 complete runs copied (RSR_I seeds 4-9 into `R8_baselines_g2`), 0 conflicts, 0 local partials, 0 Kaggle-incomplete, 0 failed runs.** The laptop RSR-I seeds 2 and 3 finished at about 16:20 (100 epochs, `metrics.json` present); counts verified: full NYSE RSR_I 10, STHGCN 10; small `rsr_i` 10, `sthgcn` 10. No seed was missing.
- **Seeds are not matched pairs.** Seed counts (10 vs 25) and platforms (laptop, Kaggle T4) differ, so full-NYSE tests are unpaired (Mann-Whitney, Holm over 5 comparisons) and the CI is a stationary block bootstrap (mean block 10, 5000 resamples) of the Sharpe difference of the seed-averaged daily series at the leak-free epoch. Small-scale arms share seeds and platform, so the paired Wilcoxon is also shown.

## 1. Full NYSE: Sharpe (mean ± std over seeds)

| arm | seeds | leak-free | best-test | leak-free / 15.87 | best-test / 15.87 | val Sharpe at selected epoch | leak-free > hold-all | leak-free > fixed-random mean | best-test > hold-all |
|---|---|---|---|---|---|---|---|---|---|
| THINK (HH) | 25 | 0.089 ± 0.300 | 2.115 ± 0.386 | 0.0056 | 0.1332 | 2.884 | 0/25 | 0/25 | 23/25 |
| TConv+DHHAN (EH) | 25 | 0.383 ± 0.758 | 2.158 ± 0.462 | 0.0241 | 0.1360 | 2.471 | 3/25 | 6/25 | 23/25 |
| STHGCN | 10 | 0.649 ± 0.443 | **3.246 ± 0.345** | 0.0409 | 0.2045 | 2.252 | 0/10 | 3/10 | 10/10 |
| RSR-I | 10 | **0.879 ± 0.443** | 2.369 ± 0.219 | 0.0554 | 0.1492 | 2.039 | 2/10 | 3/10 | 10/10 |

Baselines on the same 237 test days and masks: hold all 1737 stocks **1.531**; fixed random 5-stock set held all year **0.955 ± 0.887** (2000 draws, 5-95% [-0.48, 2.42]); random top-5 redrawn daily 0.856; constant (all-tied) prediction 0.368. THINK and EH rows reproduce `R5_g2` exactly.

- **No arm beats holding the market leak-free** (STHGCN 0/10 seeds, RSR-I 2/10, THINK 0/25, EH 3/25). No arm's mean beats the fixed random 5-stock set (0.955); RSR-I is the nearest at 0.879.
- Best-test beats hold-all for nearly every seed in every arm, but this is selection on the test year (C1).
- Validation Sharpe at the chosen epoch is 2.0-2.9 for all four arms and the 2017 value at that epoch is 0.1-0.9: the validation-to-test disconnect is the same for the baselines.
- Unannualized = ours / sqrt(252) = 15.87 (columns above; PA4, the paper's formula is not annualized as ours is, so its 1.18 is not directly comparable): leak-free RSR-I 0.055, STHGCN 0.041, THINK 0.006, EH 0.024.

## 2. Full NYSE: diagnostics at the leak-free epoch

| arm | NDCG@5 | NDCG@5 at best-test epoch | NDCG_sthan (authors' evaluator) | MSE / zero-MSE | IC | hit-rate pp | pred spread / actual | runs spread < 1e-5 |
|---|---|---|---|---|---|---|---|---|
| THINK | 0.5616 ± 0.0013 | 0.5680 | 0.758 ± 0.040 | 1.001 ± 0.003 | +0.0037 ± 0.0039 | +0.95 | 0.001 | 9/25 |
| EH | 0.5634 ± 0.0028 | 0.5686 | 0.796 ± 0.083 | 1.007 ± 0.016 | 0.0000 ± 0.0068 | -0.03 | 0.011 | 9/25 |
| STHGCN | 0.5637 ± 0.0014 | 0.5677 | 0.866 ± 0.090 | **2.160 ± 1.151** | -0.0032 ± 0.0075 | -0.54 | 0.034 | **7/10** |
| RSR-I | 0.5669 ± 0.0027 | 0.5673 | 0.865 ± 0.062 | 1.003 ± 0.003 | **+0.0106 ± 0.0027** | -2.12 | 0.008 | 5/10 |

Random NDCG@5 is 0.5639 (100 draws). Constant-prediction check: best-test equals the constant Sharpe (0.368) in 0/10 runs of both baselines.

- **No baseline predicts returns either.** NDCG@5 is within +-0.003 of random (RSR-I highest at 0.5669, +0.003). MSE is 1.00 times predicting zero for RSR-I and **2.2 times** for STHGCN (a large constant offset: predictions are not centred on zero, mean prediction -0.026 to +0.028 in several runs). Hit rates are within +-2.6 pp of 50 (RSR-I is -2.1 ± 1.6, a sign bias, not skill).
- **STHGCN's portfolio is essentially fixed.** At the leak-free epoch the daily top-5 had only **2 distinct sets over 237 days in 7 of 10 runs** (1 in one run, 13 and 23 in two), so its Sharpe is "which five stocks it landed on". Its leak-free mean 0.649 is below the fixed-random mean 0.955. RSR-I has 9-21 distinct sets in 9 runs (110 in one), also near-fixed. (Computed in a one-off check, not saved as a script.)
- **RSR-I has the only per-seed IC above zero** (+0.0106 ± 0.0027, tiny seed spread). It does not survive an ensemble check: averaging the 10 seeds' predictions gives a daily-IC mean of +0.007 with t = 0.73 over 237 days (THINK +0.004, t 0.46; EH -0.009, t -1.27; STHGCN -0.016, t -1.80). So I cannot call it signal.
- The authors' evaluator (`ndcg_sthan`, last day only) gives 0.76-0.87 for all four no-skill models, which is the range of the paper's 0.86 (p852 Table II); consistent with the NDCG-bug finding (E3), not with skill. The baselines score higher on this evaluator (0.866, 0.865) than THINK (0.758), which again reflects index-set ties and not ranking.
- Model sizes (from `R8_baselines.md`): RSR-I 5,199 parameters, THINK 1,890, STHGCN 105,381.

## 3. Full NYSE: baselines vs THINK and EH

**Leak-free** (Mann-Whitney unpaired, Holm over 5; CI of the Sharpe difference of seed-averaged daily series; verdict word per `eval/stats.py`)

| comparison | n a / n b | diff | MW p | Holm p | bootstrap CI | P(a > b) | verdict |
|---|---|---|---|---|---|---|---|
| RSR-I vs THINK | 10 / 25 | +0.791 | 0.0001 | 0.0005 | [-0.470, +2.247] | 0.93 | SEED-ROBUST ONLY |
| STHGCN vs THINK | 10 / 25 | +0.561 | 0.0020 | 0.0081 | [-0.519, +2.297] | 0.84 | SEED-ROBUST ONLY |
| RSR-I vs EH | 10 / 25 | +0.497 | 0.0298 | 0.0894 | [-0.743, +1.853] | 0.74 | NO EVIDENCE |
| STHGCN vs EH | 10 / 25 | +0.267 | 0.1492 | 0.2984 | [-0.622, +1.682] | 0.66 | NO EVIDENCE |
| RSR-I vs STHGCN | 10 / 10 | +0.230 | 0.2730 | 0.2984 | [-1.536, +1.623] | 0.65 | NO EVIDENCE |

**Best-test** (diagnostic only; no CI because `test_daily.npy` holds only the leak-free epoch): RSR-I - THINK +0.254 (Holm 0.10); STHGCN - THINK +1.131 (Holm 0.0001); RSR-I - EH +0.211 (Holm 0.10); STHGCN - EH +1.088 (Holm 0.0001); RSR-I - STHGCN -0.877 (Holm 0.003).

- **The baselines are nominally above THINK, the opposite of the paper**, and the seed-level test is strong against THINK (RSR-I Holm 0.0005, STHGCN 0.008) while the bootstrap CIs span zero, hence SEED-ROBUST ONLY. Three reasons not to read this as "RSR-I is a better model": (1) both are below hold-all and below the fixed-random mean, so the gap is between very weak portfolios (0.09 vs 0.65-0.88); (2) THINK has 9/25 near-constant runs and STHGCN 7/10, so each arm's leak-free Sharpe largely reflects which few stocks it held in 2017 (not tested stock by stock); (3) the comparison is across platforms (THINK on the laptop, STHGCN on Kaggle, RSR-I on both) with 10 vs 25 seeds, and I cannot exclude a platform effect, though a 0.5-0.8 shift from kernels alone would be large against seed std 0.3-0.76.
- Against EH the baselines are not distinguishable (Holm 0.09 to 0.30).
- STHGCN's best-test of 3.25 is far above all other arms (+1.1, Holm 0.0001). It is a max over 100 epochs of the test-year Sharpe, and C1 says this number is selection; why STHGCN's maximum is higher than the others' I did not investigate (INFERRED, not tested: larger epoch-to-epoch variation of a near-fixed portfolio). Median best-test epoch is 32, selected epoch 56.

## 4. Does the paper's ordering reproduce? (THINK 1.18 > EH 1.14 > STHGCN 1.10 > RSR-I 1.05)

| selection | full NYSE ordering by mean | small-scale ordering by mean |
|---|---|---|
| leak-free | **RSR-I 0.879 > STHGCN 0.649 > EH 0.383 > THINK 0.089** | RSR-I 0.130 > EH -0.268 > STHGCN -0.586 > THINK -0.681 |
| best-test | **STHGCN 3.246 > RSR-I 2.369 > EH 2.158 > THINK 2.115** | THINK 2.513 > EH 2.100 > RSR-I 2.096 > STHGCN 1.997 |

- **Full NYSE: no, under either selection rule.** Leak-free the observed order is the exact reverse of the paper's; best-test puts both baselines above THINK and STHGCN far above everything. One-sided tests in the paper's direction: THINK > EH p 0.74 (leak-free) and 0.57 (best-test); EH > STHGCN p 0.93 and 1.00; STHGCN > RSR-I p 0.88 leak-free but **p 0.001 best-test** (the paper's STHGCN > RSR-I holds at best-test, but THINK > STHGCN fails there with p 1.00).
- **Small scale: the paper's order shows only nominally and only at best-test.** THINK 2.513 > EH 2.100 > {RSR-I 2.096, STHGCN 1.997} (RSR-I and STHGCN are swapped relative to the paper and within noise, MW p 0.31). THINK - STHGCN +0.517 (raw p 0.026, Holm over 7 0.129), THINK - RSR-I +0.418 (raw p 0.089, Holm 0.356): not significant after correction; and the small-scale C1 control (`g2_small_results.md`) shows the best-test ordering is what shuffled labels also produce. Leak-free the order is different again (RSR-I first, THINK last, no pairwise difference significant after Holm; RSR-I vs THINK raw p 0.014, Holm 0.084).
- The paper's gaps are 0.04-0.05 Sharpe between adjacent models. Our seed std is 0.2-0.8 and one 237-day Sharpe has a standard error near 1, so even a perfectly replicating setup could not resolve those gaps; what we can say is that the ordering is not supported and the direction on full NYSE is reversed leak-free.

## 5. Small scale (309 stocks, level inputs)

| arm | seeds | leak-free | best-test | leak-free / 15.87 | best-test / 15.87 | NDCG@5 (leak-free) | MSE / zero-MSE | IC | hit pp |
|---|---|---|---|---|---|---|---|---|---|
| HH_hyper (THINK) | 10 | -0.681 ± 1.219 | 2.513 ± 0.421 | -0.0429 | 0.1583 | 0.5396 | 1.012 | -0.010 | -0.26 |
| EH_hyper | 10 | -0.268 ± 1.121 | 2.100 ± 0.976 | -0.0169 | 0.1323 | 0.5433 | 1.043 | -0.001 | +0.26 |
| EE_hyper | 10 | -0.154 ± 0.879 | 2.134 ± 0.862 | -0.0097 | 0.1344 | 0.5441 | 1.067 | -0.001 | -0.15 |
| HH_none | 10 | 0.279 ± 0.350 | 1.108 ± 0.261 | 0.0176 | 0.0698 | 0.5504 | 1.670 | -0.011 | -0.54 |
| EE_none | 10 | 0.203 ± 0.583 | 1.184 ± 0.599 | 0.0128 | 0.0746 | 0.5505 | 1.057 | -0.013 | -0.14 |
| **STHGCN** | 10 | -0.586 ± 0.770 | 1.997 ± 0.532 | -0.0369 | 0.1258 | 0.5437 | 1.699 | -0.003 | -0.12 |
| **RSR-I** | 10 | 0.130 ± 0.646 | 2.096 ± 0.494 | 0.0082 | 0.1320 | 0.5511 | 1.038 | -0.011 | -1.40 |

Hold-all 0.752; fixed random 5 stocks 0.598 ± 0.891; random-5 redrawn daily 0.505; constant-prediction Sharpe 1.924; random NDCG@5 0.5504. Arms stop early (patience 10; epochs run 11-30), so best-test is a max over unequal numbers of epochs.

Leak-free, Mann-Whitney / paired Wilcoxon (same seeds), Holm over 7, bootstrap CI:

| comparison | diff | MW p | Holm p | CI | paired Wilcoxon p | verdict |
|---|---|---|---|---|---|---|
| RSR-I vs THINK | +0.812 | 0.014 | 0.084 | [-0.085, +2.579] | 0.084 | NO EVIDENCE |
| STHGCN vs THINK | +0.096 | 0.121 | 0.485 | [-0.382, +0.691] | 0.375 | NO EVIDENCE |
| RSR-I vs EH | +0.398 | 0.162 | 0.486 | [-0.233, +2.094] | 0.557 | NO EVIDENCE |
| STHGCN vs EH | -0.318 | 0.970 | 0.970 | [-0.771, +0.410] | 0.695 | NO EVIDENCE |
| RSR-I vs STHGCN | +0.716 | 0.014 | 0.084 | [-0.133, +2.404] | 0.106 | NO EVIDENCE |
| RSR-I vs HH_none | -0.149 | 0.308 | 0.615 | [-0.848, +0.751] | 0.695 | NO EVIDENCE |
| STHGCN vs HH_none | -0.865 | 0.0028 | 0.0198 | [-1.979, -0.357] | 0.0195 | NO EVIDENCE |

- Nothing beats the market leak-free (0.752): STHGCN 1/10 seeds, RSR-I 1/10. Neither baseline is distinguishable from THINK or EH after Holm. The one raw-significant contrast, STHGCN vs HH_none (-0.865, CI excludes 0), has Holm 0.0198 > 0.01 and is against graph use: the baseline with a graph is below the no-graph THINK arm.
- Best-test: both baselines are far above HH_none (+0.99 RSR-I, Holm 0.013; +0.89 STHGCN, Holm 0.004), the same "any graph arm beats none at best-test" pattern as in `g2_small_results.md` section 1, and the shuffled-label control shows this pattern is selection.
- Diagnostics: no skill (NDCG@5 0.540-0.551 vs random 0.5504, IC -0.013 to -0.001, MSE ratio >= 1.0); STHGCN has MSE 1.7 times zero, as on full NYSE.

## 6. Caveats

- **Implementation deviations** (from `R8_baselines.md`): RSR-I is trained end-to-end (RSR pre-trains the LSTM and loads it), window 16 not RSR's 8 for NYSE, our optimizer (weight decay 5e-4, grad clip 1), our normalisation, and RSR-E and single-relation-family variants are not run. **STHGCN is mostly inferred**: the repo does not run and its paper text was not read; block layout, widths 64/16, the isolated-node handling and BatchNorm over the node axis are inferred, per-stock GRUs omitted, ranking loss instead of the tertile classification head, lookback 16 instead of 50, BatchNorm statistics from 4-day micro-batches. A STHGCN result here says "this reimplementation", not "the paper's STHGCN". Its odd diagnostics (MSE 2x zero, near-fixed portfolios) may be partly a BatchNorm or implementation effect that I did not isolate.
- Mixed platforms and unequal seed counts (see Runs); unpaired tests. THINK/EH are 25 seeds, the baselines 10; the p-value floor and power differ.
- One test year (2017), 237 days, Sharpe standard error near 1; our Sharpe is not the paper's formula so absolute levels do not compare with 1.05-1.18.
- Holm family: the 5 (full) and 7 (small) comparisons in the tables (my choice; the grid's `FAMILIES` was not used). The extra per-pair lines printed by the script (one-sided tests of the paper's order) are raw and uncorrected.
- The "distinct top-5 sets" and ensemble-IC checks were run as a one-off script and are not part of `scripts/r8_analysis.py`; the figures are quoted from that run.

## 7. Verdict

**R8 DONE. The paper's baseline comparison is NOT REPRODUCED.** On full NYSE the ordering THINK > EH > STHGCN > RSR-I does not appear under leak-free selection (it is reversed: RSR-I 0.88 > STHGCN 0.65 > EH 0.38 > THINK 0.09) nor under best-test (STHGCN 3.25 > RSR-I 2.37 > EH 2.16 > THINK 2.12); the baselines' advantage over THINK is statistically clear at the seed level (Holm 0.0005 and 0.008) but is between portfolios that are all below hold-all (1.53) and near-fixed, and the bootstrap CIs span zero. No model, baseline or THINK, predicts returns (NDCG at random, IC about 0, MSE at or above the zero predictor). The small-scale best-test shows the paper's order nominally for THINK > EH > baselines, not significant after Holm and uninformative given C1.
