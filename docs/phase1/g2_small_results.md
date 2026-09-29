# Small-scale g2 rerun: 309 NYSE stocks on the corrected hypergraph

*2026-09-29. Scope: step (a) of the g2 small-scale queue. 309 stocks (Energy/Utilities + Finance), corrected hypergraph (cache v2, 558 edges), exact eq. 14, 30 epochs, 10 seeds per arm, no tuning. Arms: HH, EE and EH × hyper / clique / none (no EH_none, it would equal EE_none).*

- Level inputs: `results/POC_sectors_g2/`. Relative inputs: `results/POC_sectors_rel_g2/`. Old-graph comparison: `results/POC_sectors_eq14/` and `results/POC_sectors_rel_eq14/`.
- `summary.md` was regenerated with `scripts/poc_sectors.py summarize --variant g2` and `--variant rel_g2`. Every number below was recomputed from `metrics.json` with a separate script (means and stds of `test.sr` and `test_oracle_sr` for all 16 arm-variants match `summary.md` to 3 decimals; 0 mismatches). The diagnostics in section 3 use the same logic as `scripts/poc_figures.py` (`diag`, `random_ndcg`), re-implemented so that the figures are untouched.
- **Leak-free** = the epoch with the best validation (2016) Sharpe, scored on 2017. **Best-test** = the maximum over epochs of the 2017 Sharpe (the paper-style upper bound; it selects on the test year). Sharpe is ours: top-5, mean/std × sqrt(252), no risk-free rate (not the paper's formula, p852 Sec. IV-B).
- Baselines on the same 237 test days: hold all 309 stocks 0.752, random 5 stocks 0.338, 5-day momentum -0.276.
- Citations: `pNNN` = page of `docs/paper/icdm22-think.pdf` (= the IEEE pages 849-853); `[A854]` = page 854.

## 1. Sharpe per arm, new graph next to old graph

Mean ± std over 10 seeds. "Old" = pre-fix graph (73 edges), exact eq. 14 (`*_eq14`). The three `none` arms use no graph, so their old and new numbers are identical, which is also a determinism check. The old graph had no EH arms.

**Level inputs (the faithful v1)**

| arm | leak-free, g2 | leak-free, old | best-test, g2 | best-test, old |
|---|---|---|---|---|
| HH_hyper (THINK) | -0.681 ± 1.219 | -0.300 ± 1.429 | 2.513 ± 0.421 | 2.190 ± 0.533 |
| HH_clique | -0.542 ± 0.433 | -0.125 ± 0.502 | 1.817 ± 0.560 | 1.272 ± 0.778 |
| HH_none | 0.279 ± 0.350 | 0.279 ± 0.350 | 1.108 ± 0.261 | 1.108 ± 0.261 |
| EH_hyper (TConv + DHHAN) | -0.268 ± 1.121 | (none) | 2.100 ± 0.976 | (none) |
| EH_clique | 0.158 ± 0.841 | (none) | 1.392 ± 0.672 | (none) |
| EE_hyper | -0.154 ± 0.879 | 0.316 ± 0.812 | 2.134 ± 0.862 | 2.075 ± 0.470 |
| EE_clique | -0.531 ± 0.267 | -0.253 ± 0.504 | 1.199 ± 0.601 | 1.161 ± 0.572 |
| EE_none | 0.203 ± 0.583 | 0.203 ± 0.583 | 1.184 ± 0.599 | 1.184 ± 0.599 |

**Relative inputs (the v2 fix)**

| arm | leak-free, g2 | leak-free, old | best-test, g2 | best-test, old |
|---|---|---|---|---|
| HH_hyper (THINK) | 0.007 ± 0.781 | 0.566 ± 1.126 | 1.924 ± 0.000 (see note) | 2.042 ± 0.294 |
| HH_clique | 0.463 ± 0.458 | 0.357 ± 0.572 | 1.924 ± 0.000 (see note) | 1.979 ± 0.163 |
| HH_none | 0.484 ± 0.370 | 0.484 ± 0.370 | 0.948 ± 0.296 | 0.948 ± 0.296 |
| EH_hyper (TConv + DHHAN) | -0.297 ± 0.249 | (none) | 1.629 ± 0.230 | (none) |
| EH_clique | 0.023 ± 0.478 | (none) | 1.064 ± 0.585 | (none) |
| EE_hyper | -0.431 ± 0.280 | -0.128 ± 0.587 | 1.709 ± 0.313 | 1.803 ± 0.531 |
| EE_clique | 0.009 ± 0.513 | 0.072 ± 0.378 | 0.988 ± 0.257 | 1.285 ± 0.409 |
| EE_none | 0.061 ± 0.288 | 0.061 ± 0.288 | 1.032 ± 0.263 | 1.032 ± 0.263 |

- **Note on the relative-input HH best-test 1.924 ± 0.000.** All 20 runs (HH_hyper and HH_clique, 10 seeds each) report exactly 1.9244. That is the Sharpe of a **constant prediction** on the test set: I checked that `evaluate_all` on an all-zero prediction gives 1.9244251900, because ties are broken by stock index and the "portfolio" is then the first five stocks in the universe, held every day. So these arms reach a "best test epoch" of 1.92 by collapsing to a constant at an early epoch (epoch 1 to 3 in every seed; the per-epoch test Sharpe sits at 1.92 for several epochs in a row). The 1.92 is a property of the five stocks and the test year, not of the model. This is my inference from the identical value plus the constant-prediction check; the collapsed epoch's predictions were not saved.
- **No arm beats holding the market (0.752) leak-free**, in either variant. The best leak-free arms are HH_none 0.484 and HH_clique 0.463 (relative) and HH_none 0.279 (level), all with seed std 0.35 to 0.46.
- **Leak-free, only one arm that uses relations exceeds the random-5 line (0.338): HH_clique with relative inputs (0.463).** With level inputs every relation-using arm is between -0.68 and +0.16.
- **Best-test protocol reproduces the paper's ordering**, as before: level inputs, THINK 2.513 > pairwise 1.817 > none 1.108. Paired Wilcoxon on best-test (raw, uncorrected, not a Holm family, diagnostic only): HH_hyper - HH_none +1.405 (p 0.002, 10/10 seeds); HH_hyper - HH_clique +0.697 (p 0.049, 8/10). Same shape in Euclidean (EE_hyper - EE_none +0.950, p 0.037). This ordering is what selection on the test year produces: with shuffled labels the best-test Sharpe is the same as with real labels (C1, old graph: 2.23 vs 2.19; recheck on g2 is queued as `POC_sectors_C1_g2`).

## 2. The paper's comparisons

### 2a. THINK (HH_hyper) vs TConv + DHHAN (EH_hyper)

- **Paper:** Table II (p852), NYSE Sharpe: THINK 1.18 ± 4e-3, TCONV + DHHAN 1.14 ± 7e-3 (NDCG 0.86 vs 0.81); both marked significant vs the baselines. Sec. V.A "Impact of Hyperbolic Temporal Convolution" (p852): replacing hyperbolic temporal convolution by Euclidean gives "significant (p < 0.01) improvements when using hyperbolic learning for representing the time-series data". The paper reports the pair, and the p < 0.01 claim is about that comparison. The paper does not say which epoch was scored or what ± is (tracker U3).
- **Ours** (Holm family of 10 comparisons per variant, from `summarize`):

| variant | leak-free diff HH - EH | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|
| level, hyperedges | -0.414 (-0.681 vs -0.268) | 0.1934 | 1.0000 | [-0.678, +0.029] | NO EVIDENCE |
| level, pairwise edges (HH_clique vs EH_clique) | -0.700 (-0.542 vs +0.158) | 0.0020 | 0.0195 | [-1.379, +0.246] | NO EVIDENCE |
| relative, hyperedges | +0.305 (+0.007 vs -0.297) | 0.3223 | 0.8262 | [-0.235, +0.826] | NO EVIDENCE |
| relative, pairwise edges | +0.439 (+0.463 vs +0.023) | 0.1055 | 0.7383 | [-0.250, +1.082] | NO EVIDENCE |

- **Plain reading:** leak-free there is **no evidence** that THINK beats TConv + DHHAN. The sign flips with the input variant (level: hyperbolic temporal conv nominally worse; relative: nominally better), which is what noise looks like. The one small raw p-value (0.002, level, pairwise arms) is in the wrong direction for the paper and does not survive Holm as a STRONG or SEED-ROBUST result (0.0195 > 0.01).
- **Best-test epoch** (paper-style, raw p, not corrected): level 2.513 vs 2.100 (diff +0.413, p 0.375, THINK higher in 6/10 seeds); relative 1.924 vs 1.629 (diff +0.295, p 0.016, 7/10), where the 1.924 is the constant-prediction artefact described above. So even under the favourable protocol the level-input comparison is not distinguishable, and the relative-input "win" is an artefact.
- The paper's difference between the two arms is 0.04 Sharpe (1.18 vs 1.14). Our seed std is 0.4 to 1.2 leak-free, and one 237-day Sharpe has a standard error near 1, so a 0.04 difference is far below what this test can resolve; the result says only that we cannot confirm or refute it at this scale.

### 2b. Hyperedges vs pairwise vs none

- **Paper:** Sec. V.C (p853), Fig. 3a: decomposing hyperedges into pairwise edges lowers NYSE Sharpe monotonically, from about 1.18 (no decomposition) to about 0.95 (everything decomposed, "THINK degenerates to hyperbolic graph attention"); Fig. 3b: removing hub hyperedges lowers it further, and "all hyperedges removed ... essentially degenerates THINK to a temporal model", about 0.88 (Fig. 3b axis read from the plot; the values are approximate). The paper has no "none" arm in Table II; the "none" numbers come from Fig. 3b. The paper's decomposition is partial (by hyperedge size), ours is full (every hyperedge to all pairs).
- **Ours** (Holm p from `summarize`, family of 10):

| variant | comparison | leak-free diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|---|
| level | HH: hyperedges vs pairwise | -0.140 | 0.4316 | 1.0000 | [-1.557, +0.560] | NO EVIDENCE |
| level | HH: hyperedges vs none | -0.961 | 0.0840 | 0.7559 | [-2.226, -0.390] | NO EVIDENCE |
| level | EE: hyperedges vs pairwise | +0.377 | 0.4922 | 1.0000 | [-0.623, +1.164] | NO EVIDENCE |
| level | EE: hyperedges vs none | -0.357 | 0.3750 | 1.0000 | [-1.558, +0.448] | NO EVIDENCE |
| level | EH: hyperedges vs pairwise | -0.426 | 0.2754 | 1.0000 | [-1.638, +0.234] | NO EVIDENCE |
| relative | HH: hyperedges vs pairwise | -0.456 | 0.0488 | 0.3906 | [-1.579, +0.562] | NO EVIDENCE |
| relative | HH: hyperedges vs none | -0.477 | 0.1602 | 0.7383 | [-1.971, +0.542] | NO EVIDENCE |
| relative | EE: hyperedges vs pairwise | -0.439 | 0.0137 | 0.1367 | [-1.624, +0.485] | NO EVIDENCE |
| relative | EE: hyperedges vs none | -0.492 | 0.0195 | 0.1758 | [-1.980, +0.723] | NO EVIDENCE |
| relative | EH: hyperedges vs pairwise | -0.321 | 0.2754 | 0.8262 | [-1.369, +0.623] | NO EVIDENCE |

  EH has no `none` arm. Two bootstrap CIs in the level family exclude 0 (HH hyperedges vs none [-2.226, -0.390], shown above; and hyperbolic vs Euclidean with hyperedges, HH_hyper vs EE_hyper, [-1.212, -0.228]) but in the direction against the paper, and the Holm-corrected Wilcoxon does not support them, so the verdict is NO EVIDENCE.
- **Plain reading:** leak-free, adding relations never helps. In 9 of the 10 comparisons the point estimate for hyperedges is below the alternative, and in the relative variant it is below in every one (-0.32 to -0.49). None is significant after Holm correction. This is not the paper's ordering (hyperedges > pairwise > none), and the direction is not established either: "relations hurt" is also NO EVIDENCE.
- **Best-test epoch** reproduces the paper's ordering in the level variant (HH 2.513 > 1.817 > 1.108; the paired raw p-values are in section 1). In the relative variant hyperedges and pairwise tie at the artefact 1.924.

## 3. Fig 1-style diagnostics, leak-free epoch (10 seeds, 2017 test)

Random-ranking NDCG@5 = 0.5507 (200 draws). MSE ratio = test MSE ÷ MSE of predicting 0 for every stock (below 1.0 would beat predicting no change). IC = mean daily rank correlation between prediction and realised return. Hit = sign hit rate minus 50 (percentage points). Spread = within-day std of predictions ÷ std of returns. Mean ± std over seeds.

**Level inputs (`POC_sectors_g2`)**

| arm | MSE ÷ zero-MSE (best-test / leak-free) | NDCG@5 (best-test / leak-free) | IC | hit rate - 50 (pp) | spread |
|---|---|---|---|---|---|
| HH_hyper | 1.006 / 1.012 ± 0.024 | 0.556 / 0.540 ± 0.007 | -0.010 ± 0.014 | -0.26 ± 1.48 | 0.015 ± 0.023 |
| EH_hyper | 1.047 / 1.043 ± 0.054 | 0.556 / 0.543 ± 0.007 | -0.001 ± 0.023 | +0.26 ± 1.49 | 0.112 ± 0.113 |
| EE_hyper | 1.028 / 1.067 ± 0.086 | 0.556 / 0.544 ± 0.007 | -0.001 ± 0.017 | -0.15 ± 1.49 | 0.107 ± 0.190 |

**Relative inputs (`POC_sectors_rel_g2`)**

| arm | MSE ÷ zero-MSE (best-test / leak-free) | NDCG@5 (best-test / leak-free) | IC | hit rate - 50 (pp) | spread |
|---|---|---|---|---|---|
| HH_hyper | 1.000 / 1.000 ± 0.001 | 0.550 / 0.551 ± 0.004 | 0.000 ± 0.011 | +1.24 ± 0.93 | < 0.00001 |
| EH_hyper | 1.011 / 1.014 ± 0.018 | 0.554 / 0.547 ± 0.003 | -0.004 ± 0.010 | -0.31 ± 1.52 | 0.002 ± 0.002 |
| EE_hyper | 1.023 / 1.009 ± 0.008 | 0.556 / 0.546 ± 0.002 | -0.001 ± 0.004 | +0.63 ± 1.39 | 0.003 ± 0.007 |

- **No arm predicts returns.** Every MSE ratio is at or above 1.000 (worse than or equal to predicting no change), NDCG@5 at the leak-free epoch is at or below random (0.540 to 0.551 vs 0.5507), IC is within ±0.010 of zero (seed std 0.004 to 0.023), and hit rates are within about 1.5 points of a coin flip. Same picture as the old graph (POC_PRESENTATION §4b).
- **Predictions are collapsed.** Level THINK: spread 1.5% of the real spread (old graph 1.4%). EH and EE keep about 11%, which is noise, not signal (their MSE is worse, 1.04 to 1.07). With relative inputs every arm collapses further (spread <= 0.3%), THINK to numerically zero: MSE ratio 1.000, IC undefined on days with identical predictions (the IC average uses the remaining days).
- **Relative-input THINK hit rate +1.24 pp is not skill.** A model that predicts the same sign for every stock scores whatever share of stock-days had that sign (2017 was an up year), so the hit rate rises without any ranking information; IC is 0.000.
- THINK, EH and EE are indistinguishable on every diagnostic: the differences are smaller than the seed std.

## 4. Does the corrected graph change any conclusion from the old graph?

**No verdict changed, and no headline conclusion changed.**

- **Verdict words.** Old graph: all 6 comparisons NO EVIDENCE in both variants. New graph: all 10 comparisons NO EVIDENCE in both variants (20 of 20). The families differ in size (6 vs 10), so Holm penalties are not identical, but no comparison is close to Holm 0.01 except the level pairwise arm HH vs EH (raw 0.002, Holm 0.0195), which is a new comparison (no EH on the old graph) and goes against the paper.
- **Old vs new, same seeds, per arm.** Leak-free Sharpe of graph-using arms moved by -0.56 to +0.11 (level: HH_hyper -0.38, HH_clique -0.42, EE_hyper -0.47, EE_clique -0.28; relative: HH_hyper -0.56, HH_clique +0.11, EE_hyper -0.30, EE_clique -0.06). Paired Wilcoxon old vs new: all p >= 0.109 except level HH_clique leak-free (p 0.049) and best-test (p 0.039), and relative EE_clique best-test (p 0.020). Those three are raw p-values out of 16 tests (8 graph-using arms × 2 measures), so they are not evidence of a change.
- **Direction, not significance, of three comparisons did change** (all still NO EVIDENCE):
  - Relative, HH hyperedges vs pairwise: +0.209 (old) to -0.456 (new).
  - Relative, HH hyperedges vs none: +0.081 to -0.477.
  - Level, HH hyperedges vs none: -0.579 to -0.961.
  With the corrected graph, relations are nominally negative in every leak-free comparison; before, the sign varied. This is a nominal shift inside the noise (CIs all span 0 or sit against the paper).
- **What is the same.** Nothing beats holding the market (0.752) leak-free. Best-test ordering hyperedges > pairwise > none still reproduces in the level variant and is stronger on the new graph (2.513 / 1.817 / 1.108, was 2.190 / 1.272 / 1.108). Models do not predict returns (MSE ratio >= 1.0, IC ~ 0, NDCG at random). Predictions are collapsed. Hyperbolic vs Euclidean is not established either way.
- **New on this graph.** The relative-input HH arms collapse to a constant prediction at their best test epoch (the 1.924 artefact, section 1); the old relative-input HH arms did not (2.042 ± 0.294, 1.979 ± 0.163). This is a new finding, not a change in a verdict: the best-test number in the relative variant was uninformative on the corrected graph.
- **Caveats.** 309 stocks, one test year (a 237-day Sharpe has standard error near 1), no tuning, 30 epochs. Tuning (step b) and A10, C1, G2, G12 (step c) on g2 are still queued and can change the picture; this note covers only the untuned step (a).

## 5. Unreadable or unknown

Nothing on the pages cited was unreadable. The Fig. 3a and 3b values quoted in 2b are read off the plot and are approximate. The paper does not state the epoch-selection rule, k in top-k, R_f, the meaning of ± (tracker U3, U7).
