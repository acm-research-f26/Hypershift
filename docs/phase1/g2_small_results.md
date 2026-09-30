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
- **Caveats.** 309 stocks, one test year (a 237-day Sharpe has standard error near 1), no tuning, 30 epochs. This section covers only the untuned step (a); tuning (step b) is in the "Tuned (step b)" section at the end (no verdict changed); A10, C1, G2, G12 (step c) on g2 are still queued.

## 5. Unreadable or unknown

Nothing on the pages cited was unreadable. The Fig. 3a and 3b values quoted in 2b are read off the plot and are approximate. The paper does not state the epoch-selection rule, k in top-k, R_f, the meaning of ± (tracker U3, U7).

## Tuned (step b): equal-budget tuning, relative inputs, 309 stocks, corrected graph

*2026-09-30. Exp `results/POC_sectors_rel_tuned_g2/`. Relative inputs only. Tuning grid per geometry: lr {5e-4, 1e-3, 3e-3} x alpha {1, 10, 30, 100}, 3 seeds (0-2), structure `hyper`, chosen on mean validation (2016) Sharpe, never test (`tuned.json`). Final 10-seed reruns of all 8 arms use each geometry's choice: **HH lr 5e-4, alpha 100; EH lr 1e-3, alpha 30; EE lr 5e-4, alpha 1**. `summary.md` was regenerated with `scripts/poc_sectors.py summarize --variant rel_tuned_g2` (force-added to git, `results/` is ignored). Every number below was recomputed from `metrics.json` and `history.jsonl` with a separate script (means, stds and Wilcoxon p for all 8 arms match `summary.md` to 3 decimals). Same definitions, baselines (market 0.752, random-5 0.338, momentum -0.276) and citations as above. "Untuned" = `POC_sectors_rel_g2` (lr 1e-3, alpha 1).*

### T1. Sharpe per arm, tuned vs untuned (mean ± std, 10 seeds; paired Wilcoxon by seed, raw p, Holm over the 8 leak-free tests)

| arm | leak-free, tuned | leak-free, untuned | diff | p (Holm) | best-test, tuned | best-test, untuned |
|---|---|---|---|---|---|---|
| HH_hyper (THINK) | +0.069 ± 0.643 | +0.007 ± 0.781 | +0.062 | 0.77 (1.00) | 1.318 ± 0.479 | 1.924 ± 0.000 (artefact) |
| HH_clique | +0.022 ± 0.945 | +0.463 ± 0.458 | -0.441 | 0.28 (1.00) | 1.562 ± 0.491 | 1.924 ± 0.000 (artefact) |
| HH_none | +0.520 ± 0.526 | +0.484 ± 0.370 | +0.036 | 1.00 (1.00) | 1.293 ± 0.262 | 0.948 ± 0.296 |
| EH_hyper (TConv + DHHAN) | +0.172 ± 0.831 | -0.297 ± 0.249 | +0.469 | 0.13 (1.00) | 1.686 ± 0.281 | 1.629 ± 0.230 |
| EH_clique | +0.342 ± 0.369 | +0.023 ± 0.478 | +0.318 | 0.19 (1.00) | 1.365 ± 0.457 | 1.064 ± 0.585 |
| EE_hyper | -0.243 ± 0.491 | -0.431 ± 0.280 | +0.188 | 0.23 (1.00) | 1.506 ± 0.186 | 1.709 ± 0.313 |
| EE_clique | +0.008 ± 0.639 | +0.009 ± 0.513 | -0.000 | 0.70 (1.00) | 1.150 ± 0.540 | 0.988 ± 0.257 |
| EE_none | +0.215 ± 0.469 | +0.061 ± 0.288 | +0.154 | 0.19 (1.00) | 1.247 ± 0.421 | 1.032 ± 0.263 |

- **Tuning changes no leak-free arm significantly** (all raw p >= 0.13, Holm 1.00). Six of eight arms move up, two down (HH_clique -0.44, with a tuned std of 0.95). Nothing beats holding the market (0.752) leak-free: the best tuned arms are HH_none 0.520, EH_clique 0.342 and EE_none 0.215 (random-5 is 0.338).
- Seed std of the leak-free Sharpe is 0.37 to 0.95. Tuned mean **validation** Sharpe is 1.66 to 2.06 for every arm (HH_hyper 1.872, EH_clique 2.060, EE_none 1.657), against test Sharpe of -0.24 to +0.52: what the search optimises does not transfer to 2017 (same disconnect as the untuned run, tracker "Validation and test years disagree").
- **The relative-input constant-prediction artefact largely disappears in best-test under tuning.** Best-test equal to 1.9244 (= the Sharpe of an all-tied prediction, i.e. the first five stocks held every day; recomputed here from `metrics.json`): untuned 10/10 HH_hyper, 10/10 HH_clique, and also 3/10 EH_hyper and 1/10 EH_clique (the earlier note counted only HH); tuned 1/10 HH_hyper, 1/10 HH_clique, 1/10 EH_hyper, 2/10 EH_clique, none elsewhere. The drop of the HH_hyper best-test from 1.924 to 1.318 (paired p 0.008) is therefore the artefact leaving, not the model getting worse. The tuned best-test values (1.15 to 1.69) are the less contaminated paper-protocol upper bounds for this setup (the HH_hyper, HH_clique, EH_hyper and EH_clique means still include their 1 to 2 runs at 1.9244).
- **Constant or near-constant predictions at the leak-free epoch remain common.** Spread (within-day std of predictions / std of returns) below 1e-5 at the validation-selected epoch: tuned HH_hyper 4/10 runs, EH_hyper 5/10, EE_hyper 1/10 (untuned: HH_hyper 10/10, EH_hyper 2/10, EE_hyper 2/10). Such a run holds the same five stocks every day, so its Sharpe is set by which five stocks it picked.

### T2. The paper's comparison: THINK (HH_hyper) vs TConv + DHHAN (EH_hyper)

Paper: Table II (p852), NYSE Sharpe 1.18 vs 1.14; Sec. V.A (p852) claims p < 0.01 for hyperbolic temporal convolution. Ours, tuned, relative inputs (Holm family of 10, from `summarize`):

| comparison | leak-free diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|
| HH_hyper vs EH_hyper | -0.102 (+0.069 vs +0.172) | 0.9219 | 1.0000 | [-0.553, +0.721] | NO EVIDENCE |
| HH_clique vs EH_clique | -0.320 (+0.022 vs +0.342) | 0.3750 | 1.0000 | [-1.361, +0.055] | NO EVIDENCE |

- Untuned it was +0.305 (Holm 0.83) and +0.439 (Holm 0.74); tuned it flips to nominally negative in both. THINK is above EH in 5 of 10 seeds (hyperedges) and 3 of 10 (pairwise). **No evidence that THINK beats TConv + DHHAN; the sign of the point estimate flips with tuning, which is what noise looks like.**
- Best-test (paper-style, raw p, diagnostic only): HH_hyper 1.318 vs EH_hyper 1.686, diff -0.368 (p 0.084, THINK higher in 2/10 seeds); pairwise +0.197 (p 0.32). Tuned, even the favourable protocol does not show THINK ahead of EH. The untuned "+0.295, p 0.016" was the artefact (HH at 1.924).
- Resolution: the paper's gap is 0.04 (1.18 vs 1.14); a single 237-day Sharpe has a standard error near 1 and our seed std is 0.6 to 0.8. This test cannot resolve a 0.04 gap.

### T3. Hyperedges vs pairwise vs none (tuned, relative; Holm family of 10)

| comparison | leak-free diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|
| HH: hyperedges vs pairwise | +0.047 | 0.9219 | 1.0000 | [-0.977, +1.182] | NO EVIDENCE |
| HH: hyperedges vs none | -0.451 | 0.1055 | 0.9492 | [-1.597, +0.261] | NO EVIDENCE |
| EH: hyperedges vs pairwise | -0.170 | 0.4922 | 1.0000 | [-1.882, +0.477] | NO EVIDENCE |
| EE: hyperedges vs pairwise | -0.251 | 0.1602 | 1.0000 | [-1.313, +0.745] | NO EVIDENCE |
| EE: hyperedges vs none | -0.459 | 0.0488 | 0.4883 | [-2.014, +0.770] | NO EVIDENCE |
| HH_hyper vs EE_hyper (hyperbolic vs Euclidean, hyperedges) | +0.312 | 0.3223 | 1.0000 | [-0.401, +0.930] | NO EVIDENCE |
| EH_hyper vs EE_hyper (hyperbolic vs Euclidean attention) | +0.415 | 0.2754 | 1.0000 | [-0.447, +0.795] | NO EVIDENCE |
| interaction (hyperedge gain in hyperbolic - in Euclidean) | +0.298 | 0.6250 | 1.0000 | [-0.442, +1.223] | NO EVIDENCE |

(EH has no `none` arm. The two HH-vs-EH rows are in T2.) All 10 comparisons are NO EVIDENCE.

- **Plain reading:** leak-free, hyperedges are not better than pairwise (HH +0.05, EH -0.17, EE -0.25) and are nominally below "no relations" (HH -0.45, EE -0.46). Tuning did not restore the paper's ordering (p853 Sec. V.C, Fig. 3a/3b). The direction is not established either (all NO EVIDENCE). Compared with untuned, the HH hyperedges-vs-pairwise contrast moved from -0.456 to +0.047, another sign flip inside the noise.
- Best-test (raw p, diagnostic): hyperedges minus none: HH +0.024 (p 0.85, THINK higher in 3/10), EE +0.258 (p 0.037, 9/10). Hyperedges minus pairwise: HH -0.245 (p 0.13), EH +0.321 (p 0.074), EE +0.355 (p 0.084). The paper's ordering shows only in the Euclidean arms and not significantly; for HH the artefact-free best-test is 1.318 (hyper), 1.562 (clique), 1.293 (none), i.e. the same.

### T4. Diagnostics at the leak-free epoch (hyper arms, tuned; 10 seeds, 2017 test)

Random-ranking NDCG@5 = 0.5507 (200 draws, same logic as `scripts/poc_figures.py`, `diag` and `random_ndcg`). MSE ratio = test MSE / MSE of predicting 0. IC = mean daily rank correlation (days with all-tied predictions are dropped by `nanmean`). Hit = sign hit rate minus 50 (pp). Spread = within-day prediction std / return std.

| arm | MSE / zero-MSE (best-test / leak-free) | NDCG@5 (best-test / leak-free) | IC | hit - 50 (pp) | spread |
|---|---|---|---|---|---|
| HH_hyper | 1.001 / 1.001 ± 0.001 | 0.554 / 0.549 ± 0.004 | -0.002 ± 0.010 | +0.89 ± 1.22 | 0.00045 ± 0.00064 |
| EH_hyper | 1.005 / 1.007 ± 0.010 | 0.554 / 0.548 ± 0.003 | -0.002 ± 0.004 | -0.32 ± 1.50 | 0.00084 ± 0.00149 |
| EE_hyper | 1.015 / 1.011 ± 0.018 | 0.554 / 0.547 ± 0.002 | -0.007 ± 0.011 | +0.00 ± 1.54 | 0.00149 ± 0.00347 |

- **Still no predictive skill after tuning.** Every MSE ratio is >= 1.000, leak-free NDCG@5 (0.547 to 0.549) is below random (0.5507), IC is within ±0.007 of zero, hit rates are within 1.5 pp of a coin flip. Untuned values were the same (section 3). Predictions are collapsed to 0.05% to 0.15% of the return spread; HH's +0.89 pp hit rate is the "same sign for every stock in an up year" effect, not ranking.
- Tuning did not move the diagnostics: THINK, EH and EE remain indistinguishable, and the differences are below the seed std.

### T5. Tuning surface (3 seeds, mean validation Sharpe at each run's best-validation epoch; ± is seed std)

Rows lr, columns alpha, `hyper` structure. **Bold** = chosen.

**HH**

| lr \ alpha | 1 | 10 | 30 | 100 | lr marginal |
|---|---|---|---|---|---|
| 5e-4 | 1.495 ± 0.29 | 1.657 ± 0.56 | 1.703 ± 0.34 | **1.791 ± 0.22** | 1.661 |
| 1e-3 | 1.067 ± 0.61 | 1.380 ± 0.44 | 1.785 ± 0.20 | 1.679 ± 0.16 | 1.478 |
| 3e-3 | 1.426 ± 0.80 | 1.344 ± 0.25 | 1.185 ± 0.40 | 1.552 ± 0.25 | 1.377 |
| alpha marginal | 1.329 | 1.460 | 1.557 | 1.674 | |

**EH**

| lr \ alpha | 1 | 10 | 30 | 100 | lr marginal |
|---|---|---|---|---|---|
| 5e-4 | 1.839 ± 0.20 | 1.688 ± 0.13 | 2.115 ± 0.56 | 2.278 ± 0.42 | 1.980 |
| 1e-3 | 1.805 ± 0.21 | 1.723 ± 0.05 | **2.402 ± 0.48** | 2.047 ± 0.37 | 1.994 |
| 3e-3 | 1.565 ± 0.23 | 1.584 ± 0.17 | 1.667 ± 0.23 | 1.810 ± 0.09 | 1.657 |
| alpha marginal | 1.736 | 1.665 | 2.061 | 2.045 | |

**EE**

| lr \ alpha | 1 | 10 | 30 | 100 | lr marginal |
|---|---|---|---|---|---|
| 5e-4 | **1.873 ± 0.48** | 1.773 ± 0.07 | 1.795 ± 0.22 | 1.337 ± 0.31 | 1.694 |
| 1e-3 | 1.818 ± 0.22 | 1.560 ± 0.15 | 1.678 ± 0.09 | 1.580 ± 0.30 | 1.659 |
| 3e-3 | 1.666 ± 0.06 | 1.564 ± 0.29 | 1.763 ± 0.36 | 1.538 ± 0.28 | 1.633 |
| alpha marginal | 1.786 | 1.633 | 1.746 | 1.485 | |

- **HH: the optimum may lie outside the grid, but this grid cannot show it.** The chosen cell is at the top edge in alpha (100) and the bottom edge in lr (5e-4). Both marginals are monotone toward that corner (alpha 1.33, 1.46, 1.56, 1.67; lr from high to low 1.38, 1.48, 1.66), so extending to alpha > 100 and lr < 5e-4 might raise validation Sharpe further. Against that, the winner (1.791) beats the runner-up (lr 1e-3, alpha 30, 1.785) by 0.006, and the within-cell seed std is 0.16 to 0.80 (n = 3, so the std of a cell mean is about 0.6 times that), so the top cells are indistinguishable and the choice is partly noise (winner's curse over 12 noisy cells). The gain from the whole search is small on validation and absent on test (leak-free +0.062, p 0.77). **Conclusion: HH's edge-of-grid choice is a real limitation of this search; the surface is flat within noise near the top, and the validation gain did not transfer to test. We did not extend the grid.**
- **EH: interior optimum.** Best cell lr 1e-3, alpha 30 (2.402), a local peak on both axes (alpha 100 gives 2.047 at the same lr; lr 3e-3 gives 1.667). Not an edge problem. The neighbouring lr 5e-4, alpha 100 (2.278) is within seed noise (± 0.4 to 0.6).
- **EE: the choice sits in the low corner of both axes** (lr 5e-4, alpha 1, 1.873); the alpha marginal is highest at 1 and the lr marginal highest at 5e-4. The EE optimum might lie at alpha < 1 and lr < 5e-4, which this grid does not cover. The whole EE surface is flat (1.34 to 1.87, most cells 1.56 to 1.80). So each of HH and EE was chosen at a boundary of a grid that extends only one way for it (HH: alpha widened upward to 100; EE: nothing below alpha 1). This weakens any HH-vs-EE reading, but that verdict is NO EVIDENCE regardless (T3).
- EH has the highest validation Sharpe (up to 2.40) but EH_hyper test Sharpe stays at 0.17. Validation Sharpe over 252 days with seed std 0.2 to 0.6 overstates what a configuration will do out of sample.

### T6. Does tuning change any conclusion?

**No verdict changed.** All 10 comparisons are NO EVIDENCE, as in the untuned g2 run (and in the old-graph tuned run, `POC_sectors_rel_tuned_eq14`). Nothing beats the market leak-free. Models still do not predict returns. Three point estimates changed: HH vs EH flipped from +0.31 to -0.10; HH hyperedges vs pairwise flipped from -0.46 to +0.05; and the HH best-test artefact (1.924) is gone. Caveats: 309 stocks, one test year, 30 epochs, 3 seeds per tuning cell, grids not extended for HH (alpha edge) or EE (low corner). A10, C1, G2 and G12 on g2 (step c) are separate.



## Step (c): A10, C1 and G12 on g2 (309 stocks, corrected graph, eq. 14, 30 epochs, no tuning)

*2026-09-30. Exps `POC_sectors_A10_g2`, `POC_sectors_rel_A10_g2`, `POC_sectors_C1_g2`, `POC_sectors_G12_hub{35,24,16,10}_g2` (all read-only here). Every number was recomputed from `metrics.json`, `test_daily.npy` and `test_{pred,gt,mask}.npy` with a separate script; the config of each run confirms the switch (`attn_dist` off vs mult, `shuffle_train_labels` True, `drop_hub_degree`). Same definitions as above: leak-free = epoch with best 2016 validation Sharpe, scored on 2017; best-test = max over epochs of the 2017 Sharpe (selects on the test year). Comparisons are paired by seed: Wilcoxon two-sided, raw p (not Holm-corrected: targeted diagnostics, not a family); bootstrap CI = stationary block bootstrap of the Sharpe difference of the seed-averaged daily return series (same routine as `summarize`). **The bootstrap CI exists only for leak-free**, because `test_daily.npy` holds the leak-free epoch only; best-test rows have no CI. **With 5 seeds the smallest possible two-sided Wilcoxon p is 2/2^5 = 0.0625**, so no 5-seed comparison here can reach p < 0.05, whatever the effect. With 10 seeds the minimum is 0.002.*

### A10. Attention without the distance term (`attn_dist=off`), HH_hyper, 10 seeds

Paper: the claim is that hyperbolic distance-guided message propagation is what lets THINK capture a node's impact (p853, text after Fig. 3a/3b, Sec. V.B/V.C). If the distance term mattered, removing it should hurt.

| inputs | leak-free, no distance | leak-free, with distance | diff (no dist - dist) | Wilcoxon p | no-dist better in | 95% bootstrap CI |
|---|---|---|---|---|---|---|
| level | -0.421 ± 1.093 | -0.681 ± 1.219 | +0.261 | 0.160 | 8/10 | [-0.01, +0.50] |
| relative | -0.033 ± 0.786 | +0.007 ± 0.781 | -0.040 | 0.695 | 5/10 | [-0.79, +0.26] |

Best-test: level 2.482 ± 0.527 (no dist) vs 2.513 ± 0.421 (diff -0.032, p 0.49, no-dist higher in 3/10). Relative: 1.924 ± 0.000 in both; all 20 runs (10 with, 10 without the distance term) sit at the constant-prediction artefact 1.9244 described in section 1, so the relative best-test carries no information.

- **The distance term has no detectable effect**, in either variant; p is at least 0.16. The level-input point estimate is nominally in favour of *removing* the distance term (+0.26, 8/10 seeds, CI just touches 0), the relative one is about 0 (-0.04). On the old graph the signs were the other way round (level -0.29, relative +0.14, both p > 0.4): the sign flips, as noise does. Both are far inside the seed std (0.8 to 1.2).
- Caveat: leak-free Sharpe of these arms is near or below zero (predictions are collapsed, section 3), so "no effect on a number that measures nothing" is the honest reading. A10 cannot say the distance term is useless in a model that works.

### C1. Shuffled training labels (leakage null), HH_hyper and EH_hyper, level inputs, 5 seeds

The training returns are permuted across stocks within each training day (`shuffle_train_labels` in `loop.py`), so no learnable signal remains. Validation and test labels are real. Compared with the real-label runs of the same seeds 0-4 in `POC_sectors_g2`.

| arm | protocol | shuffled labels | real labels | diff (shuf - real) | Wilcoxon p | shuf higher in |
|---|---|---|---|---|---|---|
| HH_hyper | leak-free | -0.695 ± 1.110 | -0.497 ± 1.526 | -0.198 | 0.625 | 2/5 |
| HH_hyper | best-test | **2.499 ± 0.592** | **2.463 ± 0.484** | +0.035 | 0.8125 | 3/5 |
| EH_hyper | leak-free | -0.255 ± 0.756 | -0.092 ± 1.231 | -0.163 | 1.000 | 2/5 |
| EH_hyper | best-test | **2.363 ± 0.537** | **2.256 ± 0.729** | +0.107 | 0.8125 | 2/5 |

Bootstrap CI of the leak-free Sharpe difference: HH [-0.41, +0.28], EH [-0.30, +0.59] (both include 0). IC (mean daily rank correlation between prediction and 2017 return, from `test_pred/gt/mask`, leak-free epoch): HH shuffled -0.011 ± 0.012 vs real -0.011 ± 0.014; EH shuffled -0.004 ± 0.016 vs real +0.001 ± 0.022. Validation Sharpe at the selected epoch: HH 1.692 ± 0.139 shuffled vs 1.864 ± 0.278 real; EH 2.091 ± 0.415 vs 1.949 ± 0.276.

- **Yes, the old-graph C1 result still holds on g2.** Old graph (HH, 5 seeds): best-test 2.23 with shuffled labels vs 2.19 with real labels. Now: 2.50 vs 2.46 (HH) and 2.36 vs 2.26 (EH). Shuffled labels give the same best-test Sharpe as real labels in both arms, with differences of +0.04 and +0.11 and p >= 0.81.
- **Reading.** A model trained on noise labels reaches a best-test Sharpe of about 2.4 to 2.5 (the paper reports 1.18 for THINK on NYSE, p852 Table II, on its own Sharpe formula and universe). So the best-test number of the paper protocol (the max over epochs of the test Sharpe) is produced by picking the luckiest epoch on the test year, and cannot be read as evidence that the model learned anything. This is also why the best-test ordering "hyperedges > pairwise > none" (section 1) cannot be taken as support for THINK. Validation Sharpe is also high (1.7 to 2.1) with noise labels, the same disconnect as in the tuning section (2016 Sharpe high, 2017 near zero).
- Shuffled labels do not lower the leak-free Sharpe or the IC either (differences within noise, IC at zero for both label types), consistent with section 3 (real-label models carry no ranking signal on 2017). This does not show that the pipeline is leak-free (that is C2/C4 and the tests); it shows the *metric* cannot tell signal from noise at this scale.

### G12. Hub removal (paper Fig. 3b), HH_hyper and EH_hyper, level inputs, 5 seeds

Hub removal drops every hyperedge that touches a node of degree >= t (levels as derived in `scripts/queues/phase1_g2_small.sh`). Edges left and the share of the 309 stocks still in at least one hyperedge (`covered_frac` in `metrics.json`; identical for both arms): full graph 558 edges (100%), t = 35: 426 (85.1%), t = 24: 194 (71.5%), t = 16: 40 (11.0%), t = 10: 16 (7.8%). Reference "none" arms (10 seeds, section 1): HH_none 0.279 leak-free / 1.108 best-test, EE_none 0.203 / 1.184 (EH_none would equal EE_none).

Sharpe, mean ± std over 5 seeds. "Full" = `POC_sectors_g2` seeds 0-4 (the same seeds).

| graph | HH leak-free | EH leak-free | HH best-test | EH best-test |
|---|---|---|---|---|
| full (558 edges) | -0.497 ± 1.526 | -0.092 ± 1.231 | 2.463 ± 0.484 | 2.256 ± 0.729 |
| hub 35 (426) | -0.537 ± 0.675 | +0.063 ± 0.612 | 2.005 ± 0.325 | 1.428 ± 0.836 |
| hub 24 (194) | -0.050 ± 0.499 | +0.251 ± 0.477 | 1.505 ± 0.461 | 1.198 ± 0.794 |
| hub 16 (40) | +0.924 ± 0.887 | +1.006 ± 0.878 | 1.976 ± 0.322 | 1.344 ± 0.989 |
| hub 10 (16) | +0.284 ± 0.631 | +0.843 ± 0.810 | 1.587 ± 0.299 | 1.152 ± 0.867 |

Figure: `docs/figures/g12_g2.png` (both protocols, mean ± std).

**Paired against the full graph** (Wilcoxon raw p; bootstrap CI for leak-free only):

| arm | level | leak-free diff (hub - full) | p | CI | best-test diff | p | full higher in (best-test) |
|---|---|---|---|---|---|---|---|
| HH | hub 35 | -0.040 | 0.81 | [-0.36, +0.82] | -0.459 | 0.44 | 3/5 |
| HH | hub 24 | +0.447 | 0.63 | [-0.27, +1.69] | -0.958 | 0.0625 | 5/5 |
| HH | hub 16 | +1.422 | 0.125 | [+0.34, +3.42] | -0.488 | 0.31 | 3/5 |
| HH | hub 10 | +0.781 | 0.63 | [-0.64, +3.00] | -0.876 | 0.0625 | 5/5 |
| EH | hub 35 | +0.155 | 0.81 | [-0.12, +1.57] | -0.828 | 0.31 | 4/5 |
| EH | hub 24 | +0.343 | 0.81 | [-0.01, +1.58] | -1.058 | 0.0625 | 5/5 |
| EH | hub 16 | +1.098 | 0.125 | [+0.53, +2.60] | -0.912 | 0.31 | 4/5 |
| EH | hub 10 | +0.935 | 0.19 | [+0.44, +2.46] | -1.104 | 0.31 | 4/5 |

("Full higher in" counts the seeds where the full graph beats the hub-dropped graph on best-test; 5/5 with p = 0.0625 is the strongest result possible with 5 seeds.)

**THINK (HH) minus Euclidean temporal (EH), paired by seed** (positive = the paper's direction):

| graph | leak-free diff | p | HH higher in | CI (leak-free) | best-test diff | p | HH higher in |
|---|---|---|---|---|---|---|---|
| full | -0.405 | 0.44 | 1/5 | [-0.65, +0.19] | +0.207 | 0.63 | 3/5 |
| hub 35 | -0.600 | 0.31 | 1/5 | [-1.34, -0.09] | +0.577 | 0.31 | 3/5 |
| hub 24 | -0.301 | 0.63 | 2/5 | [-0.89, +0.32] | +0.307 | 0.63 | 3/5 |
| hub 16 | -0.081 | 0.81 | 2/5 | [-0.84, +1.05] | +0.632 | 0.63 | 3/5 |
| hub 10 | -0.559 | 0.31 | 1/5 | [-2.05, +1.07] | +0.435 | 0.44 | 3/5 |

**Does Fig. 3b reproduce?** Paper (p853 Sec. V.C "Impact of Hyperbolic Learning", Fig. 3b): both THINK and its Euclidean variant lose Sharpe as hubs are removed and are worst when all hyperedges are gone; THINK stays above the Euclidean variant throughout (curve levels read off the plot, approximate: THINK from about 1.15 at node-degree threshold 31 to about 0.88 at 2; Euclidean roughly 1.0 to 0.83).

- **Leak-free protocol: no, the trend has the opposite sign.** Sharpe does not decline; it rises as hubs are removed (HH -0.50 to +0.92 at hub 16; EH -0.09 to +1.01), then dips at hub 10. Full vs hub 16: +1.42 (HH, CI [+0.34, +3.42], p 0.125) and +1.10 (EH, CI [+0.53, +2.60], p 0.125). The bootstrap CIs exclude 0, but the seed-level test cannot (p 0.125; 4/5 seeds agree), and the CI treats one 237-day test year as the only noise source. Hub 16 and 10 keep only 8 to 11% of stocks in any hyperedge, so the model is nearly a no-relations model there; the no-relations arms give 0.28 (HH_none) and 0.20 (EE_none) on 10 seeds, and seed std here is 0.5 to 1.5. This is not evidence that hubs hurt. THINK is *below* EH at all five graph levels (1 or 2 of 5 seeds above; diffs -0.08 to -0.60, all p >= 0.31), the wrong direction for the paper; the one CI excluding 0 (hub 35, [-1.34, -0.09]) is also against the paper and is not supported by the seed-level p (0.31).
- **Best-test protocol: the direction, not the shape.** Sharpe falls from the full graph (HH 2.46, EH 2.26) to hub 10 (HH 1.59, EH 1.15). Full beats hub 24 and hub 10 in 5/5 seeds for HH (p = 0.0625, the minimum with 5 seeds) and hub 24 in 5/5 for EH. It is **not monotone**: HH goes 2.46, 2.01, 1.51, **1.98**, 1.59 (hub 16 jumps back up by 0.47); EH goes 2.26, 1.43, 1.20, **1.34**, 1.15. THINK is above EH at every level (3/5 seeds each, diffs +0.21 to +0.63) but none is significant (p >= 0.31), and 3/5 is what a coin flip gives. Hub 10 and hub 16 are close to the no-relations reference (HH_none best-test 1.108, 10 seeds), so a decline toward "all hyperedges removed" is what one expects from a model losing its graph. C1 shows that the best-test value with *shuffled* labels is 2.4 to 2.5, so this curve cannot be read as the model using the hubs' information.
- **Verdict: Fig. 3b's monotone decline and THINK > EH are not reproduced under either protocol at a level that can be called significant.** Leak-free, the trend has the opposite sign and THINK < EH. Best-test, the direction matches the paper (decline, THINK > EH) but is non-monotone and not significant, and it is a protocol that the shuffled-label control shows to be uninformative. The paper's x-axis (node degree 31 to 2, max hyperedge size 500) is not the graph we have (max node degree 35 here, 114 on full NYSE; PA11 and `docs/phase1/fig3_degree_reconcile.md`), so the levels are not aligned even in principle.
- Caveats: 5 seeds (min p = 0.0625), one test year, 309 stocks, no tuning, 30 epochs. Hubs are removed from a graph whose hyperedges are mostly pairs (545 of 558), so "hub removal" here mostly deletes Wikidata pair edges around high-degree stocks.

## Step (c): G2, hyperedge decomposition (paper Fig. 3a), HH_hyper and EH_hyper, level inputs, 5 seeds

*2026-09-30. Exps `POC_sectors_G2_decomp_large30_g2`, `POC_sectors_G2_decomp_large15_g2`, `POC_sectors_G2_decomp_small20_g2` (read-only here; the older `POC_sectors_G2_decomp_large30` is the superseded old-graph run and is not used). Same setup, definitions and statistics as the G12 section above (309 stocks, corrected 558-edge graph, eq. 14, 30 epochs, no tuning, level inputs; leak-free = best 2016 validation epoch scored on 2017, best-test = max over epochs of the 2017 Sharpe; paired Wilcoxon two-sided, raw p; block-bootstrap CI for leak-free only). **With 5 seeds the smallest possible p is 0.0625.** Every number was recomputed from `metrics.json`, `history.jsonl` and `test_daily.npy`. The leak-free and best-test values re-derived from the per-epoch history match `metrics.json` in all 50 runs (5 seeds x 2 arms x 5 levels). The run configs confirm the switch (`decompose_mode`, `decompose_size`, `micro_batch_days`).*

Levels, ordered by how much is decomposed (edges left in the model, from `num_edges` in `metrics.json`; all 309 stocks stay covered at every level):

| level | switch | edges | note |
|---|---|---|---|
| full | none | 558 | 545 pairs + 13 larger industry edges (sizes 3 to 47) |
| small20 | `small_first`, S = 20 | 1326 | splits the seven size 13 to 20 industry edges into pairs (6 big edges left) |
| large30 | `large_first`, S = 30 | 3488 | splits every edge with more than 30 nodes (9 big edges left) |
| large15 | `large_first`, S = 15 | 4719 | 3 big edges left |
| all pairs | `*_clique` arms | 4897 | every hyperedge to all pairs, deduplicated; the queue comment expected 5067 pairs, `num_edges` says 4897 |

`large30` and `large15` ran with `micro_batch_days=1` (identical gradients, less memory). The arms stop early at different epochs (`epochs_run` 11 to 30), which matters for best-test, a maximum over however many epochs were run.

### G2a. Sharpe per level (mean ± std, 5 seeds; "full" = `POC_sectors_g2` seeds 0-4, "all pairs" = the `*_clique` arms, seeds 0-4)

| level | HH leak-free | EH leak-free | HH best-test | EH best-test |
|---|---|---|---|---|
| full (558) | -0.497 ± 1.526 | -0.092 ± 1.231 | 2.463 ± 0.484 | 2.256 ± 0.729 |
| small20 (1326) | -0.644 ± 0.581 | +0.266 ± 0.635 | 1.798 ± 0.141 | 1.774 ± 0.962 |
| large30 (3488) | -0.035 ± 0.592 | +0.242 ± 1.075 | 1.033 ± 0.470 | 1.381 ± 0.875 |
| large15 (4719) | -0.456 ± 0.038 | +0.205 ± 0.704 | 1.245 ± 0.557 | 1.952 ± 0.271 |
| all pairs (4897) | -0.424 ± 0.487 | +0.546 ± 0.916 | 1.638 ± 0.351 | 1.673 ± 0.473 |

(All pairs on all 10 seeds, section 1: HH -0.542 / 1.817, EH +0.158 / 1.392.) Figure: `docs/figures/g2_g2.png` (both protocols, mean ± std).

### G2b. Each decomposed level minus the full graph (paired by seed)

| arm | level | leak-free diff | p | CI (leak-free) | best-test diff | p | full higher in (best-test) |
|---|---|---|---|---|---|---|---|
| HH | small20 | -0.147 | 0.63 | [-0.56, +0.78] | -0.665 | 0.0625 | 5/5 |
| HH | large30 | +0.463 | 0.63 | [-0.37, +1.69] | -1.430 | 0.0625 | 5/5 |
| HH | large15 | +0.042 | 0.63 | [-0.65, +1.62] | -1.219 | 0.0625 | 5/5 |
| HH | all pairs | +0.073 | 0.63 | [-0.74, +1.46] | -0.825 | 0.125 | 4/5 |
| EH | small20 | +0.358 | 0.31 | [-0.27, +1.79] | -0.482 | 0.0625 | 5/5 |
| EH | large30 | +0.334 | 0.125 | [-0.32, +1.31] | -0.875 | 0.0625 | 5/5 |
| EH | large15 | +0.297 | 0.44 | [-0.23, +1.55] | -0.304 | 0.44 | 4/5 |
| EH | all pairs | +0.638 | 0.44 | [+0.07, +2.00] | -0.583 | 0.125 | 4/5 |

### G2c. THINK (HH) minus Euclidean temporal (EH) at each level (positive = the paper's direction)

| level | leak-free diff | p | HH higher in | CI (leak-free) | best-test diff | p | HH higher in |
|---|---|---|---|---|---|---|---|
| full | -0.405 | 0.44 | 1/5 | [-0.65, +0.19] | +0.207 | 0.63 | 3/5 |
| small20 | -0.910 | 0.31 | 1/5 | [-1.69, -0.05] | +0.025 | 0.81 | 2/5 |
| large30 | -0.277 | 0.63 | 2/5 | [-0.53, +0.41] | -0.349 | 0.81 | 3/5 |
| large15 | -0.661 | 0.125 | 1/5 | [-0.92, +0.09] | -0.707 | 0.125 | 1/5 |
| all pairs | -0.970 | 0.0625 | 0/5 | [-1.61, -0.26] | -0.035 | 1.00 | 3/5 |

### Does Fig. 3a reproduce?

Paper (p853 Sec. V.C "Impact of Hypergraph Learning", Fig. 3a; values read off the plot, approximate): NYSE Sharpe falls steadily as hyperedges are decomposed into pairs, from about 1.18 (THINK) and 1.10 (Euclidean THINK) with no decomposition to about 0.95 and 0.92 when everything is decomposed; the THINK curve stays above the Euclidean curve at every level. The text: each hyperedge of degree n is decomposed into C(n,2) pairs "in increasing order of hyperedge degree", and "the worst performance is achieved when all hyperedges are decomposed". The x-axis (hyperedge degree 500, 15, 9, 5, 3) reads like a shrinking maximum remaining degree, which points to decomposing large edges first, while the text says increasing order. The paper does not settle which (INFERRED, not in paper: either reading is possible); we ran both (`large_first` 30 and 15, `small_first` 20). The paper's levels belong to full NYSE, whose largest edge has 500 nodes, so they cannot be matched on our 309-stock graph (max size 47).

- **Leak-free protocol: no.** There is no decline. HH is flat and noisy (-0.50, -0.64, -0.04, -0.46, -0.42 across the five levels, seed std 0.04 to 1.5) and every decomposed level is within noise of the full graph (diffs -0.15 to +0.46, all p = 0.63, all CIs include 0). EH goes the other way: it rises from -0.09 (full) to +0.21 to +0.55 at every decomposed level (diffs +0.30 to +0.64, p 0.125 to 0.44; the all-pairs CI [+0.07, +2.00] excludes 0 but the seed-level test does not, and the CI treats one 237-day test year as the only noise). THINK is below the Euclidean model at all five levels (-0.28 to -0.97, HH higher in 0 to 2 of 5 seeds), the wrong direction for the paper. The largest gap (all pairs, -0.97, HH lower in 5/5 seeds, p 0.0625, CI [-1.61, -0.26]) is the strongest result 5 seeds allow, and it is against the paper.
- **Best-test protocol: the direction shows, the shape does not.** Every decomposed level is below the full graph on average (HH 2.46 falling to 1.03 to 1.80, EH 2.26 falling to 1.38 to 1.95). The full graph beats the decomposed level in 5/5 seeds in 5 of the 8 arm-level pairs (p 0.0625, the 5-seed minimum); the other three are 4/5 (p 0.125 to 0.44). It is **not monotone**: ordered by edges left, HH goes 2.46, 1.80, 1.03, 1.25, 1.64 (the fully pairwise arm is above large30) and EH goes 2.26, 1.77, 1.38, 1.95, 1.67 (large15 rebounds to 1.95, the second-best level). Per seed, the best-test curve declines at every step in only 1 of 5 seeds for HH and 1 of 5 for EH. **THINK above Euclidean does not hold either:** HH minus EH is +0.21 and +0.03 at full and small20, and -0.35, -0.71, -0.04 at large30, large15 and all pairs; none is significant (p >= 0.125).
- **C1 caveat applies in full.** Best-test is a maximum over epochs of the test-year Sharpe. With shuffled training labels the full graph gives 2.50 (HH) and 2.36 (EH) (section C1), the same as the real-label 2.46 and 2.26. So the full graph's high best-test is not information, and the decline when the graph is decomposed cannot be read as "hyperedges carry information that pairs lose". There is also a confound in the best-test number: the arms stop at different epochs (HH mean epochs run: full 17.4, small20 20.2, large30 13.8, large15 18.8, all pairs 21.0), and a maximum over fewer epochs tends to be lower. I did not quantify this.
- **Verdict: Fig. 3a (monotone decline with decomposition, THINK above Euclidean at every level) is not reproduced under either protocol.** Leak-free shows no decline for THINK and a rise for EH, with THINK below EH throughout. Best-test shows a decline in direction (largest at the first decomposition steps) but not monotone, and THINK is above EH only at the two least decomposed levels, not significantly. This is the same picture as G12 (Fig. 3b). The paper's curves are smooth and nearly monotone with narrow error bands (25 runs, p853 Fig. 3 caption; band width read off the plot, roughly 0.02 to 0.03, approximate). Our seed std is 0.04 to 1.5, and a single 237-day Sharpe has a standard error near 1.
- Caveats: 5 seeds (min p 0.0625), one test year, 309 stocks, no tuning, 30 epochs, levels defined for our graph (edges of 3 to 47 nodes; 545 of 558 edges are already pairs), so decomposition here touches only 13 hyperedges, although the edge count grows from 558 to 4897. Relative inputs were not run for G2.
