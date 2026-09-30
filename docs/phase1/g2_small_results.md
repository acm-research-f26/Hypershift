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
