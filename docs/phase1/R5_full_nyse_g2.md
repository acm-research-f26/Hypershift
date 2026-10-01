# R5 and the full-NYSE arms of A5 and A9: `R5_g2`

*2026-10-01. Full NYSE (1737 stocks, 4350 hyperedges, corrected graph), exact eq. 14, paper protocol (`norm: paper`, 100 epochs, level inputs, lr 1e-3, alpha 1, `batch_days 8`), train 2013-15, val 2016, test 2017 (237 days). Script: `scripts/r5_g2_analysis.py` (`CUDA_VISIBLE_DEVICES=-1`, recomputes everything from `metrics.json`, `history.jsonl`, `test_*.npy`). Figure: `docs/figures/r5_g2.png`.*

- **Leak-free** = the epoch with the best validation (2016) Sharpe, scored on 2017. **Best-test** = max over epochs of the 2017 Sharpe (the paper-style upper bound; it selects on the test year, see C1 in the tracker). Sharpe is ours: top-5, mean/std x sqrt(252), R_f = 0. The paper writes top-k, no annualization, R_f (p852 Sec. IV-B); k and R_f are unstated (U7, PA4), so our numbers are not like-for-like with its 1.18.
- Citations: `pNNN` = page of `docs/paper/icdm22-think.pdf`; `[A854]` = page 854.
- **Where the runs came from.** HH (THINK) 25 seeds: laptop RTX 3050 (`phase1_gpu_3.sh`). EH 25 seeds (Kaggle s2), EE 10 and HE 10 seeds (Kaggle s1, merged earlier): Kaggle T4. All 70 runs: `attn_score eq14`, `norm paper`, `epochs_run 100`, 1737 nodes, 4350 edges, identical test masks. Kaggle and laptop seeds are valid seeds, not bit-copies, so "seed k" of HH and of EH is not a matched pair in any strong sense; the seed pairing in the Wilcoxon below is nominal (a Mann-Whitney check is in Table 3 and agrees).
- **Merge of s2** (this task): 34 complete runs copied (EH seeds 0, 2-24 and STHGCN full NYSE seeds 0-9 into `R8_baselines_g2`), 0 conflicts, 1 LOCAL-PARTIAL (`R5_g2/EH/seed_1`, from an earlier laptop worker, no `metrics.json`, last write 09:48, five hours stale, no live EH job), 10 KAGGLE-INCOMPLETE (all RSR_I, failed). I moved the partial to `results/_partials/R5_g2/EH/seed_1` and re-merged; Kaggle's EH seed 1 landed (EH 25/25). `results/_partials` is git-ignored with `results/`.

## 1. Sharpe per arm (mean ± std over seeds)

| arm | seeds | source | leak-free | best-test | val Sharpe at the selected epoch |
|---|---|---|---|---|---|
| HH (THINK) | 25 | laptop | **0.089 ± 0.300** | **2.115 ± 0.386** | 2.884 ± 0.186 |
| EH (TConv+DHHAN, the paper's Euclidean arm) | 25 | Kaggle | 0.383 ± 0.758 | 2.158 ± 0.462 | 2.471 ± 0.237 |
| EE | 10 | Kaggle | 0.237 ± 0.752 | 2.361 ± 0.325 | 2.385 ± 0.287 |
| HE | 10 | Kaggle | 0.193 ± 0.449 | 2.533 ± 0.620 | 2.834 ± 0.292 |

**Baselines on the same 237 test days and masks:** hold all 1737 stocks **1.531** (matches HANDOFF's 1.53); random top-5 re-drawn daily 0.856 ± 0.918 (100 draws); a **fixed** random 5-stock set held all year 0.955 ± 0.887 (2000 draws, 5-95% range [-0.48, 2.42]); constant (all-tied) prediction 0.368. The fixed-random line is the right yardstick for near-constant predictors (C1).

- Validation Sharpe at the chosen epoch is 2.4-2.9 and the test Sharpe at that epoch is 0.09-0.38. Selecting on 2016 does not transfer to 2017.
- **Leak-free, no arm beats holding the market (1.53):** HH 0/25 seeds, EH 3/25, EE 0/10, HE 0/10. Against the fixed random 5-stock mean (0.96): HH 0/25, EH 6/25, EE 2/10, HE 0/10.
- **Best-test beats hold-all in 23/25 (HH), 23/25 (EH), 10/10 (EE), 10/10 (HE).** But C1 showed shuffled labels give the same best-test Sharpe as real labels (small scale, 2.50 vs 2.46), so this is selection on the test year, not skill.
- Unannualized (our daily series, divide by sqrt(252) = 15.87): leak-free HH 0.006, EH 0.024; best-test HH 0.133, EH 0.136. Whether the paper annualized is not stated (U7); a daily Sharpe of 1.18 would be 18.7 annualized and the market's daily Sharpe here is 0.096, so the paper's 1.18 is more plausibly annualized, but that is my inference, not stated in the paper.

## 2. Diagnostics at the leak-free epoch (reusing `poc_figures.py` logic)

| arm | NDCG@5 (ours) | NDCG@5 at best-test epoch | NDCG_sthan (authors' evaluator) | MSE / zero-MSE | IC | hit rate (pp over 50) | pred spread / actual | runs with spread < 1e-5 | best-test = constant Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| HH | 0.5616 ± 0.0013 | 0.5680 | 0.758 ± 0.040 | 1.001 ± 0.003 | +0.0037 ± 0.0039 | +0.95 ± 2.48 | 0.001 ± 0.004 | 9/25 | 0/25 |
| EH | 0.5634 ± 0.0028 | 0.5686 | 0.796 ± 0.083 | 1.007 ± 0.016 | 0.0000 ± 0.0068 | -0.03 ± 2.57 | 0.011 ± 0.028 | 9/25 | 0/25 |
| EE | 0.5635 ± 0.0033 | 0.5689 | 0.774 ± 0.071 | 1.006 ± 0.008 | +0.0020 ± 0.0036 | +0.04 ± 2.60 | 0.015 ± 0.020 | 2/10 | 0/10 |
| HE | 0.5622 ± 0.0011 | 0.5689 | 0.735 ± 0.059 | 1.001 ± 0.002 | -0.0007 ± 0.0041 | +2.12 ± 1.60 | 0.003 ± 0.008 | 0/10 | 0/10 |

- Random NDCG@5 is **0.5639** (100 random rankings). Leak-free NDCG is 0.5616-0.5635: at or slightly below random. Even the best-test epoch (0.568-0.569) is only +0.004 above random.
- MSE is 1.00-1.007 times predicting 0, IC about 0 (|IC| < 0.004), hit rate within noise of 50%. The models predict almost nothing; the spread is 0.1-1.5% of the actual cross-sectional spread. All four arms are indistinguishable on every diagnostic.
- The authors' evaluator (`ndcg_sthan`, last day only, index sets) gives 0.74-0.80 for these no-skill models; the paper's 0.86 (p852 Table II) is in that range, which is consistent with the NDCG-bug finding (E3), not with skill.
- Constant-prediction check: the all-tied Sharpe is 0.368. No run's best-test equals it (0/70), so the 1.924-type artefact of the small-scale relative runs does not occur here. Near-constant (spread < 1e-5) predictions at the leak-free epoch occur in 9/25 HH, 9/25 EH, 2/10 EE and 0/10 HE runs; such runs hold a handful of stocks (6-20 distinct top-5 sets over 237 days in HH), so their Sharpe reflects which stocks they picked.

## 3. The paper's comparison: THINK (HH) vs TConv+DHHAN (EH), and the rest

Paper: THINK 1.18 ± 4e-3 vs TCONV+DHHAN 1.14 ± 7e-3 (p852 Table II; Sec. V.A, p < 0.01). Ours: Wilcoxon (two-sided, paired by seed number), Holm across the family of 5 below, 95% stationary block bootstrap (mean block 10, 5000 resamples) of the Sharpe difference of the seed-averaged daily series at the leak-free epoch. HH-EH uses 25 seeds; every comparison with EE or HE uses seeds 0-9.

**Table 3. Leak-free**

| comparison | n | diff | Wilcoxon p | Holm p | bootstrap CI | a wins | Mann-Whitney p | verdict |
|---|---|---|---|---|---|---|---|---|
| **HH vs EH (the paper's)** | 25 | **-0.294** | 0.182 | 0.909 | [-1.087, +0.412] | 11/25 | 0.535 | NO EVIDENCE |
| HH vs EE | 10 | -0.021 | 1.000 | 1.000 | [-1.156, +0.996] | 4/10 | 1.000 | NO EVIDENCE |
| HH vs HE | 10 | +0.023 | 1.000 | 1.000 | [-0.316, +0.417] | 6/10 | 0.850 | NO EVIDENCE |
| EH vs EE | 10 | +0.091 | 0.557 | 1.000 | [-0.403, +0.524] | 3/10 | 0.623 | NO EVIDENCE |
| HE vs EE | 10 | -0.044 | 0.922 | 1.000 | [-1.268, +0.942] | 5/10 | 0.850 | NO EVIDENCE |

**Table 4. Best-test (raw p, diagnostic only; no CI because `test_daily.npy` holds only the leak-free epoch)**

| comparison | n | diff | Wilcoxon p | a wins |
|---|---|---|---|---|
| HH vs EH | 25 | -0.043 | 0.474 | 12/25 |
| HH vs EE | 10 | -0.258 | 0.160 | 3/10 |
| HH vs HE | 10 | -0.430 | 0.160 | 4/10 |
| EH vs EE | 10 | -0.275 | 0.193 | 4/10 |
| HE vs EE | 10 | +0.172 | 0.695 | 5/10 |

- **THINK is not better than TConv+DHHAN.** Leak-free the point estimate is -0.29 (THINK lower), not significant (Holm 0.91, CI spans zero). Best-test it is -0.04 (p 0.47, 12/25 seeds): a coin flip, with EH nominally higher. On seeds 0-9 only: leak-free 0.216 vs 0.328, best-test 2.103 vs 2.086.
- The paper's gap is 0.04. Our seed std is 0.3 (HH) to 0.76 (EH) leak-free and 0.39-0.46 best-test, so even 25 seeds cannot resolve a 0.04 gap; the result is "cannot confirm or refute at this resolution", with no hint of the paper's direction under leak-free selection.
- No geometry contrast is significant: hyperbolic vs Euclidean attention (EH vs EE), hyperbolic vs Euclidean temporal conv (HH vs HE, HH vs EH), all NO EVIDENCE with signs that disagree across contrasts.
- Seed-ensemble Sharpes (average the daily returns of all seeds): HH 0.087, EH 0.452 (25 seeds); on seeds 0-9 HH 0.224, EH 0.393, EE 0.335, HE 0.183. No ensemble beats 1.53.

## 4. Old-graph full-NYSE numbers

| | seeds | leak-free | best-test | per-seed val vs test rho |
|---|---|---|---|---|
| THINK, old graph (`E1_main/THINK_paperProtocol`) | 5 | 0.082 ± 0.465 | 2.301 ± 0.268 | 0.60 |
| EH, old graph (`R_paperProtocol/EH`) | 5 | 0.724 ± 0.638 | 1.636 ± 0.349 | 0.63 |
| EE, old graph (`R_paperProtocol/EE`) | 5 | 0.126 ± 0.738 | 1.956 ± 0.156 | 0.58 |
| **THINK, g2 (this run)** | 25 | **0.089 ± 0.300** | **2.115 ± 0.386** | 0.53 |

- **Contradicts the task's numbers:** the task and the tracker/HANDOFF quote old THINK as best-test 2.40 and leak-free -0.05. Those are the **first 4 seeds** (HANDOFF "THINK (4 seeds)": 2.4025 and -0.055, recomputed). The folder now holds 5 seeds, and with the fifth the numbers are 2.301 and 0.082. I use the 5-seed values. The tracker's R5 text says "5 seeds" with 2.40, which is the 4-seed figure.
- New vs old THINK (25 vs 5 seeds, Mann-Whitney): leak-free 0.089 vs 0.082 (p 0.67), best-test 2.115 vs 2.301 (p 0.39). **The graph correction (312 to 4350 hyperedges) did not change THINK's full-NYSE numbers.**
- Old EH was 0.72 leak-free (5 seeds); with 25 seeds on g2 it is 0.38 ± 0.76, so the old EH advantage was seed noise (seed std 0.64-0.76).

**Validation vs test per epoch (THINK, 25 seeds, 100 epochs each, Spearman over epochs).** Per-seed rho = **+0.53 ± 0.08** (range 0.28-0.67); rho of the seed-mean curves +0.66. EH +0.44 ± 0.18, EE +0.63, HE +0.61. **This differs from the tracker's -0.63 (small scale, old graph).** Here both curves drift down with training (mean val Sharpe 0.67 and mean test -0.15 across epochs, right panel of the figure), so the positive rank correlation mostly reflects a shared trend; it does not make the validation-chosen epoch good (leak-free 0.09 vs best-test 2.12). Median selected epoch: HH 52, EH 15, EE 28, HE 52; median best-test epoch 14-58. I did not detrend, so I cannot say how much of rho is the trend.

## 5. Does the paper's headline reproduce?

Paper: THINK ~1.18 > TCONV+DHHAN ~1.14 > baselines (p852 Table II).

- **(a) Best-test selection (the paper-style upper bound): no for the ordering, yes only in the trivial sense for the level.** THINK 2.115 is above 1.18 and above the market (1.53), but EH is numerically higher (2.158, diff -0.04, p 0.47), and EE (2.36) and HE (2.53) are higher still. The paper's THINK > EH is not seen, and C1 shows this number is selection, not skill. The level is not comparable anyway (our Sharpe formula, PA4).
- **(b) Leak-free selection: no.** THINK 0.089 ± 0.300, below hold-all (1.53, 0/25 seeds above), below a fixed random 5-stock set (0.96), with NDCG at random (0.5616 vs 0.5639), IC 0.004, MSE 1.001 times zero. THINK vs EH: -0.29, NO EVIDENCE.
- Net: **the headline is not reproduced** on the corrected graph at full NYSE with 25 seeds. The paper's absolute Sharpe is not comparable (PA4/U7), but the ordering THINK > EH and the existence of any skill are not supported under either selection rule. Remaining untested explanations are in Phase 2 (selection protocol, grouping, trading rule).

## 6. Caveats

- Mixed platforms: HH on the laptop, the others on Kaggle T4. Configs are identical (checked above), numerics are not bit-identical. HH vs EH therefore compares across hardware; a hardware effect cannot be excluded, but a 0.3 Sharpe shift from kernels alone would be surprising given seed std 0.3-0.76.
- EE and HE have 10 seeds (planned), so contrasts with them have p-value floor 0.002 and less power.
- Holm family: the 5 comparisons in Table 3 (my choice; the grid's `FAMILIES` was not used).
- ± is std over seeds; the paper's ± is undefined (U3), so its 4e-3 is not comparable.
- R8 (RSR-I, STHGCN vs THINK) is not covered here: RSR-I full-NYSE runs are pending.
