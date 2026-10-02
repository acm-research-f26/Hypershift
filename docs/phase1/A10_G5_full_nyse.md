# A10 and G5 on full NYSE: `R5_g2/{HH_none, EE_none, THINK_nodist}`

*2026-10-01. Same setting as `docs/phase1/R5_full_nyse_g2.md` (full NYSE 1737 stocks, corrected graph, exact eq. 14, `norm: paper`, 100 epochs, level inputs, lr 1e-3, alpha 1, `batch_days 8`; train 2013-15, val 2016, test 2017, 237 days). Script: `scripts/a10_g5_analysis.py` (CPU, imports helpers from `scripts/r5_g2_analysis.py`; recomputes everything from `metrics.json`, `history.jsonl`, `test_*.npy`). Definitions of leak-free and best-test, the Sharpe formula and the caveats on comparing with the paper's numbers are those of the R5 doc (paper p852 Sec. IV-B; U7, PA4).*

## Merge (Kaggle s4)

- `kaggle/fetch.sh s4`: kernel `tomphamdustry/hypershift-run-s4` COMPLETE. Dry-run and real merge: **30 complete runs copied, 0 conflicts, 0 local partials, 0 Kaggle-incomplete, 0 `failed.json`** (the kernel log has no failure lines). `R5_g2/HH_none`, `EE_none`, `THINK_nodist`: seeds 0-9 each, all `epochs_run` 100.
- Config check (seed 3, `metrics.json["config"]`): `HH_none` differs from `THINK_paperProtocol` only in `label` and `structure` (hyper to none); `EE_none` differs from `EE` only in `label` and `structure`; `THINK_nodist` differs from `THINK_paperProtocol` only in `label` and `attn_dist` (mult to off). Platform: Kaggle T4; the THINK reference arm ran on the laptop.

## 1. Sharpe and diagnostics per arm (mean ± std over seeds, leak-free epoch unless stated)

Baselines on the same test days: hold all 1737 stocks **1.531**; fixed random 5-stock set 0.955 ± 0.887; all-tied constant prediction 0.368; random NDCG@5 0.5639.

| arm | n | leak-free | best-test | val Sharpe at selected epoch | NDCG@5 | NDCG_sthan | IC | hit rate (pp over 50) | MSE / zero-MSE | spread < 1e-5 | best-test == constant |
|---|---|---|---|---|---|---|---|---|---|---|---|
| THINK (`THINK_paperProtocol`) | 25 | 0.089 ± 0.300 | 2.115 ± 0.386 | 2.884 | 0.5616 | 0.758 | +0.0037 | +0.95 | 1.001 | 9/25 | 0/25 |
| `THINK_nodist` (A10) | 10 | -0.023 ± 0.353 | 2.068 ± 0.423 | 2.909 | 0.5614 | 0.744 | +0.0030 | +2.13 | 1.000 | 4/10 | 0/10 |
| EE | 10 | 0.237 ± 0.752 | 2.361 ± 0.325 | 2.385 | 0.5635 | 0.774 | +0.0020 | +0.04 | 1.006 | 2/10 | 0/10 |
| `EE_none` (G5) | 10 | 0.846 ± 0.645 | 1.881 ± 0.487 | 2.458 | 0.5646 | 0.832 | +0.0133 | -0.31 | 1.008 | 2/10 | 0/10 |
| `HH_none` (G5) | 10 | 0.830 ± 0.727 | 2.035 ± 0.263 | 2.178 | 0.5657 | 0.824 | +0.0108 | +0.01 | 1.040 | 0/10 | 0/10 |

THINK on seeds 0-9 only: leak-free 0.216, best-test 2.103.

- Leak-free beats hold-all (1.531) in 0/25 (THINK), 0/10 (`THINK_nodist`), 0/10 (EE), 1/10 (`EE_none`), 3/10 (`HH_none`) seeds. Seed-ensemble Sharpe (average of the seeds' daily returns): THINK 0.087, `THINK_nodist` -0.017, EE 0.335, `EE_none` 1.097, `HH_none` 1.074; none reaches 1.531.
- The no-relation arms are nominally the best leak-free (0.83-0.85), but that is about the fixed random 5-stock mean (0.955) and well below hold-all. NDCG@5 (0.5646-0.5657) is at random (0.5639) and IC is about 0.01 (largest of all arms, still tiny). "No hyperedges is nominally better" does not mean a model that works.
- Best-test is 1.88-2.12 for every arm, above hold-all in 23/25, 9/10, 10/10, 8/10, 10/10 seeds. Best-test selects on the test year and shuffled labels reach the same level (C1), so it is not evidence of skill.
- The constant-prediction artefact (best-test equal to the all-tied Sharpe 0.368) occurs in 0/55 runs here.

## 2. Contrasts (family of 3, Holm)

Paired by seed number on seeds 0-9 (nominal pairing: the THINK arm ran on the laptop, the others on Kaggle). 95% CI is a stationary block bootstrap (mean block 10, 5000 resamples) of the Sharpe difference of the seed-averaged daily series. Mann-Whitney is unpaired. "with" = the full model (relations / distance term), "without" = the ablation. Diff = with minus without.

**Leak-free**

| comparison | n | with | without | diff | Wilcoxon p | Holm p | bootstrap CI | with wins | Mann-Whitney p (seeds 0-9) | Mann-Whitney, all 25 THINK vs 10: diff, p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G5: THINK vs `HH_none` | 10 | 0.216 | 0.830 | -0.614 | 0.049 | 0.111 | [-2.02, +0.28] | 3/10 | 0.054 | -0.741, 0.008 | NO EVIDENCE |
| G5: EE vs `EE_none` | 10 | 0.237 | 0.846 | -0.609 | 0.037 | 0.111 | [-2.21, +0.53] | 1/10 | 0.089 | -0.609, 0.089 | NO EVIDENCE |
| A10: THINK vs `THINK_nodist` | 10 | 0.216 | -0.023 | +0.240 | 0.193 | 0.193 | [-0.20, +0.63] | 7/10 | 0.186 | +0.112, 0.411 | NO EVIDENCE |

**Best-test (raw p, diagnostic only; no CI because `test_daily.npy` holds the leak-free epoch only)**

| comparison | n | with | without | diff | Wilcoxon p | with wins | Mann-Whitney p | Mann-Whitney, all 25 vs 10: diff, p |
|---|---|---|---|---|---|---|---|---|
| G5: THINK vs `HH_none` | 10 | 2.103 | 2.035 | +0.069 | 0.432 | 8/10 | 0.473 | +0.080, 0.476 |
| G5: EE vs `EE_none` | 10 | 2.361 | 1.881 | +0.480 | 0.065 | 8/10 | 0.014 | +0.480, 0.014 |
| A10: THINK vs `THINK_nodist` | 10 | 2.103 | 2.068 | +0.035 | 0.846 | 5/10 | 0.850 | +0.047, 0.841 |

## 3. Reading

- **G5 (no hyperedges), leak-free: relations nominally hurt, as at small scale; not significant after Holm.** Point estimates are -0.61 (THINK) and -0.61 (EE); paired Wilcoxon raw p 0.049 and 0.037, Holm 0.111 for both, both bootstrap CIs span zero, so the verdict word is NO EVIDENCE. The unpaired Mann-Whitney using all 25 THINK seeds gives diff -0.74 and raw p 0.008 for THINK vs `HH_none`; this test was not pre-specified and is not Holm-corrected, so I report it as supporting the direction (relations do not help leak-free), not as a verdict. Small scale had the same direction in all 4 leak-free contrasts (tracker G5; `docs/phase1/g2_small_results.md`: HH -0.96 level / -0.48 relative, EE -0.36 / -0.49, all NO EVIDENCE). The no-relation arms sit at the fixed random 5-stock mean (0.955), so this says relations add no leak-free value, not that no-relation models rank well.
- **G5, best-test: the paper's direction appears weakly.** THINK vs `HH_none` +0.07 (p 0.43, 8/10 seeds); EE vs `EE_none` +0.48 (Wilcoxon raw p 0.065, Mann-Whitney 0.014). At small scale the best-test gap was larger (HH 2.51 vs 1.11, paired p 0.002) and vanished with tuning (1.32 vs 1.29). Best-test selects on the test year (C1), so a gap here is not a reproduction of the paper's claim that relations help (Fig. 3b, p853).
- **A10 (distance term): no effect.** Leak-free +0.24 (p 0.19, CI [-0.20, +0.63], 7/10 seeds), best-test +0.04 (p 0.85, 5/10). Same conclusion as small scale ("no effect": level +0.26, p 0.16; relative -0.04, p 0.70; old graph -0.29 / +0.14). The leak-free sign is positive here and was positive in the small-scale level variant (+0.26) but not the relative one (-0.04); all are inside noise (seed std 0.30-0.35 here). NDCG@5, IC and spread of `THINK_nodist` match THINK's (0.5614 vs 0.5616; +0.003 vs +0.004; spread 0.000 vs 0.001): both make near-constant, near-zero predictions, so A10 cannot say the distance term is useless in a model that works.
- **Diagnostics.** NDCG@5 of all five arms (0.5614-0.5657) is within 0.003 of random (0.5639); MSE is 1.000-1.040 times predicting zero (`HH_none` is the worst, 1.040 ± 0.072); IC is below 0.014; hit rate is within 2.2 pp of 50 for every arm. The authors' evaluator `ndcg_sthan` gives 0.74-0.83 for these no-skill models, which is the range of the paper's 0.86 (p852 Table II), consistent with the NDCG finding (E3).

## 4. Caveats

- Mixed hardware: THINK reference arm on the laptop RTX 3050 (25 seeds), the new arms and EE on Kaggle T4 (10 seeds). "Paired by seed" is nominal, which is why Mann-Whitney is also reported.
- 10 seeds, one test year (a 237-day Sharpe has standard error near 1); the Holm family of 3 is my choice (the grid's `FAMILIES` was not used); the minimum Wilcoxon p at n = 10 is 0.002.
- The no-relation arms drop the attention layer's graph entirely (`structure: none`), so G5 changes more than the graph; the paper's hub-removal protocol is a separate ablation (G12).
- Untuned (paper protocol, like the main R5 arms). Tuning at small scale did not change any verdict.
