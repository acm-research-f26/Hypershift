# A. Scientific claims register (revision 2, 2026-10-22)

Revision 1 is preserved in git (commit `823bbec`). This revision adds an explicit **evidential status** for every claim, corrects the adjustment-leak claims, and replaces "leak-free" with information-timing labels (`LEAKAGE_AUDIT.md`).

## Evidential status codes

| Code | Meaning |
|---|---|
| **RECOMP** | The number was recomputed directly from saved outputs by `scripts/pub_verify.py`, `scripts/pubfig.py` or `scripts/pub_leak_mechanism.py`. This checks transcription and aggregation only. |
| **INDEP** | Reproduced by an implementation independent of the one that produced it. In this project this applies only in narrow senses, stated per row. **No claim has been reproduced by an outside party.** |
| **REPORT** | Supported only by an earlier project report and its scripts, not re-checked in this revision. |
| **SYNTH** | Supported by a limited synthetic experiment. |
| **MATH** | Follows algebraically from definitions. |
| **INTERP** | An interpretation, not a verified mechanism. |

**What "20 values recomputed, 19 match" means** (`CLAIMS_VERIFICATION.md`):
- The cited numbers equal what independent aggregation code computes from the saved per-run outputs.
- It does **not** validate the models, data, feature builders or evaluators that produced those outputs, and it is not a replication.
- The one mismatch is EH's planted-signal IC under coupled decay (0.090 cited in an earlier report vs 0.098 recomputed). It is unexplained and not used in the manuscript.

## Claims

| ID | Claim (manuscript wording, abbreviated) | Status | Key sources | Main limitation |
|---|---|---|---|---|
| T1 | Original-setup reproduction: THINK 0.089 vs EH 0.383 vs hold-all 1.531 (2017, validation-selected, **full-series normalisation**) | RECOMP (all three) | `results/R5_g2` | Inferred settings; one year; survivor universe; RSR adjustment convention unknown |
| T2 | Coupled weight decay collapses learning: THINK planted IC 0.074 → 0.174 (oracle 0.279) | RECOMP (THINK values); SYNTH (planted signal); mechanism probe = REPORT | `results/known_signal_analysis.json`, `results/f_f2_c1_*` | EH baseline mismatch with an earlier report (not used). Says nothing about THINK's actual settings |
| T3 | Corrected THINK vs EH indistinguishable (1.687 vs 1.436 train-period; 1.929 vs 1.146 full-series; 10 seeds) | RECOMP (means); test verdict = REPORT | `results/R5_f_*`, `F_r5f_results.md` | One year; seed SD 0.3–0.7 |
| T4 | 2017 Sharpe near 2 at the 88th–93rd percentile of random baskets | REPORT | `docs/phase1_5a/REPORT_2017.md` | Low power |
| T5 | Published-scale values appear only under test-informed selection, annualised Sharpe or a flawed NDCG; shuffled labels match the test-selected Sharpe | REPORT (protocol matrix is Kaggle-era output); INTERP regarding the paper | `docs/phase2/REPORT_CLOSURE.md` | **Does not identify the authors' procedure** |
| T6 | Out of sample 2018–2023: Sharpe about hold-all, IC near 0 | REPORT | `docs/phase1_5b`, `docs/phase1_5c` | 5 seeds; RSR-defined universe |
| T7 | RSR authors' code reproduces its own metrics; no validation-selected top-5 skill | REPORT; INDEP in the narrow sense that the authors' code is independent of ours | `docs/phase1_5/E_rsr_original.md` | Different metrics by design |
| T8 | Full-series normalisation leaks future scale (static-rank Sharpe 4.37) | REPORT (synthetic test plus real data) | `docs/phase1_5/A_evaluator.md` | — |
| L1 | Back-adjusted **levels** depend on later distributions and later prices; returns cancel only post-window factors | **MATH** | `MATH_ADJUSTMENT.md` | Assumes the backward multiplicative convention |
| L2 | Adjustment term as a score: L +0.019; L_path +0.094 [+0.086, +0.101]; L_div +0.010 (n.s.); TY −0.009 (n.s.) | RECOMP (L and the TY residual in pub_verify; all components in `pub_leak_mechanism.py` / `pubfig.py`) | `data/audit/yf_unadjusted/`, FB labels | Association, not causation; vendor- and fetch-date-specific; exploratory years |
| L2b | The TY-residual association (+0.031) is a mixture of ex-ante and ex-post information, **not** a "future-only" component | INTERP (corrected from revision 1) | `MATH_ADJUSTMENT.md` §4 | — |
| L3 | Correcting dollar volume changed two benchmark conclusions (residual − raw 0.014 → 0.009; volume features −0.009 → −0.003) | RECOMP (residual − raw); REPORT (rich − price) | `results/FBAUDIT_20261022/corrected` | Early-stopping rounds differ between refits; single seed |
| S1 | A characteristics-only learner matches the 20-feature learner on neutral rank IC (+0.026 vs +0.025 / +0.013) | RECOMP | `results/FBAUDIT_20261022/matched` | Legitimate characteristic predictability, **not** look-ahead |
| S2 | Residualised targets do not remove nonlinear/interaction characteristic predictability | MATH (span argument) + SYNTH (3 replicates) + REPORT (real placebo, A7) | `A12_SYNTHETIC.md`, `A7_STYLE_RANK_RR.md` | Real-data mechanism is INTERP |
| S3 | The neutral target adds about +0.011 for both feature sets | REPORT (A11 table) | A11 | INTERP of the cause |
| S4 | Residual-target gain mostly reproduced by ex-ante neutralisation of raw predictions | REPORT | A2, A2c | Single seed |
| M1 | Matched fitted-procedure comparison: full − characteristics −0.0017 / −0.0011, CIs about ±0.01 | RECOMP | A11 | Valid **only** as a comparison of fitted procedures |
| M2 | This LightGBM configuration did not exploit a planted feature with standalone IC 0.046 | SYNTH (one design; 3 replicates plus a 1-seed probe) | A12 | **Do not generalise** to GBMs, other features or markets |
| M3 | Encompassing and permutation controls failed synthetic validation | SYNTH | A12 | Specific implementations |
| M4 | Clark–West logic suggests a nested-model penalty; extending it to rank IC is unvalidated | INTERP | — | Requires theory or simulation |
| P1 | MDE about 0.014–0.019 (5 folds); prospective power about 0.07 at 0.006 | RECOMP (LRV curves in `pubfig.py`); REPORT (A1) | FB `daily.npz` | LRV varies about 2.8× across years |
| P2 | 77–89% of IC variance above the sampling floor | REPORT | A1 | — |
| P3 | Selection inflation (LOFO 0.0098 vs 0.0127; SPA vs reversal p 0.55) | REPORT | A3 | — |
| E1 | Turnover 36–77% of NAV per day; break-even 2–14 bp; buffer → 0.13 per day | REPORT | A3, A8 | Linear costs |
| E2 | No model book beats matched comparators; characteristics-only book +1.11 net Sharpe at 5 bp (post hoc) | REPORT | A8, A13 | No attribution; not investable; survivor universe |
| U1 | 43–58% symbol-match coverage of 2004–2012 members; not a bound | REPORT | A10 | Renames / reuse |

## Compromised claims (exclude, or cite only as cautionary examples)

- **C1:** FB "volume features hurt". Superseded by L3.
- **C2:** FB "residual target +0.0140". The contaminated-feature magnitude; superseded by L3 and S4.
- **C3:** Audit "stock-specific IC about +0.011". The metric fails the placebo test (S2).
- **C4:** BC review "model − placebo +0.002". An unmatched comparison; superseded by M1.
- **C5:** Paper-protocol (test-selected) Sharpe levels. Selection on the test year (T5).
- **C6:** Revision 1, L1: "future-only part … +0.031". The residual is not an identified future component (L2b).

## Not adequately tested

- **UT1:** incremental information of extra features beyond characteristics and reversal.
- **UT2:** other learner configurations.
- **UT3:** learned hyperedges on real data.
- **UT4:** investability.
- **UT5:** attribution of the characteristics-only books.
- **UT6:** any post-freeze data.
- **UT7:** the RSR panel's price-adjustment convention.
