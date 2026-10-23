# A. Scientific claims register

**How to read it.** Each row is a claim the manuscript may make, with the evidence and the wording it supports. "Verification" refers to `CLAIMS_VERIFICATION.md`:
- **R** = recomputed from saved outputs by `scripts/pub_verify.py` (MATCH);
- **D** = taken from a project document, and regenerable by the script named there, but not recomputed in this pass.

**Classes:**
- **REPRO** = independently reproduced within the project (more than one configuration, or a mechanism plus recomputation);
- **CONFIG** = supported within the tested configuration;
- **SUGG** = suggestive but unconfirmed;
- **COMP** = methodologically compromised;
- **UNTESTED** = not adequately tested.

**Status.** Every empirical number comes from data the project had already inspected. Nothing is confirmed on an independent holdout. The prospective test (`PROSPECTIVE_ADDENDUM.md`) has not been scored.

## Reproduction of THINK (Agarwal et al., ICDM 2022)

| ID | Publishable wording | Class | Evidence and files | Configuration / data / n | Effect [uncertainty] | Control | Verif. | Limitations / avoid |
|---|---|---|---|---|---|---|---|---|
| T1 | "In our inferred-settings reimplementation, the reported Sharpe advantage of THINK over its Euclidean-temporal ablation was not reproduced under validation-based epoch selection." | CONFIG | `docs/PHASE1_TRACKER.md` R5; `results/R5_g2/` | Full NYSE (1,737 stocks), test year 2017 (237 days), 25 seeds; original setup (coupled wd 5e-4, level inputs, full-series normalisation) | THINK 0.089 vs EH 0.383 (seed means; THINK − EH −0.29, CI [−1.09, +0.41]); hold-all 1.531 | Hold-all; EH arm; random baskets | R | One test year. Settings inferred: the paper gives no code or hyperparameters, and the authors' repository is empty. **Avoid** "THINK does not work" and "the authors made errors". |
| T2 | "Coupled L2 weight decay (5e-4 with Adam, lr 1e-3) prevented the stacked hyperbolic model from learning a planted signal. Removing it raised recovered IC from about 27% to 63–84% of the oracle." | REPRO | `docs/phase1_5/F_learnability.md`; `results/known_signal_*`, `results/f_*` | 309 stocks, real graph, planted signals at two strengths, 3–5 seeds per arm, two architectures | Planted 'high': THINK test IC 0.074 → 0.174 (oracle 0.279) | Oracle IC; no-graph arms; gradient-norm probe | R | The planted signal is synthetic, so this establishes an optimisation failure, not a real-data effect. It is consistent with the authors' earlier public code (wd 5e-4), but **avoid** claiming THINK itself used it. |
| T3 | "After correcting the optimisation and the input representation, THINK − EH remained statistically indistinguishable from zero (10 paired seeds), under both normalisations." | CONFIG | `docs/phase1_5/F_r5f_results.md`; `results/R5_f_{train,paper}/` | Full NYSE, 2017, 10 seeds | norm=train: 1.687 vs 1.436; norm=paper: 1.929 vs 1.146 (Wilcoxon, NO EVIDENCE) | EH, EE, RSR-I, STHGCN (the last three with 5 seeds) | R | Single year; seed SD 0.3–0.7. |
| T4 | "A 2017 test-year Sharpe near 2 is not distinguishable from random daily top-5 baskets in that year (88th–93rd percentile)." | CONFIG | `docs/phase1_5a/REPORT_2017.md` | 2017, 5 seeds, 10,000 random baskets | Hold-all 1.53; model 1.88–2.14; family p 0.12 | Random, beta-matched, industry-matched and label-permutation nulls | D | Low power: the seed-0 Sharpe CI is 0.28–3.48. **Avoid** "the model has no signal". |
| T5 | "Published-scale values in our protocol matrix appear only when the test year informs epoch selection, under an annualised Sharpe, or with an NDCG implementation that random predictors also pass. Shuffled training labels reach the same test-selected Sharpe as real labels." | CONFIG | `docs/phase2/REPORT_CLOSURE.md` (88-cell matrix); `docs/phase1/g2_small_results.md` (C1) | NYSE 2017; C1: 309 stocks, 5 seeds | C1 best-test 2.50 (shuffled) vs 2.46 (real); 43% of random models reach NDCG ≥ 0.86 under the buggy NDCG | Shuffled labels; random predictors | D | **We do not know the authors' selection rule** (the paper does not state it). Phrase as "can be reached by", never "was obtained by". |
| T6 | "Out of sample (2018–2023, frozen and with annual retraining), the corrected model's top-5 Sharpe was close to an equal-weight hold-all portfolio, with mean IC near zero." | CONFIG | `docs/phase1_5b/REPORT_POST2017.md`, `docs/phase1_5c/REPORT_WF.md` | Alpaca survivor-free bars, 1,647 stocks, 5 seeds | Frozen 0.43 vs hold-all 0.51; walk-forward 0.70 vs 0.69; IC 0.0002–0.003 | Four nulls, Holm | D | 5 seeds; one reimplementation. |
| T7 | "The RSR authors' public code reproduces its paper's own metrics. The same predictions show no ranking skill under a validation-selected top-5 Sharpe." | CONFIG | `docs/phase1_5/E_rsr_original.md`; Phase 2 W1 | NYSE 2017 | IRR 1.052 vs paper 1.06; leak-free Sharpe 0.45–0.93 vs hold-all 1.53 | Hold-all | D | Different metrics by design; not a claim about RSR in general. |
| T8 | "Full-series price normalisation, as in public code from this literature, leaks future price scale into inputs. A static rank on the leaked factor alone scores Sharpe 4.37 in the test year." | REPRO | `docs/phase1_5/A_evaluator.md` §4 (synthetic test plus real data) | NYSE 2017 | Sharpe 4.37 (an upper bound on what the leak carries) | Train-period normalisation | D | Describes a protocol property; **avoid** asserting it explains the paper's numbers. |

## Leakage in vendor-adjusted prices (new finding)

| ID | Publishable wording | Class | Evidence and files | Configuration / data / n | Effect [uncertainty] | Control | Verif. | Limitations / avoid |
|---|---|---|---|---|---|---|---|---|
| L1 | "Dividend back-adjusted closes embed the factor F(t) = ∏ over later ex-dates of (1 − D/P). Because P is a future price, any feature built on adjusted price **levels** (e.g. dollar volume) contains information about future returns. In our sample, the part of log F not explained by trailing yield predicted 5-day factor-neutral returns with rank IC +0.031 [+0.024, +0.037], positive in every year." | REPRO (mechanism algebraic; recomputed) | `docs/audit_20261022/A5_DIVIDENDS.md`; `data/audit/yf_unadjusted/` | 462 current S&P 500 stocks, 2019–2023 signals, yfinance download of 2026-10-22 | Future-only IC +0.0305; L itself +0.019; trailing yield −0.009 (n.s.) | Trailing yield (ex ante) | R | Magnitude depends on the vendor, the fetch date and the universe. Rank distortion of the feature is small (ρ = 0.995). **Avoid** implying the leak is large in every dataset. |
| L2 | "Removing the leak changed two conclusions of our own benchmark: the residual-target gain shrank from +0.014 to +0.009 (CI now includes 0), and an apparent harm from volume features (−0.009, 0/5 years) disappeared (−0.003)." | CONFIG | A5 part 2; `results/FBAUDIT_20261022/corrected/` | LightGBM, early-stopped, seed 7, 5 annual folds | See text | Matched refits | R (residual − raw, residual level) | Early-stopping round counts differ between refits. |

## Metric interpretation and style exposure

| ID | Publishable wording | Class | Evidence and files | Configuration / data / n | Effect [uncertainty] | Control | Verif. | Limitations / avoid |
|---|---|---|---|---|---|---|---|---|
| S1 | "Beta- and sector-neutral RankIC still rewards exposure to other characteristics. A LightGBM given only four style characteristics and sector scored at least as well as the same learner with all 20 features." | CONFIG | `A11_MATCHED_2x2.md`; `results/FBAUDIT_20261022/matched/` | 2019–2023, 3 seeds, 50 rounds | Style/neutral +0.0259 [+0.013, +0.039]; full/neutral +0.0248; full/residual +0.0132; style/residual +0.0149 | Matched learner | R | This is predictability attributable to known characteristics, which is **legitimate**; it is *not* leakage. **Avoid** "neutral RankIC is wrong". It answers a different question. |
| S2 | "Residualising the evaluation target on characteristics (linearly, or with quintile dummies) does not remove predictability from nonlinear functions and interactions of those characteristics. A style-only learner keeps most of its score after residualisation." | REPRO (algebraic; synthetic, 3 replicates; real-data placebo) | `A12_SYNTHETIC.md`; `A7_STYLE_RANK_RR.md` | Synthetic: 400 stocks, 500 evaluation dates. Real: 2019–2023 | Synthetic style-only LIN about 0.073 vs raw 0.079. Real placebo LIN +0.0088 [+0.003, +0.015], NL +0.0056 [+0.001, +0.010] | Random scores | D (synthetic) / D (A7) | An incomplete statistical adjustment, not leakage. The real-data *mechanism* (interactions) is inferred, since it is verified only on synthetic data. |
| S3 | "Training on the factor-neutral target improved neutral RankIC by about +0.011 for both feature sets, consistent with the target steering the learner toward style exposures that the metric rewards." | CONFIG | A11 | As above | full: +0.0116 [+0.004, +0.020]; style: +0.0110 | Matched features | D (A11) | The reading is an interpretation. |
| S4 | "Most of the residual-target gain over a raw target was reproduced by neutralising the raw model's predictions ex ante, without retraining." | CONFIG | A2, A2c | Leak-free; seed 7 | Both-neutralised difference −0.0002 [−0.007, +0.008] | Ex-ante neutralisation | D | Single seed. |

## Model comparison and learner power

| ID | Publishable wording | Class | Evidence and files | Configuration / data / n | Effect [uncertainty] | Control | Verif. | Limitations / avoid |
|---|---|---|---|---|---|---|---|---|
| M1 | "In a matched comparison (same learner, rounds, seeds, windows and preprocessing; nested inputs), the fitted full-feature model did not outperform the fitted style-only model: −0.0017 [−0.012, +0.009] and −0.0011 [−0.010, +0.008]. Increments of ±0.01 cannot be excluded." | CONFIG | A11 | 2019–2023, 3 seeds | as stated | Matched style-only learner | R | This is a **valid comparison of fitted procedures**. It is not a test of whether the extra features contain information (see M2). |
| M2 | "In one synthetic design, the same LightGBM configuration (15 leaves, min_child 1000, lr 0.05, 50 rounds) did not exploit a planted feature whose standalone IC was 0.046 (increment −0.004). At 300 rounds it gained +0.005; Ridge gained +0.038 but missed the nonlinear style signal." | SUGG (as a general statement) / CONFIG (for that synthetic design) | A12 probe (1 seed) plus 3 permutation replicates | Synthetic, signal-to-noise as specified | as stated | Ridge; 300 rounds | D | **Do not generalise** to all GBMs, all feature sets or all markets. It shows only that a matched null from this configuration is uninformative about feature information at comparable or smaller effect sizes. |
| M3 | "Two alternative tests, linear rank encompassing and a within-bin conditional permutation control, failed synthetic validation: false positives under the style-only design and/or no detection of planted information." | CONFIG | A12 | Synthetic, 3 replicates | Encompassing +0.010 to +0.016 under the null | Planted information | D | Specific implementations. |
| M4 | "Nested-model noise penalties are expected: under the null, the larger model's estimated extra parameters add forecast noise (Clark & West, 2007, for MSPE). Our negative full − style increments under the style-only synthetic design are consistent with an analogous penalty for RankIC." | SUGG | A12 | — | — | — | — | **Clark–West is derived for MSPE in nested linear models. Applying it to Spearman RankIC is an unvalidated adaptation** (see manuscript §9.3). |

## Statistics, power and selection

| ID | Publishable wording | Class | Evidence and files | Configuration / data / n | Effect [uncertainty] | Control | Verif. | Limitations / avoid |
|---|---|---|---|---|---|---|---|---|
| P1 | "With five annual folds, the minimum detectable paired neutral-RankIC difference was about 0.014–0.019. A 250-day prospective test would detect a 0.006 difference with probability about 0.07." | CONFIG | A1 | Long-run variance from the 2019–2023 paired series | as stated | — | D | The LRV varies about 2.8× across years (A9). |
| P2 | "77–89% of the day-to-day variance of cross-sectional RankIC exceeds the 1/√(n−1) sampling floor, so more time, not more stocks, governs power." | CONFIG | A1(c) | 462 stocks | as stated | — | D | Interpretation assumes stationarity. |
| P3 | "Selecting the best of 17 arms inflated its development estimate. The leave-one-year-out selected value was +0.0098 vs +0.0127 in-sample, and a best-of-pool test against reversal gave p = 0.55." | CONFIG | A3 | — | as stated | Hansen SPA, White RC, CSCV | D | The SPA p against zero (0.040) is within-pool only. |

## Economics

| ID | Publishable wording | Class | Evidence and files | Configuration / data / n | Effect [uncertainty] | Control | Verif. | Limitations / avoid |
|---|---|---|---|---|---|---|---|---|
| E1 | "Unbuffered daily-rebalanced neutral books traded 36–77% of NAV per day and broke even at 2–14 bp per unit of turnover. A weekly 10%/30% buffer cut turnover to about 0.13 per day." | CONFIG | A3, A8 | FB simulator; open1/close1 execution; drift; costed hedge | as stated | Random-score null | D | Linear costs; no borrow or financing costs; survivor universe. |
| E2 | "No model book outperformed matched comparators. The best net Sharpe at 5 bp in the inspected years came from a style-only learner's book (+1.11, positive in all five years), against +0.56 [−0.32, +1.44] for the residual LightGBM." | SUGG | A8, A13 | Post hoc, after selection over arms, schedules and executions | Paired differences: all CIs include 0 | RR, raw-neutralised and style books | D | **Do not** claim the style books' returns are explained by known style premia (attribution not done), or that they are investable. |

## Data and universe

| ID | Publishable wording | Class | Evidence and files | Configuration / data / n | Effect [uncertainty] | Control | Verif. | Limitations / avoid |
|---|---|---|---|---|---|---|---|---|
| U1 | "Matching current constituents to a public point-in-time membership list by ticker covers about 43–58% of 2004–2012 index members. This is an approximate symbol-match count, not a bound: renames lower it, while ticker reuse and pre-buyout symbols raise it." | CONFIG | A10, `BC_REVIEW.md` Erratum 2 | fja05680/sp500 lists | as stated | — | D | Not a bound. |

## Methodologically compromised (exclude, or present only as cautionary examples)

| ID | Original claim | Why compromised |
|---|---|---|
| C1 | FB: "volume/liquidity features hurt (0/5 years)" | Dividend leak (L2). Leak-free: no effect. |
| C2 | FB: "residual target +0.0140, 5/5 years" | Contaminated features handicapped the raw model (L2). Leak-free +0.0089, and mostly tilt removal (S4). |
| C3 | Audit: "stock-specific IC about +0.011 (style-residual metric)" | The metric fails the style placebo (S2). |
| C4 | BC review: "model − placebo ≈ +0.002" | Unmatched comparison (target, rounds and nesting differed); superseded by M1. |
| C5 | Paper-protocol Sharpe levels from best-test epochs | Test-year selection; matched by shuffled labels (T5). |

## Not adequately tested (state as open)

- **UT1:** whether the extra price/volume features carry information beyond style and reversal (question B). Untested with a learner shown to have power (M1–M3).
- **UT2:** whether other GBM configurations or model classes add value. Only one configuration per class was used.
- **UT3:** learned hyperedges on real data. Only a synthetic stage-1 test was run.
- **UT4:** investability of any neutral book: borrow costs, impact, point-in-time universe, delisting returns.
- **UT5:** style-factor attribution of the style-only books' returns.
- **UT6:** any claim on data after 2026-10-22. The prospective test is pending.
