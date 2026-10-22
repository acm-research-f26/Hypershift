# HyperShift — final research decision after matched-control verification (2026-10-22)

**Scope and status.** Everything here uses the 2019–2023 development years, which had already been inspected, so all of it is exploratory.
- No 2004–2012 prices and no post-freeze outcomes were used.
- No architecture or hyperparameter search was run.
- Each item is labelled **[verified]** (computed and checked here) or **[interpretation]** (a reasoned reading of verified numbers).

**Evidence files:**
- `A11_MATCHED_2x2.md`: matched feature × target design;
- `A12_SYNTHETIC.md`: metric and procedure validation;
- `A13_LF_AND_ECON.md`: v1-LF mask fix and paired economics;
- earlier: `A7`–`A10`, `BC_REVIEW.md`.

**Plan and ledger:** `docs/superpowers/plans/2026-10-22-matched-closure.md` and `CLOSURE_LEDGER.md`.

## Decision

| Level | Verdict |
|---|---|
| Scientific status of question B (do the extra features carry information beyond style and reversal?) | **INCONCLUSIVE.** Not answerable with the current learner and metrics |
| Resource recommendation for price-only modelling on the current-constituent S&P 500 universe | **STOP** |
| Study BC (2004–2012 backcast) | **STOP stands, and is strengthened:** its learner could not have detected the increments it was meant to test |

**Why STOP rather than "keep investigating".** Resolving question B needs a new *methods* study: a validated nested-model test with demonstrated power at increments of about 0.01. Even then:
- the development increments are at most about 0.01 in magnitude;
- the economic value on this universe is attained just as well by style-only books;
- no untouched historical data remain in this source that are free of survivorship problems.

STOP is a resource decision. **It does not establish that the effect is zero.**

**What would reopen it:**
- a synthetic-validated incremental-information test (roadmap R1) that finds an increment ≥ 0.01 on saved predictions;
- point-in-time universe data with delisting returns (R2);
- a new data type, such as point-in-time fundamentals (R3).

## 1. Was the original style-only comparison matched? No. [verified]

From the code (`scripts/bcreview_style.py`; A7):

| | Style-only "placebo" | "Full" model |
|---|---|---|
| Target | neutral | residual |
| Rounds | 50 fixed | early-stopped on validation (varies by year) |
| Inputs | 4 styles + sector one-hot | 20 FB features, **no** sector one-hot |
| Seeds | 7 | 7 |

Target, rounds and feature nesting all differed, so the A7 "model − placebo" increments are confounded and **superseded by A11**.

**What A7 still validly showed:** a learner restricted to style inputs scores positive on both residualised metrics. Synthetic data confirm this is a property of the metrics (A12) [verified]. That it works through style *interactions* on real data is [interpretation, verified only on synthetic data].

## 2. The corrected comparison (A11) [verified]

**Design.** One learner (FB LightGBM, 50 fixed rounds), seeds {7, 19, 42}, identical training window, evaluation dates and preprocessing, leak-free inputs. The style inputs are a subset of the full inputs; both sets carry a sector one-hot.

**Absolute cells:**

| Cell (features / target) | Neutral RankIC [95% CI] | Per-seed pooled |
|---|---|---|
| full / residual | +0.0132 [+0.003, +0.023] | +0.013, +0.012, +0.014 |
| style / residual | +0.0149 [+0.007, +0.023] | +0.014, +0.015, +0.016 |
| full / neutral | +0.0248 [+0.012, +0.038] | |
| style / neutral | +0.0259 [+0.013, +0.039] | |
| RR (zero-parameter) | +0.0116 [−0.007, +0.030] | |

**Contrasts:**

| Contrast | Pooled [95% CI] | 90% CI | Years > 0 |
|---|---|---|---|
| Feature increment full − style, residual target | **−0.0017** [−0.012, +0.009] | [−0.010, +0.007] | 2/5 |
| Feature increment full − style, neutral target | **−0.0011** [−0.010, +0.008] | [−0.009, +0.007] | 1/5 |
| Interaction (does the increment depend on the target?) | −0.0006 [−0.005, +0.004] | | |

- **Increments:** both are UNRESOLVED. Neither is detected, and neither is bounded within ±0.005.
- **Interaction:** not detected.
- **Target effect, same features:** neutral − residual is +0.0116 (full) and +0.0110 (style). The neutral target buys style exposure equally for both feature sets.
- **Encompassing:**
  - full given style: +0.007 to +0.012;
  - full given RR: +0.010 to +0.021.

  **These are not interpretable,** because encompassing gave false positives on synthetic style-only data (A12).

## 3. Validity of the predictive metrics

| Question | Metric | Valid? | Evidence |
|---|---|---|---|
| A. Predictability | FB neutral RankIC (beta and sector removed) | **Valid for "predicts factor-neutral returns"**, but it includes style premia, so it is not a stock-specific measure | A4b, A11 [verified] |
| B. Incremental information | Style-residual RankIC vs 0 (linear or quintile) | **Invalid.** A style-only model scores positive when returns are nonlinear in styles | A7, A12 [verified] |
| B | Matched LightGBM increment (full − style) | **Invalid with this learner.** Negative in a style-only world (the nested-model noise penalty; Clark & West 2007). Does not detect a planted signal of standalone IC 0.046 | A12, 3 replicates plus a 1-seed probe [verified] |
| B | Encompassing (full ranks residualised on style ranks) | **Invalid.** False positives in the style-only world | A12 [verified] |
| B | Conditional permutation control (extra features permuted within style-prediction bins) | **Invalid.** One false positive in 3, and no detection at b = 0.15 or 0.5 | A12 [verified] |
| C. Economic usefulness | Buffered sector-neutral, beta-hedged book with drift, open1/close1 execution, linear costs | **Accounting valid** (a 20-seed random null centres at 0). **Economically incomplete:** no borrow, financing or impact costs, and a survivor universe | A8, A13 [verified] |

**The decisive diagnostic [verified, scoped].** The FB LightGBM configuration is 15 leaves, min_child 1000, lr 0.05 and 50 rounds. In the synthetic probe:
- **At 50 rounds** it gains −0.004 from a planted feature whose standalone IC is +0.046.
- **At 300 rounds** it gains +0.005.
- **Ridge** gains +0.038 from the same feature, but misses the nonlinear style signal.

The synthetic signal-to-noise ratio (style IC about 0.08, planted about 0.046) is easier than the real case of interest (increments about 0.01), so the failure holds *a fortiori* [interpretation]. The support is narrow: three permutation replicates plus one single-seed probe.

**Consequence.** Every FB-family null of the form "features X do not help this LightGBM" is **untested with a learner shown to have power**. This covers volume features, rich vs price, full vs style, and "model class not the bottleneck". It is not evidence that the features are uninformative.

## 4. What incremental information the model demonstrably provides

**None demonstrated.**
- The model predicts factor-neutral returns at about +0.013 [verified]. A style-only learner does the same or better (+0.015 to +0.026), and so, noisily, does a zero-parameter residual reversal (+0.012).
- Whether its extra features hold information beyond style and reversal is **unknown**. The comparisons are null with CIs of ±0.01, and the learner was shown unable to extract increments of that size.

## 5. Economic evidence (A13) [verified, with caveats]

**Buffered weekly neutral book, net Sharpe at 5 bp (open1 in brackets):**

| Arm | Net Sharpe |
|---|---|
| A8 residual model | +0.56 (+0.56) |
| Matched full/residual | +0.46 (+0.40) |
| **Style/residual** | **+1.11 (+1.08)**, positive in 5/5 years |
| Style/neutral | +0.95 (+0.89) |
| Raw-neutralised | +0.43 (+0.32) |
| RR | +0.37 (+0.32) |

**Paired contrasts:**
- The A8 model minus every comparator ranges from −0.027 to +0.013 per year, and every CI includes 0.
- Matched full − style is −0.024 [−0.061, +0.011] per year.

**Reading [interpretation]:**
- No model-specific economic advantage.
- The best books are style-only. They are beta-hedged but not style-hedged, and their factor exposures were not measured. The result is post hoc, on inspected years, after selection over arms, schedules and executions, under a linear cost model.
- It is consistent with known style premia. It **explains away** the model's economics; it does not open a new line.
- **Verdict: no support for further investment in the model.**

## 6. Unresolved correctness issues

| Item | Status |
|---|---|
| A7 placebo comparison confounded | Superseded by A11 (a design flaw, not a code bug) |
| Survivor-coverage description | The 43–58% figure is a **symbol-match count, not a bound in either direction**. Renames lower it; symbol reuse (e.g. BR) and pre-buyout symbols with no yfinance history (KMI, HCA, HLT, DG) raise it. Errata appended to `BC_REVIEW.md` and `AUDIT_REPORT.md` |
| v1-LF eligibility-mask discrepancy | **Fixed** (test `test_lf_neutral_uses_post_lf_mask`). Effect on dev: −0.00026 pooled, max 0.022 per day. Class (1), pre-data. The v1-LF code hash has now changed three times, all before any post-freeze data, so the user's veto window applies |
| Deferred minors (carried) | A2 "nan%" display; two `__main__` blocks in `fbaudit_dividends.py`; the A8 bootstrap ignores fold boundaries; the economic arms use seed 7 only; the synthetic permutation false positive is single-seed (training-fit variance is not in the date bootstrap) |
| Frozen FB and v1 code | Unchanged (`test_freeze_hashes_unchanged`) |

## 7. Recommendation

**STOP** price-only modelling on the current universe. Scientific status of question B: **INCONCLUSIVE**. Keep v1 and v1-LF running passively; they are nearly free.

## 8. Roadmap (genuinely new data or new questions only)

| Pri | Item | Why | Gate |
|---|---|---|---|
| R1 | **Methods: a validated incremental-information test.** Nested-model comparison with the Clark–West adjustment, multi-seed fits, a flexible style baseline, and an incremental component with **demonstrated synthetic power at increments of about 0.01** | Without it, no "beyond style" claim on any data is testable. It is also the publishable methods contribution | Must pass S/I synthetic validation before any real data. It can then re-score saved 2019–2023 predictions (exploratory) |
| R2 | **Point-in-time universe with delisting returns** (a constituent history plus a survivorship-free vendor) | Removes survivorship, and the only route to a credible backtest | Licence cost; only if R1 or R3 is pursued |
| R3 | **Point-in-time fundamentals, monthly horizon** (GKX-type characteristics) with the matched style-only learner as primary control and RR as benchmark | The literature finds ML gains there, not in large-cap price-only data | Needs R2 data and the R1 test |
| R4 | Prospective v1 and v1-LF, passive | Free | Score once at ≥ 250 days; read NO EVIDENCE as uninformative |
| R5 | Style-factor attribution of the neutral books | Would confirm the style-premium reading | Low priority; known effects |

## 9. Publication

See `PUBLICATION_OUTLINE.md`.
