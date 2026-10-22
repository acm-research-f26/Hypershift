# Study BC v2 — corrected specification (conditional; NOT approved for the survivor universe)

**Status (2026-10-22).** This replaces `NEXT_STUDY_SPEC.md` as the design any future backcast must meet. **Decision: STOP** for the current data (`BC_REVIEW.md` §9). The specification is valid only on a point-in-time universe with delisting returns, unless every claim is downgraded to the survivor-cohort estimand in §2. No 2004–2012 prices have been downloaded or scored.

## 1. Corrected hypotheses and success/failure framework

Three separate questions, tested in a **fixed sequence**. Each is tested at α = .05 only if the previous one is SUPPORTED. This controls family-wise error without splitting α.

1. **H-info: predictive information.**
   - **Test:** the full model (frozen FB class, leak-free, 50 rounds, retrained annually) vs 0.
   - **Metric:** FB neutral RankIC (beta and sector removed).
   - **Interpretation:** this is expected to pass and is **not** evidence of anything beyond known characteristics. It gates the next test only.
2. **H-incr: incremental value. This is the primary scientific question.**
   - **Primary contrast:** full model − **matched style-only placebo**, on the neutral RankIC.
     - The placebo is the same learner, rounds and target, with inputs restricted to cs_z of beta, leak-free log_dvol_20, ret_60, vol_60, plus sector one-hot.
   - **Second contrast, Holm over the two inside this step:** full model − RR.
     - RR is the predeclared zero-parameter residual reversal: −(21-day return − beta × EW market, sector-demeaned) / vol_60.
   - **SUPPORTED** requires both of:
     - Holm p < .05 with the CI excluding 0;
     - at least 6 of 9 years with a positive increment, and every leave-one-year-out pooled increment > 0 (the stability rule, §3).
3. **H-econ: economic usefulness** (descriptive unless the universe is point-in-time with delisting returns).
   - **Book:** the A8 buffered weekly sector-neutral long/short with the equal-weight beta hedge, at open1 and close1, with costs of 5 and 10 bp per unit traded.
   - **Promising** requires all of:
     - net Sharpe ≥ 0.5 at 5 bp under both executions;
     - a break-even of at least 10 bp;
     - the model book − RR book Sharpe difference > 0, with a bootstrap CI excluding 0.

   A model that predicts better than zero but **matches RR or the placebo** never triggers GO.

**Decision table:**

| Result | Decision |
|---|---|
| H-incr SUPPORTED and H-econ promising (point-in-time data only) | **GO**: prospective, cost-aware paper-trading protocol of its own |
| H-incr not SUPPORTED, 95% upper bound of model − placebo < 0.005 | **Rules out** an economically relevant increment in that sample. Stop the model line |
| Anything else | **INCONCLUSIVE**: treated as stop for resource allocation. Not evidence of no effect |

## 2. Estimand and universe integrity plan

**Preferred universe.** Point-in-time S&P 500 members on each date, with delisting returns. Membership from a verified source (`A10_UNIVERSE.md` lists one free source and its caveats); prices and delistings from a survivorship-free vendor.

**Survivor-cohort estimand, if no point-in-time data are available.** "Among companies that are S&P 500 constituents in 2026, have yfinance history, and (variant C2) were index members on date t, the cross-sectional rank association between predictions at the close of t and their factor-neutral 5-day returns from t." Every claim must be qualified with: *conditional on survival to 2026*.

**Claims that cannot be supported on survivors:**
- historical investability;
- absolute returns or Sharpe;
- capacity;
- behaviour on failing firms;
- generalisation to the historical index (coverage 43–58%, lower bound).

**Mandatory checks before any scoring:**
1. **Symbol-history map.** For the 25 gap symbols in A10 and for renamed companies, exclude the earlier interval wherever reuse by a different company is possible (e.g. BR, AMP, CNC).
2. **Zero-volume stale-price days.** Exclude stock-days with zero volume and an unchanged close (A3).
3. **Raw data hashed at download.** Record yfinance `auto_adjust=False` Close, Adj Close, Volume, High, Low, Open and Dividends, with their sha256.
4. **Leak-free features.** Dollar volume from unadjusted Close × Volume. The truncation-invariance unit test passes.
5. **Sectors.** Current GICS is a disclosed limitation. Report a sensitivity without sector demeaning (beta-only neutral) as descriptive.

## 3. Power, regime and inference plan

- **Primary inference.** Paired daily series per year; pooled = mean of 9 year means; stationary bootstrap (block 10, B 5000).
- **Complementary inference, all predeclared:**
  - Newey–West with 10 lags;
  - stationary bootstrap with block 20;
  - a year-level t across 9 year means (8 df);
  - the leave-one-year-out table.
- **Expected power (A9).**

  | Quantity | Value |
  |---|---|
  | MDE, model − RR (dev LRV) | 0.0095 |
  | MDE under ×2 variance | 0.013 |
  | MDE under ×3 variance | 0.016 |
  | MDE with 2008 at 3 × the dev maximum | 0.011 |
  | Development increment over RR | 0.0002 |
  | Development increment over the placebo | 0.0015–0.0024 |

  The study is therefore powered only to detect increments ≥ 0.01. **State in advance:** a non-significant result does not rule out increments below about 0.01.
- **Regime rule.** If more than 50% of the pooled effect comes from one year, report the result as **regime-concentrated**. It cannot be SUPPORTED unless it passes the leave-one-year-out rule.
- **Realised LRV.** Report it, and the MDE it implies, alongside the result.

## 4. Freeze checklist (everything that must be committed before any 2000–2013 price download)

- [ ] This spec, with the universe variant chosen (point-in-time or survivor C1/C2) and its estimand sentence.
- [ ] Code: leak-free feature builder, placebo builder, RR, book simulator (`fbaudit/book.py`), analysis script. Unit tests pass, including:
  - truncation invariance;
  - a date guard (refuse ≥ 2013-01-01);
  - the placebo's inputs restricted to the five style columns;
  - a planted-signal positive control;
  - a within-date label-permutation negative control.
- [ ] sha256 of every code file in a `FREEZE_MANIFEST_BC.json`.
- [ ] The membership file and symbol-history map with their sha256, plus the exclusion list (reused symbols, stale-price days).
- [ ] Fold calendar: 2004–2012 evaluation; training from 2000 plus a 64-day warm-up; labels purged by h = 5.
- [ ] The fixed sequence, thresholds, stability rule, regime rule and decision table above.
- [ ] Cost grid, schedules, executions and the 20-seed random null for the book.
- [ ] A statement of which outcomes are uninterpretable, and why (survivorship).
- [ ] The download script, which may only run after the freeze commit exists (it checks for the manifest).
