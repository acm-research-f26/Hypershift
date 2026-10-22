# Study BC — pre-execution review and decision (2026-10-22)

**Decision: STOP.** Study BC as proposed in `NEXT_STUDY_SPEC.md` should not be frozen or run.

- **No new prices were used.** No 2004–2012 price, return or volume data were downloaded or scored for this review.
- **Basis.** Every number below comes from the already-inspected 2019–2023 development folds, all exploratory, with leak-free inputs. The universe check used point-in-time **membership** lists only, with no prices.
- **Evidence files:**
  - `A7_STYLE_RANK_RR.md`: placebo, residual reversal, increments;
  - `A8_COST.md`: cost-aware books;
  - `A9_POWER_REGIME.md`: regime and power;
  - `A10_UNIVERSE.md`: membership and identity;
  - `PROSPECTIVE_ADDENDUM.md` (correction section): v1/v1-LF.
- **Conditional design.** The design that *would* be required, if the data problem is solved, is in `STUDY_BC_SPEC_v2.md`. It is **not approved** for the survivor universe.

## 1. Independent critique of the existing audit

**What the audit got right:**
- the dividend look-ahead, with its mechanism and correction;
- tilt removal as the main effect of residual targets;
- "factor-neutral" RankIC rewarding style exposure;
- the cost dependence of neutral books;
- the near-zero power of prospective v1.

**What it got wrong (found by this review):**

1. **Its main positive finding is not established.** The audit called "stock-specific IC about +0.011 (style-residual metric)" *suggestive*, and built Study BC to confirm it. A **style-only placebo** fails that claim:
   - **The placebo.** The same LightGBM, trained the same way, with only beta, size, 60-day return, volatility and sector as inputs.
   - **Its scores:**
     - +0.0088 [+0.003, +0.015] on the audit's linear style-residual metric;
     - +0.0056 [+0.001, +0.010] on a quintile-dummy (nonlinear) version;
     - **+0.0265 [+0.014, +0.040] on FB's neutral metric**, about twice the full model's +0.0136.
   - **The increments over it are about zero:**
     - model − placebo: +0.0024 [−0.006, +0.011] (linear) and +0.0015 [−0.006, +0.009] (nonlinear);
     - model − residual reversal: +0.0002 [−0.009, +0.010] (A7).
   - **Why the metric fails.** Per-date residualisation of the *target*, whether linear or by quintile dummies, removes only additive style effects. A tree learns style *interactions* (sector × beta, volatility × momentum and so on), which survive residualisation. Spearman ranking does not remove them either.
   - **Conclusion.** "Beyond style" can only be measured as an **increment over a matched learner with style-only inputs**, never by scoring against a residualised target.
2. **The prospective addendum overstated two things** (now corrected):
   - the leak's bias direction, which is unknown;
   - v1/v1-LF agreement as validation, when both runs have about 7% power.

   v1-LF also had no code until this review. It is now implemented, hash-pinned and leak-strict.
3. **BC's power assumed stationary variance and a meaningful estimand.**
   - Neither holds: year-level LRV varies about 2.8× within 2019–2023 alone (A9).
   - The estimand fails the placebo.
4. **"Positive break-even" understated cost sensitivity.** The audit tested no turnover control. With a weekly buffer, books do clear 5 bp, but no arm is separable from another (A8).
5. **Process defect.** An audit-session `sed` edit committed a SyntaxError in `scripts/fbaudit_dividends.py`. It is now fixed, with regression test `test_audit_scripts_parse`.

## 2. Remaining methodological weaknesses (severity-ranked)

| # | Weakness | Severity | Evidence | Consequence |
|---|---|---|---|---|
| 1 | The style-residual RankIC estimand leaks style information through interactions | **Critical for BC** | A7 placebo positive on LIN and NL | A BC "GO" on H1 could be produced by style characteristics alone |
| 2 | No detectable increment of the model over matched simple controls (placebo, residual reversal) | **Critical for BC's value** | A7: increments about +0.000–0.002, CI ±0.008–0.010 | BC MDE for increments is 0.010–0.016 (A9): power about 5–10% at the dev effect |
| 3 | Survivor cohort covers at most 43–58% of the 2004–2012 index (lower bound); no delisting returns | **High** | A10 | Cannot support historical-investability or economic claims; reversal-type longs plausibly biased upward (magnitude unknown) |
| 4 | Ticker identity: 25 symbols with gaps of more than 1 year between index intervals, including possible reuse by different companies (BR, AMP, CNC); renamed members unmatched by symbol | **Medium** | A10 | Point-in-time filtering by symbol is unreliable without a symbol-history map |
| 5 | Regime heterogeneity: year LRV ratio about 2.8; strong year effects (e.g. RR +0.031 in 2019, −0.017 in 2022) | **Medium** | A9 | A pooled result can be driven by 1–2 years; year-level t has power 0.22–0.52 at θ = 0.008 for 9 years |
| 6 | Economic evidence: one arm passes D4, but its CI includes 0, it was selected on these years, and the other arms are within noise | **Medium** | A8 | No credible cost-surviving signal; neutral-book economics are cost-borderline |
| 7 | Costs: no borrow, financing or impact cost; linear costs | **Medium** for economics | A8 limitations | Net Sharpe figures are upper bounds in that respect |
| 8 | Current GICS sectors used for 2004–2012 | **Low–Medium** | A10 | Sector neutralisation is mis-specified in the early years |

## 7. Residual reversal as a baseline (deliverable 7)

- **Definition (predeclared D1, zero parameters).** RR = −(past 21-day return minus beta × the equal-weight 21-day market return, sector-demeaned) / vol_60. RR5 is the 5-day version.
- **Dev performance (exploratory):**
  - neutral RankIC +0.0116 [−0.007, +0.030];
  - linear residual metric +0.0065, nonlinear +0.0069 (CIs include 0);
  - RR − reversal16 +0.0026 [−0.002, +0.008].
- **Strong year dependence.** RR scores +0.031 in 2019 and −0.017 in 2022.
- **Relation to the other signals:**
  - RR correlates +0.71 with reversal16 and +0.22 with the model.
  - Model − RR is +0.0002 [−0.009, +0.010].
  - Partialling RR, reversal16 and l1 out of the model leaves +0.0070 [+0.0004, +0.014] on NL. But the placebo shows NL is not style-clean, so this residual is not attributable to stock-specific information.
- **Verdict.** RR is the right simple benchmark: literature-backed, zero-parameter, and leaving the model with no detectable increment. Any future "incremental value" claim must beat **both RR and the matched style-only placebo**.

## 8. Minimal cost-aware portfolio test (deliverable 8, done on dev; A8)

**Design:**
- book: sector-neutral decile long/short, beta-hedged with an implementable equal-weight basket, with drift;
- schedules: daily, weekly, and weekly with a 10%/30% buffer;
- execution: next close and next open;
- costs: 2, 5, 10 and 20 bp per unit of traded notional;
- a 20-seed random-score null to check the accounting.

**Findings:**
- **Buffering is the decisive control.** It cuts turnover from about 0.76 to 0.13 per day.
- **The residual LightGBM buffered book:**
  - gross Sharpe +0.95;
  - net +0.56 [−0.32, +1.44] at 5 bp;
  - break-even about 12 bp under both executions.
- **The other arms, buffered at 5 bp:**
  - raw-neutralised +0.43 (close1) / +0.32 (open1);
  - RR +0.37 / +0.32;
  - reversal16 +0.33 / +0.09.
- **Against the null.** Random books centre at gross Sharpe −0.04 to −0.10, SD about 0.5. The model book is about 2 null SDs above.
- **Reading.** A buffered weekly neutral book *may* clear about 5–10 bp of linear costs. That is suggestive, unconfirmed, and **not distinguishable between model and simple signals**. Residual-label returns were not used as P&L.

## 9. Decision: STOP (deliverable 9)

**Justification.** BC was designed to answer: "does the style-controlled model carry stock-specific predictive information in untouched years?" The review shows that question cannot be answered meaningfully with this design and data.

1. **The primary estimand is invalid.** The style-residual RankIC rewards style information, as the placebo shows. A positive BC result would be uninterpretable: it is what a style-only model already achieves.
2. **The valid estimand has nothing to detect.** That estimand is the increment over a matched style-only learner and over RR. Development estimates are about 0.000–0.002, against a backcast MDE of 0.010–0.016 that may be larger under 2008-type variance. A backcast would almost surely return INCONCLUSIVE, which tells us nothing beyond the dev result.
3. **The economic question cannot be answered credibly on this universe.**
   - The data cover at most 43–58% of the historical index, with no delisting returns and with symbol-identity problems.
   - Survivorship plausibly favours reversal-type longs, so only a negative result would be interpretable, and dev already supplies the negative.
4. **Cost.** Running BC costs a day, uses up the last untouched historical period in this data source, and buys no decision-relevant information.

**What is and is not concluded:**
- **Statistical conclusion (2019–2023, exploratory):**
  - the model's predictive information is not distinguishable from a style-only learner or a zero-parameter residual reversal;
  - all simple and learned signals are individually small (IC about 0.01) and regime-dependent.
- **Resource conclusion:** stop price-only prediction research on the current-constituent S&P 500 universe.
- **Not concluded:**
  - that price-only prediction is impossible;
  - that the model has no information;
  - anything about point-in-time universes, small caps, or other data.

**Recommended alternatives, in order:**
1. **Methodology write-up (no compute).** The programme's durable results:
   - reproducible nulls;
   - the weight-decay collapse;
   - the normalisation and dividend leaks;
   - neutral metrics rewarding style;
   - residualised metrics failing a style placebo;
   - matched-placebo increments as the correct estimand.
2. **Keep the prospective tests running passively.** v1 plus v1-LF, scored once at 250 or more days or later, with no interim looks. They are nearly free. Read NO EVIDENCE as uninformative.
3. **Only if prediction research must continue:** acquire **point-in-time data**: a constituent history with delisting returns (for example CRSP or a commercial survivorship-free US equities dataset) plus point-in-time fundamentals. Then run a GKX-style monthly study whose **primary contrast is full model − matched style-only placebo**, with RR as a second benchmark (`STUDY_BC_SPEC_v2.md` gives that framework). Do not start this without the data.
4. **If you nonetheless want a cheap economic look at 2004–2012 on survivors:** treat it as descriptive only, freeze `STUDY_BC_SPEC_v2.md`'s survivor-cohort variant, and label every result as **not an investable backtest**. I do not recommend it.

---

## Erratum (2026-10-22, closure review): survivor coverage bound

- **What was wrong.** This file and `AUDIT_REPORT.md`'s correction say the survivor cohort "covers **at most** 43–58%" of the 2004–2012 index while also calling the figure a "lower bound". These contradict each other.
- **Correct statement.** The measured share (43–58% of 1 July members matched among the 462 current constituents) is a **lower bound** on true coverage: members that later changed symbol are not matched by symbol. The cohort therefore covers **at least** 43–58% of the historical index, and **at most** 42–57% of members are missing.
- **Effect on the STOP decision.** The true share is unknown without a symbol-history map. The STOP reasoning does not depend on it: even full coverage of members would leave the delisting-return and placebo problems unsolved.
