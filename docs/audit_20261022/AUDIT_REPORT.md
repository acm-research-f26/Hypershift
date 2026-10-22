# HyperShift — independent scientific audit of Study FB and the research programme (2026-10-22)

**Status: exploratory audit.** Every number here was computed on 2015–2023 data that HyperShift had already inspected. Nothing in this report is confirmatory.

**Supporting files** (`docs/audit_20261022/`):

| File | What it holds |
|---|---|
| `A1_RECOMPUTE_POWER.md` | Independent recompute, power, block sensitivity, equivalence, seeds |
| `A2_MECHANISM.md` | Why the residual target helps; execution lag |
| `A2c_MECHANISM_CORRECTED.md` | A2 rerun with leak-free inputs |
| `A3_SELECTION_PORTFOLIO.md` | Selection bias, portfolio accounting, data checks |
| `A4_TARGETS.md` | Matched fixed-round target test, seeds, target ladder |
| `A4b_NEUTRAL_TARGET.md` | Training on the evaluation target |
| `A5_DIVIDENDS.md` | Dividend look-ahead |
| `A6_LITERATURE.md` | Verified literature |
| `NEXT_STUDY_SPEC.md` | Deliverable 6 |
| `PROSPECTIVE_ADDENDUM.md` | Part 12 |

**Code and outputs:**
- Code: `src/hypershift/fbaudit/`, `scripts/fbaudit_*.py`, `tests/test_fbaudit.py` (11 tests).
- Refit outputs: `results/FBAUDIT_20261022/`.
- Plan and decisions: `docs/superpowers/plans/2026-10-22-fb-audit.md`. Thresholds T1–T6 were predeclared there before any audit number existed.

**Conventions throughout:**
- "Neutral RankIC" is FB's metric: 5-day return minus beta × equal-weight market, sector-demeaned.
- "Style-residual RankIC" additionally removes, per date, the cross-sectional OLS fit on z(beta, log dollar volume, 60-day return, 60-day volatility) and sector dummies.
- Pooled = mean of the 5 fold means for 2019–2023. CIs are 95% stationary bootstrap, block 10, unless stated.

---

## Deliverable 7 — Executive decision memo

1. **Is there credible evidence of stock-specific predictive skill? Suggestive, not credible.**
   - **The best evidence.** The leak-free residual-target LightGBM has a style-residual RankIC of +0.011 [+0.003, +0.019], positive in 5/5 folds. The same model with no volume inputs scores +0.011 [+0.003, +0.020]. It keeps 89% at next-close execution (A2c, A2).
   - **Why it is not credible yet:**
     - It is the best of a 17-arm pool on years already inspected.
     - Its selection-adjusted value is +0.007 to +0.010: shrunk +0.0068, leave-one-fold-out +0.0098 (A3).
     - Its t-statistic of about 2.5 is below the t > 3 multiple-testing hurdle of Harvey, Liu & Zhu (2016).
     - It does not beat a one-line 16-day reversal: +0.006 [−0.002, +0.015] on the style-residual metric.
     - Its magnitude, about 0.01 IC, sits at the edge of what this design can resolve.
2. **Is the residual-target improvement meaningful? Mostly not as stated.** Three findings:
   - **It shrinks once the leak is removed.** It goes from +0.0140 to +0.0089 [−0.001, +0.019], because the dividend leak was handicapping the raw-target model (A5).
   - **Most of it is tilt removal.** Neutralising the raw model's predictions ex ante, with no retraining, recovers most of the gain. With both models neutralised the difference is −0.0002 (A2c).
   - **The defensible lever is controlling the style exposure of predictions, not the training target as such.**
3. **Can the current setup detect economically useful effects? No, not prospectively.**
   - The minimum detectable effect (MDE) over the five historical folds is about 0.014–0.019 for the relevant contrasts (A1).
   - The frozen prospective test has about 7% power at the development-sized effect (+0.006 vs reversal16). Reaching 80% power needs about 11,000 days.
   - A backcast on untouched 2004–2012 data reaches an MDE of about 0.008 (`NEXT_STUDY_SPEC.md`).
4. **Do the data-quality issues materially threaten the findings? One does: the dividend look-ahead.**
   - **The mechanism.** yfinance `auto_adjust` prices embed a back-adjustment factor that depends on future dividends **and future prices**.
   - **Its predictive content.** The future-only part of that factor predicts 5-day neutral returns at +0.031 IC, more than twice any model (A5).
   - **The features it enters.** It contaminates `log_dvol_20` and `amihud_20`.
   - **The conclusions it moves.** It overturned "volume features hurt" and shrank residual − raw.
   - **What survives it.** The best model's own score does not depend on it (corrected +0.0136).
   - **Other issues** (survivorship, current sectors, three stale-price stocks) are disclosed and unquantified, or small.
5. **Have simple predictive approaches been tested fairly? Partly.**
   - **What was tested well.** Ridge, LightGBM and MLP were compared on equal tuning budgets, so the comparison is fair *as specified*.
   - **What was not:**
     - LightGBM was tested in one constrained configuration.
     - Volume features were contaminated.
     - FB's metric rewards style exposure.
     - The only simple literature signal that matches the one lever found, residual reversal, was never run.
   - **Can model-class gains be ruled out?** No. The 90% upper bound on LightGBM − Ridge is +0.014, against a predeclared margin of 0.010 (A1).
6. **Why does positive neutral RankIC fail to produce positive market-neutral returns? Turnover and costs.**
   - **The books are gross-positive.**
     - The residual model's beta-neutral long/short has gross Sharpe +1.06.
     - Its book earns +10.7%/year on 1-day factor-neutral returns, Sharpe 1.33.
   - **They turn over heavily.** The books trade 36–77% of NAV **per day**.
   - **Costs consume them.** They break even at only 2–14 bp per unit of turnover, against the simulator's 10 bp (A3, A4b).
   - **No turnover control was tried on the neutral books,** so the FB conclusion "no neutral tradable signal" is overstated.
7. **What is the strongest unresolved hypothesis?** That a large-cap, price-only model with its style exposures removed has a small positive stock-specific IC (about 0.008–0.011) that survives realistic execution. It is testable in days on untouched 2004–2012 history.
8. **What is the cheapest investigation that could change our conclusions?**
   - **Already done:** this audit's zero-refit diagnostics. They changed four conclusions; see the register.
   - **Next:** the backcast in `NEXT_STUDY_SPEC.md`, about 1 day of engineering and about 1 h of CPU, with an MDE of about 0.008. Its NO-GO branch would end price-only large-cap prediction work.
9. **Recommendation. Pause new model and architecture research on this data.**
   - **Run three things only:**
     1. The P0 fixes (leak-free dollar volume, style-residual metric).
     2. The one backcast.
     3. The cost/turnover sweep on existing predictions.
   - **If the backcast is NO-GO or INCONCLUSIVE:**
     - stop price-only large-cap prediction;
     - write up the methodology (it is the programme's strongest asset: reproducible nulls, the weight-decay collapse, the normalisation and dividend leaks, the metric pitfalls);
     - move to fundamentally different data if prediction continues. Point-in-time fundamentals and broader universes are where the literature finds ML gains (Gu, Kelly & Xiu 2020; Avramov, Cheng & Metzker 2023).
   - **If GO:** a point-in-time universe with delisting returns, then a cost-aware prospective test.
10. **What would change this recommendation?**
    - **Toward continuing price-based work:** the backcast primary SUPPORTED with point estimate ≥ 0.008, plus ≥ 50% retained at next-close, plus a buffered neutral book with break-even above about 10 bp per unit of turnover.
    - **Toward stopping sooner:** a backcast 95% upper bound below 0.008.
    - **Not a reason to change it:** an INTERIM or NO EVIDENCE prospective v1 result, because v1 cannot discriminate (D2 in the addendum).

---

## Deliverable 1 — Critical research review

### 1.1 Findings by category

| Finding (programme-wide) | Category | Basis |
|---|---|---|
| FB's reported statistics reproduce from the saved predictions | **Established** (reproducibility) | A1(a). Max abs pooled difference 3e-8; refit determinism 3e-9 |
| THINK's paper advantage is not reproduced under validation selection or leak-free evaluation | **Established** within an inferred-settings reimplementation | Phases 1–2, `docs/phase2/REPORT_CLOSURE.md` (not re-audited here) |
| Weight-decay collapse of stacked hyperbolic layers | **Established** (mechanistic, planted-signal test) | `docs/phase1_5/F_learnability.md` |
| THINK's 2017 Sharpe ≈ 2 reflects market exposure, not ranking skill | **Established** | Phase 1.5a: hold-all 1.53; random high-beta baskets match |
| Graph, hypergraph, MASTER-style, pairwise and Kronos arms do not beat the static baseline or no-graph | **Unsupported** advantage, at the screen's power. Not an equivalence claim | Screen, Studies X and J |
| Dividend back-adjustment injects predictive look-ahead into dollar-volume levels | **Established** (data defect) | A5. Mechanism is algebraic; future-only part IC +0.031, 5/5 folds |
| FB "factor-neutral" RankIC rewards style exposure | **Established** (metric property) | A4b. Neutral-target model +0.0248 → +0.0047 after style removal |
| Market-neutral FB books are gross-positive but cost-dominated | **Established** for the FB simulator's assumptions | A3. Break-even 2–14 bp per unit turnover |
| Residual target beats raw target | **Compromised** in magnitude (leak), **Suggestive** in direction | A5: +0.0140 → +0.0089, CI touches 0. T4 robust to rounds; seed mean +0.0125 |
| The gain is learned stock-specific skill rather than tilt removal | **Unsupported** | A2, A2c (T2 = "mostly tilt removal" in both versions) |
| Positive stock-specific IC of about 0.01 | **Suggestive** | A2c +0.011 [+0.003, +0.019]; selection-adjusted +0.007 to +0.010; t ≈ 2.5 |
| Any model beats 16-day reversal | **Unsupported** | +0.006 [−0.006, +0.019] (A4 at 50 rounds); SPA p 0.55 over the pool |
| LightGBM / MLP beat Ridge | **Unsupported**, and **not ruled out** at δ = 0.01 | A1(d): 90% upper +0.014 / +0.021 |
| Volume and liquidity features hurt | **Compromised** | A5: corrected rich − price −0.0028, 1/5 folds |
| Volume features add ≥ 0.01 | **Unsupported**: ruled out on the contaminated features, A1(d) 90% upper −0.0008; leak-free CI [−0.011, +0.005] | |
| Longer horizons (10/20 days) do worse than reversal at the same horizon | **Suggestive** (descriptive) | FB horizon table. Overlap differs by horizon; one model config |
| Learned hyperedges fail | **Untested** as a real-data question | FB's fix failed a synthetic identity test. The v3 soft arm's real-data collapse is *consistent with* no group signal |
| Subgroup predictability | **Unsupported** | 1 of 16 CIs excludes 0, about the chance rate; terciles sorted on a contaminated feature |
| Prediction-to-trading: "no neutral tradable signal" | **Overstated**, i.e. Untested with turnover control | A3 |
| GBM model family adequately tested | **Untested** | One fixed configuration (15 leaves, depth 4, min_child 1000) |

### 1.2 Part 3: the residual-target finding in detail

**Hypothesis.** Training on factor-residual returns improves factor-neutral ranking.

**Experiment.** FB trained matched arms differing only in target: same features, cs_z inputs, LightGBM parameters, seed and selection rule, with early stopping chosen per arm on validation neutral RankIC.

- **Was early stopping a confound?** It was, partly, but not materially. At fixed 50 and 300 rounds the contrast is +0.0138 and +0.0107 (T4 pass, A4).
- **Seed variance:** 3 seeds average +0.0125; between-seed SD of pooled values 0.0007, against a date SE of 0.0056.

**Which of the seven candidate explanations hold:**

1. **Real stock-specific gain: mostly not.** The residual model's advantage over the ex-ante-neutralised raw model is +0.0021 (contaminated) or +0.0034 (leak-free), with CIs including 0. With both neutralised it is −0.0002 (A2c).
2. **Objective–metric alignment: yes, but alignment buys style exposure.** The target ladder rises monotonically as the training target approaches the evaluation target, ending at +0.025 for the neutral target itself (A4). Nearly all of that is exposure to styles the metric leaves in (A4b).
3. **Artefact of residualising both sides: no.** The evaluation target is *not* residualised on styles; only beta and sector are removed. The residual target removes more than the metric does. This asymmetry is why the residual model has *lower* style R² (0.13 vs 0.32).
4. **Variance reduction or easier optimisation: plausible and consistent with the data.** The residual model's daily IC SD is lower (0.096 vs 0.101), and early stopping chose more rounds in some years. This was not separately isolated.
5. **Removal of predictable factor components: yes, in the sense of (1).** The residual model drops style tilts that the raw model holds and that lose on the neutral metric.
6. **Leakage in target construction: no information leakage.** The OLS uses forward returns only as the dependent variable. Regressors are measured at t (beta trailing 126, features causal), and per-date coefficients are estimated across stocks. This is legitimate ex-post label construction.
   - **However,** one regressor (`log_dvol_20`) carries the dividend look-ahead.
   - The fewer-than-30-stock fallback never fires on evaluation dates (A2 section 5).
   - Current GICS sectors enter both the target and the metric.
7. **Selection bias: modest.** The residual contrast was one of 4 predeclared target contrasts (Holm 0.065). The leave-one-fold-out selected arm holds out at +0.0098 (A3).

**Is it economically meaningful?** On its own, no.
- **The best book (gross).** The residual model's beta-neutral long/short book is gross-positive (Sharpe 1.06; 1-day pure-alpha Sharpe 1.33) and turns over 45% of NAV per day.
- **Break-even.** It breaks even at 8.8 bp per unit of turnover.
- **What is missing:**
  - a turnover-controlled (buffered) neutral book;
  - a cost model sourced for large-cap US execution;
  - a point-in-time universe.

### 1.3 Part 2: what was done well (and should become permanent)

| Practice | Value | Executed correctly? | Make permanent? |
|---|---|---|---|
| Planted-signal learnability tests (Phase 1.5 D/F) | Very high: found the weight-decay collapse that invalidated all Phase 1 nulls | Yes | **Yes**, for every new model class |
| Shuffled-label and random-graph controls | High | Yes | **Yes**. Add a within-date label permutation control for tabular models |
| Normalisation-leak discovery (`norm: paper`) | High | Yes | Generalise into a **truncation-invariance test**: feature(t) computed on data cut at t equals feature(t) on full data. FB's causal test checked the code but not the vendor's adjustment |
| Matched comparisons and equal tuning budgets | High | Yes | Yes |
| Walk-forward folds with horizon-aware purging | High | Yes (FB) | Yes |
| Holm within families; stationary bootstrap | Good | Correct as implemented. CIs insensitive to block length 5–40 (A1) | Yes, plus a SPA/Reality-Check test for any best-of-pool claim |
| Factor-neutral metric | Good idea, incomplete | Removes beta and sector only | Replace with **neutral + style-residual co-reporting** |
| Power analysis | Good, came late | MDE from a proxy contrast. Per-contrast MDEs (A1) agree within about 20% | Compute **before** the design is fixed, per contrast, from pilot LRVs |
| Prespecification and freeze with code hashes | Very high | Yes. Hashes intact (test) | Yes. Also hash the data download and the feature cache |
| Documenting failures | Very high | Yes | Yes |
| Independent review and regression tests | High | Yes. The FB final review caught the rule-D leverage bug | Yes. A fresh-context reviewer for every report |

**Reproducibility vs validity.** FB is fully reproducible. Its saved predictions regenerate every table, and refits are deterministic to 3e-9. Yet two of its conclusions were invalid because of a vendor-side data property that no code test could see. **Reproducibility is necessary, not sufficient.**

### 1.4 Part 8: the negative claims, re-evaluated

| Claim | Verdict | Why |
|---|---|---|
| "Richer features do not help" | Partly supported | Leak-free rich − price is −0.0028 [−0.011, +0.005], and residual-target rich − price +0.0005. A ≥ 0.01 gain is unlikely but not formally excluded leak-free |
| "Model architecture is not the bottleneck" | Not established | The LightGBM − Ridge upper bound is +0.014. One GBM config. The screen's deep models were tested only against the static graph baseline |
| "Volume/liquidity features are useless" | Compromised, then neutral | See A5. Definitions are reasonable (dollar volume, Amihud, relative volume), but the levels were contaminated |
| "Graph relationships add no information" | Supported only as "no detectable advantage at this power" | Screen Holm 1.00. Not re-tested on the style-residual metric |
| "Longer horizons do not help" | Weak | Descriptive; different overlap per horizon; one model config; no turnover-adjusted comparison |
| "Market-neutral strategies cannot exploit the signals" | Overstated | Gross-positive; no buffer; 10 bp convention unsourced |
| "Learned hyperedges do not work" | Untested on real data | The synthetic task rewards identity-based pooling. The soft arm pools by current state, so the failure is an **architectural mismatch with the task** (FB's own reading), not evidence about real groups |
| "More training would not improve results" | Untested | Rounds 50 vs 300 change little (A4). More data (years) is the binding constraint, not training |

---

## Deliverable 2 — Methodological issue register (severity-ranked)

| # | Issue | Evidence | Severity | Affected conclusions | Recommended fix | Status |
|---|---|---|---|---|---|---|
| 1 | Dividend back-adjustment look-ahead in `log_dvol_20` and `amihud_20` (features, residual-target regressor, style exposures, subgroup sorting, prospective scoring) | A5: future-only factor IC +0.0305 [+0.024, +0.037], 5/5 folds; corrections move residual − raw by −0.0051 and rich − price by +0.0060 | **High** | Volume-feature verdict; size of residual − raw; subgroup terciles; prospective v1 inputs | Dollar volume from unadjusted Close × Volume everywhere; v1-LF sensitivity (addendum) | **Verified defect** |
| 2 | "Factor-neutral" RankIC leaves size, momentum and volatility in, so it rewards style exposure | A4b: +0.0248 → +0.0047 after ex-ante style removal | **High** | Every "stock-specific skill" reading of FB and screen RankICs | Co-report style-residual RankIC; predeclare which is primary | **Verified** (metric property) |
| 3 | Prospective v1 is nearly powerless at plausible effects | A1: P(detect) 0.07 at 0.006, 0.28 at 0.020; 11,466 days for 80% at 0.006 | **High** (informativeness) | Any reading of v1's outcome | Keep v1 as declared; read NO EVIDENCE as uninformative; backcast instead | **Verified** (calculation) |
| 4 | Residual-target gain is mostly ex-ante tilt removal, not learning | A2/A2c T2: both-neutralised difference −0.0002 | **Medium** | "The target is the lever" (FB_REPORT) | Reword as "control style exposure"; prefer ex-ante neutralisation, which is cheaper | **Verified** |
| 5 | Market-neutral verdict rests on 10 bp per unit turnover and unbuffered daily sleeves | A3, A4b: turnover 0.36–0.77/day; break-even 2–14 bp | **Medium** | "No neutral tradable signal" | Cost sweep (2/5/10 bp) and buffered neutral book on existing predictions | **Verified** (accounting) |
| 6 | Survivorship: current constituents traced back | DATA_AUDIT; not quantified | **Medium** for absolute levels and reversal-type signals; lower for matched contrasts | Absolute IC and Sharpe levels; generalisation | Point-in-time membership filter (partial); delisting returns need CRSP-type data | **Potential concern** |
| 7 | Best-arm claim made without a best-of-pool test | A3: SPA p 0.040 vs 0; RC p 0.155; vs reversal16 p 0.55; LOFO +0.0098; shrunk +0.0068 | **Medium** | "lgbm:rich:residual CI above zero" | Report SPA/RC and LOFO for any frozen choice | **Verified** |
| 8 | Same-close execution in labels and both simulators | Code; A2 lag table | **Medium** for 1-day reversal (l1 keeps 51% at next close), **Low** for the residual model (89%) | l1's "only CI > 0 block" | Report next-close and next-open labels by default | **Verified** |
| 9 | Pooled bootstrap treats the 5 years as fixed; between-year heterogeneity is not in the CI | A1: fold-t MDE differs from bootstrap MDE (ridge sector − raw 0.0143 vs 0.0086) | **Low–Medium** | CIs of contrasts with heterogeneous folds | Report fold-level t alongside | **Verified** |
| 10 | Single LightGBM configuration | FB plan; early stopping picked 10–300 rounds | **Medium** for "model class" claims | "Architecture not the bottleneck" | Only worth fixing if the backcast is GO | **Verified** (scope) |
| 11 | Stale zero-volume prices: SW, FERG, AMCR | A3: 3,251 zero-volume cells, 99% with unchanged close; 960 eligible cells 2019–2023 | **Low** | Marginal | Exclude stocks or days with zero volume and an unchanged close | **Verified** |
| 12 | Current GICS sectors used historically (2018 reclassification) | DATA_AUDIT | **Low–Medium** | Sector-demeaned targets and metric | Point-in-time sectors if data are acquired | **Potential** |
| 13 | POWER.md MDE came from a proxy contrast | A1: residual − raw MDE 0.017 vs claimed 0.021 | **Low** | Power statements | Per-contrast power (A1) | **Verified**, minor |
| 14 | Prospective scorer edge cases (crash early in a year; acquired tickers) | FB ledger; code reading | **Low** | v1 run mechanics | Class-1 fixes logged in the addendum | **Potential** |
| 15 | Simulator ignores weight drift and ramps in | FB ledger; matches `trading()` within 0.05 Sharpe | **Low** | Portfolio Sharpe second decimal | None needed for conclusions | **Verified**, minor |
| 16 | Feature cache key omits the screening module hash | FB ledger | **Low** | Reproducibility after code edits | Include it in the key | **Verified**, minor |

---

## Deliverable 3 — Independent statistical assessment (summary; details in A1–A5)

**Recompute.**
- All 18 FB family contrasts match to 3e-8 (pooled) from saved predictions using independent Spearman code.
- Refits are deterministic (3e-9).

**Residual-target advantage:**

| Version | Pooled [95% CI] | Folds positive | Notes |
|---|---|---|---|
| FB original | +0.0140 [+0.002, +0.026] | 5/5 | Fold-level t = 4.1 (4 df, p ≈ 0.015); one-sided sign test p = 0.031 |
| Fixed 50 rounds | +0.0138 | 5/5 | |
| Fixed 300 rounds | +0.0107 | 4/5 | |
| 3-seed mean | +0.0125 | 5/5 | |
| **Leak-free** | **+0.0089 [−0.001, +0.019]** | 4/5 | |
| Leak-free, after ex-ante neutralisation of both | −0.0002 [−0.007, +0.008] | — | |

**Strongest comparisons vs reversal16:**
- frozen spec (50 rounds) +0.0063 [−0.006, +0.019];
- leak-free residual on the style-residual metric +0.0064 [−0.002, +0.015].

**Power, using the actual paired series** (A1):

| Contrast | SE | MDE (5 folds) | 250-day prospective power at θ = 0.006 |
|---|---|---|---|
| Residual − raw | 0.006 | 0.017 | 0.07 |
| Residual − reversal16 | 0.0064 | 0.018 | 0.07 |
| Residual vs 0 | 0.005 | 0.014 | 0.08 |

**Dependence.**
- **Block length.** SE varies by under 15% across stationary blocks 5–40, the circular block bootstrap and Newey–West lags 5–20.
- **Common variation.** 77–89% of the daily IC variance is common day-to-day variation above the 1/√n sampling floor. A larger cross-section would buy little power; more **time** is what is needed.
- **Seeds.** Seed variance (SD 0.0007) is negligible next to date variance (SE 0.0056).

**Multiplicity and selection.**
- **Why a CI can exclude 0 while Holm p > .05.** The CI is per-contrast; Holm controls family-wise error over 4 contrasts.
- **CIs are not multiplicity-adjusted.** Read Holm-failing CIs as unadjusted.
- **The FB families are reasonable, but the programme-level history is not in any family:**
  - more than 2,000 runs, plus FB's own 17-arm pool;
  - the best-arm claim needs SPA (p 0.040 vs 0, 0.55 vs reversal16).
- **Year-to-year consistency** is moderately informative: 1 of 17 pool arms is 5/5 positive. But folds share training data and a pipeline.
- **PBO is 0.05.** This says arm *rankings* persist (style-light arms beat style-heavy ones), not that the winner has tradable skill.

**Equivalence (T1, δ = 0.010).**
- **Ruled out, gains ≥ δ** (90% upper bound below δ):
  - contaminated rich − price (−0.0008);
  - Ridge cs_rank (+0.0087) and volscaled (+0.0028);
  - LightGBM volscaled (+0.0038).
- **Not ruled out:**
  - LightGBM − Ridge (+0.014);
  - MLP − Ridge (+0.021);
  - targets;
  - most representations.

**Practical significance.**
- **The signal.** An IC of about 0.01 equals a top-minus-bottom decile neutral spread of about 0.4% per 5 days for the residual model (A2 section 5).
- **Its daily volatility.** It is large next to that spread, with an IC IR of about 0.13 per day.
- **Execution.** After 10 bp per unit of turnover at the observed turnover, it is not exploitable without turnover control.

---

## Deliverable 4 — Literature comparison

See `A6_LITERATURE.md`. Every citation there was verified by web search; no full texts were read.

**What HyperShift has not tested and the literature supports:**
1. Residual-reversal *signals* (Blitz et al. 2013; Da, Liu & Schaumburg 2014).
2. Turnover-aware (buffered) neutral books (Novy-Marx & Velikov 2016).
3. Survivor-free S&P 500 (Krauss, Do & Huck 2017).
4. Point-in-time firm characteristics at monthly horizons (Gu, Kelly & Xiu 2020).

**What the literature predicts for this setting.** Avramov, Cheng & Metzker (2023) find ML gains concentrated in hard-to-arbitrage stocks, which are absent from a large-cap universe. Krauss et al. (2017) report profits declining in recent years. Both predict what FB found: small ICs on 2019–2023 S&P 500 data.

---

## Part 9 — Information missing from the reports

| Question | Why it matters | Evidence in repo? | Could change a conclusion? | Cost |
|---|---|---|---|---|
| Point-in-time S&P 500 membership 2015–2023 | Survivorship affects levels and reversal signals | No | Yes, for absolute IC and Sharpe | Low (public change lists; verify the source) |
| Delisting returns | Survivorship correction | No | Yes, for levels | High (CRSP-type data) |
| Point-in-time GICS | Sector demeaning | No | Small | Medium |
| A sourced cost model for large-cap execution at the observed turnover | The break-even of 2–14 bp vs the assumed 10 bp | No | **Yes**: decides the neutral-book verdict | Low (literature or broker data) |
| Whether yfinance volume is consolidated or primary-exchange | Liquidity features | No | Small | Low |
| Whether FB's eligibility rule admits pre-listing or stale histories | SW/FERG/AMCR | Partly (A3) | Small | Low |
| Neutral RankIC on next-open execution for all screen arms (v3, X, J) | Same-close bias in earlier studies | Saved predictions exist | Small for model-vs-model contrasts | Low |
| Style-residual RankIC for the screen, X and J arms | Did graph arms differ in stock-specific IC? | Saved predictions exist | Unlikely (Holm 1.00 there) | Low |
| Exact data vintage (download date) of the FB OHLCV | Leak magnitude depends on the fetch date | File mtimes only (2026-09-28, 10-21) | No | — |
| Whether any earlier screening panel feature uses price × volume levels | Leak scope beyond FB | Screening panel has 5 features; not audited | Possibly | Low (code read) |

---

## Deliverable 5 — Prioritised roadmap

| Pri | Action | Question | Minimal design | Baseline / metric | Cost | Outcomes and decision | Type |
|---|---|---|---|---|---|---|---|
| **P0** | Leak-free dollar volume in all future code; v1-LF predeclared | Is any result driven by look-ahead? | Unadjusted Close × Volume; truncation-invariance unit test | — | 0.5 day | Removes a verified defect | Infrastructure |
| **P0** | Co-report style-residual RankIC; predeclare the primary metric | Is "neutral" skill stock-specific? | Per-date OLS on sector + 4 style z-scores at t | — | 1 h | Prevents style exposure being read as skill | Infrastructure |
| **P1** | **Backcast Study BC (2004–2012)** | Does a style-controlled price model have stock-specific IC ≥ 0.008 in untouched years? | `NEXT_STUDY_SPEC.md` | vs 0, style-residual RankIC; MDE 0.0084 | 1 day + 1 h CPU | GO → point-in-time universe + cost-aware prospective; NO-GO → stop price-only work | Quasi-confirmatory |
| **P1** | Cost/turnover sweep on existing predictions | Is the neutral-book failure only costs? | Buffered rule D (enter decile, exit outside 30%); costs 2/5/10 bp; arms: residual, reversal16 | Net Sharpe, break-even | 2 h | Break-even > 10 bp → the market-neutral verdict flips to "cost-dependent"; else confirms | Exploratory (inspected data) |
| **P2** | Point-in-time universe + delisting returns | Survivorship | Membership lists + CRSP/Norgate-type data | — | Days, plus a data licence | Only if BC is GO | — |
| **P2** | Point-in-time fundamentals at a monthly horizon (EDGAR XBRL or vendor) | Do characteristics beyond price add skill? | Gu, Kelly & Xiu-style feature set, monthly, matched Ridge/GBM | Style-residual IC | Weeks | Only if prediction research continues after BC | New programme |
| **P2** | LightGBM family sweep (leaves, depth, min_child) with equal-budget Ridge | Model-class bottleneck | Small grid, nested validation | vs Ridge | 0.5 day | Only if BC is GO | Exploratory |
| **P3** | Identity-based learned hyperedges (synthetic first) | Learned groups | Per-stock embeddings, FB synthetic test | — | 1–2 days | Unlikely to change direction | — |
| **P3 — abandon** | More architectures, horizons, subgroup slicing, Kronos variants, graph variants on this data | — | — | — | — | Cannot change decisions at this power | — |

---

## Part 11 — The next study

See `NEXT_STUDY_SPEC.md` (Deliverable 6).

**Hypothesis.** The frozen FB model class, with leak-free inputs and retrained walk-forward, has positive style-residual RankIC on 2004–2012.

**Design:**
- primary vs 0, threshold 0.008, MDE 0.0084 at about 2,250 days;
- four Holm secondaries, including the residual-reversal scorer and next-close execution;
- negative controls (random score, within-date permuted labels) and a planted-signal positive control;
- GO / NO-GO rules fixed before any data are fetched.

## Part 12 — The prospective experiment

See `PROSPECTIVE_ADDENDUM.md`.

- **v1 is untouched.** The manifest code hashes are verified by test.
- **The leak (D1) is class 2.** It gets a predeclared leak-free sensitivity run, v1-LF, done once at the same time as v1.
- **Low power (D2) is not a validity defect,** but it makes NO EVIDENCE uninformative.
- **No matched raw-vs-residual prospective protocol.** About 18 years would be needed.
