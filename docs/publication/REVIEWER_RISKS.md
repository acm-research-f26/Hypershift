# D. Reviewer-risk assessment

Ranked by how likely each objection is and how much damage it could do.

| # | Likely objection | Severity | Defensible response | What would strengthen it |
|---|---|---|---|---|
| 1 | **"You did not reproduce THINK because your implementation differs. The authors' code was unavailable."** | High | We agree, and say so. All unstated details are listed as inferred. We show the published numbers *can* be reached under test-informed protocols and that shuffled labels match them [T5]. We make no claim about what the authors did. Corrected settings also gave no advantage [T3]. | Contact the authors for the selection rule and Sharpe definition (an email draft exists in `docs/phase2/EMAIL_AUTHORS.md`). Report any reply. |
| 2 | **"Single test year (2017) for the reproduction."** | High | 2017 is the paper's test year. We add frozen 2018–2023 and walk-forward 2019–2023 out-of-sample tests [T6] and per-year results. | None cheap. State it plainly. |
| 3 | **"Everything is exploratory: all years were inspected."** | High | True. The paper is framed as mechanisms plus diagnostics. Each mechanism is shown by a controlled contrast (planted signal, algebraic decomposition, synthetic design) that does not depend on a selected year. The prospective test is predeclared but unscored. | Score the prospective test when due, and report it in a later version. |
| 4 | **"The dividend-leak result is dataset-specific, and its IC depends on the fetch date."** | Medium | The mechanism is algebraic and general. Magnitude is reported as dataset-specific, and we give the decomposition so others can measure it. | Repeat the decomposition on one other vendor or universe, if licence terms permit. |
| 5 | **"The learner-power result is from one synthetic design with three replicates. Gradient boosting is not weak."** | Medium–High | We claim only that **this configuration's** matched nulls are uninformative at the relevant effect size, and present it as a requirement to check, not a general weakness of GBMs [M2]. | More replicates and seeds; a sweep of signal-to-noise; report the configuration where power appears. This is a *methods* simulation, not a market experiment. It is the one addition we recommend before submission (see `RECOMMENDATION.md`). |
| 6 | **"Residualised metrics failing under nonlinear signals is obvious."** | Medium | The algebra is simple, but such metrics are used in practice as "beyond-style" evidence. We show the size of the effect in a real benchmark and supply the matched-learner control. | Cite practice examples carefully, or drop the novelty claim and keep it as a cautionary demonstration. |
| 7 | **"Clark–West does not apply to RankIC."** | Medium | We say exactly that. The rank-IC analogue is labelled an unvalidated adaptation [M4]. | Use an MSPE-based nested test for the formal comparison, or provide a simulation study. |
| 8 | **"The economic analysis lacks borrow, impact and point-in-time data."** | Medium | Economics are descriptive. We claim no investable result and no attribution of the characteristics-only books [E2, UT4, UT5]. | A style-factor attribution regression on saved book returns (cheap, uses existing outputs). Flag it as optional. |
| 9 | **"Survivorship bias invalidates S&P 500 results."** | Medium | It affects levels and generalisation. Our main claims are contrasts within the same universe, mechanisms and diagnostics. Coverage numbers are reported as symbol-match counts [U1]. | Point-in-time data. Out of scope. |
| 10 | **"Too many studies; garden of forking paths."** | Medium | We report selection-adjusted statistics (LOFO, SPA, RC, CSCV) [P3], predeclared thresholds, a ledger of every decision, and errata. We do not count runs as evidence. | Publish ledgers and plans as supplements. |
| 11 | **"Self-audit by the same team (and assistant). Errors were found late."** | Medium | We document our own errors as errata: a confounded placebo comparison, a committed syntax error, wrong coverage bounds. The headline numbers were recomputed from saved outputs (18/18 match). | An independent re-run by a third party from the released repository. |
| 12 | **"Weight-decay collapse is just known AdamW behaviour."** | Low–Medium | Decoupling is known (Loshchilov & Hutter, 2019). Our contribution is the size of the effect for stacked Poincaré layers and its consequence for reproducing nulls. | Present it briefly, as a case. |

## Claims to exclude or soften before submission

- Anything implying the authors used test-epoch selection or a particular protocol. Write "can be reached by", and never "was obtained by".
- "Returns are unpredictable", "price-only prediction is impossible" and "features carry no information".
- "The characteristics-only books earn known style premia." Not attributed.
- Any rank-IC Clark–West statistic presented as formally valid.
- "Coverage is at most / at least 43–58%."

## Added in revision 2

| # | Likely objection | Severity | Response |
|---|---|---|---|
| 13 | "Back-adjustment look-ahead is already known." | High for a standalone note | We agree; it is credited in §2 and `PRIOR_ART.md`. Our contribution is the explicit future-price derivation, the measured association, and the benchmark effect |
| 14 | "L_path predicts returns because it is built from future prices: trivially mechanical." | Medium | Yes, that is the mechanism by construction. The empirical questions are how much of the *feature* (L, dominated by L_div) carries (+0.019), and how much it moved conclusions (about 0.005). We do not present +0.094 as a discovery about markets |
| 15 | "The regression-residual 'future-only' claim was wrong." | — | Corrected in draft 2 (L2b) |
| 16 | "'Leak-free' was overstated (full-series normalisation, survivorship)." | — | Terminology replaced; per-experiment status in `LEAKAGE_AUDIT.md` |
