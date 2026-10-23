# F. EAR overlay: official decision (2026-10-23)

**Decision A: STOP (S&P 500 EAR overlay).** The single predeclared evaluation (`EVAL.md`; spec frozen at 35ae347
before any forward return) fails every CONTINUE condition:

| Condition (frozen) | Required | Result |
|---|---|---|
| Net active IR (A − B), 10 bp | ≥ 0.30 | **−0.47** [−1.11, +0.14] |
| Active FF5+MOM+ST_Rev alpha t | ≥ 1.0 | **−1.44** (−0.50%/yr) |
| Event information beyond the matched control, IR(A − C) | > 0 | **−0.08** [−0.72, +0.51] |
| Net active return, 10 bp | ≥ 0.30%/yr | **−0.52%/yr** |

## The five questions

1. **Executable without look-ahead?** Yes, after the pre-outcome redefinition. d0 is SEC's filing date, the score is
   complete at the close of d0+1, and the trade is at the next monthly open. Bins come from strictly earlier EARs
   and the SIC is dated (`TIMING_AUDIT.md`, `DATA_CHECKS.md`).
2. **About earnings information rather than generic price behaviour?** The matched control answers it. The
   event-conditioned tilt does no better than the same construction on the three pre-event sessions (A − C IR −0.08).
   In this universe and period, the earnings reaction carried no forward-return information beyond ordinary
   3-day returns.
3. **Economically worthwhile?** No. The realised overlay turnover is +2.3× NAV/yr, about 0.23%/yr at 10 bp, on top of
   a negative gross edge (about −0.3%/yr). The pre-test ceiling (+0.15–0.2%/yr net) was already thin.
4. **Value beyond the matched control?** None measurable (A − C −0.09%/yr, alpha t −0.33).
5. **What next?** See `ALTERNATIVES.md`: stop large-cap event-signal research on 2016–2023, and next test where the
   literature locates these effects, a broader point-in-time universe.

## Exploratory observations (not part of the verdict)

- **Both tilts lost.** EAR (A − B −0.52%/yr) and control (C − B −0.43%/yr) both lost, in five and six of the eight
  years respectively. In 2016–2023 S&P 500 names, buying recent industry-relative 3-day winners and selling losers
  lost money: a short-horizon reversal signature, consistent with Dai et al.'s point that reversal and PEAD trade
  against each other.
- **Factor exposures are small.** MOM +0.03, ST_Rev −0.01; R² 0.25. The losses are not explained by factor tilts.
- **Not a basis for a reversal strategy.** Price-only reversal was already tested (Study FB, RR21) and stopped on
  costs.

## What a STOP does and does not mean

- It closes the **S&P 500, monthly, long-only EAR tilt** on free data. It does not show that all earnings-related
  information is worthless.
- It is consistent with Martineau's finding that large-cap announcement prices are close to martingale after
  2006, and with the size dependence of the effect (DHS: big leg 0.38%/month against 0.94 for small; Dai et al.).
- No variant (other windows, cut-offs, holding rules, event subsets, industries) will be run on 2016–2023.
- The protected 2024-01..2026-09 window remains unspent.
