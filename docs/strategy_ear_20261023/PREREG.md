# D. Predeclared exploratory test: earnings-announcement-return (EAR) overlay in the PIT S&P 500 (frozen 2026-10-23)

**Status.** Committed before any return after a signal's availability is computed. Evidence and pre-outcome checks:
`TIMING_AUDIT.md`, `DATA_CHECKS.md`, `FEASIBILITY.md`, `LITERATURE_DECISION.md`. Base design:
`../overnight_20261024/NEXT_RESEARCH_SPEC.md`. Its departures are listed at the end and were all decided before any
outcome.

## Hypothesis and sample status

- **H:** among point-in-time S&P 500 members, the market's reaction to an earnings release (market-adjusted return
  over the release window) predicts the next months' industry-relative return. An equal-weight portfolio tilted
  from the worst to the best reactors, within industries, beats the equal-weight benchmark after costs.
- **Central question:** the reaction must carry information beyond ordinary short-horizon returns. The matched
  control (below) has to lose to the event-conditioned tilt.
- **Mechanism:** limited attention and underreaction to earnings news (literature in `LITERATURE_DECISION.md`). The
  event reaction is a better surprise measure than accounting SUE.
- **Sample:** 2016-07 to 2023-12 is exploratory. HyperShift used these years for price-only work, never with this
  signal. 2024-01 to 2026-09 is protected and is not touched by this test. Pre-2016 would need paid data and is out
  of scope.
- **Power:** IR standard error ≈ 0.37 over 7.5 years. This is a **kill-or-license test, not confirmation**.

## Signal (frozen)

| Item | Rule |
|---|---|
| Event | Original `8-K` (form exactly `8-K`) whose `items` include `2.02`, filed by the entity's CIK inside one of its dated segments (accepted between 120 days before the segment start and the segment end). Item 2.02 8-Ks within 5 trading days of a kept one are the same event; the first is kept. 8-K/A is ignored |
| Event day d0 | Trading day of SEC's official **filing date** (the next trading day if the market was closed). The acceptance clock is **not** used (inconsistent time zone; `TIMING_AUDIT.md`) |
| EAR | Σ over d = d0−1, d0, d0+1 of (r_i,d − r_m,d). r is the cleaned total return; r_m is the equal-weight return of members observed on both days. Every bar d0−2..d0+1 must be observed, otherwise no signal |
| Availability | After the close of d0+1. Usable at formation F only if d0+1 ≤ F−1; traded at the open of F |
| Bin | Fixed once, at availability. −1 if EAR ≤ the 20th percentile, +1 if EAR ≥ the 80th, else 0. Percentiles come from all finite EARs of mapped S&P 500 entities that became available in the 252 trading days before (causal); no bin with fewer than 100 such EARs (`ear.event_bins`) |
| Validity | A name's state is the bin of its latest event whose availability day falls on or after the first day of the month four calendar months before F's month. A later event replaces it, even one without a computable EAR (then no bin) |
| Industry | FF12 from the SIC in the EDGAR header of the CIK's earliest Item 2.02 filing in the window, valid from that filing date; otherwise 12 |
| Eligible | PIT member with 64 consecutive observed sessions at the close of F−1. **Entity `IR` is ineligible before 2020-03-03** (identity splice) |

## Portfolio (primary: long-only tilt overlay)

| Item | Rule |
|---|---|
| Benchmark B | Equal weight of eligible names, reset monthly |
| Strategy A | In each FF12 industry, the weight of bin −1 names moves pro rata to the industry's bin +1 names. If there are none, or once they reach the cap, it moves to its other non(−1) names. **No name may exceed 2 × its B weight**; weight that cannot move stays on the −1 names. Industry weights equal B's (`ear.tilt_weights`, cap 2) |
| Rebalance and execution | Monthly at the open of the first trading day of each month; weights drift between rebalances; names leaving the index exit at the next rebalance; delisted names are liquidated at the last close (`pit_engine.run_targets`) |
| Costs | 10 bp per unit of traded notional for A, B and C; sensitivities 20 and 30 bp. No borrow (long-only) |
| Window | Formations 2016-07 to 2023-12; returns to 2023-12-29 |

## Matched price-only control (C)

C is constructed exactly like A, with the same:
- events, availability day (d0+1) and validity;
- binning rule (its own trailing breakpoints);
- industries, cap, eligibility, rebalancing and costs.

The one difference is the measured window: **d0−4..d0−2**, the three sessions just before the event window. C
therefore has the same timing, turnover pattern and exposure limits, with information that is not the earnings
reaction. A − C measures what the earnings event adds beyond ordinary short-horizon returns of the same age.

## Metrics

- **Primary:** annualised IR of daily net active return A − B at 10 bp, with a stationary bootstrap 95% CI (block 21,
  B = 2000).
- **Factors:** A − B on FF5 + MOM + ST_Rev, Newey–West 10 lags.
- **Incremental:** IR of A − C at 10 bp, with its CI and factor alpha.
- **Reported:** costs at 20 and 30 bp; turnover of A, B and C; per-year A − B and C − B; maximum drawdown; worst
  year; exposures; coverage and bin sizes.

## Decision rule (frozen)

| Outcome | Rule | Action |
|---|---|---|
| **CONTINUE** | IR(A − B) ≥ 0.30 **and** alpha t ≥ 1.0 **and** IR(A − C) > 0 **and** net active return ≥ 0.30%/yr at 10 bp (economic bar) | Not validation. Licenses one scoring of the protected 2024-01..2026-09 window (after extending the PIT panel with the same code) and an implementation-only paper trade. No data purchase, no capital |
| **STOP** | Otherwise. This includes a primary pass with IR(A − C) ≤ 0: the event then adds nothing beyond price-only short-horizon returns, a direction already closed | Close the S&P 500 EAR overlay. No other windows, bin cut-offs, holding rules or industry schemes on 2016–2023 |

**No INCONCLUSIVE band.** Any result short of CONTINUE maps to STOP. Nothing affordable could resolve a middling result: pre-2016 needs paid data, which this prior does not justify, and the protected window is released only by a CONTINUE.

**Operating characteristics** of the primary IR ≥ 0.30 condition alone: about 21% / 34% / 70% at a true IR of
0 / 0.15 / 0.5. The joint rule is stricter.

## Departures from `NEXT_RESEARCH_SPEC.md` (all decided before any outcome)

1. **d0 from the filing date, not the acceptance clock.** The clock is unreliable (`TIMING_AUDIT.md`). The window
   d0−1..d0+1 covers both possible reaction days.
2. **Event-time bins against trailing breakpoints** replace a monthly cross-sectional re-rank. Names change state
   only at their own earnings, which removes rank churn. The pre-drift overlay turnover estimate is about 3× NAV/yr
   (`FEASIBILITY.md`).
3. **Cap of 2 × B weight per name.** Without it, a few top names in an industry with many bottom names reached
   10 × B (`FEASIBILITY.md`).
4. **Dated SIC** from filing headers replaces current SIC.
5. **Matched control:** the pre-event window d0−4..d0−2 replaces the overnight spec's placebo window ending at F−1.
   That placebo would refresh every name monthly, so it had a different age and a higher turnover, and the request
   asks for a control that is not mechanically disadvantaged.
6. **The decision rule adds IR(A − C) > 0 and the economic bar** (net ≥ 0.30%/yr at 10 bp, from the overnight spec §6). The test is about event information, not price momentum.

## Amendments

(none)
