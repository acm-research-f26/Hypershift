# C. Literature and the economic hypothesis (2026-10-23)

Verification with access levels: `LIT_EAR_VERIFY.md` (delegated agent). Two numbers were re-checked against the
PDFs: Dai et al. Table A2, and DHS's definition (via the agent's page references).

## What the proposed signal is, and how it relates to the papers

| Concept | Definition | Not the same as |
|---|---|---|
| Earnings surprise (SUE) | Actual EPS minus expected (analyst or seasonal random walk), scaled | The price reaction |
| **Announcement-window return (CAR/EAR)** | Market-adjusted return over the release window; DHS use days −2..+1 around Compustat RDQ, Dai et al. −1..+1 | Post-announcement drift |
| Post-announcement drift | Return *after* the window, conditional on the surprise or the CAR | The window return itself |

**Our signal is the DHS / Dai et al. measure, with three implementation differences:**
- the event date is SEC's 8-K Item 2.02 filing date, not Compustat RDQ;
- the window is −1..+1, as in Dai et al., rather than DHS's −2..+1, because exact filing dates make the earlier day
  unnecessary (`TIMING_AUDIT.md`);
- the portfolio is an industry-neutral long-only tilt with event-time bins, not a monthly value-weighted
  long-short sort.

It is a faithful *signal* and a modified *portfolio*.

## Evidence

| Source | Access | Finding | Bearing on the S&P 500 overlay |
|---|---|---|---|
| Daniel, Hirshleifer & Sun (NBER WP 24163 → RFS 2020) | FULL TEXT (WP) | CAR[−2,+1] PEAD factor: 0.65%/month (t 7.91), 1972–2014. Big-firm leg 0.38%/month, small 0.94. 1991–2014: 0.55%/month (t 4.59) | Supports CAR over SUE. Its big leg is above the NYSE median, not the S&P 500, and the sample ends in 2014 |
| **Dai, Medhat, Novy-Marx & Rizova (NBER WP 30917)** | FULL TEXT (WP), **Table A2 re-checked** | Value-weighted CAR[3-day] quintile spread, NYSE breaks: **0.36%/month [t 2.09], May 2001–Dec 2021**; 0.53 (t 5.45) over 1973–2021. Size is the strongest determinant (appendix figure, not read) | The best post-2001, large-cap-weighted support. Positive, but weaker, and not S&P 500-only |
| Dickerson, Julliard & Mueller (arXiv 2604.04430, 2026) | FULL TEXT (arXiv) | The DHS factor prices stock and bond cross-sections through 2022; "not driven by micro-cap stocks" (claim in an appendix that was not read) | A factor-pricing result, not large-cap tradability. JFE status not verified |
| **Martineau (2022, CFR)** | FULL TEXT | SUE drift gone for non-microcaps since 2006. The one announcement-return test: slope of BHAR[0,60] on BHAR[0,1] falls from about 1.6 to **about 1 for non-microcaps** (no continuation) | Direct evidence *against* drift after an observed reaction in large stocks |
| Subrahmanyam (2025, SSRN) | SECONDARY | Announcement-return factor t 2.18 with all stocks, **1.43 without microcaps**, 2001–2024 | Against (weak) |
| Brandt, Kishore, Santa-Clara & Venkatachalam (2007/08 WP) | FULL TEXT (2007 version) | EAR drift is largely a repeat reaction at later announcements; large firms get about 35–55% of the effect; sample to 2004 | Mixed; it implies holding through the next announcement |
| Chen & Welch (arXiv 2607.06502, 2026) | ABSTRACT | Median anomaly spread 7 bp/month post-2005 in non-micro stocks | Strong general prior of decay |
| Dai et al. on reversal | FULL TEXT | Raw one-month reversal is **short PEAD** (loading −0.54). Announcement CARs drift and do not revert | The matched control and ST_Rev attribution are informative |
| Mechanism: DellaVigna & Pollet (2009); Hirshleifer, Lim & Teoh (2009) | See `LIT_EAR_VERIFY.md` §7 | Distracted or inattentive investors underreact at announcements | Plausible for retail-heavy names, weaker for S&P 500 |

## Mechanism, and when it fails

- **Why it could work.** The window return captures the market's *initial* reading of the earnings news,
  including guidance, revenue and margins, which SUE misses. If attention is limited, the price completes its
  adjustment over the following weeks. The drift should then continue in the direction of the reaction.
- **When it fails:**
  1. Announcement-day prices are already efficient, as Martineau's slope near 1 for large stocks suggests.
  2. Liquidity-driven overreaction in the window partly reverts. This is the short-term reversal channel,
     strongest in illiquid names and high-VIX periods.
  3. Costs: the reaction has no persistence, so state changes every quarter for most binned names.

## Implication

- **The prior for the S&P 500 is weak.** The literature supports CAR-based drift as a broad-market,
  size-pooled factor. The most recent value-weighted evidence is positive but smaller (0.36%/month long-short,
  t 2.09). The best large-cap test (Martineau's regression) shows no continuation.
- **The expected net active return of our overlay is about +0.1–0.2%/yr at 10 bp**, below the economic bar
  (`FEASIBILITY.md` gate).
- **The test is worth running only as a cheap kill test.** The infrastructure exists. It answers an open question
  with direct data. Its result decides whether large-cap event drift stays on the agenda.
