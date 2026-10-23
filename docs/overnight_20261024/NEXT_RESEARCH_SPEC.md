# Next research step: earnings-announcement-return drift (EAR) overlay in the point-in-time S&P 500

**Status: design only, not implemented and not frozen.**
- Use: if the filing-text test STOPs, or in parallel once its scoring is done.
- An executing agent copies the "Frozen rules" section into `docs/strategy_ear_<date>/PREREG.md` and commits it
  **before** computing any forward return. It may fix wording; it must not change rules after seeing an outcome.

## 1. Research question

In point-in-time S&P 500 members from 2016-07 to 2023-12, does a long-only, FF12-industry-neutral overlay beat the
equal-weight PIT S&P 500 after 10 bp costs? The overlay moves weight away from the names with the most negative
market-adjusted reaction to their latest earnings release, and towards the names with the most positive reaction in
the same industry. The release is the 8-K Item 2.02, timed by its EDGAR acceptance time.

## 2. Rationale and evidence (details in `OVERNIGHT_RESEARCH.md`, section C)

- **Mechanism:** limited attention and underreaction to earnings news. The market's announcement-window reaction
  measures the surprise better than accounting SUE (Daniel, Hirshleifer & Sun 2020).
- **Primary evidence:**
  - DHS (NBER WP 24163, FULL TEXT): CAR-sorted PEAD factor 0.65%/month (t 7.91) over 1972–2014; big-firm leg
    0.38%/month; 1991–2014 subperiod 0.55%/month (t 4.59).
  - Dickerson, Julliard & Mueller (arXiv 2604.04430, FULL TEXT): the factor is one of two tradable factors with
    posterior probability > 50% over 1986–2022; its cross-sectional predictability "remains robust and does not
    appear to be driven by micro-cap stocks".
- **Contradictory evidence:**
  - Martineau (2022, FULL TEXT): no SUE drift outside microcaps since 2006.
  - Fink (2021, SECONDARY): no significant PEAD in large, liquid US stocks.
  - Subrahmanyam (SECONDARY): drift factor t 1.43 excluding microcaps, 2001–2024.
  - HyperShift's own finding: 16-day reversal is the strongest price-only signal, and it pushes the other way at a
    one-month horizon.
- **Why this test is informative:** no source shows CAR-based drift *within S&P 500 names after 2010*, so the test
  answers an open question with data already in hand.

## 3. Data (all free; nothing purchased)

| Input | Source | Point-in-time property |
|---|---|---|
| Earnings-release events | EDGAR submissions JSON for each mapped CIK. Form exactly `8-K` with `items` containing `2.02`, plus `acceptanceDateTime`. Already cached by the filing-text pipeline (`data/edgar/json`); read it, never rewrite it. Fetch missing pages only through `edgar.SecClient` (≤ 8 req/s, declared User-Agent) and only when the filing-text download is not running | Exact acceptance time (invert with `edgar.acceptance_et`) |
| Entity ↔ CIK | `data/edgar/cik_map.json` dated segments; join with `lazy_checks.signal_rows`-style segment logic | Dated |
| Prices, membership, eligibility | `data/pit/pit_sp500_2016_2023.npz` via `strategy.pit.load` | PIT; ends 2023-12-29 |
| Industry | FF12 from the CIK's EDGAR SIC (`edgar.ff12`) | Current SIC (known limitation M1) |
| Factors | Ken French daily FF5 + MOM + ST_Rev (`strategy.factors`) | Public |

**Not used:** 2024+ prices, I/B/E/S, Compustat, transcripts, any paid source.

## 4. Frozen rules (to copy into the PREREG)

| Item | Rule |
|---|---|
| Event | Original `8-K` (not `8-K/A`) whose `items` include `2.02`, filed by the entity's CIK inside a dated segment (accepted between 120 days before the segment start and the segment end). Several qualifying 8-Ks within 5 trading days count as one event: the first |
| Event day d0 | Acceptance date (ET) if accepted before 16:00 ET on a trading day; otherwise the next trading day |
| Signal | EAR = Σ over d = d0−1 … d0+1 of (r_i,d − r_m,d). r_m is the equal-weight return of members observed on day d. All three bars must be observed; otherwise no signal. (DHS use days −2…+1 because Compustat dates are imprecise; exact timestamps make −1…+1 sufficient) |
| Availability | Usable at formation F if the index of d0+1 is ≤ F−1, i.e. known at a close before F; traded at the open of F |
| Validity | The latest event whose d0+1 falls on or after the first day of the month 4 calendar months before F's month. Four months, not three, avoids the expiry dips found in `OVERNIGHT_RISK_REVIEW.md` M8 |
| Rank | Percentile of EAR among eligible names with a valid signal. **Bottom** = lowest 20%, **Top** = highest 20% (round(0.2 n), stable sort, ties by entity order) |
| Eligible | PIT member with 64 consecutive observed sessions at the close of F−1 (same as the filing-text test). Entity `IR` from 2017-05-12 to 2020-03-02 is excluded (identity splice, I5), declared before any outcome |
| Benchmark B | Equal weight of eligible names, reset at each F |
| Strategy A (primary, "tilt") | Start from B. In each FF12 industry, set Bottom names to 0 and give their weight pro rata to that industry's Top names. If the industry has no Top name, give it pro rata to its non-Bottom names; if it has no non-Bottom name, keep B. Industry weights equal B's |
| Execution | Monthly at the open of F; weights drift between rebalances (`pit_engine.run_targets`); names leaving the index exit at the next F; delisted names are liquidated at the last close |
| Costs | 10 bp per unit traded for A and B; sensitivities 20 and 30 bp |
| Window | Formations 2016-07 to 2023-12; returns to 2023-12-29 |
| Primary metric | Annualised IR of daily net active return A − B at 10 bp; stationary bootstrap 95% CI (block 21, B = 2000, seed 0) |
| Factor check | Active return on FF5 + MOM + ST_Rev, Newey–West 10 lags; report `n` = number of days used |
| **Decision** | **CONTINUE** if IR ≥ 0.30 **and** alpha t ≥ 1.0; otherwise **STOP**. The same thresholds as the filing-text test keep the two tests comparable |

**Declared diagnostics (non-gating, computed once):**
1. Avoid-only variant (Bottom → all non-Bottom in the industry, as in the filing-text test).
2. Top − B and Bottom − B legs.
3. Per-year active returns.
4. Overlay turnover.
5. Alpha without ST_Rev, which shows the reversal collision.
6. Event-time average market-adjusted return of Bottom and Top from d0+2 to d0+60 (equal weight).

## 5. Pre-outcome checks (no forward returns; must pass before the single scoring)

| ID | Check | Pass |
|---|---|---|
| E1 | `items` is present for 8-K rows in the cached submissions JSON (also in older `files` pages) | Present for ≥ 99% of 8-K rows 2016–2023 |
| E2 | Coverage: eligible member-quarters with an 8-K 2.02 event | ≥ 85% in every year 2016–2023; list the missing names (e.g. firms releasing earnings only with the 10-Q) |
| E3 | Timing sanity: the 2.02 acceptance precedes the same fiscal quarter's 10-Q/10-K acceptance | ≥ 95% of matched quarters; report the median gap |
| E4 | Acceptance-time mix: pre-open (< 09:30), intraday, post-close (≥ 16:00) | Reported (informs the d0 rule); intraday share flagged if > 20% |
| E5 | Signal coverage at formations under the 4-month validity | ≥ 75% mean in every year; report the minimum month |
| E6 | Panel ends ≤ 2023-12-31; no 2024+ date read | Hard assertion in the code (as in `pit.load`) |
| E7 | Tests (synthetic) | d0 before/after the close; availability strictly before F; duplicate 8-Ks collapsed; 8-K/A ignored; segment join; splice exclusion; tilt weights keep industry sums |

## 6. Holding period, costs and power

- **Holding:** a name's status lasts until its next earnings release (about quarterly). Turnover is driven by the
  earnings calendar, as in the filing-text test. Expected overlay turnover is about 2–3× NAV/yr: the tilt trades both
  legs, so 20–30 bp/yr of costs at 10 bp.
- **Minimum economically meaningful improvement:** net active ≥ 0.3%/yr at 10 bp. Below that, implementation risk
  dominates.
- **Power:** 7.5 years gives an IR standard error of about 0.37 and a minimum detectable IR of about 1.0. The joint
  rule has a CONTINUE probability of about 16% / 28% / 64% at a true IR of 0 / 0.15 / 0.5. This is a
  **kill-or-license test, not confirmation.**

## 7. Evaluation sample and what follows

- **2016-07 to 2023-12:** exploratory. This window's prices were used by price-only HyperShift work, but never with
  this event-conditioned signal.
- **If CONTINUE:**
  - If the filing-text strategy has STOPped, its protected window 2024-01 to 2026-09 is released and may be scored
    **once** for EAR under the frozen rules, after a PIT panel extension built with the same code. 2024+ was spent
    only for the Study J arms.
  - If the filing-text strategy is CONTINUE, keep that window for it; EAR confirmation is then prospective only
    (signal dates after the EAR PREREG commit, ≥ 12 months) plus an implementation-only paper trade.
- **Pre-2016:** needs Norgate (not authorised). Not part of this step.

## 8. Stopping rule

- STOP the EAR line if the decision rule fails.
- STOP also if a pre-check fails and cannot be fixed by a dated amendment **before** scoring.
- One scoring only. No variation of window length, quintile cut-off, CAR window or industry scheme after an outcome.
- A STOP closes "large-cap event-conditioned earnings drift on free data". The remaining public-information
  candidates (insiders, 8-K events) have lower power in the S&P 500 (`OVERNIGHT_RESEARCH.md` ranking), so a STOP
  here argues for pausing S&P 500 signal research until a broader point-in-time universe exists.

## 9. Engineering (about 1 working day; no bulk downloads)

1. `src/hypershift/strategy/ear.py`:
   - `events_from_submissions(sub)`: form, items, acceptance;
   - `event_day(acc_et, cal)`;
   - `ear(panel, i, d0)`;
   - `signals_at` (reuse `lazyprices.signals_at` with `months=4`);
   - `tilt_weights(eligible, bottom, top, industry)`.
2. `tests/test_ear.py` covering E7.
3. `scripts/ear_checks.py` (E1–E6, no forward returns) and `scripts/ear_eval.py` (mirror `lazy_eval.py`; output
   `EVAL.md`).
4. Order of work: PREREG commit → code and tests → checks → amendment if needed → single scoring.
