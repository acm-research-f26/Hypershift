# Implementability of S&P 500 filing-text changer avoidance (design facts and synthetic arithmetic)

No returns of the strategy were computed or read.
- **Facts** come from the code at a14fd5d, from the timestamps verified in `tests/test_edgar.py`, and from SEC rules
  (general knowledge, labelled).
- **Volumes and turnover** are synthetic calculations (`tools/turnover_sim.py`) under stated assumptions.

## 1. When filings arrive and when the strategy can act

| Item | Fact or estimate | Basis |
|---|---|---|
| EDGAR acceptance hours | 06:00–22:00 ET on business days. A filing accepted after 17:30 gets the next business day's filing date, but its acceptance time is still exact | SEC rule (general knowledge) |
| Typical acceptance times | AAPL 10-K 2023-11-02 18:08 ET; MSFT 10-Q 2026-01-28 16:07 ET; AAPL 10-Q 2026-07-31 06:01 ET. Large caps often file after the close or before the open, near the earnings release | `test_edgar.py` (verified against index pages) |
| Deadlines, large accelerated filers | 10-K: 60 days after fiscal-year end; 10-Q: 40 days after quarter end | SEC rule (general knowledge) |
| Seasonality (December FYE, most of the index) | 10-K: mid-February to 1 March. 10-Q: late April–early May, late July–early August, late October–early November | Follows from the deadlines |
| Availability in the backtest | First trading day strictly after the acceptance date (`edgar.py:72-74`) | Code |
| Earliest trade | Open of the first formation (first trading day of the month) at least one session after availability (`lazyprices.py:21`) | Code |
| **Delay from acceptance to order** | 2 to about 23 trading days; on average about half a month. A filing accepted on the second-to-last session of a month trades at the next month's open; one accepted on the first day of a month waits about 22 sessions | Arithmetic on the rule |

## 2. Signal volumes (estimate)

- **Filings:** about 500 firms × 4 periodic reports ≈ 2,000 per year.
- **10-K season:** with roughly 80% December FYE (assumption) and filings spread over the 25–60-day window, about
  400 10-Ks over 5 weeks, i.e. ~80 per week (~16 per business day). Real filings bunch near the deadline, so peak
  days are higher.
- **10-Q seasons:** about 400 over 3 weeks, ~130 per week (~27 per day), three times a year.
- **Quiet periods:** close to zero filings outside these four windows (off-cycle filers only).

## 3. What drives turnover

Synthetic calendar with N = 500, an assumed FYE mix of 80% December and 7/7/6% September/June/March, the frozen
`signals_at` / `changers` / `avoid_weights`, flat prices, so turnover is pure signal turnover:

| Persistence of a firm's similarity rank across consecutive filings (ρ) | Overlay one-way turnover (× NAV/yr) | With a 4-month validity window (counterfactual) |
|---|---|---|
| 0 | 2.26 | 2.12 |
| 0.5 | 1.90 | 1.57 |
| 0.8 | 1.58 | n/a |

- **Turnover comes from filing events, not from the monthly rebalance as such.** A firm can only change status when it
  files (about 4 times a year). The monthly formation just batches those changes.
- The holding period does not set turnover: a firm that stays in the bottom quintile across filings stays avoided at
  no cost.
- **The 3-month validity rule adds trading.** At the February formation a December-FYE firm whose Q3 10-Q became
  available in October has expired, but its 10-K is not yet available. Synthetic coverage by formation month is
  0.63 (Feb), 0.84 (May), 0.78 (Aug) and 0.75 (Nov), against about 1.0 otherwise. Changers drop out and re-enter a
  month later, adding about 0.1–0.3× NAV/yr.
- **Cost:** at 10 bp per unit traded, 1.6–2.3× NAV/yr costs **16–23 bp/yr of active return**. The ceiling stated
  before scoring was 30–40 bp/yr gross (`LIT2_SYNTHESIS.md`), so costs can absorb half or more of it. Cost
  sensitivity is the first-order economic risk.
- The equal-weight benchmark trades too (drift reset and index changes). It is charged the same 10 bp, so the active
  return nets out its common part.

## 4. Concentration, capital and liquidity

- **Breadth:** about 100 changers per formation (20% of the names with a signal), fewer in February (synthetic
  minimum 59).
- **Position sizes:** each changer moves from about 0.2% to 0, and its industry peers rise by about 25% of their
  weight. If most of an industry's names are changers, a name can reach several times its benchmark weight (M3 in
  the risk review). Read the realised maximum in EVAL.
- **Liquidity:** all names are S&P 500 members. At $10M NAV a changer trade is about $20k, negligible against typical
  daily dollar volume. Capacity is not binding for any plausible personal or small-fund size.
- **Capital:** none required for the paper trade (implementation only).

## 5. Operating the pipeline

| Component | Cost / volume | Failure risk | Mitigation |
|---|---|---|---|
| Discovery of new 10-K/10-Q filings | One daily EDGAR form index, or about 500 submissions-JSON polls per day at ≤ 8/s (about 1 minute). Free | EDGAR outage or late index; SEC 403 for a missing User-Agent | Retry the next day. The rule already waits ≥ 1 session; reconcile against the daily index |
| Document fetch | About 2,000 documents/yr, 1–10 MB each. Free | Rate limits | 8 req/s with a declared User-Agent (as now) |
| Parsing | Seconds per document | **Filing-agent format changes (C1, I1)** | Alarm when the parse-error rate or median token count per filing agent moves; parse bytes or strip the XML declaration |
| Identity | CIK changes after mergers or reorganisations; dual-class shares; ticker reuse | Wrong or missing signal (`IR` splice, I5) | CIK, not ticker, as the key; dated segments; a new CIK means no prior, so no signal |
| Missing filings | NT 10-K late filers; fiscal-year changes | No signal; the name is held neutrally | Alert if no filing arrives within 5 days after the deadline |
| Compute and storage | Trivial (token counts are stored, not raw HTML) | — | — |

**Immutable records** (required before any paper trade):
- An append-only JSONL with one line per filing: accession, CIK, form, acceptance time (ET), document URL and
  sha256, token count, prior accession, similarity, code commit.
- One line per formation: eligible set, changers, target weights.
- One line per order and simulated fill: open price source.
- Commit the log daily, so git history gives tamper evidence, or keep a hash chain over the lines.
- Never overwrite a past signal; corrections are new lines that reference the old one.

**Long-only vs long-short.**
- Long-only needs no borrow, locate or recall handling, and keeps the benchmark's industry weights.
- The long-short diagnostic would add borrow cost (50 bp/yr assumed), recall risk, a beta hedge and roughly twice the
  traded notional, for a leg (stable reporters) whose alpha reverts quickly in event time (CMN).
- Long-only is the right implementation.

## 6. The three largest obstacles and the cheapest fixes

1. **Extraction fragility across filing agents** (C1 verified; I1 plausible). One vendor's XML declaration silently
   removes a large class of documents, and whitespace conventions shift tokens.
   - Fix now: the C1 patch and a parse-rate alarm per filing agent. Cost: hours.
   - Future spec (not this test): whitespace-insensitive extraction, i.e. join element text with a space.
2. **Costs relative to the edge.** 16–23 bp/yr of overlay costs against a 30–40 bp/yr gross ceiling.
   - Cheapest fix (future spec only): a hysteresis buffer, e.g. stay avoided until the rank rises above 30%, plus
     the 4-month validity window. In the synthetic case that removes the February round trips.
3. **Latency and batching.** Information waits up to about 23 sessions for the monthly formation, while CMN's alpha
   accrues over months, so this matters less than costs.
   - Cheapest fix if it ever matters: a mid-month formation in filing-season months only.

None of these is a reason to change the running test. They matter only if it says CONTINUE.
