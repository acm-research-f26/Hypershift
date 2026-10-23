# D. SEC filing pipeline: what was built and how timing and identity are protected

## Components

| File | Role | Tests |
|---|---|---|
| `src/hypershift/data/edgar.py` | SEC client, acceptance times, availability rule, text extraction, cosine similarity, original-form filter, prior-year matching, SIC→FF12, company-name matching | `tests/test_edgar.py` (30) |
| `src/hypershift/strategy/lazyprices.py` | Formation dates, signal selection with expiry, within-form changers, industry-neutral avoidance weights | `tests/test_lazyprices.py` |
| `src/hypershift/strategy/pit_engine.py` (`run_targets`) | Long-only target-weight engine: next-open fills, drift, costs, delisting liquidation | `test_run_targets_equals_equal_weight_benchmark` |
| `scripts/lazy_cik_map.py` | Dated entity→CIK map with verification (`CIK_MAP.md`) | Name-match tests |
| `scripts/lazy_signals.py` | Resumable download (2015–2023 only); term counts plus provenance; similarity table | — |
| `scripts/lazy_checks.py` | Pre-outcome checks P1, P3–P6 (`PRECHECKS.md`) | — |
| `scripts/lazy_eval.py` | The single predeclared evaluation (`EVAL.md`) | — |

## Timing protections

- **Acceptance time, not period end or filing date.** The SEC submissions JSON labels `acceptanceDateTime` as UTC,
  but the value is Eastern time plus twice the UTC offset (+8 h EDT, +10 h EST). This was verified against
  index-page "Accepted" times on 8 filings across both DST regimes. `edgar.acceptance_et` inverts it; check P1
  re-verifies 12 random filings.
- **Next-day availability.** A filing is usable only from the first trading day *after* its acceptance date
  (`available_index`). An after-close, weekend or holiday filing therefore cannot drive a same-day trade, and an
  early-morning filing waits one day (conservative).
- **Monthly formation.** At the first trading day of month M, only filings available by the last trading day of M−1
  are used, and trades execute at that day's open.
- **Originals only.** Exact form strings `10-K` and `10-Q`; `/A`, `10-KT` and others are excluded, so an amendment
  can never replace what was available at the time.
- **Prior-year comparisons** must be accepted before the current filing, with a report date 330–400 days earlier.
  Fiscal-year changes and new registrants yield no signal.
- **Protected period.** No filing accepted after 2023-12-31 is requested.

## Identity protections

- **Entities** come from the PIT panel (Alpaca `asof` series plus fja05680 membership).
- **CIK by month.** The CIK listed for the ticker in monthly Wikipedia revisions (2016–2024), so re-registrations
  get dated segments (Google→Alphabet, Tyco→JCI, Xerox Holdings, APA, Baker Hughes, TechnipFMC, Ingersoll Rand).
- **Every CIK is verified on SEC data.** The name must match (`edgar.name_match`: equal names, a word subset, or a
  word-aligned prefix). Candidates found by name search must pass a strict rule: equal names or a subset of at
  least two words. And the CIK must have an original 10-K/10-Q accepted within 200 days before at least 90% of the
  member months assigned to it.
- **Defects caught during the build and fixed before any signal was used:**
  - "Dow Inc." → Churchill **Downs** (name search);
  - "Endo" → **Endonovo**;
  - "Paramount" → **Paramount Gold Nevada** (9 months of CBS/ViacomCBS).
- **Result:** 99.8% of member-months mapped. The unmapped entities are AABA (an investment company), SBNY (files
  with the FDIC), WYND, ENDP and UBER (1 month).
- **Today's tickers never decide identity on their own.** The current SEC ticker table is only a fallback
  candidate and must pass the same historical checks.

## Text protections

- **Only the primary document is used.** Hidden inline-XBRL blocks (`ix:header`, `display:none`), scripts and
  styles are removed, as are tables with more than 15% numeric characters (Cohen et al.'s rule).
- **Tokens:** lowercase alphabetic words of 2+ letters; no stemming; stop words kept.
- **Provenance.** Each document is stored as term counts with its URL, the sha256 of the raw bytes and the
  retrieval time. Raw HTML is not kept; it can be re-fetched and checked against the hash.

## Known limitations

- **SIC codes are current, not dated.** FF12 uses each entity's latest CIK.
- **Wikipedia's CIK column is community-maintained.** The SEC checks guard against errors in it.
- **Several delisted entities' CIKs keep filing after their last bar** (Pepco, Southern Co Gas, Level 3,
  Columbia Pipeline, KSU and others), because the subsidiaries keep reporting. Signals from those filings are
  irrelevant: the entity is no longer eligible.
- **The text rule treats boilerplate and substantive text alike**, as in the paper. Check P6 reports how much
  of the measure is document-length change.
