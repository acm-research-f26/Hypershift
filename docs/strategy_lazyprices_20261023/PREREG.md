# Predeclared exploratory test: filing-text changer avoidance in the S&P 500 (frozen 2026-10-23)

**Status.** Committed before any filing text is downloaded in bulk and before any return of this strategy is
computed. Every later change needs a dated amendment below and is reported as a deviation.

## Hypothesis and what this test can and cannot show

- **H (narrow):** in point-in-time S&P 500 members, firms whose latest 10-K/10-Q differs most from the same report a
  year earlier ("changers") subsequently underperform their industry peers. An equal-weight S&P 500 portfolio that
  avoids them therefore beats the equal-weight benchmark after costs.
- **Economic mechanism:** investor inattention to subtle disclosure changes (Cohen, Malloy & Nguyen 2020).
- **Prior:** in-sample, the effect was **insignificant in S&P 500 stocks** (Cheng, Liu, Qiao & Wang, JFQA 2026,
  1996–2017). The realistic expectation is a net active IR of about 0.1–0.2 (`LIT2_SYNTHESIS.md`).
- **Sample labels:**
  - 1995–2014 is the literature's discovery sample;
  - **2016–2023 is HyperShift exploratory data, partly post-publication** (the working paper was circulated in 2010; the JF paper appeared in 2020);
  - 2024-01 to 2026-09 is **protected** (unscored for this strategy; not fetched now);
  - observations after this freeze are prospective.
- **Power.** Seven and a half years give an IR standard error of about 0.37 and a minimum detectable IR of about 1.0.
  This is an exploratory **kill-or-license test, not confirmation**. Operating characteristics of the CONTINUE rule
  (IR ≥ 0.30) at true IR 0 / 0.15 / 0.5 are about 21% / 34% / 70%. A false CONTINUE costs one look at the protected
  window plus an implementation-only paper trade.
- **A STOP closes only this implementation** (S&P 500 changer avoidance). It is not a verdict on filing text in
  general.

## Signal (frozen)

| Item | Rule |
|---|---|
| Forms | Original `10-K` and `10-Q` only (exact form strings). Excluded: `/A` amendments, `10-KT`, `10-K405`, `10-KSB`, and anything filed under another form. An amendment never replaces the original |
| Document | The filing's `primaryDocument` (main 10-K/10-Q body). Exhibits and attached files are not used |
| Comparison | The same CIK and the same form, original, accepted before the current filing, with report date (period end) between 330 and 400 days before the current report date: the prior-year 10-K, or the same fiscal quarter's 10-Q a year earlier. If several match, the latest accepted. No match (new registrant, fiscal-year change, CIK change after a merger) gives **no signal** |
| Text extraction | Parse HTML with lxml. Remove `ix:header` blocks and every element styled `display:none` (hidden inline-XBRL facts), plus `script` and `style`. Remove every `table` whose text has numeric characters above 15% of letters plus digits (Cohen et al.'s rule). Take the remaining text, unescape entities, and lowercase |
| Tokens | `[a-z]{2,}` (alphabetic, 2+ letters). Numbers and single letters dropped. No stemming. Stop words kept (as in the paper's main results) |
| Measure | **Cosine similarity of raw term-frequency vectors** (the paper's first measure). No other measure, section, encoder or tokenisation is computed for the decision |
| Timing | Acceptance time in US/Eastern, from the submissions JSON with SEC's double-converted timestamp inverted (verified against index-page "Accepted" times). A filing becomes **available on the first trading day strictly after its acceptance date (ET)**, never at the same day's close |
| Validity | At a formation date F (first trading day of a month), a firm's signal is its latest original 10-K/10-Q available on or before the last trading day before F, provided it became available within the three calendar months before F's month. Otherwise the firm has no signal |
| Rank | Among eligible firms with a signal, percentile rank of similarity **within form type** (10-K signals and 10-Q signals ranked separately), to avoid a seasonal form tilt. **Changers = bottom 20%** in each form |

## Identity

- **Entities** come from the PIT panel (`data/pit/pit_sp500_2016_2023.npz`).
- **CIK mapping** (`CIK_MAP.md`, verified before scoring):
  1. current SEC ticker→CIK for the entity's tickers;
  2. otherwise company name to EDGAR.

  A CIK is accepted only if its 10-K/10-Q filings cover the entity's membership span. Delisted entities' CIKs must
  stop filing near the last bar; one that keeps filing is flagged as an identity break.
- An unmapped entity has no signal and is held neutrally.

## Portfolio (primary: long-only avoidance)

| Item | Rule |
|---|---|
| Eligible | PIT S&P 500 member and 64 consecutive observed sessions at the close before F |
| Benchmark B | Equal weight of all eligible names, reset at each F |
| Strategy A | B's weights, with changers set to 0 and their weight redistributed pro rata to the non-changers **in the same Fama–French 12 industry** (from the CIK's EDGAR SIC code; unmapped names go to "Other"). Industry weights therefore equal B's. Maximum weight per name is about 1.25 × B's. If an industry has no non-changer, its changers keep their weight |
| Rebalance and execution | Monthly at the **open of F**, with weights drifting between rebalances (`pit_engine`), next-open fills. A name that leaves the index exits at the next rebalance; delisted names are liquidated at their last close |
| Costs | 10 bp per unit of traded notional for both A and B. No borrow (long-only). Sensitivities 20 and 30 bp |
| Window | Formations 2016-07 to 2023-12; returns to 2023-12-29 |
| Primary metric | **Annualised IR of daily net active return (A − B) at 10 bp**, with a stationary bootstrap 95% CI (block 21 days, B = 2000) |
| Factor check | Active return on FF5 + MOM + ST_Rev, Newey–West 10 lags |

## Decision rule (primary)

| Outcome | Rule | Action |
|---|---|---|
| **CONTINUE** | Net active IR ≥ 0.30 **and** active FF alpha t ≥ 1.0 | Not validation. Licenses (i) one scoring of the protected 2024-01 to 2026-09 window under this frozen spec, and (ii) an implementation-only prospective paper trade. No data purchase is triggered |
| **STOP** (this implementation) | Otherwise | Close S&P 500 changer avoidance. The broader-universe question remains a separate INVESTIGATE item |

## Declared diagnostics (non-gating; computed once)

1. **Leg decomposition:** equal-weight changers minus B, and equal-weight top-quintile non-changers minus B
   (industry-neutral as in A).
2. **Long-short diagnostic:** equal-weight non-changers (top 20% within form) minus changers. Monthly, 10 bp costs,
   50 bp/yr borrow, unhedged; its beta is reported.
3. Costs at 20 and 30 bp; realised turnover of A and of the active part (traded notional per year); maximum active
   weight; per-year active returns; active factor exposures.

## Pre-outcome checks (text and mechanics only; no returns)

These must pass before the single scoring run. A failure is fixed by amendment before scoring, or stops the test.

| ID | Check | Pass condition |
|---|---|---|
| P1 | Timestamp inversion on a sample of index pages | Exact match |
| P2 | CIK map coverage and verification | At least 90% of eligible member-months mapped; every mapped CIK passes the filing-span check; identity breaks listed |
| P3 | Signal coverage | At least 75% of eligible member-months have a valid signal in each year 2017–2023 |
| P4 | Extraction quality | No primary document below 1,000 tokens left unexplained; similarity distribution by year and form reported |
| P5 | Inline-XBRL transition | Mean similarity of each firm's first inline-XBRL filing within 1 SD of the other filings; otherwise fix extraction |
| P6 | Length artefact | Report correlation of similarity with abs(Δ log token count). Informative only, but a correlation stronger than −0.8 means the measure is mostly length: fix by amendment before scoring |
| P7 | Fixtures | Sample covering an amendment (ignored), a fiscal-year change (no comparison), a renamed company (same CIK) and a delisted company (filings stop) |

## Amendments

**A1 (2026-10-23, after the first pre-outcome check run and before any return was computed).** The first
`PRECHECKS.md` failed P1, P3, P4 and P5.
- **Extraction bug (cause of P3–P5).** 4,096 documents raised lxml's "Unicode strings with encoding declaration are
  not supported" and were stored with zero tokens, giving similarity 0. These were mostly inline-XBRL filings from
  2019–2021, whose files begin with `<?xml … encoding=…?>`.
  - **Fix:** the declaration is stripped before parsing (`edgar.extract_tokens`; test
    `test_extract_tokens_handles_xml_encoding_declaration`).
  - Failed or empty documents are re-fetched and re-parsed.
  - The text rules themselves are unchanged.
- **P1 time zone.** SEC's JSON `acceptanceDateTime` is inconsistent: some filings are double-converted (+8/+10 h),
  others are correct UTC. So the inverted clock time matched the index page for 8 of 12 sampled filings, and was
  4–5 h early for the other 4.
  - Only the **availability day** enters the signal. EDGAR accepts filings only between 06:00 and 22:00 ET, so a
    4–5 h early clock time never moves the date earlier.
  - **Added guard:** a filing is also never available before the trading day after SEC's official `filingDate`
    (`available_index(..., filing_date)`, tested).
  - **P1's pass condition is restated as an exact match of the availability day** with the one computed from the
    index-page time. Clock times are still reported.
- **Cost if wrong:** none for timing (the guard is more conservative). The extraction fix changes which documents
  have text. Every outcome is still uncomputed.

**A2 (2026-10-23, before any return was computed). From the independent pre-results review
(`../Hypershift-overnight/docs/overnight_20261024/OVERNIGHT_RISK_REVIEW.md`).**

| Item | Problem | Resolution |
|---|---|---|
| I3: implementation deviated from this spec | When a firm's latest filing had no prior-year comparison, `signals_at` silently used an older filing still inside the window. This spec says such a filing gives **no signal** | Fixed: every 2016–2023 original filing enters the signal table, with NaN similarity when it has no comparison or its document failed, and the latest filing always wins (test `test_latest_filing_without_comparison_gives_no_signal`) |
| I2: check hygiene | P4–P6 included failed rows | These checks are now computed on valid comparisons only, and the counts of no-comparison and failed filings are reported |
| I1: formatting | Text between adjacent HTML blocks is joined without a separator, so a template or filing-agent change can move tokens. The measured effect is small (same-document cosine 0.9997–0.9999) | The frozen text rule is **unchanged**. A diagnostic is reported next to the verdict: the share of filing-agent switches among changer pairs compared with all pairs |
