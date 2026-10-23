# Pre-results review of the filing-text changer-avoidance test (2026-10-24, overnight)

Scope: code and documents only, at commit a14fd5d (`PREREG.md`, `edgar.py`, `lazyprices.py`, `pit_engine.py`,
`study.py`, `factors.py`, `scripts/lazy_{signals,checks,eval,cik_map}.py`). No outputs were read: not `EVAL.md`,
`PRECHECKS.md`, `signals.csv`, token files or `results/`. Two inputs were read: the PIT panel's membership and raw
closes for a few entities (identity checks), and four public 10-K primary documents fetched from SEC (8 requests in
total). The main checkout's copies of the reviewed files are identical to a14fd5d (CRLF ignored).

Classes:
- **CRITICAL**: a verified defect that can change the economic conclusion; fix it before interpreting.
- **IMPORTANT**: a verified deviation, or a plausible material risk; run the diagnostic before interpreting.
- **MINOR**: document it; no action needed for this verdict.

## Summary

| ID | Class | Issue | Verified | Action before interpreting |
|---|---|---|---|---|
| C1 | **CRITICAL** | DFIN inline-XBRL documents raise in `extract_tokens`, so the pair gets no signal | Yes: 2 real Microsoft 10-Ks | Apply the patch, re-fetch only the errored documents, rerun the checks |
| I1 | IMPORTANT | Tokens depend on whitespace between HTML block elements, so a filing-agent or template switch can look like a text change | Mechanism yes; size on changers no | Agent-switch share among changers |
| I2 | IMPORTANT | P4–P6 include errored rows (similarity 0, 0 tokens) | Yes (code) | Recompute P4–P6 without them |
| I3 | IMPORTANT | The signal falls back to an older filing when the latest one has no comparison (PREREG says no signal) | Yes (test) | Count the fallbacks |
| I4 | IMPORTANT (interpretation) | Alpha t ≥ 1 is the binding gate; PREREG's operating characteristics cover IR only | Arithmetic | None; read verdicts with this in mind |
| I5 | IMPORTANT (panel reuse) | Entity `IR` carries Gardner Denver prices while its membership and CIK are Ingersoll-Rand plc (2017-05-12 to 2020-03-02) | Yes (panel) | None for this verdict; fix before reusing the panel |
| M1–M7 | MINOR | Current (not dated) SIC; dual-class double rows; 1.25× not a cap; leg diagnostic not industry-neutral; subsidiary name matches; silent factor-date truncation; Exhibit-13 10-Ks | Mixed | Notes only |

Checked and found correct: listed at the end.

---

## C1 (CRITICAL). Documents with an XML encoding declaration fail to parse

- **Location:**
  - `src/hypershift/data/edgar.py:86-90`: bytes are decoded to `str`, then `lxml.html.fromstring(doc)`.
  - `scripts/lazy_signals.py:27-31`: the exception is caught, and an empty count with `error` set is stored.
  - `scripts/lazy_signals.py:61`: a later run re-fetches only accessions *without* a token file.
  - `scripts/lazy_checks.py:33`: errored rows are dropped from the signal.
- **Failure mode:** lxml refuses a Python `str` whose XML declaration names an encoding (`ValueError: Unicode strings
  with encoding declaration are not supported`). Workiva's `<?xml version="1.0" ?>` has no encoding attribute and
  parses. DFIN ActiveDisclosure output starts with `<?xml version="1.0" encoding="ASCII"?>` and fails.
- **Verified on real filings (2026-10-24):**

  | Document | Accession (agent prefix) | First bytes | Frozen extractor |
  |---|---|---|---|
  | MSFT 10-K FY2023 | 0000950170-23-035122 (DFIN) | `<?xml version="1.0" encoding="ASCII"?><!-- DFIN New ActiveDisclosure ...` | raises ValueError |
  | MSFT 10-K FY2022 | 0001564590-22-026876 (DFIN) | XML declaration with encoding | raises ValueError |
  | AAPL 10-K FY2023 | 0000320193-23-000106 (self, Workiva) | `<?xml version="1.0" ?>` | parses |
  | AAPL 10-K FY2017 | 0000320193-17-000070 | `<DOCUMENT>` SGML header, HTML 4 | parses |

  Scope of the claim: verified on DFIN inline-XBRL 10-Ks from 2022 and 2023 only. How many S&P 500 filings it hits,
  from which year, and whether other agents emit an encoding attribute are **not** known. The error count in the
  main checkout answers that (diagnostic below).
- **Impact on the economic conclusion:** every affected document gives no signal, and the firm is held neutrally.
  This is not a look-ahead, but:
  1. coverage loss is concentrated by filing agent and by period (inline-XBRL years);
  2. P3 (≥ 75% coverage each year) may fail; if it passes, the test runs on a sample selected by filing agent;
  3. together with I3, firms with a failed latest filing carry a stale older signal for up to 3 months.

  The verdict would then rest on a different, smaller, non-random population than the one PREREG describes. **Do not
  interpret a verdict computed before this is fixed**, whatever its sign.
- **Smallest diagnostic (main agent; counts errors only, no returns):**
  ```
  python -c "import csv,collections as C; r=[x for x in csv.DictReader(open('data/edgar/signals.csv')) if x['error']]; print(len(r)); print(C.Counter((x['acceptance_et'][:4], x['accession'][:10]) for x in r).most_common(12)); print(C.Counter(x['error'][:60] for x in r).most_common(3))"
  ```
- **Fix:** `patches/C1_xml_declaration_and_refetch.patch` (applies to a14fd5d with `git apply --check`).
  1. `extract_tokens`: on `ValueError` only, strip the leading `<?xml ...?>` and parse again. Every document that
     parsed before takes the same path, so **every stored token file stays valid**. Verified: with the patch, all
     of `tests/test_edgar.py` passes, and the C1 cases in `tests/test_sec_integrity.py` turn to XPASS. On the real
     Microsoft 10-Ks, FY2022 gives 43,773 tokens and FY2023 gives 42,535; their year-over-year cosine is 0.99930.
  2. `lazy_signals.py`: an accession whose token file records an error is fetched again.

  **Rerun trap:** without (2), a plain rerun of `lazy_signals.py` re-fetches nothing, because errored token files
  exist. The alternative is to delete the errored token files first. Then rerun `lazy_signals.py`, which rewrites
  `signals.csv`, and `lazy_checks.py`.
- **Process:** this is an implementation fix, not a change of rule (PREREG already says "Parse HTML with lxml").
  Log it as a dated amendment before scoring.

## I1 (IMPORTANT, design). Block-element whitespace changes tokens

- **Location:** `edgar.py:103`. `root.text_content()` joins text nodes without separators.
- **Failure mode:** `<div>Risk Factors</div><div>Our business</div>` gives `factorsour` when the source has no
  newline between the divs, and `factors` + `our` when it has one. Adjacent table cells merge the same way
  (`<td>Name</td><td>Title</td>` gives `nametitle` in both cases). Two years of the same text can therefore differ
  only because the filing agent or the template changed.
- **Measured** with `tools/extraction_probe.py`, comparing the frozen extraction with one that separates every
  element by a space:

  | Document | Frozen tokens absent from spaced extraction | cos(frozen, spaced), same document | Year-over-year cos (frozen) |
  |---|---|---|---|
  | AAPL FY2023 | 0.90% | 0.99968 | 0.99921 (vs FY2022) |
  | AAPL FY2022 | 0.79% | 0.99975 | |
  | MSFT FY2023 | 0.14% | 0.99994 | 0.99930 (vs FY2022) |

  For Apple, formatting alone moves 1 − cos by about 40% of its yearly change. Apple and Microsoft are among the most
  stable filers, so this does **not** show how large formatting effects are relative to the changer threshold.
- **Not a code defect:** "take the remaining text" is what PREREG says. It is a validity risk for the signal's
  meaning.
- **Smallest diagnostic (no returns):** share of bottom-quintile (changer) pairs whose accession prefix (filing-agent
  CIK, first 10 characters) differs between the current filing and its prior, against the share among all pairs.
  - Self-filers make the prefix an imperfect proxy for the filing agent.
  - A ratio well above 1 means part of "changing" is presentation.
  - P5 (first inline-XBRL filing) covers one such switch.
- **Action:** report the ratio next to the verdict. It qualifies a CONTINUE (see the morning template). It cannot
  rescue a STOP.

## I2 (IMPORTANT). Pre-checks P4–P6 include errored rows

- **Location:** `scripts/lazy_checks.py:90` (`ntok`), `:99` (`simall`), `:108` (P5 mask), `:118` (`dl`). All four use
  every record, including rows whose `error` is set. Those rows have similarity 0.000000 and `n_tokens` 0.
- **Failure mode:** with C1 active, many first inline-XBRL filings are errored rows at similarity 0. P5 can then fail,
  or pass, for the wrong reason, and P6's correlation is distorted. P4 lists the rows as "documents under 1,000
  tokens", but **"parse error" is not an acceptable P4 explanation**: it leaves a sample selected by filing agent and
  by period.
- **Fix:** `patches/I2_prechecks_exclude_errored_rows.patch` keeps errors out of P4–P6 and still reports their count.
  Rerun the checks after C1.

## I3 (IMPORTANT, spec deviation). Fallback to an older filing

- **Location:** `src/hypershift/strategy/lazyprices.py:21` (`np.isfinite(s)` in the selection), together with
  `scripts/lazy_signals.py:84` (filings without a prior never reach `signals.csv`) and `scripts/lazy_checks.py:33`
  (errored rows dropped).
- **Failure mode:** PREREG's Comparison and Validity rows say the signal is the *latest* original filing, and that a
  filing without a comparison gives *no signal*. The pipeline cannot see such a filing, so `signals_at` silently uses
  the next older one still inside the 3-month window.
- **Verified:** `test_latest_filing_without_comparison_gives_no_signal` (strict xfail).
- **Impact:** small once C1 is fixed. Typical triggers are fiscal-year changes and CIK changes, and the fallback
  filing is at most about two months older. Before C1 is fixed, it hits every firm whose latest document errored.
- **Diagnostic:** for each formation, count the entities whose signal accession is not their latest available
  original 10-K/10-Q. The latest filing comes from the submissions cache; the comparison needs no returns.
  - If the count is below about 2% of signals, note it as a deviation.
  - If it is larger, amend before scoring: emit rows with NaN similarity for unmatched filings and let NaN block older
    rows.

## I4 (IMPORTANT, interpretation). What the CONTINUE rule can detect

- Over 7.5 years, the standard error of the IR is about 1/√7.5 = 0.37. The alpha t-statistic is about the residual IR
  × √7.5, so **t ≥ 1 needs a residual IR of about 0.37 or more**. That is a stricter gate than IR ≥ 0.30 whenever the
  factor model explains little of the active return.
- The approximate probability of CONTINUE under the joint rule is 16% / 28% / 64% at a true IR of 0 / 0.15 / 0.5.
  PREREG quotes 21% / 34% / 70% for the IR condition alone.
- The literature prior is a net IR of 0.1–0.2 (`LIT2_SYNTHESIS.md`). Under it, **STOP is the most likely outcome even
  if the effect is real**, and a CONTINUE at IR 0.3–0.5 carries roughly a 1-in-3 to 1-in-2 chance of being noise.
- This is not a defect: PREREG labels the test kill-or-license. The morning template uses these numbers.

## I5 (IMPORTANT for panel reuse; MINOR here). Identity splice in entity `IR`

- **Verified from the panel:**
  - `IR` is a member on every date from 2016-01-04 to 2023-12-29 and has no bars before 2017-05-12, which is Gardner
    Denver's IPO date.
  - Its raw closes are $22.93 (2017-06-01), $32.34 (2018-06-01) and $35.77 (2020-02-03).
  - Entity `TT`, which is Ingersoll-Rand plc renamed Trane Technologies, has raw closes of $89.60, $88.27 and
    $136.19 on the same dates, but is a member only from 2020-03-03.
  - So for about 2.5 years the Ingersoll-Rand plc membership slot is priced with Gardner Denver, a non-member. The CIK
    map gives `IR` CIK 1466258 (Ingersoll-Rand plc) until 2020-04. Signal and returns therefore belong to different
    companies.
- **Mechanism:** fja05680 lists the ticker, and the ticker passed from Ingersoll-Rand plc (renamed TT) to Gardner
  Denver (renamed IR) on 2020-03-02. Alpaca's `asof` series for `IR` follows the newer company.
- **Impact here:** one name in about 500 (equal weight about 0.2%), present in both A and B. It moves the active
  return only in months when `IR` is a changer. Negligible for the verdict.
- **Panel:**
  - This is a real identity break that `INFRA_CHECKS.md` missed: its "IR from 2016-01-04 (342 days)" entry was
    classified as missing bars.
  - Nine other entities have membership before their first bar (AABA, WYND, XL, BHGE|BKR, ARNC, FTI, UAA, AMCR, APTV).
    They are renames or missing bars, not splices, except for ARNC's last 2 days (2020-04-02/03, new Arconic Corp).
  - Generic check for any reuse: membership starting before the first bar, with the price level jumping against the
    CIK's company.

## MINOR

- **M1. Current, not dated, SIC.**
  - Location: `lazy_eval.py:28`; `lazy_cik_map.py:160,204`.
  - The FF12 industry comes from the latest CIK's SIC today. SIC changes are rare; a look-ahead in classification
    only.
- **M2. Dual-class double observations.**
  - GOOG/GOOGL, FOX/FOXA and NWS/NWSA are simultaneous members on all 2,012 panel days; DISCA/DISCK on 1,579; UA/UAA
    on 1,561. Each pair shares one CIK.
  - So one filing produces two rows (`test_one_accession_shared_by_two_share_classes...`). This is consistent with an
    equal-weight benchmark that holds both classes, but the filing counts twice in the within-form rank.
  - `CIK_MAP.md`'s "disjoint membership" wording is wrong for these pairs.
- **M3. The 1.25× maximum is not a cap.**
  - PREREG's "Maximum weight per name is about 1.25 × B's" describes the typical case. It is not enforced:
    4 changers out of 5 in an industry give the fifth name 5× B (test).
  - `lazy_eval` reports the realised maximum active weight. Read it.
- **M4. Leg diagnostic is not industry-neutral.**
  - Location: `lazy_eval.py:39-40`.
  - PREREG diagnostic 1 says "(industry-neutral as in A)"; the code uses plain equal weights. The generated text
    discloses this. Non-gating.
- **M5. Subsidiary look-alikes accepted.**
  - Strict `name_match` accepts reporting subsidiaries whose names extend the parent's ("Duke Energy" vs
    "DUKE ENERGY CAROLINAS, LLC"; strict xfail).
  - Only the 5 fallback CIKs depend on name matching. Diagnostic: list those 5 with their EDGAR names.
- **M6. Silent factor-date truncation.**
  - `factors.attribute` (`factors.py:45`) keeps only dates present in the Ken French files. `lazy_eval` does not
    print `att['n']`.
  - If the files end before 2023-12-29, the alpha t-statistic covers a shorter sample without notice.
  - Diagnostic: `att['n'] == len(dates)`.
- **M7. Exhibit-13 10-Ks.**
  - Some large firms (often banks) incorporate the MD&A and financials by reference to Exhibit 13, so the primary
    document is a short wrapper. Their 10-K similarity is then high and uninformative. A switch to or from an
    integrated 10-K looks like a large change.
  - This follows from the PREREG design (primary document only, as in Cohen et al.). Diagnostic: changers with
    |Δ log tokens| > 0.7 (P6 inputs).

## Checked and found correct (no action)

- **Timing:**
  - The acceptance time is inverted from the submissions JSON (tested, plus new DST and year-end cases).
  - A filing is available on the first trading day strictly after its acceptance date (`edgar.py:72-74`). It is
    usable at formation F only if available by F−1 (`lazyprices.py:21`), and it is traded at the open of F.
  - So the earliest trade is the open of the session after the availability day. That is at least one full session
    after acceptance, never a same-day close (new test).
  - Exchange holidays with EDGAR open (Good Friday) are handled (new test).
- **Amendments** never become the current or prior document. **The prior must have been accepted before the current
  filing**, so no future document is retrieved (new tests). A missing report date or prior gives no match without an
  error.
- **Historical identity** comes from dated Wikipedia revisions of the constituent table, not today's ticker table.
  Today's SEC ticker table is used only in the 5-CIK fallback, under strict name matching and a filing-span check.
  CIK segments stop predecessor and successor filings from leaking across a CIK change (new test on `signal_rows`).
- **Portfolio:**
  - Weights are feasible: long-only, sum to 1, and each industry keeps the benchmark's weight.
  - Costs are charged on traded notional for both A and B.
  - Execution is at the open after drifting close(t−1)→open(t); the open index never uses the same day's close
    (tested upstream).
  - Delistings go to cash at the last close and the cash is redeployed at the next rebalance (new test). Halted names
    keep their weight.
  - The benchmark is an implementable equal-weight S&P 500 with the same costs.
- **Statistics:**
  - Stationary bootstrap with block 21 and B = 2000 (`study.py:80-83`), as in PREREG.
  - Newey–West with 10 lags on FF5 + MOM + ST_Rev.
  - Every parameter was frozen in `PREREG.md` (commit 4f2ae6f) before the EDGAR code (73bf7dc) and before any
    download.
