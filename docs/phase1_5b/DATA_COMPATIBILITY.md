# Post-2017 data compatibility spec and gate (Phase 1.5b, plan Task 1)

Status: SPEC FIXED 2026-10-04, written before any overlap statistic was computed. Section 9 (results and verdict) is appended after the audit run; Sections 1-8 are not edited after that except to mark an item `AMENDED` with a reason.

Scope rule (plan R1): the audit may count coverage and availability for 2018+ but must not compute any model, strategy or portfolio return for any date in 2018+. Price-derived statistics are computed only on the overlap window 2015-01-02..2017-12-08 (and the 2017 saved-prediction survivor-bias check, R8).

## 1. Identity rule

- Node = one of the 1,737 ordered RSR NYSE tickers (`data/raw/rsr/data/NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv`, sha256 `fb14b5ef35894a1d7d8a80f3ba1c750c0a628c6e3d8d2c3e75dad2fc0dc11701`). Node index = position in that file. The universe is frozen: delisted names are masked, new listings are excluded (R9).
- Yahoo symbol candidates for RSR ticker `T`, tried in this order: (1) `T`; (2) `T` with `.` replaced by `-` (class shares, e.g. `BRK.B` -> `BRK-B`); (3) for RSR preferred-style names `ROOT-X`, `ROOT-PX`. The first candidate that returns data AND passes the identity test below is used. A ticker that returns data but fails the test is REJECTED (masked for all dates), never "repaired".
- Identity test (predeclared, per ticker), on the winning price convention of Section 2 over the overlap window 2015-01-02..2017-12-08: matched stock-days >= 250 and share of matched stock-days with |r_new - r_RSR| < 1e-4 is >= 0.95. This is the guard against ticker reuse, mergers and splicing: a different security under the same symbol does not reproduce three years of daily returns.
- Continuity rule (availability only, no returns): after the last observed day of the overlap window, if a ticker has a gap of >= 60 consecutive trading days and then reappears, the series is truncated at the start of the gap (treated as delisted). Reappearance after a long gap is the signature of ticker reuse that the 2015-2017 overlap test cannot see.
- Name continuity: Yahoo company names are NOT used as evidence (the names are current, not historical). Name checks are listed as not performed.

## 2. Price convention candidates and the predeclared R5 overlap test

Candidates, per ticker, from `yf.download(..., auto_adjust=False, actions=True)`:

- (a) raw close = Yahoo `Close` multiplied by the product of all later split ratios (undoes Yahoo's split adjustment using the `Stock Splits` events).
- (b) split-adjusted close = Yahoo `Close` (auto_adjust=False; verified on AAPL 2020-08-31 4:1 in the audit).
- (c) split+dividend-adjusted close = Yahoo `Adj Close`.

Test: on the RSR calendar (Section 5), for every ticker and every date t in 2015-01-02..2017-12-08 where the close is observed on t and on the previous RSR calendar day in BOTH sources, compare r_new(t) = c_t / c_{t-1} - 1 with r_RSR(t) = the RSR `gt` (close-to-close return of the stored normalized close). A stock-day matches when |r_new - r_RSR| < 1e-4. Share = matched / total matched-eligible stock-days, pooled over all tickers with data. The winner is the candidate with the highest pooled share. Pass threshold: winner share >= 0.95. If the winner is below 0.95 the gate FAILS for that source.

The R5 test is the only evidence used to choose the convention; price level is irrelevant because inputs are `input_mode=relative` (R4: each window is divided by its last close, loop.py:58-62). Required for the chosen series (R4): prices consistent inside every 16-day window and MA30 lookback (no unadjusted split inside a window). If candidate (a) wins, splits inside windows are a defect that must be reported and the affected tickers listed (RSR would have the same defect; this would be a finding about RSR, not a licence to splice).

## 3. RSR return definition (reproduced exactly)

`gt[t] = (c[t] - c[t-1]) / c[t-1]` if the close is observed on both t and t-1, else 0 with mask 0 on the missing side (`src/hypershift/data/rsr.py:parse_eod`). Close-to-close price return of the chosen convention; no risk-free rate. `gt` is a placeholder where unobserved and is never used without the mask.

## 4. Fill and mask semantics

- Missing close (no bar for an RSR-calendar date) -> mask 0; all five feature cells for that date are the fill value 1.1 after normalization (rsr.py:FILL; `renormalize_train` re-writes 1.1 on masked cells). Mask is keyed on the close only.
- Delisting: the series ends; every later date is masked. No terminal (delisting) return is invented and no zero return is written as a real observation. The delisting return itself is UNKNOWN from Yahoo (bars simply stop); this is a documented survivor-bias source (Section 7).
- MAs with a missing close inside the window: how RSR computed them is **UNKNOWN**. Checked 2026-10-04 on the RSR files: for rows where a window contains a missing close, MA5/10/20/30 matched none of (i) mean of observed closes in the calendar window, (ii) forward-filled calendar window, (iii) mean of the last w observed closes (hit rates 3-9% of 335-1861 windows per MA each, ffill best on a small sample). Rule for the new source (INFERRED, not RSR-derived): MA_w(t) = mean of the last w calendar-day closes after forward-filling within the series (no fill before the first observation); a window with fewer than w prior observations is not computed (the feature cell is masked as 1.1, mask 0). Windows containing a filled day are a small share (RSR requires >= 98 percent coverage per ticker) and are reported as a count.
- Feature order (frozen): `[MA5, MA10, MA20, MA30, close]`; MAs are trailing means of the last w closes including day t (verified against RSR files: max abs error < 1e-5 on full windows; docs/phase1_5/C_data_graph.md).
- Scale: `norm=train` divides each stock by its max observed close over the training dates (RSR rows < 756, i.e. dates <= 2015-12-31). For the new source the scale is the max of the NEW source's own closes over the dates up to 2015-12-31 (no RSR prices are used). It is a fixed per-ticker constant, so a future price cannot change an earlier feature. Irrelevant to relative mode (R4) but fixed for completeness.

## 5. Calendar and windows

- RSR calendar: `NYSE_aver_line_dates.csv`, sha256 `2a883ed7238a42057b9259a85c9319f3a75699c39c31477559c6ac51b9aa3796`, 1,274 dates 2012-11-19..2017-12-08; RSR row 0 = date index 29 = 2013-01-02 (docs/phase1_5/C_data_graph.md). Split indices 756 / 1008 = 2016-01-04 / 2017-01-03, T = 1,245.
- Post-2017 calendar: union of dates observed in the downloaded bars on or after 2017-12-11 (US trading days actually printed by Yahoo; a date is a trading day if >= 50 percent of covered tickers have a bar). No synthetic dates. Primary window 2018-01-02..2023-12-29 (last Yahoo trading day of 2023); the audit downloads through 2023-12-31.
- Warm-up: >= 46 trading days before 2018-01-02 (16-day window + MA30 lookback = 16 + 29 + 1) taken from the NEW source only. No RSR/new splice anywhere; the first usable 2018 window ends on 2018-01-02.

## 6. Graph and node order

Hypergraph = cached graph v2 (`data/raw/rsr/data/hypergraph_cache/NYSE_industry-wiki.json`, 4,350 hyperedges, max node degree 114; CLAUDE.md; sha256 recorded in Section 9) keyed on the frozen 1,737-node order. Masked nodes keep their index and edges; the model sees them as missing (mask 0, fill 1.1) exactly like RSR missing days. Ticker order is never sorted, filtered or reindexed.

## 7. Source ladder and labelling

1. Confirmatory: CRSP via WRDS (permanent IDs, delisting returns), only if the user obtains access. Not available now.
2. Pilot: Yahoo Finance via yfinance (version in Section 9). **Exploratory and survivor-biased**: Yahoo does not retain most delisted tickers; delisting returns are absent; prices are as-of the download date (revised adjusted prices); symbols may have been reassigned (guarded by Section 1). Any result built on it must carry this label.

## 8. Not tested / out of scope

Name continuity against historical company names; delisting returns; intraday bars; fundamentals; Yahoo-vs-CRSP agreement (no CRSP data). A pass here supports only an exploratory pilot.

## 9. Results and verdict

(To be filled after the audit: coverage table 2018-2023, R5 shares per convention, rejected tickers, R8 survivor-bias numbers, gate verdict.)
