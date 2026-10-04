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

## 9. Results and verdict (audit run 2026-10-04; `scripts/post2017_data_audit.py`; numbers in `post2017_audit_results.json`, per-ticker table `post2017_audit_tickers.csv`)

Source: yfinance 1.7.0, downloaded 2026-10-04T05:54Z, 2014-11-01..2023-12-31, `auto_adjust=False, actions=True`. 1,856 candidate symbols for the 1,737 tickers, 1,123 returned data, 733 returned nothing (a retry recovered 0, so these are genuinely absent, not transient failures). Graph v2 cache sha256 `859d44173f0dd61c896d63a22af84455fecae4a3d66461a118cf80ce802e197c`. Raw bars: `data/raw/yahoo_post2017/` (git-ignored).

**Calendar check.** On 2014-11-19..2017-12-08 the Yahoo trading-day set equals the RSR date vector exactly (0 days only in one, 0 only in the other). 796 Yahoo trading days precede 2018-01-02 (>= 46 required).

**R5 overlap test (predeclared; 741 overlap days; pooled over the first symbol with data of each of the 1,123 tickers = 817,171 matched-eligible stock-days).**

| convention | matched | share | tickers passing identity (>= 250 days, share >= 0.95) |
|---|---|---|---|
| (a) raw close (undo Yahoo splits) | 793,072 | **0.9705** | 967 |
| (b) split-adjusted `Close` | 793,070 | 0.9705 | 967 |
| (c) split+dividend `Adj Close` | 779,168 | 0.9535 | 865 |
| (d) POST HOC: genuine splits only, spin-off pseudo-splits undone | 793,097 | 0.9705 | 967 |

On the 967 identity-passing tickers the pooled shares are (a) 0.9943, (b) 0.9942, (c) 0.9782, (d) 0.9943; the median per-ticker share is 0.9973 for (a), (b), (d) and 0.9811 for (c).

Predeclared winner: (a) raw, share 0.9705 >= 0.95. The margin over (b) is 2 stock-days of 817,171 (a statistical tie), so the predeclared rule does not discriminate (a) from (b). Findings that do discriminate (post hoc, labelled as such):

- RSR is NOT dividend-adjusted ((c) is worse: 0.9535 vs 0.9705; 102 fewer tickers pass).
- RSR IS adjusted for genuine splits (2:1, 5:1, 1:5, 3:2 and similar: the RSR return on those days equals the split-adjusted return) but NOT for Yahoo's non-integer "splits" that are really spin-offs or special distributions (ratios such as 1.081, 1.193, 1.319, 1.398, 1.841; the RSR return keeps the drop). Of 46 split events in the overlap on identity-passing tickers, raw matches RSR on 31 (the pseudo-splits), split-adjusted on 10 (the genuine splits), neither on 5. (a) and (b) are tied because each is wrong on one of the two event types; (d) is right on both and has the highest share (793,097), although its margin (+25 stock-days) is also tiny.
- R4 consequence: (a) raw has an unadjusted genuine split inside windows (e.g. AFL 2018-03-19 2:1: adjusted Close 45.24 -> 44.70, raw 90.49 -> 44.70), which R4 forbids and RSR itself does not contain. The pilot construction should therefore be (d) or (b); this is a post-hoc amendment that needs the user's or reviewer's sign-off before use. The predeclared (a) must not be used for model inputs.
- Yahoo `Close` with `auto_adjust=False` is split-adjusted (AFL above; AOS 2016-10-06 2:1: (b)/(d) return 0.0107 = RSR 0.0108, raw -0.4946).

**Identity.** Of 1,737 tickers: 614 have no Yahoo data (delisted, renamed or preferreds; this includes all 2018-2026 attrition), 20 have data but < 250 matched days, 136 have >= 250 days and share < 0.95 (median share 0.90; 104 of them between 0.80 and 0.95, mostly closed-end funds and ADRs; the cause is UNKNOWN: dividend adjustment does not explain it, none of them passes under (c)), and 967 pass. 11 failures have share < 0.02 (reused or different security, e.g. TEN, B, ACH, WES, COR). The rejection list is `post2017_audit_tickers.csv` (`identity_ok=False`). The continuity (>= 60-day gap) rule truncated 0 tickers. Name continuity was not checked (Section 8).

**Coverage of the frozen 1,737 nodes, 2018-2023 (bar availability only; no returns computed).**

| year | trading days | nodes with any Yahoo bar | identity-pass nodes with a bar | eligible all year | mean / min / max daily eligible |
|---|---|---|---|---|---|
| 2018 | 251 | 1104 | 967 | 967 | 967 / 967 / 967 |
| 2019 | 252 | 1104 | 967 | 967 | 967 / 967 / 967 |
| 2020 | 253 | 1110 | 967 | 967 | 967 / 967 / 967 |
| 2021 | 252 | 1115 | 967 | 967 | 967 / 967 / 967 |
| 2022 | 251 | 1118 | 967 | 967 | 967 / 967 / 967 |
| 2023 | 250 | 1123 | 967 | 967 | 967 / 967 / 967 |

The eligible set is 967 of 1,737 nodes (55.7 percent), constant over the six years: every name that left the market before the download date is absent from Yahoo, so the panel holds survivors only, with zero attrition. The 19 extra "any bar" nodes that appear over 2018-2023 (1104 to 1123) are all identity failures (probably symbols reused by later listings; not verified). Eligible nodes on 2018-01-02: 967 (all with >= 30 prior closes).

**R8 survivor bias (2017 test, saved `R5_f2_alpha0_train/HH` predictions, seeds 0-4, read-only; top-5 chosen among the subset only, same days, same predictions).**

| universe | nodes | mean daily eligible (2017) | top-5 Sharpe, mean of 5 seeds (per seed) | hold-all Sharpe |
|---|---|---|---|---|
| full RSR universe | 1737 | 1731 | 1.977 (1.88, 1.93, 1.89, 2.04, 2.14) | 1.531 |
| covered subset (identity pass = survivors) | 967 | 966 | 2.249 (2.04, 1.53, 2.66, 3.18, 1.84) | 1.762 |

Restricting to the survivors raises the 2017 top-5 Sharpe by +0.27 and hold-all by +0.23 (top-5 minus hold-all: 0.446 full, 0.487 covered), and widens the seed spread. A survivor-only 2018-2023 test would therefore be biased upward in level; this check does not identify the direction of the bias on ranking skill.

### Gate verdict

**PASS as an EXPLORATORY, SURVIVOR-BIASED Yahoo pilot, with conditions; not a confirmatory source.** Basis: the predeclared pooled R5 share of the winning convention is 0.9705 (>= 0.95), identity is verified on 967 tickers, the calendar is identical on the overlap, and the causal-feature tests are green (`tests/test_post2017_data.py`, 15 tests). Conditions:

1. Price construction must be (d) genuine-split-adjusted close (or (b)), with no dividend adjustment. The predeclared winner (a) is inadmissible under R4. The switch from (a) is a post-hoc amendment awaiting sign-off; the audit cannot separate (a), (b) and (d) by more than 25 stock-days.
2. Universe = 967 of 1,737 nodes (55.7 percent). The other 770 are masked for all dates 2018+ (614 no data, 20 short history, 136 failed identity). Graph v2 and node order are kept; masked nodes keep their slots.
3. Survivors only; no delisting returns; the level of any Sharpe is upward-biased (R8 above: +0.23 to +0.27 on the 2017 test).
4. The 136 identity failures with share 0.80-0.95 are unexplained (UNKNOWN). If their cause is price-vintage noise, identity may be too strict, which only shrinks coverage and does not admit bad names.

If the user later obtains CRSP/WRDS access, that source should replace this pilot; the same R5 and identity tests apply unchanged.

## Amendment A1 (2026-10-04, Claude Code orchestrator, under the user's "do what you think is best")

**The pilot uses convention (d): adjust genuine stock splits only; leave Yahoo's non-integer spin-off "splits" unadjusted; no dividend adjustment.** This replaces the predeclared winner (a).

Why:
- The predeclared R5 rule could not tell the candidates apart: (a) 0.9705 vs (b) 0.9705, a 2-stock-day margin.
- (a) is inadmissible under R4, because it leaves genuine splits unadjusted inside input windows.
- On the 46 split events in the overlap, RSR behaves like (d): raw matches the 31 spin-off-type events and split-adjusted matches the 10 genuine splits. That makes (d) the closest reading of RSR's semantics.

Why this is not outcome-dependent:
- The decision uses only 2015-01-02 to 2017-12-08 overlap evidence, made before any 2018+ return, strategy result or inference was computed.
- It changes input construction, not model selection or the test statistic.
- It is recorded here before the freeze manifest.

Status: still **exploratory, survivor-biased** (967 of 1,737 nodes pass identity; R8 bias: 2017 top-5 Sharpe +0.27 and hold-all +0.23 on the survivor subset). Every 2018+ comparison therefore uses hold-all, random and matched nulls on the **same 967-name eligible subset**.
