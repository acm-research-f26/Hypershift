# H. EAR round handoff (2026-10-23)

## Findings

1. **Timing.**
   - SEC's acceptance clock (submissions JSON) is unreliable, but the official filing date is reliable.
   - The market reaction falls on d0 or d0+1 for 77% of 16,663 events, at 2.3–2.4× normal size. Pre-event days sit
     at baseline, so delayed 8-Ks don't matter.
   - Exhibit release dates equal the 8-K filing date in 9/9 readable samples.
   - Decision **REDEFINE BEFORE OUTCOMES**: d0 = filing date; window d0−1..d0+1; trade at the next monthly open.
2. **Data.**
   - The `IR` splice is excluded; the other 7 CIK-change entities were checked.
   - Dated SIC from filing headers: 653 CIKs, 9 FF12 differences from current SIC.
   - Event-time bins and a 2× cap were fixed before outcomes.
3. **Literature.**
   - The signal is the DHS / Dai et al. CAR. Post-2001 value-weighted evidence is mildly positive (Dai et al.:
     0.36%/month, t 2.09; re-checked in the PDF).
   - Martineau's large-cap regression shows no continuation.
   - The expected net was thin, so the test was run as a cheap kill test.
4. **Result: STOP.**
   - Net active IR −0.47 [−1.11, +0.14]; alpha t −1.44; −0.52%/yr.
   - Event information beyond the matched control: IR −0.08.
   - Both the EAR tilt and the control lost, a short-horizon reversal signature.
5. **Decisions.**
   - Stable reporters: ARCHIVE.
   - Broader universe: INVESTIGATE, starting with the S&P 400.
   - Next hypothesis: filing-text avoidance in the PIT S&P 400, with a feasibility probe first.

## Files

All in `docs/strategy_ear_20261023/`:
- `TIMING_AUDIT.md`
- `DATA_CHECKS.md`
- `LIT_EAR_VERIFY.md`, `LITERATURE_DECISION.md`
- `FEASIBILITY.md`
- `PREREG.md` (frozen at 35ae347)
- `EVAL.md`
- `STRATEGY_DECISION.md`
- `ALTERNATIVES.md`

**Code:**
- `src/hypershift/strategy/ear.py`
- `scripts/ear_{timing_audit,sic,common,checks,eval}.py`
- `tests/test_ear.py` (7)

**Data (git-ignored):** `data/edgar/ear_events.json`, `data/edgar/sic_hist.json`.

## Protected and untouched

- 2024-01..2026-09 was not read.
- No data purchased.
- No variants run after the result.

## Outstanding blockers

None for this round. The next step needs a free S&P 400 membership history; its quality is unverified (Wikipedia
revisions since 2010-12).

## Single next action

Run the **S&P 400 feasibility probe** in `ALTERNATIVES.md`:
1. Parse 3 Wikipedia revisions (2016-07, 2020-01, 2023-12) for table structure and the CIK column.
2. Check Alpaca `asof` bars for 20 deleted S&P 400 names.
3. Report against the stated gate.

No outcomes. If the probe passes, build the S&P 400 panel and run the frozen filing-text spec once.
Suite: 575 passed, 2 xfailed in 684s (2026-10-23).
