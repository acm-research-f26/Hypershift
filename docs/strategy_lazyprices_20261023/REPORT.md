# Filing-text round: report and decisions (2026-10-23)

**Bottom line.**
- **The test.** S&P 500 changer avoidance (long-only, FF12-industry-neutral, full 10-K/10-Q cosine similarity
  against the year-earlier report) was run once, under a spec frozen before any download, on point-in-time members
  over 2016-07 to 2023-12.
- **Result.** Net active IR at 10 bp **−0.01 [−0.66, +0.68]**; active alpha t **−0.13** → **STOP**.
- **What stops.** Only this implementation. Filing text as an information source is not closed; its documented home
  is non-S&P 500 stocks, which our data does not cover.
- **Next.** The strongest free alternative, specified and ready to freeze, is an **earnings-announcement-return
  drift overlay** keyed on 8-K Item 2.02 acceptance times (`../overnight_20261024/NEXT_RESEARCH_SPEC.md`).

## A. Previous strategy: closure and infrastructure

**Style-only LightGBM book: STOP as a capital candidate.**

| Measure | Value |
|---|---|
| Net Sharpe at 10 bp | +0.11 [−0.61, +0.90] |
| Factor alpha t | +0.15 |
| Net Sharpe at 20 / 30 bp | −0.41 / −0.92 |

- **Not proven zero.** The edge is indistinguishable from zero.
- **Causes of the fall from +1.19 are confounded.** Universe, training procedure, sectors, exit rule, data source and
  turnover all changed (append-only addendum in `../strategy_pit_20261023/STRATEGY_PIT_REPORT.md`).
- **Infrastructure checks** (`INFRA_CHECKS.md`):
  - the sector proxy used up to 62 later returns for 18 entities; `sectors_causal` was added;
  - opening fills never use the same day's close (tested);
  - no traded position sat on a neutralised corporate-action day;
  - failure flags reconcile: 4 flagged, 2 members at their last bar;
  - identity breaks: FBHS→FBIN (negligible weight), and the `IR` entity carrying Gardner Denver prices from 2017-05
    to 2020-03 (I5; fix before reusing the panel).

## B. Research synthesis

`LIT2_SYNTHESIS.md` and `LIT2_VERIFY.md`, plus the independent overnight review
`../overnight_20261024/OVERNIGHT_RESEARCH.md`. Key points:
- **Lazy Prices** (1995–2014, broad universe). Both legs carry alpha in calendar time, but only the changer side
  persists in event time (NBER WP, p. 17).
- **Cheng, Liu, Qiao & Wang** (JFQA 2026). The effect is **insignificant in S&P 500 stocks** even in-sample.
- **No post-publication evidence.** No peer-reviewed post-publication test exists; unreviewed large-cap replications
  are null.
- **LLM-disclosure work.** The overnight review reports that the flagship LLM-disclosure paper was withdrawn after a
  failed replication.
- **Sadlo.** Not found under that title.

## C. Strategy specification

`PREREG.md`, frozen at commit 4f2ae6f before any filing download. Two pre-outcome amendments followed; both were made
before any return was computed and are justified in `PREREG.md` and `LEDGER.md`:
- **A1:** fixed an XML-declaration extraction bug and added a filingDate availability guard;
- **A2:** the latest filing without a comparison gives no signal; checks use valid rows only; a formatting
  diagnostic is reported.

## D. SEC pipeline

`PIPELINE.md`.
- **Code:** `src/hypershift/data/edgar.py` and `src/hypershift/strategy/lazyprices.py`.
- **Timing:**
  - acceptance time with SEC's inconsistent JSON time zone inverted;
  - next-trading-day availability;
  - filingDate guard;
  - original forms only.
- **Identity:** dated CIKs from monthly Wikipedia revisions, each verified on SEC filings and names; 99.8% of
  member-months mapped. Three wrong-company matches were caught and fixed.
- **Tests:** `test_edgar.py`, `test_lazyprices.py`, `test_pit.py`, and the merged overnight `test_sec_integrity.py`.
  Together: 85 pass; 2 strict xfails mark known limitations (I1 block whitespace, M5 subsidiary names).
- **Pre-outcome checks** (`PRECHECKS.md`), all passing:
  - no filing became available early;
  - coverage 92–95% in every year;
  - 0 parse errors;
  - the inline-XBRL transition is neutral (−0.15 SD);
  - the length correlation is −0.64.

## E. Evaluation (`EVAL.md`, run once)

| Cost | Active ann. | Tracking error | Active IR [95% CI] |
|---|---|---|---|
| 10 bp | −0.01% | 0.71% | **−0.01** [−0.66, +0.68] |
| 20 bp | −0.12% | 0.72% | −0.17 |
| 30 bp | −0.24% | 0.72% | −0.34 |

- **Factor-adjusted.** Active alpha −0.03%/yr (t −0.13), R² 0.09. All factor exposures are near zero, so the overlay
  is genuinely neutral.
- **Per-year active return** ranges from −1.0% (2020) to +0.6% (2018), with no persistent sign.
- **Declared diagnostics (non-gating):**
  - changers trail the benchmark by −0.40%/yr (IR −0.12);
  - stable reporters lead it by +1.19%/yr (IR +0.34, CI [−0.43, +1.15]);
  - the long-short spread is +1.09%/yr (IR +0.19, alpha t +0.42).

  Directionally they agree with the literature but are far from significant. **They are not a basis for a re-test on
  these data:** choosing the stable-reporter leg now would be selection on the outcome.

## F. Economic assessment

- **No improvement over an implementable benchmark.** The overlay's gross active edge (≈ +0.1%/yr) is consumed by
  its own trading: +1.19× NAV/yr of extra turnover, or about 12 bp/yr at 10 bp.
- **Ceiling vs result.** The pre-stated ceiling of 0.3–0.4%/yr was not reached.
- **Operationally fine.** Maximum active weight 0.28%; changers number 68–100 per month, spread across forms and
  industries. Capacity and operations are not the problem; the signal is absent at this cap level.

## G. Alternatives, ranked

Full table in `../overnight_20261024/OVERNIGHT_RESEARCH.md`.
1. **Earnings-announcement-return drift overlay.** Keyed on 8-K Item 2.02 acceptance times; free data already
   cached; exact timestamps.
   - Support: Daniel–Hirshleifer–Sun (big-firm leg 0.38%/month), and Dickerson–Julliard–Mueller (2026: robust to
     2022, not microcap-driven).
   - Against: Martineau finds no large-cap SUE drift since 2006, and HyperShift's own 16-day reversal pushes the
     other way.
2. **Opportunistic insider purchases** (Form 4, free, 2003+). Value-weighted evidence exists, but signals are sparse
   in S&P 500 names and post-2007 persistence is unverified.
3. **Filing text outside the S&P 500** (the documented home of Lazy Prices). Needs a broader point-in-time universe
   (Alpaca delisted small caps plus a causal liquidity rule) and a cost model for less liquid names.

## H. Decisions

| Item | Decision | Reason |
|---|---|---|
| S&P 500 changer avoidance | **STOP** | Predeclared rule (IR −0.01 < 0.30; alpha t −0.13 < 1.0). No tone, section or leg variants on 2016–2023 |
| Protected 2024-01..2026-09 window for filing text | **Not spent** | Only a CONTINUE licensed it |
| Filing text in non-S&P 500 stocks | **INVESTIGATE** (not now) | The literature locates the effect there; it needs a broader PIT universe and a cost model, so do it only if F1 also fails |
| EAR drift overlay | **BUILD → TEST** (next) | Best expected decision value per unit of work; the spec is ready |
| Norgate, Sharadar, any paid data | **DEFER** | No candidate has passed a kill test |
| SEC/EDGAR pipeline, CIK map, PIT engine | **BUILD** (done; keep) | Reused directly by the EAR test |
| Capital | — | Not an outcome of this round |

## I. Next agent action (bounded)

**Assignment.** Execute `docs/overnight_20261024/NEXT_RESEARCH_SPEC.md` (EAR drift overlay):
1. Copy its frozen rules into `docs/strategy_ear_<date>/PREREG.md` and commit it before computing any forward return.
2. Build the 8-K 2.02 event table from the cached submissions JSON (`edgar.SecClient`, at most 8 requests/s for
   missing pages only), using `cik_map.json` segments.
3. Run the declared pre-outcome checks: event coverage, timing, and the exclusion of the IR splice.
4. Run the single 2016-07..2023-12 evaluation with `pit_engine.run_targets`.

**Inputs:** `data/pit/pit_sp500_2016_2023.npz`, `data/edgar/cik_map.json`, `data/edgar/json/`, Ken French factors.

**Outputs:** `PREREG.md`, `PRECHECKS.md`, `EVAL.md`, `REPORT.md`, the event-table script, and tests.

**Stopping rule:** the spec's own thresholds. A STOP closes the large-cap event-drift idea; no variants on 2016–2023.
