# G. Stable reporters, broader universe, and the next hypothesis (2026-10-23)

## Decision B: stable reporters → ARCHIVE (no test, holdout not spent)

| Consideration | Evidence |
|---|---|
| Our observation | Top-20% stable reporters minus B: +1.19%/yr, IR +0.34 [−0.43, +1.15]. This was a *declared diagnostic* of the filing-text test, chosen to be reported but not chosen as the hypothesis; it is the best of four reported numbers, i.e. selected |
| Mechanism | Weak. In Cohen–Malloy–Nguyen the non-changer leg's alpha "quickly reverts to zero" in event time, while the changer leg persists. There is no clear reason why *unchanged* text should keep predicting outperformance |
| Independent evidence | One supporting paper (Adosoglou et al. 2021, full text): long-only Doc2Vec non-changers with a momentum filter added "to improve our five-factor alphas", about 40 stocks a year, 1999–2018, about 20 annual observations, bag-of-words versions null. No post-2014 or large-cap test found |
| Value of a test | Prospective only: the protected window must not be spent on a selected diagnostic. A credible prospective test needs years of signal dates |

**Archive it.** It may be revived only if new external evidence (a post-2015, out-of-sample, large-cap result)
appears; then it would be frozen and scored prospectively.

## Decision C: broader point-in-time universe → INVESTIGATE (feasibility probe first; the recommended direction)

**Why move.** Three S&P 500 information tests have now failed:
- style book: net Sharpe +0.11;
- filing-text avoidance: IR −0.01;
- EAR: IR −0.47, and no event information beyond the control.

Every literature source locates these effects in smaller stocks:
- Cheng–Liu–Qiao–Wang: Lazy Prices significant **only outside the S&P 500**, small caps about 60 bp/month;
- DHS: CAR drift 0.94%/month in small firms vs 0.38 big;
- Martineau: large-cap SUE drift gone since 2006;
- CMP: insider effects larger in the bottom size half.

Continuing with large-cap event strategies has low expected value.

| Universe | Effect size (literature) | Costs | Free PIT membership | Prices incl. delisted | Engineering | Verdict |
|---|---|---|---|---|---|---|
| **S&P 400 (mid caps)** | Between large and small; Lazy Prices firms average $2.5–3.5B (a mid-cap profile) | About 1.5–2× large caps (FIM: small caps 21 vs large 11 bp impact) | Wikipedia "List of S&P 400 companies" revisions since **2010-12** (quality NOT VERIFIED) | Alpaca SIP with `asof` (to be verified for delisted mid caps) | Reuse the S&P 500 pipeline: about 3–5 days | **Best next step** |
| S&P 600 (small caps) | Larger | About 2×+; borrow harder | Wikipedia revisions only from **2018-08**: too short for 2016–2023 | As above | Same | Not now (history too short) |
| Liquidity-filtered US universe (e.g. top 1,500 by trailing dollar volume) | Largest breadth | Variable; needs a spread model | Constructible causally from Alpaca's full symbol list, *if* inactive symbols are complete (unverified) | Same caveat | Larger (symbol master, delistings, cost model): 2–3 weeks | Later, if S&P 400 shows promise |

**Power.** Adding about 400 mid caps roughly doubles cross-sectional breadth. That lowers the tracking error of a
diversified overlay, but each test still has 7.5 years (IR standard error about 0.37). Breadth helps the *size* of
the detectable effect only through larger expected IR, not more years.

## Strongest next hypothesis: filing-text changer avoidance in the PIT S&P 400 (design only)

**Why this one ranks first:**
- It is the only candidate whose primary verified evidence explicitly says the effect lives outside the S&P 500
  (Cheng et al. Table 12, read before any HyperShift outcome).
- The full signal pipeline already exists and is validated: EDGAR timing, CIK map, text extraction, pre-checks.
- It is low turnover: the S&P 500 overlay added 1.2× NAV/yr.
- It is free.

**Alternatives considered:**
- *Opportunistic insider purchases* (Form 4, free): post-2008 replications show about 60–70% weaker predictability
  (SECONDARY), signals are sparse, and it needs a new Form 4 parser.
- *Guidance changes and analyst revisions:* need paid I/B/E/S or unverified parsing.

| Item | Specification |
|---|---|
| Research question | Does long-only, FF12-neutral avoidance of the bottom 20% (within form) of 10-K/10-Q cosine similarity beat the equal-weight PIT S&P 400 after 20 bp costs, 2016-07..2023-12? |
| Rationale and evidence | Inattention to disclosure changes (CMN). Significant only outside the S&P 500 (Cheng et al.). Contradiction: no post-publication peer-reviewed test; general anomaly decay (Chen & Welch: 7 bp/month post-2005 non-micro) |
| Signal | **Exactly the frozen filing-text rules** (`../strategy_lazyprices_20261023/PREREG.md` plus amendments A1/A2). No new text choices |
| Data (free) | S&P 400 membership from monthly Wikipedia revisions 2016–2024. Prices: Alpaca SIP `asof` per membership interval (same builder). CIKs: the revision tables' CIK column if present, otherwise SEC ticker table plus name search with the strict name rule and filing-coverage verification. SIC from filing headers |
| Feasibility gate (must pass before any outcome) | Membership snapshots within ±5% of 400 names in ≥ 95% of months, and additions/deletions reconciled against the change table; ≥ 97% of member-days with bars, including delisted names; ≥ 90% of member-months CIK-mapped; the filing-text pre-checks P1–P7 |
| Costs | 20 bp per unit traded (primary), 30/40 bp sensitivities; long-only, no borrow |
| Benchmark | Equal-weight PIT S&P 400 |
| Decision | CONTINUE if net active IR ≥ 0.30, alpha t ≥ 1.0 and net active ≥ 0.30%/yr at 20 bp; otherwise STOP. CONTINUE releases one protected 2024–2026 scoring of this S&P 400 strategy |
| Minimum meaningful improvement | +0.30%/yr net |
| Power | IR standard error about 0.37: a kill-or-license test |
| Stopping rule | One scoring. A STOP closes filing-text avoidance in large and mid caps on free data |
| Engineering | About 3–5 days: membership builder from revisions, panel build (reuse `pit_build_panel.py` with a membership source switch), CIK map rerun, about 15k additional filings to download (about 1 h at SEC limits) |

**First action (bounded): the feasibility probe only.**
1. Parse 3 S&P 400 Wikipedia revisions (2016-07, 2020-01, 2023-12) for table structure and the CIK column.
2. Check Alpaca `asof` bars for 20 deleted S&P 400 names.
3. Report against the gate.

No outcomes.
