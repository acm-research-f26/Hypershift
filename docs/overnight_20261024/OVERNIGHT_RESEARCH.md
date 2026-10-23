# Overnight literature review: public-information signals for a point-in-time S&P 500 book (2026-10-24)

## Purpose and ground rules

Decision supported: **which single hypothesis to test next if the filing-text changer-avoidance test STOPs.**

- This document extends `docs/strategy_lazyprices_20261023/LIT2_VERIFY.md` (L2) and
  `docs/strategy_pit_20261023/LIT_VERIFY.md` (L1, 40 papers). Entries verified there are cited, not re-done.
- **Access labels:**
  - FULL TEXT: body read tonight, or in L1/L2 as stated.
  - ABSTRACT ONLY.
  - SECONDARY: summary, blog or search snippet; unverified.
- No number below was produced by this review; every number comes from the cited source at the stated access level.
- A strong historical result is treated as evidence about the past, not about current profitability.

### Hard constraints for ranking (the HyperShift situation)

- **C-a, free point-in-time timestamps.** EDGAR acceptance times are free and exact. I/B/E/S, transcripts and paid
  news are excluded unless purchased.
- **C-b, S&P 500 applicability.** The only point-in-time price universe is the Alpaca S&P 500 panel for 2016–2023.
  Pre-2016 prices need Norgate, which is not authorised.
- **C-c, post-2010 evidence in large caps.** Without it, the realistic prior is near zero.

---

## A. SEC filing-text changes

| Source | Access | What it shows | Relevance |
|---|---|---|---|
| Cohen, Malloy & Nguyen, "Lazy Prices" (NBER WP 25084; JF 2020) | FULL TEXT (L2 §1) | 1995–2014, all 10-K/10-Q. Value-weighted Q5−Q1 cosine 5-factor alpha 0.34%/month (t 2.53). Both legs carry alpha in calendar time; in event time the changer (short) side persists and the long side reverts. 86% of changes are negative in tone; the 14% positive changes predict positive returns. No announcement reaction | The discovery sample ends in 2014; no S&P 500 table and no net-of-cost result |
| Cheng, Liu, Qiao & Wang, "Attentive Options Traders" (JFQA, accepted March 2026) | FULL TEXT (L2 §3) | 1996–2017 optionable firms. The effect is "significant only for non-S&P 500 stocks". Stronger for negative-tone changes | **Strongest direct prior against the running test** |
| Padysak (2020), "The Positive Similarity of Company Filings and the Cross-Section of Stock Returns" (SSRN 3690461) | SECONDARY (Quantpedia summary, read tonight) | About the largest 1,000 US stocks, Brain similarity data, Feb 2007–May 2020. For **positive-language** similarity the pattern reverses: lowest-similarity decile 1.14%/month vs highest 0.73%; bottom − top 5.47%/yr, volatility 6.48%, return/volatility 0.84. For uncertainty language the original direction holds (no numbers given). Monthly hold, no costs, no t-statistics | Sign depends on *what* changed. A tone-blind full-document cosine mixes opposite-signed effects, a plausible reason for a weak full-document result in large caps |
| Sadlo, "Copy-Paste Outperformance: Lazy Investors and Copied Reports" | **NOT FOUND** | Searched tonight: two web-search queries (exact title and author+topic), Google Scholar (exact title: no match; one 2025 Korean MD&A paper surfaced), Semantic Scholar API (HTTP 429). L2 had SSRN returning 403. Semantic Scholar lists a "Sven Sadlo" with a 2016 SSRN paper on retrieving annual reports (10.2139/SSRN.2870309), not this title | **Do not cite.** Existence of the paper under this title is unconfirmed |
| Two GitHub replications (S&P 100 / S&P 500, current constituents) | SECONDARY (L2 §4a) | Full-document 10-K cosine long-short about zero in large caps; Risk-Factors-only +15 bp/month (p 0.08) | Survivorship-biased; the Risk Factors result lies inside our test window, so choosing it now would be selecting on outcomes |
| QuantConnect student study (Brain Language Metrics, 100 most liquid US stocks, 2020-01 to 2026-06) | SECONDARY (read tonight) | Long-only top-25 by filing similarity with a max-Sharpe optimiser: Sharpe 0.558 vs SPY 0.533; 9 of 25 parameter cells beat SPY; no costs, in-sample parameter choice | Recent large-cap evidence of **no meaningful edge** (weak evidence) |
| Brown & Tucker (2011, JAR) | ABSTRACT ONLY (L2) | The price reaction to MD&A modifications weakened over time | Decaying attention to changes |
| LLM applications. Kim, Muhn & Nikolaev, "Bloated Disclosures" (arXiv 2306.10224) | ABSTRACT + arXiv comment (read tonight) | **Withdrawn**: "A co-author attempted to independently replicate key results ... the analyses did not yield results supporting the reported findings" | A negative data point for LLM-on-disclosure return claims |
| Lopez-Lira & Tang, "Can ChatGPT Forecast Stock Price Movements?" (arXiv 2304.07619 v6, Oct 2025) | ABSTRACT ONLY (read tonight) | News headlines, not filings. GPT-4 scores predict drift "especially for small stocks and negative news"; "Strategy returns decline as LLM adoption rises" | Any LLM-text edge is concentrated outside large caps and is being competed away |

**Direction of the effect.**
- Negative changes carry the persistent effect (CMN tone split; Cheng et al. stronger for negative tone).
- Positive-tone changes may have the opposite sign (CMN 14% positive changes; Padysak, SECONDARY).
- *Implication:* a tone-blind cosine is a noisy mixture. A tone-signed change is the natural refinement, but choosing
  it after seeing 2016–2023 would be selection on outcomes. It would need the protected window or prospective data.

**Avoidance vs shorting.** CMN's persistent side is the changer (short) side, so long-only avoidance captures it
without borrow, but only by underweighting. The active effect is about ¼ of the spread (`LIT2_SYNTHESIS.md`).

**Large caps and recent years.**
- No peer-reviewed post-2014 test was found (L2, re-searched tonight).
- Cheng et al. place the effect outside the S&P 500 even in-sample. Recent large-cap practitioner evidence is null.

**Main failure reason:** the large-cap market already prices disclosure changes. Option traders react (Cheng et al.),
and EDGAR attention and LLM adoption have risen.

## B. Insider trading (Form 4)

| Source | Access | What it shows |
|---|---|---|
| Cohen, Malloy & Pomorski (2012, JF), "Decoding Inside Information" | FULL TEXT (L1 §15, NBER WP) | 1986–2007. Routine trades (same calendar month in past years) are uninformative; opportunistic trades earn 82 bp/month value-weighted (t 2.15) |
| Lakonishok & Lee (2001, RFS 14(1) 79–111) | ABSTRACT ONLY (read tonight) | 1975–95: "very little market movement" when insiders trade or report; cross-sectional predictability "driven by ... smaller firms"; information is in purchases, "insider selling appears to have no predictive ability" |
| Ali & Hirshleifer (2017, JFE, 10.1016/j.jfineco.2017.09.002) | SECONDARY (search summaries of the abstract) | Insiders identified by profits on trades before quarterly earnings announcements; value-weighted long-short four-factor alpha above 1%/month, including the short side |
| Kravet & Lee (2026), "When are Insider Purchases Credible Signals..." (SSRN 7320700) | SECONDARY (CLS Blue Sky summary, read tonight) | 2003–2022, more than 97,000 executive/director purchases. Five-day reaction from the Form 4 filing "approximately 2%"; 60-day abnormal return 3.4% after non-acquisition purchases vs 0.3% after post-acquisition purchases. No size split or time trend reported |
| arXiv 2602.06198 (2026), insider purchases in microcaps | SECONDARY (read tonight; single-author style, unrefereed) | Form 4 purchases 2018–2024, $30M–$500M firms; 60% of lines accepted at or after the 16:00 close; median reporting lag 2 days. The calendar-time alpha falls from 2.10%/month to 0.68% (t 1.78) once the first event day is skipped. A liquid-universe extension "fails every matched-comparison gate" |
| Aalto MSc thesis replication of CMP (2012) | SECONDARY (search snippet; the PDF returned 403) | Opportunistic buys still predict returns, but alphas about ¼–⅓ of the original |

**Mechanism:** insiders hold private information; opportunistic (non-calendar) purchases reveal it.

**Information and timing.**
- Form 4 is due within 2 business days of the trade (Sarbanes-Oxley §403, since 29 Aug 2002).
- XML Form 4 exists from 2003. The SEC "Insider Transactions Data Sets" are quarterly, flattened bulk files from the
  XML Forms 3/4/5. Their existence was confirmed tonight (secondary listings); their **start date is not verified**.
- Acceptance timestamps are exact. The 2023 Rule 10b5-1 amendments add a Form 4 checkbox for plan trades (general
  knowledge; not verified tonight).

**Holding period, turnover, S&P 500 fit.**
- Holding: 1–6 months. Turnover is event-driven and sparse.
- **Open-market purchases are rare in S&P 500 firms.** Most large-cap insider activity is sales, much of it under
  10b5-1 plans. A purchase-based tilt touches few names per month, so breadth and power are low.
- Sales are weak or uninformative (LL 2001), except via the Ali–Hirshleifer opportunism trait (SECONDARY).

**Costs and data:** free (EDGAR). Medium engineering: parse Form 4 XML or the bulk sets, map reporting owners and
issuer CIKs (the issuer CIK is direct), classify routine vs opportunistic from 3 years of history.

**Main failure reason:** the S&P 500 has too few informative events; the evidence lives in small firms; the
documented alphas decayed (thesis, SECONDARY); and a large part of the reaction happens on the first event day.

## C. Earnings information

| Source | Access | What it shows |
|---|---|---|
| Martineau (2022, CFR 11, 613–646), "Rest in Peace PEAD" | FULL TEXT (L1 §12) | Drift after **SUE**-type surprises absent for all but microcaps since 2006; announcement-day response 6× larger in 2016–2019 than in 1984–1990 |
| Daniel, Hirshleifer & Sun (2020, RFS; NBER WP 24163) | **FULL TEXT (read tonight)** | PEAD factor sorted on the **4-day market-adjusted return around the latest announcement** (days −2 to +1; Compustat RDQ at least 2 trading days before month end; excluded if no announcement in 6 months). 2×3 NYSE size × 20/80 surprise breakpoints, value-weighted, monthly. 1972–2014: 0.65%/month, t 7.91; **big-firm leg 0.38%/month vs small 0.94%**; 1991–2014 subperiod 0.55%/month, t 4.59. The kitchen-sink alpha is 0.58% (t 6.76) |
| Dickerson, Julliard & Mueller, "The Co-Pricing Factor Zoo" (arXiv 2604.04430, April 2026; replication package says JFE forthcoming) | **FULL TEXT (read tonight)** | 1986–2022. The DHS PEAD factor is one of only two tradable factors with posterior probability > 50% in the stock+bond SDF. Footnote 26: "Despite its reduced time series predictability in most recent data (see, e.g., Martineau (2022)), ... the cross-sectional predictability of the post-earnings announcement drift within a portfolio context remains robust and does not appear to be driven by micro-cap stocks" (Internet Appendix IA.4, not read) |
| Hirshleifer, Peng & Wang (2025, RFS 38(3) 883–937; NBER WP 30860) | ABSTRACT ONLY (read tonight) | Higher social-network centrality gives stronger immediate reactions and **weaker drift**. This supports the attention mechanism; it is not evidence of large-cap drift |
| Subrahmanyam (2025), reconciliation of PEAD disagreement | SECONDARY (UCLA Anderson brief, L2 §5(v); the title "Keeping it Simple" and SSRN 5930255 not re-confirmed tonight) | Feb 2001–Dec 2024: drift factor t 2.18 with all stocks, t 1.43 excluding microcaps; the disagreement is attributed to how drift is measured and to microcap inclusion |
| Fink (2021), "A review of the PEAD" | SECONDARY (search snippet; ResearchGate 403) | "for large, liquid US stocks no significant PEAD can be observed anymore", while across sizes recent evidence "typically still confirms a PEAD" |
| Heitz, Narayanamoorthy & Zekhnini (2020 WP), "The Disappearing Earnings Announcement Premium" | SECONDARY (search snippet; PDF 403) | The US announcement premium "has disappeared ... in recent years"; robust internationally; attributed to a US-specific 2004 change |
| Frazzini & Lamont (2007, NBER 13090); Savor & Wilson (2016, JF) | FULL TEXT (L1 §13–14) | Announcement-month premium (to 2004/2009) |
| Analyst revisions (Stickel 1991; Barber et al. 2001) | SECONDARY (L1 §39, L2 §5(iv)) | Needs I/B/E/S (paid); no recent large-cap evidence found |
| Earnings-call language / LLMs | see section A | Transcripts are paid; LLM evidence is small-cap-tilted, decaying, partly withdrawn |

**Reconciliation, and the key unknown.** The papers measure different things.
- Martineau measures drift after an *accounting* surprise (SUE).
- DHS and DJM use the *market's announcement-window reaction* (CAR). A CAR-based surprise already embeds whatever the
  market understood, and DHS argue it better captures persistence.
- DJM say the CAR-based factor is robust to 2022 and not microcap-driven, but the factor still holds the small half
  of the market with equal weight to the big half.
- **No source read tonight shows CAR-based drift inside S&P 500 names after 2010.** That is the open question, and it
  can be tested on the data HyperShift already has.

**Information and timing.**
- **8-K Item 2.02** (Results of Operations) is furnished when earnings are released (within 4 business days by rule;
  in practice usually concurrent). The EDGAR acceptance timestamp gives a free, exact announcement time for S&P 500
  firms from 2004 on. One caveat (secondary law-firm note): the SEC may back-date the deemed receipt time after EDGAR
  outages.
- The announcement-window return comes from the PIT price panel. The signal is known at the close of day +1 and
  tradable the next open.

## D. Other public information

| Signal | Access | Evidence | Assessment |
|---|---|---|---|
| Risk-factor (Item 1A) changes | L2 (CMN Table VII: EW 5-factor about 188 bp/month, post-FY2005 subsample); GitHub S&P 500 +15 bp (SECONDARY) | The strongest section in CMN, but its large-cap support is post-2015, inside our window | Selection on outcome if adopted now; a candidate only for prospective data |
| MD&A changes | CMN Table VII (MD&A EW 0.20%/month, t 2.81); Brown & Tucker (ABSTRACT) | Weaker than Risk Factors; reaction decaying | Low |
| 8-K events (e.g. Item 4.02 non-reliance, 5.02 officer departures) | Lerman & Livnat (2010, RAS 15(4)), ABSTRACT ONLY | "some items have significant return drifts after the SEC filings" | Free and exact timestamps, but rare events in the S&P 500, so low breadth and power |
| Regulatory text rewrites | General knowledge: SEC Reg S-K Items 101/103/105 amendments effective 9 Nov 2020; COVID risk factors in 2020 filings | Mass rewrites in FY2020 10-Ks and 2020 10-Qs | Not a signal: a confound for any text-change measure (see the risk review) |

---

## Ranked comparison: the five most promising hypotheses

The ranking rests on the three constraints (C-a timestamps, C-b S&P 500 fit, C-c post-2010 large-cap evidence), then
on power and cost. "Prior net IR" is a judgement, not an estimate.

| Rank | Hypothesis | Mechanism | Info and exact timing | Holding / turnover driver | Post-2010 large-cap evidence | S&P 500 fit | Data cost | Complexity | Cost sensitivity | Main failure reason | Prior net IR (judgement) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **Earnings-announcement-return drift (EAR) overlay**: underweight names with the most negative 3-day market-adjusted reaction to their latest 8-K Item 2.02 release, within industry | Limited attention / underreaction to earnings news | 8-K 2.02 acceptance time (free, exact); prices from the PIT panel; signal at close of day +1 | 1–3 months; turnover from the quarterly announcement cycle (about 4 refreshes per name per year) | Mixed: DJM (FULL TEXT) robust to 2022, not microcap-driven; DHS big-firm leg 0.38%/month to 2014; Martineau (SUE) none for large caps since 2006 | Direct: every member files 8-K 2.02 | **Zero.** Submissions JSON already cached for all mapped CIKs; PIT panel in hand | Low: reuses `lazyprices`/`pit_engine` | Medium (about 2× NAV/yr active, comparable to the running test) | Large-cap drift arbitraged away (Martineau); collides with 1-month reversal | 0.1–0.3 |
| 2 | Opportunistic insider purchases (CMP classification) as a tilt | Private information | Form 4 within 2 business days; acceptance time exact | 1–6 months; event-driven, sparse | Weak: small-firm concentration (LL 2001); decay (SECONDARY) | Poor: few purchases in large caps | Free; medium parsing | Medium | Low (sparse) | Too few events, so no power; the first-day reaction is not capturable | 0–0.1 |
| 3 | Tone-signed filing change (negative-tone changers only; Risk Factors section) | Inattention to negative disclosure | Same as the running test | Quarterly | Only SECONDARY and in-window | Same panel | Free | Low–medium (sentiment dictionary, sectioning) | Medium | Outcome-selected if chosen after the current result; prospective only | 0–0.2 |
| 4 | Earnings-announcement premium (overweight expected announcers) | Attention-driven buying / announcement risk | Expected date from the prior-year 8-K 2.02 (free) | 1 month; high, calendar-driven | Reported disappeared in the US (SECONDARY) | Direct | Free | Low | High | Premium gone in the US after 2004 | about 0 |
| 5 | 8-K adverse-event avoidance (4.02, 5.02 CFO exits, NT 10-K late filings) | Slow incorporation of governance/accounting red flags | 8-K acceptance (free, exact) | 3–6 months; sparse | Abstract-level only | Few events per year in the S&P 500 | Free | Low | Low | No power; events often accompanied by a same-day price move | 0–0.1 |

Excluded on cost or evidence:
- Analyst revisions and transcripts: paid data.
- LLM tone of filings or calls: cost, small-cap tilt, a withdrawn flagship paper, decay with adoption.
- Lazy Prices in a broader universe: needs a point-in-time small/mid-cap price panel that does not exist; logged as
  INVESTIGATE.

## Choice: the strongest alternative if the filing-text test STOPs

**H1: earnings-announcement-return drift in the point-in-time S&P 500, as an industry-neutral long-only avoidance
overlay keyed on 8-K Item 2.02 acceptance times** (not implemented; spec in `NEXT_RESEARCH_SPEC.md`).

Why this one:
1. It is the only candidate that meets all three constraints. Timestamps are free and exact, every S&P 500 member
   files 8-K 2.02, and there is post-2010 evidence: DJM, full text, a factor robust to 2022 and not microcap-driven.
2. It costs nothing new. The submissions JSON the running test already cached lists every 8-K with its items and
   acceptance time, and the PIT panel and the engine are reusable unchanged.
3. Its information differs from HyperShift's price-only models: it conditions on a dated corporate disclosure, and
   only the announcement-window reaction is price-based.
4. Contradictory evidence names the exact risk (Martineau; Fink, SECONDARY), so the test is a clean kill-or-license
   with a pre-stated failure mode.

What would make it fail: the large-cap drift is gone after 2006 (Martineau), or it is masked by 1-month reversal,
which HyperShift found to be the strongest price-only signal (16-day reversal, FB study).

## Sources consulted tonight (beyond L1/L2)

- [NBER WP 24163, Daniel–Hirshleifer–Sun](https://www.nber.org/papers/w24163.pdf) (full text)
- [arXiv 2604.04430, Co-Pricing Factor Zoo](https://arxiv.org/pdf/2604.04430) (full text); [replication package](https://data.mendeley.com/datasets/sxrmkvt3k8)
- [Quantpedia summary of Padysak (2020)](https://quantpedia.com/the-positive-similarity-of-company-filings-and-the-cross-section-of-stock-returns/)
- [QuantConnect filing-stability study](https://www.quantconnect.com/research/20966/filing-language-stability-as-a-selection-signal/)
- [arXiv author page: Bloated Disclosures withdrawal](https://arxiv.org/a/kim_a_2.html); [arXiv 2304.07619 v6](https://arxiv.org/abs/2304.07619v6)
- [CLS Blue Sky on Kravet & Lee (2026)](https://clsbluesky.law.columbia.edu/2026/09/22/when-are-insider-purchases-credible-signals-of-private-information/); [arXiv 2602.06198](https://arxiv.org/html/2602.06198v2)
- [Lakonishok & Lee (2001), RePEc](https://ideas.repec.org/a/oup/rfinst/v14y2001i1p79-111.html); [Ali & Hirshleifer, AQR page](https://www.aqr.com/About-Us/AQR-Insight-Award/2018/Opportunism-as-a-Firm-and-Managerial-Trait-Predicting-Insider-Trading-Profits-and-Misconduct)
- [Hirshleifer–Peng–Wang, NBER 30860](https://www.nber.org/papers/w30860); [UCLA Anderson brief](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/)
- [Lerman & Livnat (2010), RePEc](https://ideas.repec.org/a/spr/reaccs/v15y2010i4d10.1007_s11142-009-9114-7.html)
- Searches that failed to verify: Sadlo (Google Scholar, web, Semantic Scholar 429); Heitz et al. PDF (403); Fink
  review (403); Aalto thesis (403).
