# Literature verification for the S&P 500 earnings-announcement-return (EAR) overlay (2026-10-23)

Decision supported: whether an industry-neutral, long-only avoidance overlay keyed on the announcement-window
return (CAR/EAR) of the latest 8-K Item 2.02 release (hypothesis H1 in `docs/overnight_20261024/OVERNIGHT_RESEARCH.md`)
has literature support in S&P 500 names after 2005–2010, plus six side questions.

## Access legend

- **FULL TEXT (version):** body read tonight (or in an earlier file, as stated). The version is named because
  several papers differ across versions.
- **FULL TEXT (OCR):** text recovered by rendering the PDF and running Windows OCR (the NBER PDF has a broken font
  map). Numbers are read from running text, not from tables, and OCR errors are possible.
- **ABSTRACT ONLY.**
- **SECONDARY:** summary, search snippet or blog. Unverified.
- **NOT VERIFIED:** looked for, not confirmed.

Reused, not redone (earlier files): Cohen-Malloy-Nguyen (CMN) 2020, Cheng et al. (JFQA), Martineau abstract/intro,
Frazzini-Israel-Moskowitz (FIM) 2012 costs, Novy-Marx-Velikov (NMV) costs, Lakonishok-Lee 2001, Kravet-Lee 2026,
arXiv 2602.06198, Padysak, GitHub replications. Source files: `OVERNIGHT_RESEARCH.md` (OR), `LIT_VERIFY.md` (L1),
`LIT2_VERIFY.md` (L2). No SEC site was contacted. PDFs are in `C:\Users\Hi\.claude\jobs\49450b0d\tmp\lit3\`.
Wikipedia was queried through its public API. Printed page numbers below are the working-paper page numbers.

---

## 1. Daniel, Hirshleifer & Sun (DHS), "Short- and long-horizon behavioral factors"

- **Citation:** Review of Financial Studies 33(4), 2020 (volume/pages not re-checked). NBER WP 24163, December 2017;
  the PDF carries an SSRN stamp (SSRN 3086063). https://www.nber.org/papers/w24163
- **Access:** FULL TEXT (NBER WP 24163, Dec 2017). The 2020 RFS print version was not read.

**Verified definition of the PEAD factor (WP pp. 12–13, Section 1.1):**
- Signal: the **four-day cumulative abnormal return around the most recent quarterly earnings announcement date**,
  sum over days d = -2 to +1 of (R_i,d - R_m,d), where R_m is the market return and day 0 is the announcement date.
  It follows Chan, Jegadeesh & Lakonishok (1996). The text writes the sum as a simple (arithmetic) sum of
  differences.
- Announcement date: **Compustat quarterly item RDQ.** Valid daily returns are required on at least two of the four
  days. RDQ must be at least **two trading days before month end**.
- Staleness: the latest announcement is used. If there is no announcement in the past six months the firm is excluded.
- Universe: all NYSE/AMEX/NASDAQ common stocks (CRSP share codes 10, 11), **excluding financial firms**.
- Portfolio, monthly, a 2x3 sort in the Fama-French (1993) fashion:
  - size: firm is Small or Big by the **NYSE median** of month t-1 market equity;
  - CAR group: independent sort on CAR at the end of month t-1 with **NYSE 20% / 80% breakpoints** into L, M, H;
  - six value-weighted portfolios, held for the current month;
  - PEAD(t) = (r_SH + r_BH)/2 - (r_SL + r_BL)/2, so the two size legs get equal weight.
- Holding implication (derived from the definition, not a quote): each month the whole cross-section is re-sorted on
  the latest CAR, so a name stays in a leg for up to about six months after its announcement, until a newer one
  replaces it.
- Motive for CAR over SUE (footnote 15, p. 13): SUE is a noisy proxy for expected earnings and ignores persistence;
  a return-based measure does not, and cites Brandt et al. (2008) that CAR-type measures better forecast returns.
- Sample: July 1972 to December 2014 (N = 510 months; Table 1, Panel A).

**Verified results:**

| Item | Value | Where |
|---|---|---|
| PEAD mean, 1972:07–2014:12 | 0.65%/month, t 7.91, monthly Sharpe 0.35 (highest t among the factors) | Table 1 Panel A; text p. 15 |
| Big-firm leg PEADB = r_BH - r_BL | 0.38%/month | footnote 18, p. 15 |
| Small-firm leg PEADS = r_SH - r_SL | 0.94%/month | footnote 18, p. 15 |
| 1972–1990 | 0.77%/month, t 7.32 | footnote 19, p. 16 (Internet Appendix Table A1) |
| 1991–2014 | 0.55%/month, t 4.59 | footnote 19, p. 16 |
| Decay of CAR portfolios | return predictability "much weaker or insignificant just two quarters after portfolio formation" (Table 5) | Section 3, footnote 33 area |
| PEAD loadings | PEAD beta coefficients are insignificant in all return-prediction models; the PEAD characteristic (CAR) correlates below 0.05 with the estimated PEAD beta | footnote 33 |

The "big-firm leg" is every NYSE-median-and-above firm, not the S&P 500. No post-2014 result is in this WP.
The kitchen-sink alpha of 0.58% (t 6.76) in OR was not re-read tonight.

**Implication for the S&P 500 EAR overlay:** The signal to copy is a market-adjusted [-2,+1] CAR around the latest
announcement, re-sorted monthly, with firms dropped after six months. The literature that supports "CAR drift" is
factor-level, equal-weights big and small, ends in 2014, and its big leg is 0.38%/month in a pooled size half, not
S&P 500 names after 2006.

---

## 2. Dickerson, Julliard & Mueller (DJM), "The Co-Pricing Factor Zoo"

- **Citation:** Alexander Dickerson (UNSW), Christian Julliard (LSE, CEPR), Philippe Mueller (Warwick). arXiv
  2604.04430v1 [q-fin.PR], 6 April 2026. https://arxiv.org/abs/2604.04430
- **Venue/status:** the arXiv version is a preprint. The UCLA Anderson brief (SECONDARY) calls it "in press" at the
  Journal of Financial Economics, and OR reports that the replication package says JFE forthcoming. The arXiv text
  itself cites a different Dickerson et al. (2025) paper and does not state a venue for this one. JFE status is
  **NOT VERIFIED from a primary source.**
- **Access:** FULL TEXT (arXiv v1).

**Verified:**
- Sample: 1986:01 to 2022:12 (T = 444); out-of-sample evaluation July 2004 to December 2022.
- The stock "PEAD" factor is the DHS factor: the factor table attributes it to "Daniel et al. (2020a)", described
  as the short-term behavioral factor reflecting post-earnings-announcement drift. Footnote 9 says all factors come
  from the authors' websites or public repositories. The arXiv text does not reconstruct the factor, so its
  construction is exactly the DHS definition above (inference from the attribution).
- Main text: PEAD "exhibits a particularly high posterior probability" of being in the stock SDF (cites Bryzgalova
  et al. 2023, Avramov et al. 2023). PEAD and the bond analogue PEADB are the only two tradable factors with
  posterior probability above 50% in the joint bond-stock SDF. They have the highest Sharpe ratios in the full
  sample (Internet Appendix Table IA.III); PEAD has the highest Sharpe among stock factors in the first half of the
  sample and "one of the highest" in the second (IA.IV, not read).
- **Footnote 26 (p. 21, verbatim):** "Despite its reduced time series predictability in most recent data (see,
  e.g., Martineau (2022)), we document remarkable stability of the post-earnings announcement drift for forming
  long-short corporate bond and stock portfolios across subsamples in Internet Appendix IA.4. That is, the
  cross-sectional predictability of the post-earnings announcement drift within a portfolio context remains robust
  and does not appear to be driven by micro-cap stocks."
- Robustness caveat (p. 47 area): under the lowest prior-shrinkage setting, PEAD's posterior probability drops below
  its prior threshold.

**NOT VERIFIED:** Internet Appendix IA.4 (the only place the "not micro-cap" claim is documented) was not available.
The claim is a statement about a **value-weighted factor portfolio that still gives equal weight to the small half**
(DHS design), not about S&P 500 names.

**Implication for the S&P 500 EAR overlay:** DJM is evidence that the DHS factor still prices cross-sections through
2022. It does not show that the big-firm or S&P 500 leg earns drift after 2006, and its pricing is joint with the
small half. Treat it as a factor-pricing result, not a large-cap tradability result.

---

## 3. Martineau (2022), "Rest in Peace Post-Earnings Announcement Drift"

- **Citation:** Critical Finance Review 11(3–4), 613–646. CFR draft PDF (the draft is numbered pages 1–46).
  https://cfr.ivo-welch.org/published/papers/martineau2021rest.pdf
- **Access:** FULL TEXT (CFR draft, re-read tonight for the CAR question; earlier read in L1 §12).

**Surprise measures actually used:**
- Analyst surprise (Eq. 1): actual EPS minus the median of analyst forecasts issued in the prior 90 days, scaled by
  price five trading days before the announcement, winsorized at 1/99%. I/B/E/S sample 1984–2019.
- Random-walk surprise (seasonal change in EPS) from Compustat, 1973–2019 (593,654 announcements).
- Surprises enter as **decile ranks**; Table 3 regresses BHAR[0,1] (Panel A) and BHAR[2,60] (Panel B) on the rank.
- BHAR is a buy-and-hold return net of the matched Fama-French 25 size/book-to-market portfolio (Eq. 4).
- "All-but-microcap" = market cap above the NYSE 20th percentile.

**Does he test announcement-return (CAR/EAR)-based drift? Partly, in one test, not as a sort:**
- Section 5 / **Figure 10** (draft p. 34): a 2-year rolling "unbiasedness regression" of **BHAR[0,60] on BHAR[0,1]**
  (Eq. 6) on the I/B/E/S sample, 1990–2019. The slope is 1 if the announcement-day price is a martingale; above 1
  means underreaction. Reported readings:
  - full sample: about 1.4 (1990) to about 1.2 recently (trend 1.32 - 0.006 per period unit as printed, p 0.003);
  - **all-but-microcap: from about 1.6 to "almost one in recent years"** (trend 1.38 - 0.012, units as printed, p < 0.001):
    "announcement day prices for these stocks are approximately martingale";
  - microcaps: no trend (1.24 + 0.002, units as printed, p 0.361), "prices do not fully reflect all the earnings news at the
    time of announcements."
- R-squared of that regression rises from about 10% to about 25% (all-but-microcap) and 20% (microcap) in recent
  years (Figure 10 Panel B).
- This is the only test conditioned on the announcement return. It conditions on a 2-day announcement return,
  not the DHS [-2,+1] CAR, and the slopes are read from a figure, not a table.
- A text search of the CFR draft finds **no citation to Daniel-Hirshleifer-Sun and none to Brandt et al.**. The only
  related citation is Hirshleifer et al. (2009), used for the benchmark-adjusted return.

**Large-cap statements, verbatim from the draft:** abstract: "For large stocks, PEAD have been non-existent since
2006 but has only disappeared recently for microcap stocks." Introduction: "since 2006, analyst earnings surprises
fail to positively predict post-announcement returns over 60 days for all-but-microcap stocks, and since 2016 for
microcap stocks"; announcement-date responsiveness is six times (all-but-microcap) and three times (microcap) larger
in 2016–2019 than in 1984–1990.

**Implication for the S&P 500 EAR overlay:** The paper's conclusion for large stocks rests on SUE-type sorts plus one
announcement-return regression whose slope converges to 1 for non-microcaps. That regression is the closest direct
evidence on CAR-conditioned drift in large stocks, and it points to no drift. It does not run the DHS CAR sort.

---

## 4. Brandt, Kishore, Santa-Clara & Venkatachalam, "Earnings announcements are full of surprises"

- **Citation:** working paper (Duke/UCLA/NBER), never found in a journal. Cited by DHS as 2008. SSRN 909563 (search
  result); the SSRN page itself returned 403.
- **Access:** FULL TEXT of the **June 2007 version** (UCLA host:
  https://www.anderson.ucla.edu/documents/areas/fac/finance/ear.pdf). **The versions differ:** the SSRN abstract
  (January 2008, SECONDARY via search) reads 7.55% a year, 1.3% above SUE, "no reversal after 3 quarters", about 12.5%
  combined. The June 2007 text reads 6.3%, 0.7% above SUE, and about 11% combined. Use the June 2007 numbers below.

**Verified (June 2007 text):**
- EAR = three-day [-1,+1] abnormal return, stock minus its size/book-to-market Fama-French portfolio (Eq. 4).
  Sample 1987–2004, Compustat Industrial Quarterly (RDQE dates) and CRSP; SUE from a seasonal random walk, with
  I/B/E/S as a robustness check.
- Quintile breakpoints use the prior quarter's observations to avoid look-ahead; portfolios form the day after the
  announcement.
- EAR drift: 60-day long-short abnormal return 2.8%, 120-day 3.7%, 180/240-day 5.4% / 6.3%. SUE: about 3%, 4.6%,
  5.7%, 5.6% (Section 4.1, Table 2 Panel A). EAR and SUE are largely independent; combined about 11% a year.
- **Timing of the drift (important):** EAR strategy returns are concentrated "around the earnings announcements in
  the following four quarters" ("a set of stairs"), while SUE returns accrue gradually. The long-short three-day
  return at each of the next four announcements is about 3%. The quarter-1 three-day spread is 2.67% (1987–1995) and
  3.14% (1996–2004), rising over time (Table 5), where the SUE spread falls.
- **Reversal caveat (p. 15):** the EAR strategy earns more around announcements than over the whole period
  "suggest[ing] perhaps that some of the abnormal returns generated during the earnings announcements reverse
  subsequently."
- Monthly implementation (Table 6): re-sorting every month, the EAR hedge is 6.7% a year, against 11.7% annualized
  when the one-quarter-ahead announcement returns are annualized instead. The authors say monthly re-sorting closes
  positions before the next announcement.
- **Large firms (Section 4.4.2, Table 7):** top 1,000 CRSP firms by size. The profitability of the EAR sort "is
  considerably reduced, yet far from eliminated": one quarter ahead, the EAR5-EAR1 three-day spread is 89 bp to
  1.4% (versus at least 2.5% in the full sample); four quarters ahead about 2.0% (versus almost 4.0%). SUE spreads are
  negative in 70% of the large-firm cells. Table 7 is again subsequent-announcement-window returns, not
  between-announcement drift.
- Predecessor point: Chan et al. (1996) used announcement returns only as a proxy for earnings surprise and did not
  study subsequent return patterns.

**Implication for the S&P 500 EAR overlay:** The paper's EAR "drift" is mostly a repeat reaction at the next four
announcements, not a steady drift between them. A 1–3 month avoidance overlay that holds no position through the next
announcement would capture the smaller between-announcement part. Large-firm results end in 2004 and are a fraction
of the full-sample size. There is also a hint of announcement-date reversal in the same table.

---

## 5. Chan, Jegadeesh & Lakonishok (1996), "Momentum Strategies"

- **Citation:** Journal of Finance 51(5), 1681–1713. NBER WP 5375, December 1995.
- **Access:** FULL TEXT (OCR) of NBER WP 5375 (Dec 1995). The PDF's text layer is garbled; the OCR text was read for
  Sections II–V. Earlier files had only the abstract (L1 §5). A CFA Digest summary (SECONDARY) confirms the sample.
- Sample: all domestic primary stocks on NYSE, Amex, Nasdaq; January 1977 to January 1993 (CFA Digest and Table
  notes).

**Verified definition and results (running text, Section II.B and Tables 3–8):**
- **ABR** is the abnormal return **relative to the equally-weighted market index cumulated from two days before to
  one day after the most recent past announcement date** of quarterly earnings. This is the window DHS adopt.
- Table 4 (sort on ABR): stocks with large favorable announcement returns outperform those with large unfavorable
  ones by **5.9% in the first six months and 8.3% in the first year** (full sample). The text also says the news in
  the announcement return "continues to be played out at the next announcement following portfolio formation", with
  an 8.8% spread in a panel (Table 4 Panel D) whose cell I could not pin down through OCR; that 8.8% is not used.
- The announcement-return effect is shorter-lived than price momentum. Conditioning on prior return, high-ABR beats
  low-ABR by 2.8% in six months and 3.8% in a year (Table 6 Panel A).
- Fama-MacBeth (Table 7): six-month slope on ABR 3.7% (rank scale), below SUE and REV6 (about 6%).
- **Large stocks (Section V.A, Table 8):** sample restricted to market cap above the **NYSE median**. One-year spread
  on ABR is **4.3%** (full sample 8.3%); on SUE 2.9% (full 7.5%); on analyst revisions 7.6% (full 9.7%). Two-way
  price + announcement-return spread 8.4%, price + SUE 7.7%, price + revisions 8.5%.
- Abstract: "There is little evidence of subsequent reversals in the returns of stocks with high price and earnings
  momentum."

**Implication for the S&P 500 EAR overlay:** The original CAR-sort evidence is 1977–1993 and "large" means above the
NYSE median, which is far broader than the S&P 500. It supports a smaller but present CAR drift in 1990s large firms,
not a post-2005 one.

---

## 6. Evidence on CAR/EAR drift in large caps after 2005–2010; reversal at one month

No source read tonight isolates S&P 500 names after 2006 for a CAR-based sort.

| Source | Access | Finding | Size / period |
|---|---|---|---|
| Dai, Medhat, Novy-Marx & Rizova, "Reversals and the Returns to Liquidity Provision" (NBER WP 30917, Feb 2023; I found no print venue) | FULL TEXT (WP) | PEAD defined as quintile sort (NYSE breaks, value-weighted) on the **three-day market-adjusted CAR around the latest announcement**: **0.53%/month (t 5.45), 1973–2021**; **post-decimalization (May 2001–Dec 2021): 0.36%/month (t 2.09)**, Table A2 | All NYSE/AMEX/NASDAQ common stocks, VW. No size-split table in the main text; Appendix Figure D3 says size is the strongest determinant of PEAD and momentum strength (figure not read) |
| Subrahmanyam (2025), SSRN 5930255, "Keeping it Simple: How Can Post-Earnings Return Drift Exist and Not Exist Simultaneously?" | SECONDARY (UCLA Anderson brief, re-read tonight; SSRN page 403) | An "off-the-shelf earnings factor" from abnormal returns around announcements: t 2.18 with all stocks, **t 1.43 excluding microcaps**, Feb 2001 to Dec 2024; the brief says microcaps explain the drift | The brief does not name the factor. It is "based on abnormal returns around earnings announcements", which matches DHS but is not confirmed |
| Martineau (2022) | FULL TEXT | Announcement-return regression slope near 1 for all-but-microcap (see §3) | 1990–2019 |
| Chen & Welch, "What Useful Alphas?" (arXiv 2607.06502, 8 July 2026) | ABSTRACT + first page | About 200 published anomalies: median long-short 48 bp/month through 2005 all stocks; **7 bp** post-2005 in non-micro (top 90% of cap) stocks | General anomaly study. Whether announcement-return or PEAD signals are in its set, and their individual values, were **not checked** |
| Fink (2020 WP), "A Review of the PEAD" (U. Graz WP 2020-04; the JBEF 2021 citation is not verified from a primary source) | FULL TEXT (WP, Nov 2020) | Lists abnormal announcement-window returns as a "comprehensive surprise measure" that predicts drift (Brandt et al., CJL, Zhou & Zhu 2012, Gerard 2012); notes PEAD magnitude declining (Chordia et al. 2014; Martineau; Richardson et al. 2010). The OR quote "for large, liquid US stocks no significant PEAD ... anymore" is **not in the WP text** (it came from a search snippet of the published version) | Review; no new tests |
| Kettell, McInnis & Zhao (April 2022 WP), "Why Has PEAD Declined Over Time?" | ABSTRACT + intro | The decline is explained by falling persistence of SUE, not only by arbitrage | SUE-based, not CAR-based |
| Luthra, SSRN 7137218 (2026) | SECONDARY (snippet) | 40 earnings events, 10 large companies, 2025–26: mean 20-day drift +0.67% (SD 6.29%, t 0.67) | Too small to inform anything |

**One-month reversal colliding with drift (FULL TEXT, Dai et al. 2023):**
- Standard one-month reversal (REV, losers minus winners, NYSE breaks, value-weighted, 1973–2021): 31 bp/month
  (t 1.68). Among firms that **announced in the prior month**: 6 bp (insignificant). Among non-announcers: 51 bp
  (t 2.54). Table 1.
- Removing each stock's three-day announcement CAR from last month's return (REVX): 54 bp/month (t 2.96), no
  announcer/non-announcer gap. Industry-relative REVX (IRRX): 108 bp/month (t 9.35) (Table 2).
- Regression of REV on IRRX, PEAD and industry momentum: PEAD loading **-0.54** (t -17.4); the three explain 87% of
  REV's variance. Post-decimalization (Table A2): REV 18 bp (t 0.62), IRRX 58 bp (t 3.29). (The text states "t-statistic
  of 0.18" for REV; the table shows 0.62. Table value used.)
- Reading (corrected): the announcement-CAR part of last month's return **does not revert; it drifts** (same
  direction). A raw reversal signal buys last month's losers, which includes names with a large negative announcement
  CAR, so raw reversal is **short PEAD** (loading -0.54) and its announcer leg is diluted to about zero. Removing the
  CAR (REVX) lifts reversal from 31 to 54 bp/month. So a raw reversal tilt (such as HyperShift's 16-day reversal) and
  a CAR-drift overlay conflict in the same book unless the reversal input has announcement CARs stripped out.
- Related, SECONDARY: Hameed & Mian (2015) cited by Dai et al. for weaker reversal among announcers; a UCLA Anderson
  summary says reversals are about 70% weaker in months after announcements (not verified against the paper).

**Tangential, not evidence against (ABSTRACT, April 2008 WP):** Aboody, Lehavy & Trueman,
"Limited Attention and the Earnings Announcement Returns of Past Stock Market Winners" (UCLA WP, April 2008; I did not
confirm the print venue): stocks with the strongest prior 12-month returns earn a market-adjusted +1.58% in the five
trading days before announcement and **-1.86% in the five days after**, significant after transaction costs. They
attribute it to individual-investor attention, not to news. The sort is on past 12-month returns and the windows are
five days, not an announcement CAR, so it does not test the overlay.

**Implication for the S&P 500 EAR overlay:** No S&P 500-specific test was found. The strongest primary post-2001
evidence (Dai et al.) is a value-weighted CAR quintile spread, rebalanced monthly (so large caps carry most of the
weight and the horizon matches the overlay), of 0.36%/month (t 2.09, May 2001 to Dec 2021), down from 0.53 for the full
sample; it is mildly positive. An ex-microcap factor test (Subrahmanyam, SECONDARY, factor unnamed) is insignificant
(t 1.43). These two are in tension and neither is large-cap-only. The reversal result matters for book design: the CAR
signal and a raw reversal signal are opposite bets on last month's announcers.

---

## 7. Mechanism papers: why drift could follow an already-observed reaction

**DellaVigna & Pollet (2009), "Investor Inattention and Friday Earnings Announcements"**
- Journal of Finance 64(2), 709–749 (SECONDARY via search results). NBER WP 11683, "Investor Inattention, Firm
  Reaction, and Friday Earnings Announcements", October 2005. https://www.nber.org/papers/w11683
- **Access:** FULL TEXT of the WP abstract and sample description. The journal numbers were not seen in a primary
  source.
- WP abstract (verbatim figures): delayed response as a share of the total response is **60% on Friday versus 40%
  on other weekdays**; abnormal announcement-day volume is **10% lower** on Fridays; Friday announcements have a 45%
  higher chance of a negative surprise and 50 bp lower abnormal return. Sample: CRSP stocks matched to Compustat and
  I/B/E/S announcements, January 1995 to June 2004.
- SECONDARY (search summary of the JF abstract): "15% lower immediate response and 70% higher delayed response"
  and 8% lower volume. These differ from the WP (60/40 split, 10% volume), so they come from a different version.
  **JF figures NOT VERIFIED.**

**Hirshleifer, Lim & Teoh (2009), "Driven to Distraction: Extraneous Events and Underreaction to Earnings News"**
- Journal of Finance 64(5), 2289–2325 (SECONDARY; DOI 10.1111/j.1540-6261.2009.01501.x).
- **Access:** FULL TEXT of the MPRA version (MPRA 3110, April 16, 2007), not the JF print.
- Verified: more same-day announcements by other firms weaken the immediate price and volume reaction and strengthen
  drift. Drift-based Good-minus-Bad portfolio, EW, March 1995–December 2004, five quintiles by number of same-day
  announcements: 3-factor alpha **1.64%/month for the high-news quintile (significant at 1%)** against **0.77% and
  insignificant for the low-news quintile**; high-minus-low fund-of-funds alpha 0.86% (10% level). The paper says there
  is "no discernible" drift when there is little competing news. Distraction hits positive-surprise firms more.

**Implication for the S&P 500 EAR overlay:** Both papers use 1995–2004 CRSP samples, before decimalization, algorithmic
trading and the rise of intraday news processing. They explain why drift could follow a visible reaction and predict
more drift on Fridays and busy announcement days, which is a testable interaction for the overlay (not run here), but
they say nothing about S&P 500 names after 2010.

---

## 8. Stable reporters (separate decision): independent evidence beyond CMN

- **CMN event-time behaviour (reused, L2 Fig. 7, p. 18):** non-changer (Q5) alpha "quickly reverts to zero, while
  the negative [changer] alpha persists ... up to 6 months out"; "a significant portion of the return spread ...
  comes from the short side." Both legs earn calendar-time alpha at the 3-month hold. VW Q5 5-factor alphas: Jaccard
  +0.23 (t 2.11), Cosine +0.23 (2.23), MinEdit +0.21 (1.84), Simple +0.15 (1.15).
- **Adosoglou, Lombardo & Pardalos (2021), "Neural network embeddings on corporate annual filings for portfolio
  selection", Expert Systems with Applications 164, 114053 (DOI 10.1016/j.eswa.2020.114053).** Access: **FULL TEXT
  (author postprint, University of Parma)**, upgraded from abstract-only in L2.
  - Sample: 10-Ks 1999–2018, CRSP monthly returns, **market cap above $300 million**, 45,516 firm-years. Annual
    portfolios, held April–March (12 months), annual rebalance.
  - Selection: cosine similarity of PV-DM Doc2Vec vectors above 0.95 ("Semantic Similarity Portfolio", about 60 firms
    a year); the "Non-struggling Portfolio" (NSP) adds a positive-prior-year-return filter, added "to improve our
    five-factor alphas" (about 40 firms a year).
  - Results: NSP value-weighted **3-factor alpha 9.75% a year (t 3.24), 5-factor alpha 8.45% (t 2.61)**; SSP long-only
    11%/yr (t 2.75) before the momentum and liquidity controls, 5-factor VW alpha not significant. Bag-of-words
    versions "show no significant abnormal returns". Regressions use about twenty annual observations.
  - Caveats, from the text: hyperparameter and model choice among PV-DM, PV-DBOW and Word2Vec; tiny annual samples;
    the study overlaps the CMN era; not an independent post-2014 test; no cost estimate beyond a remark that
    turnover is low. Tilt: statistically significant HML loading (value tilt).
- "Lazy Network" (Complexity 2022, article 9430919): known only from search summaries (reported up to 95 bp/month
  five-factor alpha). SECONDARY, NOT VERIFIED.
- Out-of-sample 2022–2026 work: **none found.** Searches returned no peer-reviewed post-2021 test of the non-changer
  premium. The near items are the null GitHub replications and QuantConnect study in OR/L2 (SECONDARY), Padysak
  (SECONDARY; positive-language similarity reverses sign), the withdrawn "Bloated Disclosures" (OR), and a Chinese
  crash-risk study (PLOS ONE 2023; similarity and crash risk, not returns).
- Cheng et al. (JFQA, reused): effect only outside the S&P 500.

**Implication for the S&P 500 EAR overlay:** None directly (separate decision). For the stable-reporter idea: the
only supporting independent paper is long-only, small-N, tuned, and ends in 2018; there is no independent large-cap
evidence after 2014.

---

## 9. Size concentration; small/mid-cap universe; costs (separate decision)

**Where the effects live (by size):**

| Effect | Evidence | Access |
|---|---|---|
| SUE drift | Martineau: none for all-but-microcap since 2006, none for microcaps since 2016 | FULL TEXT |
| CAR drift (DHS) | big leg 0.38%/mo vs small 0.94%/mo, 1972–2014 | FULL TEXT (WP) |
| CAR drift (Brandt) | top-1,000 CRSP firms: next-announcement spreads about 35–55% of the full-sample size; 1987–2004 | FULL TEXT (2007 ver.) |
| CAR drift (CJL) | above-NYSE-median: one-year ABR spread 4.3% vs 8.3% full; 1977–1993 | FULL TEXT (OCR) |
| Drift factor ex-microcap | t 1.43 vs 2.18 (Subrahmanyam, 2001–2024) | SECONDARY |
| Lazy Prices | Cheng et al.: only outside the S&P 500, small-cap about 60 bp/month; CMN mean firm size about $2.5–3.5B (a mid-cap profile) | L2 FULL TEXT |
| Insider purchases | Lakonishok & Lee: predictability driven by smaller firms (L1/OR ABSTRACT). CMP Table IX (below). Jeng, Metrick & Zeckhauser (NBER 6913): small-vs-large difference not significant (SECONDARY snippet) | mixed |

**Chronology caution:** every one of these samples ends before the 2015–2023 window the S&P 500 test can use. Evidence for
S&P 400/600 effects specifically: none found beyond size-sorted results above.

**Free historical constituent membership for S&P 400 / S&P 600 (Wikipedia, checked via the MediaWiki API tonight):**
- "List of S&P 400 companies" exists since **31 December 2010** ("first draft", 34,954 bytes; 61,890 bytes on
  1 January 2011; sizes suggest a table but I did not inspect the early revisions), 1,375 revisions through 22 October 2026. It has a "Selected past and announced changes to the list
  of S&P 400 components" section.
- "List of S&P 600 companies" was created on **27 August 2018** as a stub and expanded the same
  day (120,489 bytes at 06:56 UTC; size suggests a table, not inspected), with a "Recent and announced changes" section. At least 500 revisions by 2023
  (my first query stopped at 500 without pagination; a later paginated pass hit HTTP 429, so the total is not known).
- For comparison, "List of S&P 500 companies" starts 14 September 2005 (moved list, 11,486 bytes).
- **Limits:** each revision shows the list as the editors saw it on that date, so a snapshot series gives approximate
  point-in-time membership only from 2011 (S&P 400) and late 2018 (S&P 600), with no membership before those dates.
  Snapshot accuracy, delay between S&P announcements and edits, and how complete the "changes" tables are were **NOT
  VERIFIED.** The API is rate limited. This does not provide the price panel the OR note says is also missing.

**Round-trip costs for small/mid caps (reused, not redone):**
- FIM 2012 NBER draft (L1 §24, FULL TEXT of the 2012 draft; 2018 per-size figures NOT VERIFIED): 1998–2011 market
  impact averages **11.21 bp for large caps vs 21.27 bp for small caps** (equal-weighted trade average); mean
  implementation shortfall 13 bp overall; patient AQR execution, not retail.
- NMV (L1 §25 and re-read tonight, FULL TEXT of the August 2015 WP): effective spreads are higher for smaller stocks
  (their Figure 1 gives medians by size rank; the numbers were not extracted); for a mid-turnover strategy the net
  Sharpe ratios converge across size bins (ValMomProf: gross 1.14 / 1.05 / 0.84 for micro / small / large; net
  0.87 / 0.85 / 0.64). NMV's bins: micro = bottom 20% NYSE, large = above NYSE median.

**Implication for the S&P 500 EAR overlay:** The effects documented in the literature are larger the smaller the stock,
so an S&P 500 test is the strict case. A move to S&P 400/600 would raise the effect and the cost together, and the
free membership history starts only in 2011 / 2018.

---

## 10. Next alternatives: insiders, guidance, analyst revisions

**Opportunistic insider purchases.**
- Cohen, Malloy & Pomorski (2012), "Decoding Inside Information", JF 67(3), 1009–1043 (per EconPapers). NBER WP 16454.
  **FULL TEXT (NBER WP, Oct 2010).** Sample classification from 1986, tests 1989–2007 (the text also says 1986–2007).
  - Opportunistic trades: VW abnormal return 82 bp/month (t 2.15), routine ≈ 0 (reused, L1 §15). Firms with an
    opportunistic trade: mean market cap about $1.6B, median about $117M (the insider universe: $3.3B / $330M)
    (Table I Panel A as extracted; column layout garbled, so read as the two universes).
  - **Table IX (size split):** top half of market cap: opportunistic-buy coefficient 0.55 (t 2.31); bottom half 0.80
    (t 2.78) in one-month return regressions (percent per month). Column assignment is read from a garbled text
    layout and should be confirmed. "Large" is the top half of the insider sample, not the S&P 500.
- **Post-2008:** Aalto MSc thesis replication (2008–2024 Form 4): predictability 60–70% weaker than CMP
  (SECONDARY; the PDF returned 403 and the fetch tool was refused). OR gave "¼–⅓ of the original", which agrees
  (30–40% of the original). Not peer-reviewed, no size or subperiod split. Cline, Gokkaya & Liu (2017) "The persistence of
  opportunistic insider trading": title only, content NOT VERIFIED. Kravet & Lee (2026) and arXiv 2602.06198
  (reused from OR; SECONDARY): first-day capture and liquid-universe failure.
- **Large caps:** CMP size split above (top-half result still positive and significant, 1989–2007). Nothing found for
  S&P 500 names after 2008.

**Management guidance changes.** Milian (2018), "The Information Content of Guidance and Earnings", European
Accounting Review 27(1), 105–128. **SECONDARY** (RePEc snippet via search): guidance bundled with earnings is
underweighted until the next announcement; one snippet says about 20% of the earnings drift reverses at the next
announcement. Sample, size split and recent-decade results NOT VERIFIED. Fink's review (FULL TEXT, footnote 9) says
guidance and analyst revisions "can produce PEAD-like effects", citing Milian.

**Analyst revisions.** Reused: Stickel 1991 and Barber et al. 2001 (SECONDARY, L1 §39, L2 §5(iv)): 1986–1996 gross
above 4% a year, no net-of-cost excess. CJL 1996 (FULL TEXT OCR, §5): the revisions sort is the strongest of the
three surprise measures in the large-stock sample (one-year spread 7.6% vs 9.7% all), 1977–1993. Recent-decade
large-cap evidence: none found; I/B/E/S is paid.

**Implication for the S&P 500 EAR overlay:** None of these three supplies a free, large-cap, post-2010 signal. Insider
purchases come closest on data cost but are sparse in S&P 500 names.

---

## Summary table

| # | Source | Access | Verified finding | Bears on S&P 500 EAR overlay |
|---|---|---|---|---|
| 1 | Daniel-Hirshleifer-Sun, RFS 2020 / NBER 24163 | FULL TEXT (WP Dec 2017) | PEAD = 2x3 NYSE-median size x 20/80 sort on 4-day [-2,+1] market-adjusted CAR around latest RDQ; VW, monthly, 6-month staleness; 0.65%/mo (t 7.91) 1972–2014; big leg 0.38 vs small 0.94; 1991–2014 0.55 (t 4.59) | Gives the exact signal; big leg is the NYSE-median half, ends 2014 |
| 2 | Dickerson-Julliard-Mueller, arXiv 2604.04430 | FULL TEXT (arXiv v1) | DHS PEAD factor has posterior probability above 50% in the 1986–2022 SDF; fn 26: robust and "not driven by micro-cap stocks" (IA.4 not read); JFE status NOT VERIFIED | Factor-level support only; size claim not shown for large caps |
| 3 | Martineau, CFR 2022 | FULL TEXT (CFR draft) | SUE (analyst, random-walk) deciles, drift gone for non-microcaps since 2006; the one announcement-return test (BHAR[0,60] on BHAR[0,1]) has slope falling from about 1.6 to about 1 for non-microcaps; no DHS or Brandt citation | Closest direct evidence against CAR-conditioned drift in large stocks |
| 4 | Brandt et al., WP | FULL TEXT (June 2007) | EAR drift concentrated at next four announcements; 6.3% a year; large-firm spreads reduced to about half; possible reversal; 1987–2004 | Drift is mostly a repeat reaction; short-window overlay catches less |
| 5 | Chan-Jegadeesh-Lakonishok 1996 | FULL TEXT (OCR, NBER WP) | ABR window days -2..+1 vs equal-weighted market; 5.9% (6m) / 8.3% (1y) spread; above-NYSE-median 4.3% (1y); 1977–1993 | Original CAR evidence, old and broad "large" |
| 6a | Dai-Medhat-Novy-Marx-Rizova, NBER 30917 | FULL TEXT (WP) | CAR-sort PEAD 0.53%/mo (t 5.45) 1973–2021, 0.36% (t 2.09) post-May 2001, all stocks; reversal offsets drift: REV loading -0.54 on PEAD, announcer reversal 6 bp | Strongest primary post-2001 evidence: VW all-stock CAR sort still positive (mildly) post-2001; raw reversal is short PEAD, so a reversal tilt conflicts with the overlay |
| 6b | Subrahmanyam 2025, SSRN 5930255 | SECONDARY | factor t 2.18 all vs 1.43 ex-microcap, 2001–2024 | Against large-cap drift |
| 6c | Aboody-Lehavy-Trueman WP 2008 | ABSTRACT | Past 12-month winners: +1.58% before, -1.86% after announcement (5 days each); not a CAR sort | Tangential |
| 6d | Chen & Welch, arXiv 2607.06502 | ABSTRACT | Non-micro post-2005 median anomaly 7 bp/month; PEAD-specific value NOT VERIFIED | General decay prior |
| 7a | DellaVigna-Pollet 2009 / NBER 11683 | FULL TEXT (WP abstract) | Friday: delayed response 60% vs 40% of total; volume 10% lower; 1995–2004. JF figures (15%, 70%) SECONDARY | Mechanism, old sample |
| 7b | Hirshleifer-Lim-Teoh 2009 / MPRA 3110 | FULL TEXT (2007 version) | High-news days alpha 1.64%/mo vs 0.77% insignificant; 1995–2004 | Mechanism; testable interaction |
| 8 | Adosoglou et al. 2021 | FULL TEXT (postprint) | NSP VW 3-factor 9.75%/yr (t 3.24), 5-factor 8.45% (t 2.61), mcap > $300M, 1999–2018, about 40 stocks; tuned; no independent post-2014 test found | None for EAR; weak for stable reporters |
| 9 | Wikipedia S&P 400 / 600 pages | API checked | S&P 400 list since 2010-12-31; S&P 600 since 2018-08-27; quality NOT VERIFIED | Free membership exists only from 2011 / 2018 |
| 10 | Cohen-Malloy-Pomorski 2012; Aalto thesis; Milian 2018 | FULL TEXT (CMP WP); SECONDARY (others) | CMP top-half-cap opportunistic buy 0.55 (t 2.31) vs bottom half 0.80; thesis 60–70% weaker 2008–2024; guidance and analyst-revision recent large-cap evidence NOT FOUND | Alternatives lack free large-cap post-2010 support |

## Bottom line for H1 (S&P 500 EAR overlay)

1. **Exact DHS signal:** four-day market-adjusted CAR over days -2..+1 around the latest Compustat RDQ announcement,
   re-sorted monthly, dropped after six months, VW with NYSE 20/80 breakpoints within NYSE-median size halves.
2. **Post-2006 large-cap support for CAR-based drift:** no S&P 500-specific test found. A value-weighted all-stock CAR
   sort is mildly positive post-2001 (Dai et al.: 0.36%/month, t 2.09, large caps carry most weight); DJM's factor-level
   claim has its size appendix unread; an ex-microcap factor test (Subrahmanyam, SECONDARY, factor unnamed) is
   insignificant (t 1.43). Mixed, and none of it is large-cap-only.
3. **Martineau scope:** SUE-type sorts (analyst and random-walk, decile ranks, 60-day BHAR) plus one announcement-return
   unbiasedness regression, whose slope converges to about 1 for non-microcaps.
4. **Evidence against the S&P 500 version:** Martineau's regression; Subrahmanyam t 1.43 (SECONDARY); Brandt's
   large-firm results (smaller, mostly at future announcements, hint of reversal), all pre-2005. **Reversal
   collision:** Dai et al. show the announcement CAR drifts rather than reverts, so raw reversal is short PEAD (loading
   -0.54); the overlay and a raw reversal signal conflict unless CARs are removed from the reversal input.
