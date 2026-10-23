# Literature verification for S&P 400 filing-text feasibility

Date: 2026-10-24. Reuses (not redone): `docs/strategy_lazyprices_20261023/LIT2_VERIFY.md`, `docs/strategy_ear_20261023/LIT_EAR_VERIFY.md`, `docs/overnight_20261024/OVERNIGHT_RESEARCH.md`, `docs/strategy_pit_20261023/LIT_VERIFY.md`.

Access legend: **FULL TEXT (version)** = PDF read via `pdftotext`; **ABSTRACT ONLY** = abstract from a primary metadata source (Crossref, Semantic Scholar, publisher page); **SECONDARY** = summary, search snippet, blog; **NOT VERIFIED** = not read, not claimed.

Process note: I was told not to contact sec.gov. One WebFetch call (a "skip this" prompt on the SEC staff small-cap liquidity PDF) did fetch that URL by mistake. I did not read the saved file. The SEC numbers below come from a search-result snippet and are labelled SECONDARY.

---

## 1. Sadlo (2020/2021), "Copy-Paste Outperformance: Lazy Investors and Copied Reports"

- **Citation:** Sadlo, Sven(-Philip), SSRN working paper, DOI 10.2139/ssrn.3748216 (SSRN abstract id 3748216). Semantic Scholar gives year 2020. Crossref lists "posted 2021" and "created 2021-02-26", author given as Sven-Philip Sadlo. Not peer reviewed (Crossref type: posted-content, preprint).
- **Access:** ABSTRACT ONLY. The abstract was retrieved from the Semantic Scholar API and matches the Crossref record. SSRN returned 403. The openAccessPdf link only points to the DOI resolver. **Full text not obtained.** Earlier "not found" was wrong; the record exists.
- **Abstract, as returned (Semantic Scholar and Crossref agree):** replicates Cohen-Malloy-Nguyen (2020) "for an updated and investable sample of S&P 1500 constituents between 1996 and June 2020 (3,668 firms, 208,425 reports). The quintile of firms with least similar reports still generates negative 3- and 5-factor alpha of up to -5.12% per year." "In contrast to Cohen et al. (2020), I find a pronounced asymmetry: 'changers' underperform but 'no changers' do not outperform." The explanation is inattention: "'changers' continue to underperform in the attention-grabbing universe of S&P 500 firms." Because changers "earn lower future margins and returns on invested capital, these results challenge semi-strong market efficiency for probably the most widely followed stocks in the world."
- **What the abstract says about S&P 500 firms:** only that changers "continue to underperform" in the S&P 500 subset. It gives no S&P 500 magnitude, t-statistic, weighting, or sub-period. The headline -5.12% per year is for the S&P 1500 sample. Whether the S&P 500 result is statistically significant is NOT VERIFIED.
- **Extracted from abstract:** sample S&P 1500 constituents (this covers S&P 400 mid caps), 1996 to June 2020; 3,668 firms, 208,425 reports (annual and quarterly). Which leg drives returns: the short (changer) leg; the long (no-changer) leg does not outperform. The similarity measure is NOT stated in the abstract. Portfolio construction, weighting, costs and turnover are NOT VERIFIED. The "up to" wording means the maximum across specifications; the best case is -5.12% a year.
- **Implication for S&P 400 filing-text avoidance:** this is the only located source claiming the effect in an S&P-constituent (including mid-cap) universe, but the evidence is an unreviewed abstract. The short leg is the one that works, so a long-only avoidance (exclude changers) fits the claimed asymmetry. Magnitudes in the S&P 400 are NOT VERIFIED. Obtain the full text before relying on it.

---

## 2. Cheng, Liu, Qiao & Wang, "Attentive Options Traders: Textual Changes to 10-Ks and Option Volatility Smirk(s)", JFQA

- **Authors (settled):** the accepted-manuscript PDF title page (https://jfqa.org/wp-content/uploads/2026/03/24613_Attentive_Options_Traders.pdf) lists **four** authors: Hua Cheng (Sun Yat-sen University International School of Business and Finance; University of Texas at Austin), Steve Liu (University of Rhode Island), Zheng Qiao (Xi'an Jiaotong University; corresponding author), Z. Jay Wang (University of Oregon). The Cambridge/JFQA listing (via WebFetch of the Cambridge page) also gives four authors in that order, JFQA First View, online 7 April 2026, pp. 1-35. So "Cheng, Liu and Qiao" is incomplete: the paper has **four authors, including Z. Jay Wang**. The acknowledgements thank editor Thierry Foucault.
- **Access:** FULL TEXT (JFQA accepted manuscript, March 2026 upload). Sample period 1996 to 2017 (text, around the data section and Table 1 note). Sample: stocks with traded options, value-weighted portfolios.
- **Table 12 reconstruction.** The `-layout` text interleaves panels; `-raw` gives a clean, ordered reading, and it agrees with the paper's own text on p. 40 (Panel B about 60 bp per month; significant only for non-S&P 500). Values are percent per month; t-statistics in parentheses. Q5 is the most similar firms, Q1 the least similar; Q5-Q1 is long similar, short changers. Page numbers: text on manuscript p. 40; Table 12 on p. 61.

| Panel | Measure | Q1 | Q5 | Q5-Q1 (t) |
|---|---|---|---|---|
| A Large cap | Excess return | 0.556 (1.64) | 0.746 (2.17) | 0.190 (1.09) |
| A | 3-factor alpha | -0.098 (-0.70) | 0.092 (0.53) | 0.190 (1.11) |
| A | 4-factor alpha | -0.070 (-0.51) | 0.126 (0.74) | 0.196 (1.11) |
| A | 5-factor alpha | -0.057 (-0.44) | 0.142 (0.86) | 0.199 (1.11) |
| B Small cap | Excess return | -0.118 (-0.30) | 0.407 (1.24) | 0.525 (2.33) |
| B | 3-factor alpha | -0.907 (-4.82) | -0.292 (-1.60) | 0.615 (2.87) |
| B | 4-factor alpha | -0.823 (-4.91) | -0.214 (-1.14) | 0.609 (2.75) |
| B | 5-factor alpha | -0.830 (-4.86) | -0.211 (-1.15) | 0.619 (2.81) |
| C S&P 500 | Excess return | 0.451 (1.36) | 0.664 (1.91) | 0.213 (1.06) |
| C | 3-factor alpha | -0.206 (-1.56) | 0.017 (0.10) | 0.223 (1.09) |
| C | 4-factor alpha | -0.182 (-1.39) | 0.071 (0.43) | 0.253 (1.26) |
| C | 5-factor alpha | -0.175 (-1.36) | 0.086 (0.53) | 0.260 (1.30) |
| D Non-S&P 500 | Excess return | 0.339 (0.80) | 0.635 (1.83) | 0.296 (1.43) |
| D | 3-factor alpha | -0.502 (-3.36) | -0.143 (-1.34) | 0.359 (2.01) |
| D | 4-factor alpha | -0.462 (-3.46) | -0.100 (-0.92) | 0.362 (2.18) |
| D | 5-factor alpha | -0.452 (-3.54) | -0.150 (-1.42) | 0.302 (2.02) |

  Q2 to Q4 cells omitted; they are in the PDF. Panel C is firms in the S&P 500 at any time during the portfolio construction period; Panel D is firms never in the S&P 500 (table note). Panels A and B split on the median fiscal-year-end market cap and keep only firms staying in the same size group.
- **Reading:** the earlier worry (S&P 500 panel cells inconsistent with the text) is resolved; it came from the scrambled layout. Panel C Q5-Q1 is 0.21 to 0.26 per month (t 1.06 to 1.30), Panel D 0.30 to 0.36 (t 1.43 to 2.18). The S&P 500 point estimates are only about 0.04 to 0.10 pp below the non-S&P 500 ones. The gap in significance mostly reflects larger standard errors in Panel C (t-statistic roughly 1.1 to 1.3 for estimates of 0.21 to 0.26 vs about 2.0 for 0.30 to 0.36), not a clearly smaller effect. The small-cap panel (0.52 to 0.62) is about double either.
- **Formal difference test:** none found. I searched the full text for "interaction", "difference" and "test" near Table 12. Interactions appear only in the Fama-MacBeth smirk regressions, not in size or S&P tests. The text (p. 40) only compares significance across separate subsamples: "no significant difference in returns between Q1 and Q5 for large-cap stocks ... the effect is significant only for non-S&P 500 stocks." The table has no Panel C minus Panel D row. Significant in one group and not in the other is not evidence that the two differ. Verdict: the paper reports **separate subsamples only**; the S&P 500 vs non-S&P 500 difference is untested, and the S&P 500 Q5-Q1 point estimates (0.21 to 0.26, same sign) cannot be called zero.
- **Implication for S&P 400 filing-text avoidance:** the S&P 500 estimate is smaller than the non-S&P 500 estimate and statistically insignificant, but it is positive and about 70 to 85 percent of the non-S&P 500 magnitude; the S&P 400 sits in the non-S&P 500 bucket here (never-in-S&P 500 includes mid caps), so its sign and rough size are plausible but not isolated. Sample ends in 2017, optionable stocks only.

---

## 3. Cohen, Malloy & Nguyen (2020), "Lazy Prices", Journal of Finance

- **Access:** FULL TEXT (NBER WP 25084, read tonight: https://www.nber.org/system/files/working_papers/w25084/w25084.pdf). JF version doi 10.1111/jofi.12885 not read. Earlier notes in `LIT2_VERIFY.md` stand.
- **New checks:**
  - The text (p. 6 of the WP) says "we perform a number of robustness checks across firm size, time, industry, firm-events", and that effects "show up in value-weighted returns across the universe of all publicly traded firms (and so are not concentrated in small firms)". The introduction also says the patterns "exist in large firms" (Section IV text). I found no size-terciles, S&P 500, or large-cap-only table in the main text. The 15 appendix tables I located by title (A-1 post-SOX Risk Factors; A-2 sentiment, uncertainty, litigiousness; A-3 law firms; A-4 special events; A-5 industry; A-6 Oster-bias; A-7 future 8-Ks and short interest; A-8 attention; A-9 tone; A-10 distraction; A-11 life cycle; A-12 negative words; A-13 10-K vs 10-Q; A-14 stop words; A-15 sequential risk factors) have no size or S&P split in their titles. Table contents were not all read.
  - Table III (printed in the NBER WP) reports means by similarity quintile. Mean market value of equity: Q1 (least similar, changers) 3,507,587; Q2 3,219,430; Q3 2,829,955; Q4 2,504,717; Q5 2,464,603, in $1000s, i.e. about $3.5B vs $2.5B. The text (p. 6) states "the average changer firm ... is actually larger than the average long at $3.5 B market cap (vs. $2.5 B)". The row labels in the printed table are misaligned, as in the earlier note, but the MV row is confirmed by the text.
  - No post-2014 or post-publication test is in the paper (sample to 2014).
- **Implication for S&P 400 filing-text avoidance:** the $2.5B to $3.5B means are mid-cap sized, and the claim is value-weighted survival, not an S&P 400 test. A mean of about $3B is dominated by the large-cap tail in a value-weighted portfolio, so this does not show the effect is present within mid caps by itself.

---

## 4. Replications and later studies with size splits

- **Sadlo (2020):** S&P 1500 and S&P 500 (section 1). ABSTRACT ONLY.
- **Cheng et al. (JFQA 2026):** size splits and S&P 500 split (section 2), FULL TEXT, 1996 to 2017.
- **Padysak, "The Positive Similarity of Company Filings and the Cross-Section of Stock Returns" (SSRN 3690461).** Access: SECONDARY (Quantpedia summary). Brain Company filing-similarity scores on roughly the largest 1,000 US stocks, 28 Feb 2007 to 29 May 2020; the focus is similarity of positive language, not full-document similarity. Decile 1 (lowest similarity) 1.14% a month, decile 10 0.73%; lowest-minus-highest 0.41% per month (about 5% a year); bottom-minus-top annual return 5.47%, volatility 6.48%. This is the opposite sign to Lazy Prices (low similarity of positive language does well), costs not reported in the summary, no size split. It is not a Lazy Prices replication.
- **Student/GitHub items** (earlier files): S&P 500 replication null on the full document; Risk Factors marginal. SECONDARY, unreviewed.
- **Not found:** any study splitting filing-text-similarity returns by S&P 400, mid cap, or size terciles. A direct search (size, mid-cap, post-2010) returned nothing. So no S&P 400 or mid-cap evidence exists in what I could access.
- **Implication for S&P 400 filing-text avoidance:** there is no direct mid-cap evidence. The Lazy Prices avoidance premium for S&P 400 stocks is untested in the literature available here.

---

## 5. Limits to arbitrage across size: costs

- **Frazzini, Israel & Moskowitz, "Trading Costs"** (SSRN 3229719, April 2018). Access: ABSTRACT/SECONDARY for the 2018 version; FULL TEXT only of the Dec 2012 draft (earlier note in `LIT_VERIFY.md` item 24). 2018: $1.7 trillion of live trades, 21 markets, 19 years; summaries (SECONDARY, etf.com/Alpha Architect snippets) give mean market impact just under 9 bp, median just over 6 bp, value-weighted mean just over 15 bp, with impact driven chiefly by trade size relative to volume. 2012 draft (Table II, 1998 to 2011): large caps 11.21 bp market impact vs 21.27 bp small caps. Per-size 2018 numbers NOT VERIFIED. A later Cambridge paper (SECONDARY snippet) notes the FIM sample is skewed to larger, liquid stocks.
- **Novy-Marx & Velikov (2016), "A Taxonomy of Anomalies and Their Trading Costs," RFS 29(1), 104-147.** Access: FULL TEXT (working paper, `mysimon.rochester.edu/novy-marx/research/ToAatTC.pdf`, read tonight). Costs are effective spreads from the Hasbrouck (2009) Gibbs procedure (96.5% correlation with TAQ effective spreads, per the paper). Fig. 1 (median effective spread across market-cap ranks of the largest 2,000 firms by decade) is a figure; I could not extract its values, so **no bp numbers for mid-size stocks**. Text: smaller-cap stocks are more expensive; section 7 uses size groups from NYSE breaks (micro below the 20th percentile, large above the median). For low-turnover strategies, net and gross Sharpe ratios "are not very different", even in micro caps (text, around Figs. 5 to 7). Sample ends 2013.
- **Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices,"** Journal of Finance 67(2), 719-760, DOI 10.1111/j.1540-6261.2012.01729.x. Citation verified from Crossref (volume 67, issue 2, pages 719-760, published 27 March 2012). Access: ABSTRACT ONLY. The abstract: "Daily high (low) prices are almost always buy (sell) trades"; the high-low ratio reflects variance and spread; the spread estimator uses 1-day and 2-day high-low ratios and "generally outperforms other low-frequency estimators." Suitability for daily-bar spreads: it needs daily high and low only, so it can run on free bars. Accuracy for large liquid stocks with spreads of a few bp is NOT VERIFIED (I did not read the paper's results); in general such estimators are noisy for tiny spreads and can be negative, but that is my recollection, not verified here. The alternative Abdi & Ranaldo (2017), "A Simple Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices," RFS 30(12), starting page 4437 (end page not verified) (abstract: highest correlation with TAQ effective spread among low-frequency estimators, best for less liquid stocks; SSRN 2725981 version read via Crossref; RFS print metadata from search result) also uses closes.
- **Effective spread level for S&P 400-size stocks:** I found **no source reporting a number for S&P 400 constituents or for the 2015 to 2025 period**. Closest: an SEC staff small-cap liquidity study (SECONDARY, search snippet only, about 12 years old): effective spreads "decline monotonically with capitalization" and relative spreads range 0.045% to 0.210% for stocks with $2B to $5B capitalization (4.5 to 21 bp). Bins do not map to the S&P 400, and the data are older. A JFE paper on bias in the effective spread (ScienceDirect S0304405X21001458) uses NYSE 1996 to 2015 size terciles; 403 on fetch; numbers NOT VERIFIED. So the S&P 400 cost level must be measured directly (Rule 605 or own TAQ), not taken from the literature.
- **Implication for S&P 400 filing-text avoidance:** a low-turnover annual-filing avoidance filter is cost-insensitive per NMV's low-turnover finding, so the cost argument against S&P 400 is weak; but no direct S&P 400 spread number is verified. Use mid-cap-style 4 to 21 bp (SECONDARY range) only as a rough bracket.

---

## 6. Insider purchases after 2010

- **Gangopadhyay & Yook (2021/2022), "Profits to opportunistic insider trading before and after the Dodd-Frank act of 2010,"** Journal of Financial Regulation and Compliance 30(1), 43-59 (Crossref: published online 31 July 2021; some sources date it 2022), DOI 10.1108/JFRC-02-2021-0018. Authors Partha Gangopadhyay and Ken C. Yook. Access: ABSTRACT ONLY (Crossref abstract; the Emerald page is paywalled). Abstract: trades identified with the Cohen, Malloy & Pomorski (2012) method; "opportunistic insider purchases were highly profitable before the DFA. Profits after opportunistic purchases were significantly lower after the DFA. Opportunistic insider sales were contrarian trades both before and after the DFA. However, share prices kept increasing after insiders sold their shares." Claimed as the first study comparing opportunistic insider trading profits before and after the DFA. **Sample period, sample size, magnitudes and size splits: NOT VERIFIED.** The finding of a post-2010 decline is confirmed qualitatively only; no numbers.
- **Cohen, Malloy & Pomorski (2012),** JF 67(3), 1009-1043, NBER WP 16454. FULL TEXT (WP) in `LIT_EAR_VERIFY.md`: opportunistic trades VW abnormal return 82 bp a month (t 2.15), routine about 0; size split (Table IX, column assignment from garbled text): top half of market cap coefficient 0.55 (t 2.31), bottom half 0.80 (t 2.78); "large" is the top half of insiders, not the S&P 500. Not re-read.
- **Post-publication / 2015 to 2026 studies (SECONDARY, reused from `OVERNIGHT_RESEARCH.md` and `LIT_EAR_VERIFY.md`):** Aalto MSc thesis, 2008 to 2024, predictability 60 to 70% weaker than CMP; Kravet & Lee (2026) 2003 to 2022, 60-day abnormal return 3.4% after non-acquisition purchases, no size or time trend; arXiv 2602.06198, microcaps 2018 to 2024, alpha falls 2.10% to 0.68% a month once the first event day is skipped, liquid extension fails matched gates. New in this pass: Jeon & Sulaeman (2024), "Corporate Insider Purchases and the Options Market," SSRN 4864272 (Crossref abstract, ABSTRACT ONLY): insider purchases in stocks with high options activity are followed by negligible abnormal returns; low-options-activity stocks see positive six-month abnormal returns. That bears on S&P 400 names, which have active options more often than small caps. No net-of-cost large or mid-cap study located.
- **Implication for S&P 400 filing-text avoidance:** insider purchases are not a replacement; evidence of post-2010 decline is confirmed (abstract), the signal is concentrated in smaller, low-options-activity firms, and no mid-cap net-of-cost estimate exists.

---

## 7. Alternative low-turnover public-information signals with modern evidence

No clean post-2010, US, mid/large-cap, net-of-cost study was found for any candidate. In order of evidence quality:

1. **Profitability / quality (XBRL-based, free point-in-time).** Hou, Xue, Zhang & Van Nieuwerburgh (2020), "Replicating Anomalies," RFS 33(5), 2019-2133, DOI 10.1093/rfs/hhy131 (NBER WP 23394 abstract read via WebFetch; ABSTRACT ONLY/SECONDARY): 447 anomalies replicated with NYSE breakpoints and value weights; 286 (64%) insignificant at 5%; 161 significant; the q-factor model leaves 115 of those insignificant. The abstract summary I got does not say which survive, so it does not alone show profitability works. Asness, Frazzini & Pedersen, "Quality Minus Junk" (SSRN 2312432; ABSTRACT ONLY): QMJ earns significant risk-adjusted returns in the US and 24 countries (published sample pre-2013). Novy-Marx (2013) (earlier file): large-cap profitability spread only t = 1.88. Chen & Zimmermann (2022), "Open Source Cross-Sectional Asset Pricing," Critical Finance Review 11(2), 207 (SECONDARY): replicability of original t-stats (98% of 161 clearly significant predictors exceed t = 1.96); this is replicability, not post-publication performance. Jacobs & Müller (2020 JFE; SSRN 2816490 abstract): the US is the only country with a reliable post-publication decline in anomaly long-short returns. So post-2010 persistence for large and mid caps is not established.
2. **Net stock issuance.** SECONDARY only (search snippets): survives a microcap exclusion and is the one pooled style that stays significant after multiple-testing correction in one study (title "Investment styles and the multiple testing of cross-sectional stock return predictability"); another source flags its significance as mostly pre-discovery. Numbers NOT VERIFIED.
- **Weak candidates (checked, do not recommend):** share repurchases: Fu & Huang (2016), Management Science 62(4), 964-984, abstract verified: "long-run abnormal returns following both stock repurchases and seasoned equity offerings disappear for the events in 2003-2012." Short interest: Boehmer, Huszar & Jordan (2010 JFE; SSRN 1405511 abstract) cover 1988 to 2005 and the stronger leg is low short interest (positive public information slowly incorporated); no post-2010 cross-sectional study found. Insider purchases: section 6.
- **Implication for S&P 400 filing-text avoidance:** the best modern alternative is a profitability/quality tilt, but its post-2010 mid-cap strength is not verified; treat it as a candidate for a separate test, not as established.

---

## Summary table

| # | Item | Access | Key verified finding | S&P 400 relevance |
|---|---|---|---|---|
| 1 | Sadlo (SSRN 3748216) | ABSTRACT ONLY | S&P 1500, 1996 to June 2020, 3,668 firms, 208,425 reports; least-similar quintile 3 and 5-factor alpha up to -5.12%/yr; short leg only; changers "continue to underperform" in S&P 500, no S&P 500 numbers | Only claim covering S&P mid caps; unreviewed; full text needed |
| 2 | Cheng, Liu, Qiao, Wang (JFQA 2026) | FULL TEXT (accepted ms.) | Four authors. Q5-Q1 per month: large cap 0.19 to 0.20 (t about 1.1); small cap 0.52 to 0.62 (t 2.3 to 2.9); S&P 500 0.21 to 0.26 (t 1.06 to 1.30); non-S&P 500 0.30 to 0.36 (t 1.43 to 2.18); no formal difference test | Same-sign, smaller S&P 500 effect; S&P 400 falls in the non-S&P 500 group |
| 3 | Cohen-Malloy-Nguyen (2020) | FULL TEXT (NBER WP) | No size split table; Table III mean MV $3.5B (Q1) to $2.5B (Q5); VW effect claimed | Mid-cap-sized means, not an S&P 400 test |
| 4 | Replications by size | mixed | Padysak: opposite-sign positive-language measure, largest 1,000 stocks 2007 to 2020; no mid-cap or S&P 400 study found | None direct |
| 5 | Costs: FIM, NMV, Corwin-Schultz | FULL TEXT (2012 draft, NMV WP) / ABSTRACT | FIM 2012: 11.21 bp large vs 21.27 bp small impact; NMV: low-turnover strategies barely affected by costs; CS (2012) JF 67(2), 719-760 verified; S&P 400 spread number not found; SEC snippet 4.5 to 21 bp for $2B to $5B (SECONDARY, old) | Costs probably small for a low-turnover filter; spread must be measured |
| 6 | Gangopadhyay & Yook (JFRC 30(1), 43-59) | ABSTRACT ONLY | Opportunistic purchases profitable before DFA, significantly lower after; no magnitudes or size split | Not a substitute |
| 7 | Alternatives | ABSTRACT / SECONDARY | Buyback abnormal returns vanish 2003 to 2012 (Fu-Huang); profitability/quality best candidate but post-2010 mid-cap evidence unverified | Candidate only |
