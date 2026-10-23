# LIT2 verification: Lazy Prices and related literature

Date: 2026-10-23. Prepared for the frozen-spec basis of a filing-similarity strategy.

Access legend:
- FULL TEXT: body text read (version stated).
- ABSTRACT ONLY: abstract/landing page only.
- SECONDARY: third-party summary, blog, or GitHub write-up. Not peer reviewed and unverified.

Tools: `pdftotext` and `pdftoppm` worked. Tables in the NBER PDF were read as rendered images where the text layer was garbled. Anything not confirmed is marked NOT VERIFIED.

---

## 1. Cohen, Malloy & Nguyen, "Lazy Prices"

- **Citation (as given by requester):** Journal of Finance 75(3), 2020. Journal volume, pages and DOI NOT VERIFIED (a search did not return them). NBER WP 25084; SSRN 1658471.
- **URL read:** https://www.nber.org/system/files/working_papers/w25084/w25084.pdf
- **Access:** FULL TEXT, NBER WP 25084, "September 2018, Revised March 2019" (30 pp. body plus tables/figures/appendix). The published JF version was NOT read. Numbers below are NBER-version numbers and may differ slightly from the JF version (for example the abstract headline, the sample period or the table layout).

### Sample and universe (a)
- Sample period: all 10-K, 10-K405, 10-KSB and 10-Q filings on EDGAR, 1995-2014 (abstract; Sec. II, p. 11).
- Return and accounting data: CRSP monthly returns, Compustat, I/B/E/S, Loughran-McDonald dictionary (p. 12). The text states no explicit filters on share code, price or size. It calls the sample "the universe of firms ... not a small, illiquid or otherwise selected subset" (pp. 4-5). NOT VERIFIED whether CRSP common stock (share codes 10/11) or price screens were applied.
- Counts (Table I): 345,639 filing observations in Panel A; 327,130 observations with similarity scores in Panel B. Mean 10-K length is 44,430 words and mean 10-Q length is 15,724 words (p. 14). Number of distinct firms NOT VERIFIED (not stated in the body).

### Documents and comparison (b)
- Both 10-K and 10-Q. Comparison is year-on-year: a 10-K against the prior-year 10-K, and a 10-Q against the same-quarter 10-Q one year earlier (p. 14 and fn. 18, p. 16). The authors say seasonality makes this the most comparable pairing.
- Appendix Table A-13 splits the sample. It gives 10-Q-only and 10-K-only runs with Sim_Cosine, equal-weight, 5-factor alpha Q5-Q1: 10-Q-only 0.49 (t 6.80), holding 3 months; 10-K-only about 0.36 (t 2.61) on the visible 3-factor row, holding 9 months. The text says the main result is "similar". The 10-K panel's 5-factor row was not fully verified.
- Main portfolio tests (Table II) use the full document. Section-level tests (Table VII) use each Item separately.
- Sections (Fig. 8, Sec. IV.B-C): MD&A (10-K Item 7; 10-Q Item 2), Risk Factors (Item 1A), Legal Proceedings (Item 3), Market Risk (Item 7A), Controls and Procedures, Other Information.
- Item 7 (MD&A) has the lowest average similarity, so most changes occur there (Fig. 9). The most return-informative section is Risk Factors, followed by Legal Proceedings and Item 7A. The authors call the section results "suggestive but not definitive" (p. 22).
- Risk Factors, Table VII, 5-factor alpha Q5-Q1 (Jaccard): EW 0.02 (t 2.76) and VW 0.0156 (t 2.57). The text equates the EW figure to "188 bp per month" (0.0188; the table is rounded to 0.02). VW Jaccard 3-factor is 0.0165 (t 2.75), i.e. about 165 bp.
- Item 1A only exists in 10-Ks from FY2005, so the Risk Factors sample is much shorter than the full 1995-2014 span. Appendix Table A-1 reruns it for 2003-2014. Its table text is garbled; its numbers were NOT VERIFIED.
- Comparison for MD&A (Table VII, EW Jaccard, 5-factor): 0.20 (t 2.81). For MD&A the text gives 11-22 bp per month.

### Text preprocessing (c)
- Only main 10-K/10-Q text is extracted. Tables with more than 15% numeric characters, HTML tags, XBRL tables, exhibits, ASCII-encoded PDFs, graphics, XLS and other binaries are removed (p. 11, following Loughran-McDonald stage-one parse).
- Stop words are NOT removed in the main results. Appendix Table A-14 re-runs after removing stop words using the L-M cleaned data. The text says results are "even larger and more significant" (p. 28).
- Stemming and case handling: NOT STATED in the text read.
- Items are identified by many iterated regular expressions on "Item" + number + title (p. 14).
- Values are winsorised at the 1st and 99th percentiles (p. 15).

### Similarity measures (d)
Four measures, all scaled 0 to 1 with 1 meaning identical (pp. 12-14). Table I Panel B means (all N = 327,130): Cosine 0.872 (sd 0.191, median 0.947); Jaccard about 0.395 (sd 0.191); MinEdit about 0.376 (sd 0.171); Simple about 0.146 (sd 0.093). The mapping of the means to the Jaccard/MinEdit/Simple labels came from garbled table text and is NOT VERIFIED.
- Cosine: term-frequency vectors over the union of terms, dot product over the product of Euclidean norms.
- Jaccard: intersection over union of the word sets (binary). The worked example in the paper gives 0.83 and 0.25.
- Minimum edit distance: smallest number of word operations to turn one document into the other, then scaled into [0,1]. The exact normalisation was not extracted.
- Simple: word-level `diff` / Word Track Changes. Words in changes, additions and deletions are counted and divided by the average document size, then feature-scaled to [0,1].
- Primary measure: no single primary measure. All four appear in every table. Jaccard is used for Fig. 7, Table VI and Table X. Simple is used for Table V and the Fama-MacBeth magnitude example. The text says all four deliver "this pattern" (p. 17).
- Correlations (Table I Panel C): all pairs above 0.5 except Simple vs Cosine, which is 0.19 per the text (p. 15); the garbled table shows 0.2076.
- Measure choice matters for significance. See Table II: Sim_Simple equal-weight excess-return spread is only 0.18 (t 1.20), while its 3- and 5-factor alphas are 0.28 and 0.27 (t 3.22 and 3.01).

### Portfolio construction (e)
- Each month, firms are sorted into quintiles "based on the prior month's distribution of similarity scores across all stocks" (text, p. 16). The Table II caption says "prior year's distribution". This is an internal inconsistency in the NBER version. NOT VERIFIED which applies.
- Q1 = lowest similarity ("changers"); Q5 = highest ("non-changers"). L/S is long Q5, short Q1.
- Entry: a stock enters in the month after the public release of a 10-K or 10-Q. Each firm is held for 3 months, with monthly rebalancing (overlapping portfolios), per Table II. Appendix A-13 uses 9 months for 10-K-only.
- Equal-weight (Panel A) and value-weight with lagged market cap (Panel B).
- December-FYE firms use Q1/Q2/Q3 10-Qs (about April/May, July/Aug, Oct/Nov) and the 10-K (Feb/Mar). Off-cycle firms are handled analogously.
- Hold period and persistence: Fig. 7 plots event-time returns for up to 6 months. The abstract and conclusion say returns "continue to accrue out to 18 months, and do not reverse". The 18-month support was not located in a table; NOT VERIFIED.

#### Table II headline numbers (monthly %, t in parentheses; read from the rendered table)
Equal-weighted Q5 minus Q1:

| Measure | Excess ret | 3-factor alpha | 5-factor alpha |
|---|---|---|---|
| Cosine | 0.31 (3.13) | 0.34 (4.45) | 0.32 (4.21) |
| Jaccard | 0.38 (2.65) | 0.44 (4.56) | 0.42 (4.31) |
| MinEdit | 0.36 (2.69) | 0.48 (5.96) | 0.45 (5.46) |
| Simple | 0.18 (1.20) | 0.28 (3.22) | 0.27 (3.01) |

Value-weighted Q5 minus Q1:

| Measure | Excess ret | 3-factor alpha | 5-factor alpha |
|---|---|---|---|
| Cosine | 0.34 (2.53) | 0.35 (2.63) | 0.34 (2.53) |
| Jaccard | 0.56 (3.75) | 0.54 (4.08) | 0.46 (3.44) |
| MinEdit | 0.39 (2.31) | 0.46 (3.06) | 0.37 (2.45) |
| Simple | 0.50 (2.69) | 0.58 (3.59) | 0.51 (3.14) |

- The 5-factor model is market, size, value, momentum and liquidity (not FF5 profitability/investment). The abstract's "up to 188 bp" is the Risk Factors section result, not the full-document result.
- The intro's "34-58 bp per month ... t = 3.59" is the VW range across measures; 58 bp is the Simple 3-factor alpha.
- Fama-MacBeth (Table IV): Sim_Simple coefficient 0.0292 (t 2.11) in column 12; text says a one-standard-deviation drop in similarity means returns 36 bp lower per month. Other columns of Table IV were not parsed cleanly.
- The paper has no FF5 (profitability/investment) alphas and no Newey-West discussion in the text read. NOT VERIFIED for the JF version.

### Leg decomposition (f)
Value-weighted (Panel B) quintile alphas, 5-factor, with t:

| Measure | Q1 (changers) | Q5 (non-changers) | Q5-Q1 |
|---|---|---|---|
| Jaccard | -0.23 (-2.20) | +0.23 (2.11) | 0.46 (3.44) |
| Cosine | -0.12 (-1.38) | +0.23 (2.23) | 0.34 (2.53) |
| MinEdit | -0.17 (-2.02) | +0.21 (1.84) | 0.37 (2.45) |
| Simple | -0.36 (-3.49) | +0.15 (1.15) | 0.51 (3.14) |

3-factor VW alphas: Jaccard Q1 -0.32 (-2.97), Q5 +0.23 (2.01); Simple Q1 -0.39 (-3.89), Q5 +0.19 (1.45).

Equal-weighted 5-factor: Cosine Q1 -0.12 (-1.75), Q5 +0.21 (3.28); Jaccard Q1 -0.14 (-1.84), Q5 +0.28 (3.57); MinEdit Q1 -0.15 (-2.14), Q5 +0.30 (4.11); Simple Q1 -0.06 (-0.89), Q5 +0.21 (2.68).

Reading, from the numbers and the text:
- The text says "the return spread shows up on both sides: changers experience negative future returns and non-changers positive" (p. 17). So non-changers do earn positive abnormal returns.
- The long leg is roughly as large as the short leg in equal-weight, and statistically stronger there (t 3 to 4). In value-weight the short leg is larger and more significant for Jaccard and Simple. The long-leg value-weight 5-factor alpha is only marginal for MinEdit (t 1.84) and not significant for Simple (t 1.15).
- Event time (Fig. 7, p. 18): "any positive alpha on the Q5 long side quickly reverts to zero, while the negative alpha persists and increases up to 6 months out -- never reversing". The text then says "a significant portion of the return spread ... comes from the short side" (p. 18).
- Conclusion: both legs carry alpha in calendar time at 3-month holding, but the long-leg alpha is short-lived in event time and the persistent component is the short (changer) side. Intermediate quintiles Q2 to Q4 show near-zero alpha, so this is not a monotone gradient in every column.
- Excess returns (raw) Q1 is not significant for Jaccard (EW 0.59, t 1.48; VW 0.23, t 0.64). The short leg shows up only in the factor-adjusted numbers. A long-only Q5 captures about 0.8-1.0% monthly raw return in EW, and that includes the market.

### Size, large caps, costs, turnover (g)
- Value-weight results are similar to or larger than equal-weight, so the authors argue the effect is not a small-cap result (pp. 6, 17-19). That is a value-weight claim on the whole CRSP-type universe. It does not isolate an S&P 500 or large-cap subsample. No size-tercile or top-cap table was found in the body. Appendix tables were skimmed only for A-1, A-13 and A-14; others NOT VERIFIED.
- Table III (means by quintile): average market value of equity about $3.5B (Q1) vs $2.5B (Q5), in $1000s as printed (3,507,587 and 2,464,603). The four row labels in the printed table look misaligned (rows reading "Monthly turnover 0.0663 ... Shorting fees 71.7 ... Sentiment 0.0016"), so which row holds turnover vs shorting fee is NOT VERIFIED. The text says Q1 has "lower average shorting costs" and "very modest turnover".
- No transaction-cost-adjusted returns and no net-of-cost Sharpe are given. The authors argue costs and limits to arbitrage are not important because turnover is modest.
- Mean firm size of $2.5B-$3.5B indicates mid cap, not S&P 500 large cap (inference from the means; they are not an S&P-only statistic).
- Persistence and reversal: Fig. 7 (Jaccard) shows no reversal to 6 months. 12-18 months is asserted in the text. Announcement-window return is zero (Fig. 7-B; fn. 12 says sub-sections are also zero).

### What predicts the changes; real effects; options (h)
- Table V (Sim_Simple on change content; firm and time fixed effects): lower similarity goes with more negative-sentiment words, more uncertainty words, more litigiousness words, and CEO/CFO-change mentions. Coefficients examined: Sentiment of Change -2.49 (t -37.83); other column labels were garbled and are not mapped here.
- 86% of changes are negative in tone; the 14% positive changes predict positive returns (p. 5).
- Appendix Table A-2 (text, p. 26): Jaccard 5-factor L/S alpha is 71 bp (t 3.29) in the high-litigiousness subsample and 72 bp (t 3.51) in the high-uncertainty subsample. Results are concentrated in low-sentiment, high-uncertainty, high-litigiousness subsamples.
- Real effects (Table X, 2-quarter-ahead): lower similarity predicts lower operating income, net income and sales. Appendix A-7 (text, p. 27): predicts more future 8-Ks, higher short interest, negative earnings surprises, bankruptcies ("suggestive").
- Attention proxies: SEC EDGAR log data (IPAccessMultipleYear). Interaction coefficient -0.0953 (t 2.05) in Table VIII column 8, a 22% reduction per 1-sd. High-distraction filing days have a 47 bp spread (t 2.75) vs 30 bp (t 1.43) on low-distraction days (fn. 23).
- Table IX: the effect is concentrated among firms whose text does not make explicit comparisons to prior periods ("compared to last year"); about one third of filings contain such phrases.
- Options: the paper contains no option-market test.

### Caveats for the frozen spec
- The sample ends in 2014, so the paper contains no post-2014 or post-publication evidence. Publication date: working paper circulated 2010 (SSRN 1658471), JF 2020.
- Table II text and caption disagree on the quintile reference distribution (prior month vs prior year).
- Main-text sample-size filters are not stated; CRSP universe filters, penny-stock exclusion, and the minimum number of stocks per quintile are NOT VERIFIED.
- "188 bp" is Risk Factors only and rests on a post-FY2005 sub-sample.
- The paper reports no S&P 500 or large-cap-only result, no net-of-cost return, and no Sharpe ratio.

---

## 2. Sadlo, "Copy-Paste Outperformance: Lazy Investors and Copied Reports"

- **Citation (partly verified):** Sven-Philip Sadlo (author name per search result "Sven-Philip Sadlo"; a Semantic Scholar entry lists "Sven Sadlo", year 2020). A search snippet describes an SSRN listing: 62 pages, posted 26 Feb 2021. Exact SSRN ID, venue (working paper vs thesis) and date: NOT VERIFIED. A direct SSRN fetch returned HTTP 403 and Semantic Scholar's API returned HTTP 429.
- **Access:** NONE. Could not read the abstract or the text.
- Signal, sample, leg decomposition, magnitudes: NOT VERIFIED. Do not cite any number from this paper.
- The title suggests a copy-paste (similarity) signal similar to Lazy Prices, but that is an inference from the title only.

---

## 3. Cheng, Liu, Qiao and Wang, "Attentive Options Traders: Textual Changes to 10-Ks and Option Volatility Smirk"

- **Citation:** Hua Cheng (Sun Yat-sen University International School of Business and Finance; University of Texas at Austin), Steve Liu (University of Rhode Island), Zheng Qiao (Xi'an Jiaotong University; corresponding author) and Z. Jay Wang (University of Oregon). Journal of Financial and Quantitative Analysis, accepted paper (editor Thierry Foucault). Note the requester listed three authors; there are four. Volume, issue and DOI NOT VERIFIED. The PDF is dated March 2026 (PDF metadata: created 18 Mar 2026).
- **URL:** https://jfqa.org/wp-content/uploads/2026/03/24613_Attentive_Options_Traders.pdf
- **Access:** FULL TEXT, JFQA accepted-manuscript PDF (about 2,400 lines of extracted text). Tables 11-13 are garbled in the text layer, so only text statements are used below where table cells are ambiguous.
- **Sample:** optionable firms with OptionMetrics data, 1996-2017, 10-Ks from EDGAR, cosine similarity between consecutive 10-Ks (Sec. III.A). 10-Ks only (10-Q results "similar but weaker", fn. 1).
- **Findings:**
  - Lower 10-K cosine similarity (more textual change) is followed by larger increases in the OTM-put minus ATM-call volatility smirk (Xing, Zhang, Zhao 2010 definition).
  - The effect is stronger for negative-tone changes, high abnormal idiosyncratic volatility, low analyst coverage and high short-sale costs.
  - Lazy Prices (Table 11, propensity-matched, value-weight, quintiles on the prior month's similarity, entry one month after filing, 9-month hold): for optionable stocks the Q5-Q1 alpha is 46 to 58 bp per month, significant at 5%. Row cells read: three-factor 0.495 (t 2.32), four-factor 0.463 (t 2.09), five-factor 0.575 (t 2.38), 0.561 (t 2.37); I could not map the printed rows to the factor labels with certainty.
  - For stocks WITHOUT options, Q1 and the long-short alphas are not significant (Q5-Q1 about 0.13 to 0.25, t about 0.3 to 0.6 in the garbled cells).
- **Large-cap evidence (Table 12, optionable stocks, value-weight; text, p. 40):** "no significant difference in returns between Q1 and Q5 for large-cap stocks"; small-cap Q1 underperforms Q5 by about 60 bp per month in risk-adjusted returns. "The effect is significant only for non-S&P 500 stocks." Large-cap Q5-Q1 cells appear as 0.19 to 0.20 (t about 1.1). The S&P 500 panel cells in the garbled text were inconsistent with this sentence, so I rely on the text sentence only and flag NOT VERIFIED for the S&P 500 panel numbers.
- **Interpretation:** the authors say stock-market informed trading is more efficient for large caps because option signals are more liquid and monitored. This is a direct warning that the Lazy Prices stock signal may be absent in large, optionable stocks in 1996-2017.
- **Interaction with options:** the stock return predictability is stronger for stocks with larger smirk increases (Table 13 double sort).
- **Recent-period attenuation:** no explicit post-2015 sub-period test seen in the text read. NOT VERIFIED.
- **Counter-note (not in this paper):** the claim "lazy-prices effect concentrated in firms without options" is false here. It is the reverse: the effect is concentrated in optionable firms.

---

## 4. Replications, post-publication evidence, critiques

### 4a. Direct replications of the Lazy Prices signal (all SECONDARY)
No peer-reviewed post-publication replication was found by these searches. Two GitHub projects were read. Both are one-author, unreviewed, use current constituents (survivorship bias, which the authors acknowledge), and give dates running to 2026 that I could not independently verify.

1. https://github.com/iqueipopg/lazy-prices (SECONDARY): S&P 100 current members, FY2007-2025 10-Ks, portfolio months March 2009 to September 2026 (211 months), yfinance adjusted prices, French factors. Cosine TF-IDF on the full document, quintiles within fiscal-year cohort, long Q5 / short Q1, equal weight, 12-month hold, 10 bp cost per unit one-way turnover. Result: gross L/S mean -0.92% per year, Sharpe -0.11 (95% bootstrap [-0.56, 0.38]); FF5+MOM alpha -0.93% per year (t -0.49). Every quintile has a positive FF5+MOM alpha, which the author blames on survivorship. Placebo percentile 33. Best of 54 variants: Sharpe 0.54, deflated Sharpe 0.13. Subperiods: 2009-2016 -3.85% per year; 2017-2026 +1.44% per year. Caveats from the author: about 19 stocks per leg; full-document cosine median 0.996, so little variation; Item 7 missing for 23% of pairs and Item 1A for 8%.
2. https://github.com/Shardul-Pandit/Lazy-Prices (SECONDARY): 495 current S&P 500 members, 9,553 10-Ks (2005-2026), 9,035 year-over-year comparisons, monthly returns March 2006 to September 2026, equal-weight quintiles. Full 10-K cosine: L/S -10 bp per month (t -1.19), six-factor alpha -6 bp (t -0.81). Risk Factors (Item 1A) cosine: L/S +21 bp (t 2.08), six-factor alpha +15 bp (p 0.08), 12 bp after costs (t 1.31). The Risk Factors spread is 31 bp (t 2.11) after 2015 and 7 bp (t 0.59) before. Jaccard and edit-distance Risk Factors versions show no effect; MD&A has the opposite sign; embedding similarity has no spread. The author does not treat Risk Factors as a reliable anomaly. Both are consistent with the Cheng et al. result that large caps show little effect.

### 4b. Related and later literature
- **Adosoglou, Lombardo & Pardalos (2021)**, "Neural network embeddings on corporate annual filings for portfolio selection," Expert Systems with Applications 164, 114053. Access: ABSTRACT ONLY (University of Parma repository page). Sample 1998-2018 10-Ks; Doc2Vec-type embeddings; "Semantic Similarity Portfolio" long non-changers, with a positive-prior-year-return filter. Reports up to about 10% per year risk-adjusted alpha, long leg only, no sub-period breakdown on the page. Post-2015 and large-cap results NOT VERIFIED. Full-text read not done. A follow-up, "Lazy Network" (Complexity, 2022, article 9430919), is only known from a search summary (SECONDARY): reported up to 95 bp monthly five-factor alpha on a least-disrupted portfolio.
- **Brown & Tucker (2011)**, "Large-Sample Evidence on Firms' Year-over-Year MD&A Modifications," Journal of Accounting Research 49(2), 309-346, doi 10.1111/j.1475-679X.2010.00396.x. Access: ABSTRACT ONLY (RePEc/IDEAS). Findings per abstract: modification scores rise with economic change; price reactions to 10-K filings rise with the score but analyst revisions do not; modification scores and the price reaction weakened over the decade even as MD&A length grew. Cohen et al. quote this paper and say they replicate the null announcement effect.
- **Loughran & McDonald (2016)**, "Textual analysis in accounting and finance: a survey," Journal of Accounting Research 54(4), 1187-1230, doi 10.1111/1475-679X.12123. Access: ABSTRACT ONLY via search result. A PDF copy exists at https://code.webartifex.biz/alexander/whu-textual-analysis/raw/branch/main/lectures/papers/Loughran_McDonald_RAS_2016.pdf; I did not read it. Any warnings about similarity measures, document length or boilerplate: NOT VERIFIED. The abstract says only that textual analysis is less precise than quantitative methods.
- **Hoberg & Lewis (2017); Hanley & Hoberg (2019):** not searched in detail. NOT VERIFIED.
- **Open Source Asset Pricing (Chen & Zimmermann, Critical Finance Review 2022, 27(2), 207-264)** and **Jensen-Kelly-Pedersen factor list:** searches did not surface a 10-K similarity entry or a documentation page. Whether Lazy Prices is a listed predictor, and its post-publication performance, NOT VERIFIED. Check openassetpricing.com directly.
- **General post-publication decay (SECONDARY, from search snippets):** McLean & Pontiff report average decay of about 26% out of sample and about 35% post-publication; Jensen-Kelly-Pedersen say returns decay but stay positive. Ivkovic and Zekhnini (March 2026 working paper) argue decay is heterogeneous. None of these tests the filing-similarity signal specifically.
- **Statement:** no peer-reviewed post-2014 or post-publication test of the Lazy Prices stock signal was found. The only post-sample evidence found is (i) the unreviewed GitHub replications above (null on the full-document measure in large caps, a marginal Risk Factors effect) and (ii) the Cheng et al. result that the effect is absent in large-cap and S&P 500 optionable stocks over 1996-2017 (still inside or near the original sample).

---

## 5. Brief notes on alternative information signals (all abstract or snippet level; labelled)

- **(i) Opportunistic insider purchases.** Cohen, Malloy & Pomorski (2012), "Decoding Inside Information," Journal of Finance 67, 1009-1044 (NBER w16454). ABSTRACT/SECONDARY. Data are Thomson Reuters insider filings, January 1986 to December 2007. Opportunistic trades earn 82 bp per month value-weight abnormal return; routine trades about zero. No post-2008 decay or replication evidence was found in these searches. NOT VERIFIED.
- **(ii) Earnings-announcement premium.** SECONDARY. Heitz, Narayanamoorthy and Zekhnini ("The Disappearing Earnings Announcement Premium") report the US premium has disappeared in recent years, tied to the 2004 disclosure regulation, while it stays robust internationally. Heater, Liu, Tan and Zhang report average announcement returns falling from 0.30% in the 1990s to -0.30% in the 2010s. No update by Frazzini-Lamont or Savor-Wilson was found. Not read beyond snippets.
- **(iii) Management earnings guidance changes.** Not searched. NOT VERIFIED.
- **(iv) Analyst revisions.** SECONDARY. Barber, Lehavy, McNichols & Trueman (2001) covers 1986-1996: gross abnormal returns above 4% per year with daily rebalancing, no net-of-cost excess return. A 2005 follow-up (1996-2003, about 438,000 recommendations) shows the share of buys peaking at 74% in 2000 and falling after. No paper showing decay in recent decades was found. NOT VERIFIED.
- **(v) Martineau (2022) rebuttals.** Source: UCLA Anderson Review brief (https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/), read via fetch, SECONDARY. Martineau, "Rest in Peace Post-Earnings Announcement Drift," Critical Finance Review 11 (2022), 613-646: drift disappears in non-microcaps from 2001 and is gone by 2006. The brief names two later papers: (a) Dickerson, Julliard and Mueller, "The co-pricing factor zoo," Journal of Financial Economics (in press); the brief's reference list gives the author list inconsistently, so the author names are NOT VERIFIED; it concludes the earnings-drift factor remains a major factor. (b) Hirshleifer, Peng and Wang (2025), "News diffusion in social networks and stock market reactions," Review of Financial Studies 38(3), 883-937; drift with t about 14, microcaps not filtered, weaker with social connectedness. Reconciliation: Subrahmanyam (2025), "Keeping it Simple: How Can Post-Earnings Return Drift Exist and Not Exist Simultaneously?", SSRN 5930255. Using Feb 2001 to Dec 2024 data, the earnings-drift factor has t = 2.18 with all stocks and t = 1.43 excluding microcaps; he says microcaps drive the drift. Only the brief was read, not the papers.

---

## Summary table

| Paper | Access | Key verified finding | Large-cap relevance | Status |
|---|---|---|---|---|
| Cohen-Malloy-Nguyen, Lazy Prices | FULL TEXT, NBER WP 25084 (Mar 2019 rev.) | VW Q5-Q1 5-factor alpha 0.34-0.51%/mo (t 2.5-3.4), EW 0.27-0.45% (t 3.0-5.5); 3-month hold, monthly overlap; both legs, short leg more persistent | Whole-universe VW only; no S&P 500 or net-of-cost result; sample ends 2014 | Verified, NBER version only |
| Sadlo, Copy-Paste Outperformance | NONE | Existence only (SSRN, about Feb 2021 / SemSch 2020) | n/a | NOT VERIFIED |
| Cheng-Liu-Qiao-Wang, Attentive Options Traders (JFQA) | FULL TEXT (accepted ms., Mar 2026) | Effect only in optionable stocks; Q5-Q1 46-58 bp/mo; large-cap and S&P 500 not significant, small-cap about 60 bp | Direct evidence the effect is weak in large caps (1996-2017) | Verified (table cells partly garbled) |
| GitHub replications (2) | SECONDARY | Full-document L/S about zero in S&P 100 / S&P 500; Risk Factors +15 bp (S&P 500, p 0.08) | Direct but unreviewed, survivorship biased | Unreviewed |
| Adosoglou et al. 2021 | ABSTRACT ONLY | Up to 10%/yr alpha, 1998-2018, long leg | Not reported | Partial |
| Brown & Tucker 2011 | ABSTRACT ONLY | MD&A modification measure; price reaction weakened | n/a | Partial |
| Loughran & McDonald 2016 | ABSTRACT ONLY | Survey; no similarity-specific content verified | n/a | Partial |
| Opportunistic insiders (CMP 2012) | ABSTRACT/SECONDARY | 82 bp/mo VW, 1986-2007 | No post-2008 evidence found | Partial |
| Earnings announcement premium | SECONDARY | Disappeared in US after 2004 per one paper | n/a | Partial |
| PEAD rebuttals (UCLA brief) | SECONDARY | HPW 2025 RFS; DJM JFE; Subrahmanyam: t 2.18 all stocks, 1.43 ex-microcaps | Drift mostly microcaps | Partial |
