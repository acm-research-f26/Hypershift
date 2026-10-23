# Literature verification (2026-10-23)

Method: WebSearch for citation/abstract; open PDFs (NBER, author sites, publisher-hosted copies) downloaded and text-extracted with pdftotext, then grepped. Numbers below were read in the text unless tagged otherwise.

**Access legend**
- FULL TEXT = I extracted and read text of the PDF. The version is stated; it is the published version only where noted. A working-paper (WP) version can differ from the journal version.
- ABSTRACT ONLY = abstract / publisher or RePEc page, no body text read.
- SECONDARY = only described by other sources.
- "NOT VERIFIED" = I could not confirm it. Journal volume/pages not seen in a source I opened are marked NOT VERIFIED rather than filled from memory.

Page/table references for FULL TEXT papers are approximate where I only have the extracted line, and are labelled "text" in that case.

---

## 1. Jegadeesh (1990)
- **Citation:** Jegadeesh, N. (1990), "Evidence of Predictable Behavior of Security Returns," Journal of Finance 45(3), 881-898.
- **DOI/URL:** https://ideas.repec.org/a/bla/jfinan/v45y1990i3p881-98.html (citation from RePEc listing).
- **Access:** SECONDARY (RePEc citation plus later papers describing it). Abstract text not retrieved.
- **Key finding:** Described by later papers (a Karachi study and a Wits thesis) as showing short-term (about 1-month) return reversal, alongside Lehmann (1990). NOT VERIFIED from the paper itself; no numbers seen.
- **Holding/data:** NOT VERIFIED (monthly reversal is the standard description).
- **Large-cap/cost notes:** NOT VERIFIED.
- **Caveats:** Nothing beyond the citation is confirmed.

## 2. Lehmann (1990)
- **Citation:** Lehmann, B. (1990), "Fads, Martingales, and Market Efficiency," QJE 105(1), 1-28.
- **DOI/URL:** NBER WP 2533, https://www.nber.org/papers/w2533 (QJE page numbers from RePEc).
- **Access:** FULL TEXT of NBER WP 2533 (March 1988 draft). The PDF is OCR-garbled but readable. Not the QJE version.
- **Key finding:** Abstract (RePEc and WP): "winners" and "losers" of one week reverse the next week. The author reports arbitrage profits that survive corrections for bid-ask spreads and plausible transaction costs, and attributes it to inefficiency in the market for liquidity around large price changes. Return-level numbers NOT VERIFIED (OCR too poor to trust). The WP text mentions CRSP daily data from 1962.
- **Holding/data:** Weekly rebalance, one-week formation and holding, CRSP daily returns. Sample end date NOT VERIFIED.
- **Large-cap/cost notes:** The paper says profits persist after plausible costs; magnitudes NOT VERIFIED. Later work (Lo-MacKinlay, Nagel) says part of this is cross-autocovariance / liquidity provision.
- **Caveats:** The costs assumed in 1990 are not comparable with today's execution.

## 3. Jegadeesh & Titman (1993)
- **Citation:** Jegadeesh, N., Titman, S. (1993), "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency," JF 48(1), 65-91.
- **DOI/URL:** https://ideas.repec.org/a/bla/jfinan/v48y1993i1p65-91.html. PDF copy: floridapsc.com (scan).
- **Access:** FULL TEXT (published JF scan, OCR'd).
- **Key finding:**
  - Sample: Jan 1965 - Dec 1989, NYSE/AMEX (text, Table I note).
  - Decile (equal-weight) winners minus losers earn significant profits over 3-12 month holding periods (abstract).
  - Table I, J=6 formation, K=6 holding: buy-sell 0.95%/month (t=3.07) if formed immediately; 1.10%/month (t=3.61) if formed 1 week after measurement. Panel B (skip a week) is higher for every strategy, because of bid-ask and price-pressure effects.
  - Part of the first-year abnormal return dissipates in the following two years (abstract).
- **Holding/data:** Overlapping K-month holdings, monthly rebalance, J-month ranking, CRSP prices only. Average turnover of the 6/6 rule is 84.8% semiannually (text).
- **Large-cap/cost notes:** After a 0.5% one-way cost the risk-adjusted return is 9.29% per year and still significant. It is also significantly positive in each of three size-based subsamples (text). Whether the large-cap tercile is significant on its own: NOT VERIFIED.
- **Caveats:** Equal-weighted deciles on 1965-89; this is not a large-cap post-2000 result.

## 4. Carhart (1997)
- **Citation:** Carhart, M. (1997), "On Persistence in Mutual Fund Performance," JF 52(1). Pages NOT VERIFIED.
- **DOI/URL:** https://www.afajof.org/page/Top50Cited (listing).
- **Access:** ABSTRACT ONLY (via search summary of the AFA listing) plus secondary summaries.
- **Key finding:** Persistence in fund returns is mostly explained by common factors and expenses. Hendricks-Patel-Zeckhauser "hot hands" is mostly Jegadeesh-Titman one-year momentum, and funds do not earn higher returns by following momentum. The remaining persistence is concentrated in the worst funds. Secondary sources credit this paper with the four-factor model (Fama-French plus momentum). No numbers seen.
- **Holding/data:** Mutual fund data. Not an equity trading-signal paper.
- **Large-cap/cost notes:** NOT VERIFIED.
- **Caveats:** Choi and Zhao (2021, CFR) is cited by secondary sources for finding no significant persistence in 1994-2018. NOT VERIFIED here.

## 5. Chan, Jegadeesh & Lakonishok (1996)
- **Citation:** Chan, L., Jegadeesh, N., Lakonishok, J. (1996), "Momentum Strategies," JF 51(5), 1681-1713.
- **DOI/URL:** https://ideas.repec.org/a/bla/jfinan/v51y1996i5p1681-1713.html (also NBER WP 5375).
- **Access:** ABSTRACT ONLY (the NBER PDF text extraction was garbled).
- **Key finding (abstract):** Past returns and past earnings surprises each predict large drifts in future returns after controlling for the other. Risk, size and B/M do not explain the drifts. There is little evidence of reversal. Analysts' forecasts respond sluggishly to past news. A CFA Digest summary reports momentum over 6-month and 1-year horizons. No numbers verified.
- **Holding/data:** 6-12 month holds (secondary, CFA Digest). Needs analyst forecasts / quarterly earnings (SUE, forecast revisions).
- **Large-cap/cost notes:** NOT VERIFIED.
- **Caveats:** Details (sample, return magnitudes) NOT VERIFIED.

## 6. Asness, Moskowitz & Pedersen (2013)
- **Citation:** Asness, C., Moskowitz, T., Pedersen, L.H. (2013), "Value and Momentum Everywhere," JF 68(3), 929-985.
- **DOI/URL:** 10.1111/jofi.12021; PDF https://pages.stern.nyu.edu/~lpederse/papers/ValMomEverywhere.pdf
- **Access:** FULL TEXT (JF-formatted copy, vol. LXVIII no. 3).
- **Key finding:** Value and momentum premia exist in 8 markets/asset classes, with a common factor structure; value and momentum are negatively correlated with each other (abstract). The US stock sample runs Jan 1972 - Jul 2011; it excludes ADRs, REITs, financials and stocks priced under $1 (text). Per-market Sharpe ratios are in the paper's Tables (e.g., the combo 50/50 columns show Sharpe up to about 1.07 in one panel); I did not tie these to specific markets, so treat as NOT VERIFIED.
- **Holding/data:** Monthly rebalance, 12-2 month momentum, B/M value (specification detail otherwise NOT VERIFIED).
- **Large-cap/cost notes:** Universe restricted to the largest stocks per market (exact cutoff NOT VERIFIED). No net-of-cost results verified.
- **Caveats:** The sample ends 2011.

## 7. Novy-Marx (2013)
- **Citation:** Novy-Marx, R. (2013), "The Other Side of Value: The Gross Profitability Premium," JFE 108(1), 1-28.
- **DOI/URL:** 10.1016/j.jfineco.2013.01.003; PDF https://www.johnhcochrane.com/s/Novy_marx_OSoV.pdf
- **Access:** FULL TEXT of the June 2012 manuscript (not the JFE print).
- **Key finding:** Sample July 1963 - Dec 2010, financials excluded. The most profitable firms earn 0.31%/month more than the least profitable (t=2.49); the FF3 alpha of the spread is 0.52%/month (t=4.49) (text). Among the largest stocks, the profitability spread is 26 bp/month (t=1.88), versus 14 bp/month (t=0.95) for value (text). Realized annual Sharpe ratios are 0.27 and 0.14; the two large-cap strategies have correlation -0.58 and do well together (text).
- **Holding/data:** Annual rebalance (Compustat gross profit / assets).
- **Large-cap/cost notes:** The large-cap profitability spread is only marginally significant alone (t=1.88). No costs in the paper (Novy-Marx & Velikov cover that).
- **Caveats:** Manuscript version.

## 8. Fama & French (2015)
- **Citation:** Fama, E., French, K. (2015), "A five-factor asset pricing model," JFE 116(1), 1-22.
- **DOI/URL:** PDF https://tevgeniou.github.io/EquityRiskFactors/bibliography/FiveFactor.pdf
- **Access:** FULL TEXT (JFE print copy).
- **Key finding (abstract):** The 5-factor model (size, value, profitability, investment) beats FF3. Its main failure is the low average returns of small stocks that behave like firms investing a lot despite low profitability. With profitability and investment added, HML is redundant for describing average returns in their sample. I did not extract the sample period.
- **Holding/data:** Annual factor rebalance (Compustat).
- **Large-cap/cost notes:** Not an implementation paper; no costs.
- **Caveats:** Useful as attribution model, not a trading signal.

## 9. Frazzini & Pedersen (2014)
- **Citation:** Frazzini, A., Pedersen, L.H. (2014), "Betting Against Beta," JFE (volume/pages NOT VERIFIED).
- **DOI/URL:** NBER WP 16601, https://www.nber.org/papers/w16601
- **Access:** FULL TEXT of NBER WP 16601 (Dec 2010 draft).
- **Key finding:** The BAB factor (long leveraged low-beta, short high-beta) has a US Sharpe ratio of 0.75 over 1926-2009, about twice that of value and 40% above momentum; global equities 0.85 (text). The sample is CRSP common stocks from Jan 1926 to Dec 2009 (text).
- **Holding/data:** Monthly rebalance, betas from rolling returns.
- **Large-cap/cost notes:** Needs leverage; no net-of-cost results verified.
- **Caveats:** The WP differs in places from the JFE paper.

## 10. Ang, Hodrick, Xing & Zhang (2006)
- **Citation:** JF 61(1), 259-299.
- **DOI/URL:** 10.1111/j.1540-6261.2006.00836.x; NBER WP 10852 https://www.nber.org/papers/w10852
- **Access:** FULL TEXT of NBER WP 10852.
- **Key finding:** High idiosyncratic-volatility stocks have "abysmally low" average returns. The WP states a difference of -1.06%/month between the highest and lowest idiosyncratic-vol quintiles (text). Sensitivity to VIX changes: the 5-1 spread is -1.04%/month (-1.15% controlling for MKT, -0.83% under FF3) (text). The sample starts Jan 1986 (VIX data) (text).
- **Holding/data:** Monthly, value-weighted portfolios.
- **Large-cap/cost notes:** Excluding small growth firms leaves FF3 alpha of -0.44%/month for the zero-cost portfolio (text). Costs not analysed.
- **Caveats:** WP version.

## 11. Bernard & Thomas (1989)
- **Citation:** Bernard, V., Thomas, J. (1989), "Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?" Journal of Accounting Research 27 (Supplement), pages reported inconsistently (1-36 vs 1-48): NOT VERIFIED.
- **DOI/URL:** NOT FOUND.
- **Access:** SECONDARY.
- **Key finding (secondary):** The top-minus-bottom SUE decile spread is positive in 41 of 48 quarters in 1974-1985 (some sources say 1974-86), and in 11 of 16 quarters with a negative NYSE market return; read as delayed response, not a risk premium. Numbers not checked against the paper.
- **Holding/data:** Quarterly earnings (SUE) required.
- **Large-cap/cost notes:** NOT VERIFIED.
- **Caveats:** None of this was read in the paper.

## 12. Martineau (2022)
- **Citation:** Martineau, C. (2022), "Rest in Peace Post-Earnings Announcement Drift," Critical Finance Review 11(3-4), 613-646.
- **DOI/URL:** https://ideas.repec.org/a/now/jnlcfr/104.00000122.html; PDF https://cfr.ivo-welch.org/published/papers/martineau2021rest.pdf
- **Access:** FULL TEXT (CFR draft PDF, "CFR (draft), 1-46").
- **Key finding:**
  - Abstract: prices fully reflect earnings surprises on the announcement date; "For large stocks, PEAD have been non-existent since 2006 but has only disappeared recently for microcap stocks."
  - Intro: since 2006, analyst earnings surprises fail to positively predict post-announcement returns over 60 days for "all-but-microcap" stocks (market cap above the NYSE 20th percentile), and since 2016 for microcaps.
  - Announcement-date response to surprises is 6x (all-but-microcap) and 3x (microcap) larger in 2016-2019 than in 1984-1990.
  - Data: Compustat 1973-2019 (593,654 announcements) and I/B/E/S 1984-2019. Drift is measured as 60-trading-day cumulative returns.
- **Holding/data:** 60-day post-announcement window. Needs I/B/E/S consensus or Compustat earnings.
- **Large-cap/cost notes:** Directly relevant: no PEAD in large caps after 2006 on this evidence. Table 3 coefficients were not extracted.
- **Caveats:** A UCLA Anderson brief (secondary) says two 2025 accepted papers contradict this, and a UCLA working paper attributes the disagreement to research-design choices. NOT VERIFIED beyond that brief.

## 13. Frazzini & Lamont (2007)
- **Citation:** Frazzini, A., Lamont, O. (2007), "The Earnings Announcement Premium and Trading Volume," NBER WP 13090 (journal publication NOT VERIFIED).
- **DOI/URL:** https://www.nber.org/papers/w13090
- **Access:** FULL TEXT.
- **Key finding:** Stocks earn higher returns in months with a scheduled announcement. Long-short return is 61 bp/month (about 7%/year), t over 5 (text, 1973-2004). Monthly strategies earn 7-18%/year with higher Sharpe than other popular anomalies. The premium is "strong in large capitalization stocks" and persists since 1927 (text). The explanation offered is attention-driven buying by small investors.
- **Holding/data:** Monthly; requires expected announcement dates (Compustat 1971-2004).
- **Large-cap/cost notes:** Costs not netted. Premium reported to survive in large caps (text).
- **Caveats:** Sample ends 2004; WP version. Post-2004 decay not verified here.

## 14. Savor & Wilson (2016)
- **Citation:** Savor, P., Wilson, M. (2016), "Earnings Announcements and Systematic Risk," JF 71(1), 83-138.
- **DOI/URL:** 10.1111/jofi.12361; WP https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Draft20111215p_edited.pdf
- **Access:** FULL TEXT of the Dec 2011 WP; the published abstract was read on a search page.
- **Key finding:** Published abstract: firms scheduled to report earn an annualized abnormal return of 9.9% (abstract via search, not the WP). WP: the premium is persistent across stocks up to 20 years; early announcers earn more; announcement betas explain 37% of the cross-section of average returns across test portfolios; the sample runs to 2009.
- **Holding/data:** Weekly announcer-minus-non-announcer portfolio.
- **Large-cap/cost notes:** NOT VERIFIED.
- **Caveats:** The 9.9% appears in the published abstract, not in the WP I read.

## 15. Cohen, Malloy & Pomorski (2012)
- **Citation:** JF (volume/pages NOT VERIFIED).
- **DOI/URL:** NBER WP 16454, https://www.nber.org/papers/w16454
- **Access:** FULL TEXT of NBER WP (Oct 2010).
- **Key finding:** Routine insider trades (over half of all insider trades) carry no information. A portfolio of opportunistic trades earns value-weight abnormal returns of 82 bp/month (9.8% annualized, t=2.15); the equal-weight figure was not extracted (text). Routine-minus-opportunistic differences are in Table III. The sample is 1986-2007 (text).
- **Holding/data:** Monthly portfolios; needs insider filings (Form 4 / Thomson). Routine = same calendar month in at least a number of past years.
- **Large-cap/cost notes:** Value-weighted result of 82 bp is the headline, so not driven by microcaps. Costs NOT VERIFIED.
- **Caveats:** WP version; post-2007 persistence NOT VERIFIED.

## 16. Asquith, Pathak & Ritter (2005)
- **Citation:** JFE 78(2), 243-276.
- **DOI/URL:** https://ideas.repec.org/a/eee/jfinec/v78y2005i2p243-276.html; PDF from Ritter's site.
- **Access:** FULL TEXT (JFE print copy).
- **Key finding:** In 1988-2002, short-sale-constrained stocks (high short interest, low institutional ownership) underperform by 215 bp/month equal-weighted (significant) but only 39 bp/month value-weighted (insignificant) (abstract/text). For most stocks, constraints are unlikely to bind.
- **Holding/data:** Monthly; needs short interest and 13F institutional ownership.
- **Large-cap/cost notes:** The value-weighted result being insignificant means the effect lives in small stocks.
- **Caveats:** Sample ends 2002.

## 17. Engelberg, Reed & Ringgenberg (2018)
- **Citation:** JF 73(2), 755-786; DOI 10.1111/jofi.12601.
- **DOI/URL:** WP https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/03/riggenberg.pdf
- **Access:** FULL TEXT of the Oct 2014 WFA WP (only the abstract and sample were examined).
- **Key finding:** Stocks with more short-selling risk (fee volatility, recall risk) have lower returns, less price efficiency and less short selling. Data cover July 2006 - Dec 2011 (text); loan fees have mean 85 bp and 99th percentile 1,479 bp per annum.
- **Holding/data:** Needs securities-lending data (Data Explorers); not free.
- **Large-cap/cost notes:** Directly relevant to borrow cost of shorts. Specific return magnitudes NOT VERIFIED.
- **Caveats:** Short sample.

## 18. Rapach, Ringgenberg & Zhou (2016)
- **Citation:** JFE 121(1), 46-65; DOI 10.1016/j.jfineco.2016.03.004.
- **Access:** ABSTRACT ONLY (via search).
- **Key finding:** Aggregate short interest predicts market returns, with annual out-of-sample R2 reported as 12.89% (in-sample) and 13.24% (out-of-sample) in the abstract; utility gains over 300 bp per year. A 2024 student thesis (secondary) says the power disappears excluding the GFC; not peer reviewed.
- **Holding/data:** Market-level, monthly / annual.
- **Large-cap/cost notes:** Not relevant to cross-sectional single-stock ranking.
- **Caveats:** Optional paper.

## 19. McLean & Pontiff (2016)
- **Citation:** JF 71(1), February 2016 (page range NOT VERIFIED).
- **DOI/URL:** PDF https://Www.Gwern.net/doc/economics/2016-mclean.pdf
- **Access:** FULL TEXT (JF print copy).
- **Key finding:**
  - 97 predictors: portfolio returns are 26% lower out-of-sample and 58% lower post-publication; implied publication effect 32% (abstract).
  - Table I: average monthly return 0.582% in-sample, 0.402% out-of-sample pre-publication, 0.264% post-publication.
  - The average gap between sample end and publication is 56 months (text).
  - Post-publication declines are greater for predictors with larger in-sample returns; returns are higher for portfolios concentrated in high-idiosyncratic-risk and low-liquidity stocks (abstract).
  - Short interest rises post-publication, consistent with informed trading (section on short interest; mean short interest 3.45%, median 0.77%).
- **Holding/data:** Monthly portfolios, equal weights unless the original study used value weights. Sample through 2013.
- **Large-cap/cost notes:** Costs not netted.
- **Caveats:** Earlier WP versions had different numbers (82 characteristics, about 35% decay).

## 20. Shumway (1997)
- **Citation:** JF 52(1), 327-340.
- **DOI/URL:** https://ideas.repec.org/a/bla/jfinan/v52y1997i1p327-40.html
- **Access:** ABSTRACT ONLY (search summary) plus secondary.
- **Key finding:** CRSP omits delisting returns, and the omitted returns are large (abstract). The -30% value for missing performance-related delistings is described by secondary sources as Shumway's standard adjustment; I did not see it in the paper. Shumway & Warther (1999) use -55% for Nasdaq (secondary).
- **Holding/data:** CRSP delisting file.
- **Large-cap/cost notes:** NOT VERIFIED.
- **Caveats:** The -30% attribution is NOT VERIFIED from the primary text.

## 21. Brown, Goetzmann, Ibbotson & Ross (1992)
- **Citation:** RFS 5(4), 553-580; DOI 10.1093/rfs/5.4.553.
- **Access:** ABSTRACT ONLY (partial, via RePEc).
- **Key finding:** In a survivorship-truncated sample, the volatility-return link produces apparent predictability; numerical examples suggest the effect can be strong enough to explain the evidence on predictability. Later rejoinder (1995 WP, secondary): with differing variances, survivorship-biased samples show spurious persistence.
- **Holding/data:** Fund / index data.
- **Large-cap/cost notes:** NOT VERIFIED.
- **Caveats:** Magnitudes NOT VERIFIED.

## 22. Jensen, Kelly & Pedersen (2023)
- **Citation:** JF 78(5), 2465-2518; DOI 10.1111/jofi.13249.
- **DOI/URL:** CBS-hosted publisher's version PDF.
- **Access:** FULL TEXT (final published version).
- **Key finding:** Bayesian model; the majority of factors replicate, cluster into 13 themes, work out of sample in 93 countries (abstract). The replication rate rises to 82.4% under their hierarchical model (text); slightly lower (81.5%, 79.8%) under alternative priors (text). The global dataset has 153 factors.
- **Holding/data:** Monthly factor portfolios.
- **Large-cap/cost notes:** Not an implementation paper; no net-of-cost evidence.
- **Caveats:** The model-based replication rate differs from HXZ's naive count.

## 23. Chen & Zimmermann (2022)
- **Citation:** Critical Finance Review 11(2), 207-264.
- **DOI/URL:** https://cfr.ivo-welch.org/published/papers/chen2021open.pdf
- **Access:** FULL TEXT (CFR working-paper/accepted PDF).
- **Key finding:** 319 characteristics; of the 161 clearly significant originally, 98% of long-short portfolios have t above 1.96. The 44 mixed-evidence ones average t of 2. Reproduced vs original t-stat slope is 0.88, R2 82% (abstract of this PDF). The Fed FEDS page cites 0.90 / 83%, so numbers differ between versions.
- **Holding/data:** Open-source code and data portal.
- **Large-cap/cost notes:** Not addressed.
- **Caveats:** Version-dependent numbers.

## 24. Frazzini, Israel & Moskowitz (2018)
- **Citation:** "Trading Costs," SSRN 3229719 (April 2018).
- **DOI/URL:** https://papers.ssrn.com/abstract=3229719
- **Access:** ABSTRACT ONLY for 2018; FULL TEXT of the Dec 2012 NBER-conference draft "Trading Costs of Asset Pricing Anomalies" (preliminary).
- **Key finding:**
  - 2018 abstract (via search): $1.7 trillion of live trades, 21 developed markets, 19 years; costs an order of magnitude below prior studies.
  - 2012 draft (text, Table II): 1998-2011, 5.3M orders, 9,128 stocks. Mean market impact 12.18 bp, mean implementation shortfall 13 bp. Dollar-weighted: 18.65 bp (MI), 20.20 bp (IS).
  - By size: large caps 11.21 bp market impact vs 21.27 bp small caps.
  - Short-term reversal is the strategy most constrained by trading costs.
- **Holding/data:** One institution's execution data (AQR).
- **Large-cap/cost notes:** About 11 bp one-way impact for large caps (equal-weighted trade average, 2012 draft). The 2018 per-size figures were NOT VERIFIED.
- **Caveats:** The costs are for a sophisticated, patient executor; do not transfer to retail.

## 25. Novy-Marx & Velikov (2016)
- **Citation:** RFS 29(1), 104-147.
- **DOI/URL:** https://www.nber.org/papers/w20721; PDF https://mysimon.rochester.edu/novy-marx/research/ToAatTC.pdf
- **Access:** FULL TEXT of the Aug 2015 working paper (not the RFS print).
- **Key finding:** Most anomalies with turnover below 50%/month generate significant net spreads when designed to mitigate costs; only two of those above 50% one-sided monthly turnover do (industry momentum plus industry-relative reversals; industry-relative reversals among low-vol stocks) (text). A buy/hold spread is the best mitigation. Round-trip costs for typical VW strategies exceed 50 bp; costs reduce the realized spread by more than 1% of monthly one-sided turnover (e.g., 20% turnover costs at least 20 bp/month) (text). Momentum's profitability disappears at about $5bn of new capital per side (text). Sample is July 1963 - Dec 2013 (July 1973 for a sub-period); effective-spread estimates from a matched procedure.
- **Holding/data:** Monthly or annual rebalancing, NYSE-breakpoint decile sorts.
- **Large-cap/cost notes:** Short-term reversal has turnover almost three times momentum (text). Net results for plain reversal are not shown as significant.
- **Caveats:** WP version. Costs are effective-spread based.

## 26. Gu, Kelly & Xiu (2020)
- **Citation:** RFS (volume/pages NOT VERIFIED).
- **DOI/URL:** NBER WP 25398, https://www.nber.org/papers/w25398
- **Access:** FULL TEXT of the NBER WP (Sept 2019 revision).
- **Key finding:**
  - Data: 1957-2016, nearly 30,000 stocks, 94 characteristics (text). Test split NOT VERIFIED.
  - Table 1 monthly out-of-sample R2: All 0.40% for NN3; Top 1000 0.70% for NN3 (0.63% RF); Bottom 1000 0.45%. So gains are not concentrated in small stocks.
  - Table 7: best 10-1 VW strategy (NN4) returns 2.3%/month with annualized Sharpe 1.35. Equal weight Sharpe is 2.45; excluding microcaps (below the NYSE 20th percentile) EW is 1.69 (text referencing Tables A.9, A.10).
  - The paper concludes ML is "most valuable for forecasting larger and more liquid stock returns and portfolios" (text).
  - Dominant predictors: reversal, momentum, liquidity, volatility, valuation.
- **Holding/data:** Monthly decile portfolios.
- **Large-cap/cost notes:** Drawdown / turnover are in Table 8 (not extracted); no explicit net-of-cost Sharpe was verified.
- **Caveats:** Gross returns; not a daily-horizon study. Note that Avramov et al. disagree about hard-to-arbitrage concentration.

## 27. Avramov, Cheng & Metzker (2023)
- **Citation:** Management Science 69(5); DOI 10.1287/mnsc.2022.4449. Pages NOT VERIFIED.
- **Access:** ABSTRACT ONLY (via search).
- **Key finding (abstract):** Deep-learning signals earn profits from difficult-to-arbitrage stocks and high limits-to-arbitrage states; dropping microcaps, distressed stocks, or high-volatility periods substantially weakens profits. Reasonable trading costs make ML performance worse (high turnover, extreme tangency weights). No numbers read.
- **Holding/data:** Monthly.
- **Large-cap/cost notes:** The abstract directly argues ML gains shrink without microcaps and after costs (conflicts in emphasis with GKX).
- **Caveats:** Abstract only; magnitudes NOT VERIFIED.

## 28. Krauss, Do & Huck (2017)
- **Citation:** EJOR 259(2), 689-702; DOI 10.1016/j.ejor.2016.10.031.
- **DOI/URL:** PDF https://www.iwf.rw.fau.de/files/2016/03/03-2016.pdf
- **Access:** FULL TEXT of FAU Discussion Paper 03/2016 (not the EJOR print).
- **Key finding:** S&P 500 constituents (survivor-bias adjusted), Dec 1992 - Oct 2015, daily, k=10 long/short. Ensemble earns 0.45%/day before costs and 0.25%/day after 0.05% per share per half-turn costs (text). Sub-periods (after costs): 12/92-03/01 >200%/yr, Sharpe 6.7; 04/01-08/08: 22%/yr, Sharpe 0.51; 09/08-12/09: >400%/yr, Sharpe 4.5; last sub-period (printed "01/01 to 10/15", most likely 01/2010 to 10/2015): -14% to -25%/yr after costs, but still positive before costs. So yes, profits decayed and turned negative after costs in the recent period.
- **Holding/data:** One-day-ahead, daily rebalance, lagged returns only.
- **Large-cap/cost notes:** Large-cap universe with an assumed 5 bp/half-turn; high turnover.
- **Caveats:** WP version; the paper's cost assumption (5 bp) is optimistic relative to NMV / FIM for most investors.

## 29. Blitz, Huij, Lansdorp & Verbeek (2013)
- **Citation:** Journal of Financial Markets 16(3), 477-504; DOI 10.1016/j.finmar.2012.10.005.
- **Access:** SECONDARY / ABSTRACT ONLY (Robeco / EUR repository pages; PDF fetch failed).
- **Key finding (search summary of abstract):** Ranking on residual returns (versus raw returns) avoids dynamic factor exposures and roughly doubles the risk-adjusted return; profits survive trading costs "including in a large-cap sample" post-1990. Third-party write-ups give a rolling 36-month FF3 residual, universe above NYSE median market cap and above $1, sample Jan 1926 - Dec 2008, gross Sharpe 1.28 vs 0.62. These are NOT VERIFIED from the paper.
- **Holding/data:** Monthly.
- **Large-cap/cost notes:** The abstract-level claim is positive for large caps after costs; magnitudes NOT VERIFIED.
- **Caveats:** Sample ends 2008.

## 30. Da, Liu & Schaumburg (2014)
- **Citation:** Management Science 60(3), 658-674.
- **DOI/URL:** NY Fed Staff Report 513 (earlier version, "Decomposing Short-Term Return Reversal"): https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr513.pdf
- **Access:** FULL TEXT of Staff Report 513 (2011). The published abstract is from a search summary.
- **Key finding:** Staff report: strategy capturing the residual (non-fundamental) component earns a risk-adjusted return three times the standard reversal over 1982-2009; the standard strategy has a 3-factor alpha of 0.33%/month in the sample (text). Published version: four times (abstract via search). Uses analyst forecast revisions as the cash-flow-news proxy.
- **Holding/data:** Monthly; needs I/B/E/S forecast revisions.
- **Large-cap/cost notes:** NOT VERIFIED.
- **Caveats:** The multiplier changed between versions (3x vs 4x).

## 31. Nagel (2012)
- **Citation:** Nagel, S. (2012), "Evaporating Liquidity," RFS (volume/pages NOT VERIFIED).
- **DOI/URL:** NBER WP 17653.
- **Access:** FULL TEXT of the NBER WP.
- **Key finding (abstract):** Short-term reversal returns proxy for returns to liquidity provision; the expected return is strongly time-varying and highly predictable with VIX. Expected returns and conditional Sharpe increase enormously with VIX, e.g., in 2007-09. Even industry-level reversals earn high returns at high VIX. The coefficient magnitudes were not extracted.
- **Holding/data:** Daily/weekly reversal strategy; needs VIX (from 1990).
- **Large-cap/cost notes:** NOT VERIFIED.
- **Caveats:** Implication: reversal profits are state-dependent, and low-VIX periods are weak.

## 32. Harvey, Liu & Zhu (2016)
- **Citation:** RFS (volume/pages NOT VERIFIED).
- **DOI/URL:** NBER WP 20592.
- **Access:** FULL TEXT of the NBER WP (Oct 2014).
- **Key finding:** A new factor needs a t-ratio greater than 3.0 (abstract); Fama-MacBeth's original t=2.57 is the starting reference; they argue most claimed research findings in financial economics are likely false.
- **Caveats:** WP version.

## 33. White (2000)
- **Citation:** Econometrica 68(5), 1097-1126; DOI 10.1111/1468-0262.00152.
- **Access:** FULL TEXT (journal print copy via UW-Madison).
- **Key finding:** Test the null that the best model found in a specification search has no predictive superiority over a benchmark (Reality Check, bootstrap). Applied to S&P 500 technical-trading rules (application data around 1990-1994, text).
- **Large-cap/cost notes:** n/a.

## 34. Hansen (2005)
- **Citation:** J. Business & Economic Statistics 23(4), 365-380.
- **Access:** SECONDARY (PDF fetch failed; abstract not retrieved).
- **Key finding (secondary):** SPA test; improves power over White's Reality Check, which is asymptotically biased (Song 2012 summary). No numbers.

## 35. Bailey & Lopez de Prado (2014)
- **Citation:** "The Deflated Sharpe Ratio," Journal of Portfolio Management (volume/pages NOT VERIFIED).
- **DOI/URL:** SSRN 2460551; PDF https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- **Access:** FULL TEXT (SSRN version; abstract read).
- **Key finding:** DSR corrects for selection bias under multiple testing and for non-normal returns; separates legitimate findings from flukes. Formula details not extracted here.

## 36. Hou, Xue & Zhang (2020)
- **Citation:** RFS (volume/pages NOT VERIFIED).
- **DOI/URL:** NBER WP 23394.
- **Access:** FULL TEXT of the NBER WP (May 2017).
- **Key finding (abstract):** 447 anomaly variables. With NYSE breakpoints and value-weighted returns, 286 (64%) are insignificant at 5%, including 95 of 102 liquidity variables (93%); with t>3, 380 (85%) are insignificant. Of 161 significant ones, the q-factor model leaves 115 alphas insignificant. Sample January 1967 - December 2014 (text). Cause cited: microcaps and equal weighting.
- **Caveats:** WP numbers; the published version may differ.

## 37. Chordia, Subrahmanyam & Tong (2014)
- **Citation:** JAE 58(1), 41-58; DOI 10.1016/j.jacceco.2014.06.001.
- **Access:** ABSTRACT ONLY (SMU repository abstract via search).
- **Key finding:** Most anomalies attenuated; average anomaly-portfolio returns about halved after decimalization; decline tied to hedge fund AUM, short interest, and share turnover. A later paper (Auer & Rottmann, CESifo 2018, secondary) says the US evidence is not fully robust.
- **Caveats:** Numbers beyond "about halved" NOT VERIFIED.

## 38. Lo & MacKinlay (1990)
- **Citation:** RFS 3(2), 175-205 (RePEc) or 175-206 (MIT page).
- **DOI/URL:** NBER WP 2977.
- **Access:** FULL TEXT of the NBER WP.
- **Key finding (abstract):** Contrarian profits need not reflect overreaction: with serially independent individual returns, cross-autocovariances make contrarian portfolios profitable. Evidence that large-stock returns lead small-stock returns.

## 39. Analyst forecast revisions (Stickel 1991)
- **Citation:** Stickel, S. (1991), "Common Stock Returns Surrounding Earnings Forecast Revisions: More Puzzling Evidence," The Accounting Review 66(2), 402-416. It is NOT a Journal of Finance paper (a 1992 JF Stickel paper exists on analyst reputation, a different topic).
- **Access:** SECONDARY (a Brazilian paper citing it; the PDF I downloaded was that paper, not Stickel).
- **Key finding (secondary):** Upward consensus revisions are followed by positive abnormal returns that persist 3 to 12 months. NOT VERIFIED from Stickel.
- **Alternative with text read:** Da, Liu & Schaumburg (item 30) uses revisions as a cash-flow-news proxy.

## 40. Cohen, Malloy & Nguyen (2020), "Lazy Prices"
- **Citation:** JF (volume/pages NOT VERIFIED).
- **DOI/URL:** NBER WP 25084.
- **Access:** FULL TEXT of NBER WP (Sept 2018, rev. March 2019).
- **Key finding:** 10-K/10-Q filings, 1995-2014. Abstract: shorting "changers" and buying "non-changers" earns up to 188 bp/month alphas (t=2.76); intro: value-weighted abnormal returns of 34-58 bp/month (up to 7%/year, t=3.59) over the following year, accruing out to 18 months with no reversal. Returns accrue only when the information later appears (no announcement effect).
- **Holding/data:** Stocks enter the month after filing release, held 3 months, rebalanced monthly (table notes). Needs EDGAR text and parsing.
- **Large-cap/cost notes:** The VW figure (34-58 bp) is far below the headline 188 bp. Costs NOT VERIFIED. Sample ends 2014, post-publication decay NOT VERIFIED.

---

## Summary table

| # | Paper | Access | Verified |
|---|---|---|---|
| 1 | Jegadeesh 1990 | SECONDARY | partial (citation only) |
| 2 | Lehmann 1990 | FULL TEXT (WP, OCR) | partial |
| 3 | Jegadeesh-Titman 1993 | FULL TEXT | Y |
| 4 | Carhart 1997 | ABSTRACT ONLY | partial |
| 5 | Chan-Jegadeesh-Lakonishok 1996 | ABSTRACT ONLY | partial |
| 6 | Asness-Moskowitz-Pedersen 2013 | FULL TEXT | Y |
| 7 | Novy-Marx 2013 | FULL TEXT (manuscript) | Y |
| 8 | Fama-French 2015 | FULL TEXT | Y (abstract-level) |
| 9 | Frazzini-Pedersen 2014 | FULL TEXT (WP) | Y |
| 10 | Ang et al. 2006 | FULL TEXT (WP) | Y |
| 11 | Bernard-Thomas 1989 | SECONDARY | partial |
| 12 | Martineau 2022 | FULL TEXT | Y |
| 13 | Frazzini-Lamont 2007 | FULL TEXT (WP) | Y |
| 14 | Savor-Wilson 2016 | FULL TEXT (WP 2011) | Y |
| 15 | Cohen-Malloy-Pomorski 2012 | FULL TEXT (WP) | Y |
| 16 | Asquith et al. 2005 | FULL TEXT | Y |
| 17 | Engelberg et al. 2018 | FULL TEXT (WP) | partial |
| 18 | Rapach et al. 2016 | ABSTRACT ONLY | partial |
| 19 | McLean-Pontiff 2016 | FULL TEXT | Y |
| 20 | Shumway 1997 | ABSTRACT ONLY | partial (-30% unconfirmed) |
| 21 | Brown et al. 1992 | ABSTRACT ONLY | partial |
| 22 | Jensen-Kelly-Pedersen 2023 | FULL TEXT | Y |
| 23 | Chen-Zimmermann 2022 | FULL TEXT | Y |
| 24 | Frazzini-Israel-Moskowitz 2018 | ABSTRACT (2018) + FULL TEXT (2012 draft) | partial |
| 25 | Novy-Marx-Velikov 2016 | FULL TEXT (WP) | Y |
| 26 | Gu-Kelly-Xiu 2020 | FULL TEXT (WP) | Y |
| 27 | Avramov et al. 2023 | ABSTRACT ONLY | partial |
| 28 | Krauss et al. 2017 | FULL TEXT (WP) | Y |
| 29 | Blitz et al. 2013 | ABSTRACT/SECONDARY | partial |
| 30 | Da-Liu-Schaumburg 2014 | FULL TEXT (2011 WP) | partial |
| 31 | Nagel 2012 | FULL TEXT (WP) | Y (qualitative) |
| 32 | Harvey-Liu-Zhu 2016 | FULL TEXT (WP) | Y |
| 33 | White 2000 | FULL TEXT | Y |
| 34 | Hansen 2005 | SECONDARY | partial |
| 35 | Bailey-Lopez de Prado 2014 | FULL TEXT (SSRN) | Y |
| 36 | Hou-Xue-Zhang 2020 | FULL TEXT (WP) | Y |
| 37 | Chordia et al. 2014 | ABSTRACT ONLY | partial |
| 38 | Lo-MacKinlay 1990 | FULL TEXT (WP) | Y |
| 39 | Stickel 1991 | SECONDARY (it is Accounting Review, not JF) | partial |
| 40 | Cohen-Malloy-Nguyen 2020 | FULL TEXT (WP) | Y |

Papers that did not exist as described: none outright. Stickel 1991 is in The Accounting Review, not the Journal of Finance. Most FULL TEXT entries are working-paper versions; numbers may differ from the journal print.
