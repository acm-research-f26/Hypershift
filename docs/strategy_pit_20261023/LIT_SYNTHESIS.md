# B. Literature synthesis: what the evidence says about where an edge could be

Source verification: `LIT_VERIFY.md` (2026-10-23). Each entry there gives its access level: FULL TEXT (usually a working-paper
version), ABSTRACT ONLY or SECONDARY, plus what could not be verified. Below, a number is quoted only if
`LIT_VERIFY.md` saw it in the paper. Claims resting on abstracts or secondary sources are marked *(abstract)* or
*(secondary)*. Two headline facts were re-checked independently:
- Martineau 2022: CFR 11(3-4), 613–646; abstract "PEAD have been non-existent since 2006" for large stocks.
- McLean & Pontiff 2016: 26% lower out-of-sample, 58% lower post-publication.

## Cross-cutting lessons (these set the priorities)

1. **Published anomalies decay.**
   - McLean & Pontiff (2016): across 97 predictors, returns are 26% lower out-of-sample and 58% lower post-publication; the monthly mean falls from 0.582% to 0.264%.
   - Chordia, Subrahmanyam & Tong (2014) *(abstract)*: anomaly returns roughly halved after decimalisation.
   - **Planning prior:** any published effect is worth well under half its paper value today, before costs.
2. **Most of the evidence lives in small, illiquid stocks and equal weights.**
   - Hou, Xue & Zhang (2020): with NYSE breakpoints and value weights, 64% of 447 anomalies are insignificant (85% at t > 3), including 93% of liquidity variables.
   - Asquith, Pathak & Ritter (2005): the short-constraint effect is 215 bp/month equal-weighted but an insignificant 39 bp value-weighted.
   - Avramov, Cheng & Metzker (2023) *(abstract)*: ML profits come from hard-to-arbitrage stocks and weaken without microcaps.
   - Gu, Kelly & Xiu (2020) disagree: out-of-sample R² is 0.70% for the top 1,000 stocks vs 0.45% for the bottom 1,000. That is a statistical fit, not net profit.
   - **For an S&P 500 universe, assume the value-weighted, large-cap version of each effect, which is the weaker one.**
3. **Turnover is the binding constraint.**
   - Novy-Marx & Velikov (2016): of anomalies with one-sided turnover above 50% a month, only two survive costs, both industry-relative reversal variants. Short-term reversal turns over about 3× as much as momentum.
   - Frazzini, Israel & Moskowitz (2012 draft): large-cap market impact is about 11 bp one-way for a patient institution, and short-term reversal is the strategy most constrained by costs.
   - Krauss, Do & Huck (2017): daily ML long-short on S&P 500 constituents was hugely profitable before 2010 but loses 14–25%/yr after 5 bp half-turn costs in 2010–2015.
   - **HyperShift's own result (no net edge from price-only short-horizon learners) is what this literature predicts.**
4. **Survivorship and delisting bias are large enough to create the effects we found.**
   - Brown, Goetzmann, Ibbotson & Ross (1992) *(abstract)*: survivor-truncated samples produce spurious predictability.
   - Shumway (1997) *(abstract)*: omitted delisting returns are large. The often-quoted −30% default could **not** be confirmed in the paper and is used here only as a labelled sensitivity.
   - In HyperShift: the survivor equal-weight benchmark showed +3.1% FF alpha (t 2.6), and the candidate's short leg loses in exactly the names whose failed peers are missing.
5. **Multiple testing.**
   - Harvey, Liu & Zhu (2016): a new factor needs t > 3.
   - White (2000) Reality Check, Hansen (2005) SPA *(secondary)*, Bailey & López de Prado (2014) deflated Sharpe.
   - Replication is better for well-known factors: Jensen, Kelly & Pedersen (2023), 82% under their Bayesian model; Chen & Zimmermann (2022), 98% of clearly significant predictors reproduce. These cover gross factor premia, not net, large-cap, post-publication trading.

## Direction by direction

Columns, kept short: mechanism · information · holding period · when it works · survives costs? · transfers to liquid US
large caps? · already tested in HyperShift? · new idea.

| Direction | Mechanism | Information | Holding | Conditions | Cost survival | Large-cap transfer | HyperShift | Idea / verdict |
|---|---|---|---|---|---|---|---|---|
| Short-term reversal (Lehmann 1990; Jegadeesh 1990 *(secondary)*; Lo & MacKinlay 1990) | Liquidity provision; part is lead-lag cross-autocovariance (Lo–MacKinlay) | Prices | 1 week to 1 month | Expected return rises strongly with VIX (Nagel 2012) | Worst of all anomalies (NMV; FIM) | Weak net | **Yes.** 16-day reversal beat every learner gross (FB). RR21 IR +0.19 at 10 bp, negative at 20 bp | Only state-dependent (VIX) versions are plausible. Low power: high-VIX regimes are rare. STOP as standalone |
| Residual / industry-relative reversal (Blitz et al. 2013 *(abstract)*; Da, Liu & Schaumburg 2014) | Removing factor or cash-flow news isolates the non-fundamental part | Prices; analyst revisions for Da et al. | Monthly | Large-cap claim after costs *(abstract)* | Better than raw reversal | Partly | **Yes** (RR21 sector/beta residual; residual targets in FB) | Already in the style book's target. No new information |
| Momentum (Jegadeesh & Titman 1993; Carhart 1997 *(abstract)*; Asness et al. 2013) | Under-reaction | Prices | 3–12 months | Crashes after rebounds (not verified here) | Yes (low turnover; JT: 9.29%/yr after 0.5% one-way, 1965–89) | Weaker | **Yes.** MOM tilt shows no net edge vs EW, 2016–2023 (survivor panel) | A cheap factor exposure, not alpha. STOP as a product |
| Low beta / low idiosyncratic volatility (Frazzini & Pedersen 2014; Ang et al. 2006) | Leverage constraints; lottery demand | Prices | Monthly | — | Yes (low turnover) | AHXZ: −0.44%/month FF3 alpha without small growth | **Yes.** LOWVOL tilt no net edge; vol_60 and beta are style-book inputs | Factor tilt. STOP as alpha |
| Profitability / value / quality (Novy-Marx 2013; Fama & French 2015) | Risk or mispricing of profitable firms | Fundamentals (XBRL / Compustat) | Annual | — | Yes | Large-cap profitability spread 26 bp/month, t = 1.88 (Novy-Marx) | **No** | Already priced as RMW/CMA and buyable via ETFs. Use as attribution, not as a research target |
| PEAD / SUE (Bernard & Thomas 1989 *(secondary)*; Chan, Jegadeesh & Lakonishok 1996 *(abstract)*) | Delayed reaction to earnings news | Quarterly EPS, filing dates | 60 days | **Gone for large caps since 2006** (Martineau 2022; 2025 rebuttals unverified) | — | No | No (planned as H-B in `NEW_DATA.md`) | **STOP H-B.** The best available evidence says the effect is gone in exactly our universe |
| Earnings-announcement premium (Frazzini & Lamont 2007; Savor & Wilson 2016) | Attention-driven buying or systematic announcement risk | **Announcement dates** (predictable from the prior year) | Announcement month or week | FL: 61 bp/month, "strong in large caps", 1973–2004; SW: 9.9%/yr abnormal *(abstract)* | **Doubtful**: a long-announcers book replaces most of each leg every month (~20× NAV/yr, like the failed style book). Halved for decay and net of ~2%/yr costs, a net Sharpe near 0.2 | Claimed yes | **No** | **F candidate #3.** Free dates (EDGAR 8-K Item 2.02 from Aug 2004) but high turnover; test only as a low-cost overlay on another book |
| Filing-text change, "Lazy prices" (Cohen, Malloy & Nguyen 2020) | Inattention to subtle 10-K/10-Q changes | EDGAR filing text | 3 months; no reversal to 18 months | VW 34–58 bp/month (up to 7%/yr), 1995–2014 | Plausibly (quarterly) | VW figure given | **No** | **F candidate #1.** Quarterly signal, 3-month hold (about 4× NAV/yr), a value-weighted claim, free text from about 1996, so the longest testable history. Heavier engineering (text diffing) |
| Insider trading (Cohen, Malloy & Pomorski 2012) | Opportunistic (non-routine) insiders know more | SEC Form 4 | Monthly | VW 82 bp/month (t 2.15), 1986–2007 | Plausibly | VW headline, so not microcap-driven | **No** | **F candidate #2.** Free structured Form 4 data (XML from 2003). Signals are sparse in large caps; post-2007 persistence unverified |
| Short interest / lending (Asquith et al. 2005; Engelberg et al. 2018; Rapach et al. 2016) | Short-sale constraints; informed shorting | Short interest (FINRA, free, semi-monthly); lending fees (paid) | Monthly | Effect concentrated where constraints bind | — | **Weak:** VW insignificant (APR) | No | DEFER: the constraint barely binds in S&P 500 names; the useful data (fees) are paid |
| ML on characteristics (Gu, Kelly & Xiu 2020; Avramov et al. 2023; Krauss et al. 2017) | Nonlinear combination of known signals | Characteristics | Monthly (GKX) / daily (Krauss) | Avramov: limits to arbitrage | Krauss: negative after costs 2010–15 | Contested | **Yes, exhaustively** (THINK, screen A–J, FB, style book) | STOP architecture work. Our nulls agree with Krauss and Avramov |
| Analyst revisions (Stickel 1991 *(secondary)*; Da et al. 2014) | Sluggish incorporation of analyst news | I/B/E/S (institutional) | 3–12 months *(secondary)* | — | — | — | No | DEFER: no affordable point-in-time source for an individual |

## What this implies for priorities

1. **Stop spending on price-only, short-horizon, high-turnover signals in large caps.** The literature (NMV, FIM, Krauss) and
   HyperShift's five studies agree: the gross effect is small and costs take it.
2. **New information beats new models.** All three remaining candidates add information the program never had (filing
   text, insider filings, event dates) and come with a value-weighted or large-cap claim in the original paper. They are
   ranked by **turnover** and **years of free point-in-time history**, the two quantities that decide whether an edge
   can survive costs and be tested at all:

   | Candidate | Est. turnover | Free information history | Rank |
   |---|---|---|---|
   | Lazy prices (10-K/10-Q text change) | ~4× NAV/yr (quarterly, 3-month hold) | EDGAR full text, about 1996 onward | 1 |
   | Opportunistic insider purchases | Low (monthly, sparse signals) | Form 4 XML, 2003 onward | 2 |
   | Earnings-announcement premium | ~20× NAV/yr | 8-K Item 2.02 dates, Aug 2004 onward | 3 |
3. **Expect decay.** Plan with less than half the published magnitude (McLean–Pontiff). For Lazy Prices' value-weighted
   34–58 bp/month that is roughly 15–30 bp/month. A long-short Sharpe near 0.3–0.5 is a realistic
   ceiling, and confirming it needs 30+ years of point-in-time data. **History length, not model choice, decides
   whether any of these can be tested.**
4. **PEAD (the H-B plan in `NEW_DATA.md`) is withdrawn:** Martineau (2022) reports no large-cap drift since 2006.
