# HyperShift: point-in-time validation and next direction (2026-10-23)

**Bottom line.**
- **The style book is stopped.** The only surviving candidate (style-only LightGBM sector-neutral long/short book)
  was re-tested under a predeclared rule on a free, validated point-in-time S&P 500 universe that includes failed and
  acquired names. Net Sharpe +0.11 [−0.61, +0.90] at 10 bp; FF alpha t +0.15. Decision: **STOP.**
- **The earlier edge was the survivor universe.** With GICS sectors instead of the proxy, the same arm gives −0.17.
  The equal-weight survivor alpha (+3.1%, t 2.6) becomes −0.7% (t −0.5) on the true universe.
- **No price-based candidate is left.**
- **Next bet: new information, not new models.** The best is filing-text change ("Lazy Prices": low turnover,
  value-weighted claim, free history from about 1996). It can only be *confirmed* on pre-2016 data, which needs
  Norgate Platinum ($630/yr). That purchase is your decision, and it is worth making only after a frozen spec passes
  a free 2016–2023 kill screen.

Files in `docs/strategy_pit_20261023/`:
- `PREREG.md`: predeclared spec plus amendment A1, committed before any number;
- `PIT_DATA.md`: panel build and validation;
- `DATA_ASSESSMENT.md` (C);
- `LIT_VERIFY.md` and `LIT_SYNTHESIS.md` (B);
- `STYLE_PIT.md` (D, E);
- this report (A, F–I);
- `STRATEGY_PIT_LEDGER.md`: rulings.

## The standard question, answered

| Question | Answer |
|---|---|
| What economic advantage are we pursuing? | Until today: nonlinear interactions of beta, liquidity, 60-day return and volatility (the style book). **Stopped.** Next: investor inattention to subtle changes in 10-K/10-Q text (Cohen, Malloy & Nguyen 2020), a slow information-diffusion effect that needs no fast trading |
| What evidence supports its existence? | Style book: none once survivorship is removed. Lazy Prices: one peer-reviewed paper, value-weighted 34–58 bp/month in 1995–2014 (full text read, working-paper version), no reversal to 18 months; post-publication decay unverified |
| What could invalidate it? | Post-publication decay (McLean–Pontiff: −58% on average), large-cap concentration of the effect being too weak, and text-diff measurement choices that can be tuned (a multiple-testing risk). All three are testable |
| What is the cheapest reliable way to test it? | Free EDGAR text + the free Alpaca point-in-time panel for a 2016–2023 *kill screen* under a frozen spec. Then **one** holdout on 1997–2015 with Norgate prices, which needs your authorisation for $630 |
| How will the result change what we do? | Killed in 2016–2023 (net Sharpe < 0): stop strategy research on large caps and keep only factor exposure bought cheaply. Not killed: ask for Norgate and run the once-only holdout. Passing that holdout is the first evidence that could justify paper trading, and then capital |

## A. Current evidence: claims checked against code and data

| Earlier claim (`docs/strategy_20261022/`) | Check | Status now |
|---|---|---|
| Style book +1.19 net Sharpe (survivors, 3 seeds) | Reproduced as arm O in the old setting. Under PIT with retraining: P +0.11 / −0.17; paper-spec universe S +0.34 / −0.34 | **Superseded: survivor artefact** |
| "Under PIT membership the edge halves to +0.61" | That run scored only names known to survive to 2026, and its engine keeps weights of names that leave the index (exits gated by eligibility). Stuck weights explain about 0.1 of the Sharpe (E6); the survivor-conditioned universe explains the rest | **Corrected:** the true PIT edge is about 0 |
| Equal-weight survivor alpha +3.1% (t 2.6) is a survivorship signature | PIT equal weight −0.7% (t −0.5); survivors inside the same PIT data +1.5% (t 0.95) | **Confirmed** |
| PRICE3, MOM/LOWVOL tilts, RR21 alone: STOP | RR21 on PIT +0.36 at 10 bp, −0.02 at 20 bp | **Confirmed** |
| H-B (PEAD/SUE) is the best new-data hypothesis | Martineau (2022, full text): no large-cap drift since 2006 | **Withdrawn** |
| Norgate trial first, then Platinum | The trial has only 2 years of history; Alpaca made the 2016–2023 PIT test free | **Revised** (C) |
| Paper trade "exits names that leave the index" | `scripts/paper_trade.py` uses `tradable = fb.eligible` (64-session rule only) and never applies the membership filter its spec requires. The spec's outcome rule ("`engine.run` on logged weights") is also not implementable as written | **Code ≠ spec.** Moot: the paper trade is retired (H) |
| `engine.run` is correct for survivor panels | True for the survivor panel; under PIT eligibility it keeps unhedged stubs. `strategy/pit_engine.run_pit` fixes exits and liquidates delisted names; it reproduces `engine.run` when every name is always eligible (test) | New engine for every PIT test; the frozen engine is untouched |

## B. Literature synthesis

See `LIT_SYNTHESIS.md`: 40 papers, each with an access level in `LIT_VERIFY.md`. 26 were read in full text (mostly
working-paper versions), 8 as abstracts only, 5 from secondary sources. Two headline facts were re-checked
independently. Implications for priorities:
1. **Price-only, short-horizon, high-turnover signals in large caps are exhausted.** Five HyperShift studies agree
   with Novy-Marx & Velikov, Frazzini–Israel–Moskowitz and Krauss et al.
2. **Plan with less than half of any published effect.** McLean & Pontiff measure −58% post-publication. Most
   anomalies live in small, equal-weighted stocks (Hou, Xue & Zhang).
3. **PEAD is dead in large caps** (Martineau).
4. **The remaining candidates bring new information.** Rank them by turnover and by years of free point-in-time
   history.

## C. Data assessment

See `DATA_ASSESSMENT.md`.
- **Alpaca SIP plus fja05680 membership is validated for 2016–2023:**
  - 695 of 695 membership intervals have bars;
  - 99.7% of member-days are covered;
  - returns agree with yfinance on survivors (median correlation 1.000);
  - delisted and failed names are included.
- **Defects found and handled:**
  - several spin-offs are recorded as reverse splits; returns were rebuilt from raw bars (PREREG A1);
  - `asof` splices histories around some mergers;
  - there are no terminal returns after delisting;
  - there is no point-in-time classification.
- **Not validated, and blocked on your decision:** anything before 2016.
  - Norgate Platinum: $630/yr, 1990 onward, delistings and constituents; its trial has only 2 years of history.
  - Sharadar: needs your login for a quote.
  - CRSP: institutional.
- **Nothing purchased.**

## D. Style-strategy evaluation (predeclared kill test)

See `STYLE_PIT.md`. Window 2018–2023, weekly, next-open execution, 10 bp costs plus 50 bp/yr borrow, PIT engine.

| Arm | Net Sharpe @10 [95% CI] | @20 | FF alpha (t) | Traded/yr |
|---|---|---|---|---|
| **P: PIT retrain (primary)** | **+0.11 [−0.61, +0.90]** | −0.41 | +0.2% (+0.15) | 19.2 |
| S: paper-spec universe (survivors ∩ PIT) | +0.34 [−0.48, +1.11] | −0.23 | +0.6% (+0.39) | 21.3 |
| O: survivor-trained saved predictions (2019–23) | +0.20 [−0.54, +0.96] | −0.08 | −0.3% (−0.19) | 12.5 |
| RR21 (zero parameters) | +0.36 [−0.42, +1.19] | −0.02 | +2.1% (+0.91) | 29.7 |
| R: Ridge (degenerate by construction; see note) | −0.84 | −1.00 | −6.5% (−2.42) | 12.9 |
| N: random scores | −2.72 | −4.60 | −7.7% (−6.80) | 72.1 |

- **Sensitivities for P** (none changes the decision):
  - GICS sectors for survivors: −0.17;
  - next-close execution: +0.10;
  - −100% terminal return on failures: +0.08.
- **Paired contrasts:** P − S −0.21 [−0.98, +0.65]; P − RR21 −0.27.
- **Decision: STOP** (net Sharpe < 0.30).
- **Power and selection:**
  - the minimum detectable Sharpe is about 1.1, so this was a kill test, as declared;
  - under the survivor estimate (1.19), P(STOP) was about 1.5%;
  - selection on the overlapping survivor years biases upward, so the STOP is conservative.

## E. Sanity checks

| ID | Result | Consequence |
|---|---|---|
| E1 | PIT equal-weight FF alpha −0.7% (t −0.5), vs +3.1% (t 2.6) on survivors | Universe fix works |
| E2 | 430/462 survivors match yfinance; median share of days within 0.5% = 1.000 | Data consistent |
| E3 | 27 one-day moves above 40% before cleaning; 7 were corporate-action errors (5 spin-offs recorded as reverse splits, the UA share dividend, an AIV adjustment error). 12 member-days zeroed; the rest are dated news | Fixed before scoring (A1) |
| E4 | Label-maturity guard in `pit.train_signals` (test) | Passed |
| E5 | Random scores −7.6%/yr against a cost-plus-borrow drag of 7.5%/yr | Engine accounting correct |
| E6 | Frozen engine vs PIT engine: P +0.22 vs +0.11, O +0.24 vs +0.20 | Stuck weights inflated earlier PIT figures slightly |

No other check was run: none could have changed the decision.

## F. Three strongest alternative directions

Ranked by turnover (cost survival) and years of free point-in-time information (testability). Details in `LIT_SYNTHESIS.md`.

1. **Lazy Prices: year-on-year change in 10-K/10-Q text.**
   - **Mechanism:** inattention.
   - **Evidence:** value-weighted 34–58 bp/month, 1995–2014, no reversal to 18 months.
   - **Turnover:** about 4× NAV/yr (quarterly signal, 3-month hold).
   - **Information:** EDGAR full text, free, about 1996 onward.
   - **Risks:** decay after the 2018–2020 publication; a multiple-testing temptation in the similarity measure, which must be fixed in advance (cosine on the paper's definition).
2. **Opportunistic insider purchases** (Cohen, Malloy & Pomorski 2012).
   - **Evidence:** value-weighted 82 bp/month (t 2.15), 1986–2007.
   - **Turnover:** low.
   - **Information:** Form 4 XML, free, 2003 onward.
   - **Risks:** signals are sparse in S&P 500 names; persistence after 2007 is unverified.
3. **Earnings-announcement premium** (Frazzini & Lamont 2007; Savor & Wilson 2016).
   - **Evidence:** 61 bp/month, "strong in large caps", to 2004.
   - **Information:** 8-K Item 2.02 dates, free, from August 2004.
   - **Risk:** about 20× NAV/yr turnover, so the net Sharpe is near 0.2 after decay. Test it only as an overlay (timing tilt) on another book.

Not chosen:
- short interest: the value-weighted effect is insignificant, and fee data is paid;
- analyst revisions: no affordable point-in-time source;
- profitability, value, quality: factor premia, cheaply bought;
- more models on price data.

## G. Resource allocation

**Power arithmetic.** A realistic net large-cap edge is a Sharpe of 0.3–0.5. Testing it:

| History | Detectable Sharpe | Power at a true Sharpe of 0.4 |
|---|---|---|
| 2016–2023 (8 years) | about 1.0 | about 20% |
| 1997–2015 (19 years) | about 0.64 | about 40% |
| 1997–2023 (27 years) | about 0.54 | about 55% |

No affordable dataset *confirms* such an edge. The allocation therefore maximises information per dollar and accepts
that the final answer may stay "not shown".

| Rank | Action | Cost | Value |
|---|---|---|---|
| 1 | **Freeze a Lazy Prices spec on priors**: universe, similarity measure, quarterly rebalance, value or equal weight, costs and thresholds, before touching data | 1 day | Prevents the multiple-testing failure that sank earlier work |
| 2 | **EDGAR ingestion**: CIK↔ticker history, 10-K/10-Q text, filing timestamps, and SIC codes. The SIC codes replace the 73% sector proxy | 1–2 weeks, $0 | Needed for both the screen and the holdout |
| 3 | **2016–2023 kill screen** on the existing Alpaca PIT panel. STOP if net Sharpe < 0 | Hours of CPU | Cheap. It removes a dead idea but cannot confirm a live one |
| 4 | **If not killed: you authorise Norgate Platinum ($630/yr)** for a once-only 1997–2015 holdout under the frozen spec | $630 + 2–3 days | **The first point where money buys decisive information** |
| 5 | Insider (F2) and announcement-timing overlay (F3) | Only after 1–4 | Second chances, sharing the EDGAR infrastructure |
| — | Capital | — | Only after a passed holdout plus 12 months of paper trading |

## H. Decisions

| Item | Decision | Reason |
|---|---|---|
| Style-only book as a capital candidate | **STOP** | Predeclared kill test failed (+0.11; −0.17 with GICS); no variant reproduces under PIT |
| Paper trade of the style book (first signal 2026-11-30) | **STOP** (retire before start) | Its own capital rule required a PIT re-test with Sharpe ≥ 0.5 and alpha t ≥ 2, which has now failed. It can never lead to capital, and its code omits the membership filter its spec requires. Spec and code stay frozen as a record |
| PEAD/SUE (H-B) | **STOP** | Martineau 2022: no large-cap drift since 2006 |
| Price-only learners, reversal, momentum and low-vol as alpha, architectures | **STOP** | Confirmed again on PIT data; literature agrees |
| PIT data infrastructure (`data/pit_sp500`, `strategy/pit`, `pit_engine`) | **BUILD** (done; keep) | Free, validated, reusable for every future kill screen |
| Lazy Prices (F1) | **TEST** (spec, then EDGAR, then kill screen) | Best mix of prior evidence, turnover and free history |
| Norgate Platinum | **DEFER**, pending your authorisation after F1 passes the screen | $630 buys decisive information only for a frozen candidate that survived the screen |
| Insider purchases (F2), announcement overlay (F3) | **DEFER** | After F1; same infrastructure |
| Sharadar quote | **DEFER** (needs your login) | Ask only if fundamentals or insider history before 2003 become necessary |
| 2024–2026 pseudo-holdout; 1990–2014 history | **Protected** (untouched) | Not spent on an underpowered or failed candidate |

## I. Next execution plan

1. **Spec (day 1).** Write `docs/strategy_lazyprices/PREREG.md`. It must fix:
   - the S&P 500 PIT universe;
   - the 10-K/10-Q similarity measure (cosine on word counts as in the paper; no alternatives);
   - quarterly rebalance with entry the month after filing and a 3-month hold;
   - long-short quintiles, value and equal weight both reported, equal weight primary;
   - costs of 10 bp plus borrow;
   - FF5 + MOM + ST_Rev attribution;
   - thresholds: kill if net Sharpe < 0 on 2016–2023; CONTINUE to the Norgate request if ≥ 0.3 and alpha t ≥ 1;
   - the holdout rule.

   Commit it before downloading text.
2. **EDGAR module (week 1–2).** `src/hypershift/data/edgar.py`. Tests:
   - a filing is visible only after its acceptance time;
   - the CIK↔ticker mapping survives renames;
   - the similarity is computed only against the same form one year earlier;
   - SIC codes come out per CIK.

   Respect SEC fair-access limits (declared User-Agent, ≤ 10 requests/second).
3. **Kill screen (week 2).** Run on `data/pit/pit_sp500_2016_2023.npz` with `pit_engine.run_pit` and report once.
4. **Decision point.**
   - Killed: write the closing report and stop large-cap strategy research.
   - Not killed: ask you to authorise Norgate Platinum, then build `data/norgate.py` and run the 1997–2015 holdout once.

Commands to reproduce today's results:

```bash
.venv/Scripts/python.exe scripts/pit_build_panel.py        # Alpaca credentials in env; ~25 min first run (cached after)
.venv/Scripts/python.exe scripts/strategy_pit.py           # STYLE_PIT.md (~20 min; scores cached in results/STRATEGY_PIT_20261023)
.venv/Scripts/python.exe scripts/strategy_pit_sector_check.py
```
