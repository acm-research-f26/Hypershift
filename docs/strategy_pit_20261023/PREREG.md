# Predeclared PIT re-test of the style-only book (written 2026-10-23, before any PIT outcome was computed)

**Status.** Committed before the point-in-time (PIT) panel is scored. Any change after this commit needs a dated
amendment below, with its reason, and is reported as a deviation.

## What this test is and is not

- **Is:** a *kill test* of the only surviving candidate (style-only LightGBM sector-neutral LS book) on a universe
  free of survivorship: every S&P 500 member on each date 2016–2023, including names later acquired, renamed or
  failed. It answers: *does the survivor-panel edge (+1.19 net Sharpe, 3 seeds) survive once the failed names are
  present?*
- **Is not:** confirmation. 2018–2023 was inspected (on the survivor panel) when the candidate was chosen. Six
  years give a Sharpe standard error of about 1/√6 ≈ 0.41 and a minimum detectable Sharpe (80% power, two-sided
  5%) of about 2.8/√6 ≈ 1.14. Only a STOP result is decisive.
- **Operating characteristics of the STOP rule (net Sharpe < 0.30):** if the true Sharpe were 1.19, P(STOP) ≈ 1.5%;
  if 0.6, ≈ 23%; if 0, ≈ 77%.

## Data (validated before this spec; `PIT_DATA.md`)

| Item | Rule |
|---|---|
| Membership | fja05680 `components.csv` (latest snapshot on or before each date) |
| Prices | Alpaca SIP daily bars, 2016-01-04 to 2023-12-29. Each membership interval (ticker, start, end) is queried with `asof` = its last member day, which maps the ticker to the company that held it then (renames, ticker reuse). Entities with identical daily returns are merged; membership is the union |
| Returns / labels | `adjustment=all` close and open |
| Dollar volume | `adjustment=raw` close × raw volume (never adjusted × raw) |
| Protected periods | No bar after 2023-12-31 is fetched or used. 1990–2014 untouched (Alpaca has no data before 2016) |
| Sectors (one rule for every name) | Each entity is assigned the GICS sector whose equal-weight return series (built from survivor names with current GICS, leaving the entity itself out) correlates most with the entity's first 126 observed daily returns. Survivors and non-survivors get the same rule. Agreement with current GICS on survivors is reported |
| Beta | Trailing 126 sessions vs the equal-weight return of observed PIT members (screening formula, clip ±3) |
| Eligibility | PIT member on date t AND 64 consecutive observed sessions (FB rule) |

## Engine correction (found while preparing this test)

`strategy.engine.run` keeps the weight of any name that is not *eligible* at a rebalance. On the survivor panel
this rarely bound; under PIT eligibility every index deletion becomes a permanent, unhedged stub, and a delisted
name earns 0 forever. The test therefore uses a new `strategy.pit_engine.run_pit` (engine.py is frozen by the
paper-trading manifest and is not edited):

- entry requires eligibility at t−1, observed at t and a finite execution price;
- exit requires only observed at t and a finite price: a held name that left the index is sold at the next rebalance;
- a held name whose bars end (delisting) is liquidated at its last close plus a terminal return (below), and the
  proceeds go to cash.

With all names always eligible and observed, `run_pit` reproduces `engine.run` exactly (test).

**Terminal (delisting) returns.** Base: 0 beyond the last close (right for cash mergers). Sensitivity: −100% for
*failure delistings* (last close below 50% of the close 60 sessions earlier), and −30% for every non-merger-looking
delisting under the same rule (Shumway 1997 style).

## Arms (fixed; no configuration search)

All arms: features = cs_z of beta, log 20-day dollar volume (raw), 60-day return, 60-day volatility, plus sector
one-hot; target = FB residual target (5-day return residualised per date on sector + the four characteristics);
all cross-sections (cs_z, residual regression, training rows) use PIT eligibility at their own signal date.

| Arm | Description |
|---|---|
| **P (primary)** | Historical retrain simulation on the true PIT universe. LightGBM, FB `LGBM_PARAMS`, 50 rounds, seeds 7/19/42, predictions rank-averaged. Retrained at the first signal of each year 2018–2023 on every signal whose 5-day label matured by that signal's close (data from 2016-01-04) |
| R | Same as P with Ridge (alpha = 1.0, fixed), the matched linear model on the same inputs |
| RR21 | 21-day residual reversal (`strategy.signals.rr21`), zero parameters, same book |
| S | Paper-spec universe: P's procedure restricted to the 462 survivor tickers (2026 constituents) ∩ PIT members; training also restricted. P − S isolates survivorship inside one data source |
| O | The survivor-trained saved predictions (FBAUDIT matched `style_residual`, seeds 7/19/42; 2019–2023 only) on PIT eligibility with `run_pit` |
| N | Negative control: random scores (seed 0) |

**Book (all arms).** `ls_weights`: sector-neutral long top 10% / short bottom 10% per sector (sectors with ≥ 10
eligible names), 30% hold band, 0.5 gross per leg, equal-weight beta hedge. Signal every 5 trading days starting
with the first trading day of 2018; trade at the next open. Costs 10 bp per unit traded; borrow 50 bp/yr.
Evaluation window: 2018-01-02 to 2023-12-29 (O: 2019–2023).

## Metrics and tests

- Primary: arm P annualised net Sharpe at 10 bp + borrow, stationary bootstrap 95% CI (block 21, B = 2000).
- FF5 + MOM + ST_Rev daily attribution, Newey–West 10 lags.
- Paired: daily net return differences P − R and P − S, Sharpe of the difference with bootstrap CI.
- Reported, non-gating: 5/20/30 bp, next-close execution, per-year, long vs short leg, turnover, max drawdown,
  terminal-return sensitivities.

## Decision rule (arm P)

| Outcome | Rule | Action |
|---|---|---|
| **STOP** | net Sharpe < 0.30 at 10 bp | Stop the style book as a capital candidate. Paper trade continues only as an implementation exercise, if at all |
| **CONTINUE** | net Sharpe ≥ 0.60 AND FF alpha t ≥ 1.5 AND mean daily P − R ≥ 0 | Not validation. Justifies asking the user to authorise Norgate Platinum ($630/yr) for a once-only 1990–2014 holdout under a frozen spec |
| **INCONCLUSIVE** | otherwise | DEFER: no data purchase justified by this candidate alone; paper trade continues |

## Sanity checks (predeclared; each can change a decision)

| ID | Check | Consequence |
|---|---|---|
| E1 | PIT equal-weight benchmark FF alpha (survivor panel: +3.1%, t 2.6) | If t ≥ 2 on PIT, the universe fix is suspect: investigate before reading any strategy result |
| E2 | Alpaca vs yfinance daily returns on survivor names, 2016–2023 | Median correlation < 0.99 → data problem; stop |
| E3 | Entity merges; every one-day move > 40% listed and inspected | Unexplained splices → fix before scoring |
| E4 | Label-maturity guard (code assertion + test) | Hard failure |
| E5 | Random-score control (arm N) | Sharpe far from the cost drag → engine problem |
| E6 | `engine.run` vs `run_pit` on identical scores | Measures how much the earlier PIT number (+0.61) depended on stuck weights |

## Amendments

**A1 (2026-10-23, after building the panel, before any strategy or benchmark number was computed).** Check E3
found that Alpaca's `adjustment=all` series are wrong around several spin-offs: Alpaca records them as reverse
splits (TGNA 2017-06-01 as 1:3, WRK 2016-05-16 as 1:6, UTX 2020-04-03 as 1:2, DLPH/APTV 2017-12-05 as 1:3). That
produced fake one-day returns of −70% to −89%; AIV's adjusted close was 0.83 against a raw 32. Changes:
- **Returns** are rebuilt from `adjustment=raw` bars (`pit_sp500.clean_returns`, tested). Alpaca's adjustment is
  used on a day only if the adjusted return is within ±15% and no larger in size than the raw return. That covers
  dividends and consistent splits. Otherwise the day gets a zero return.
- **Unrecorded share-ratio gaps** (raw overnight gap within 1% of 1/2, 1/3, 1/4, 1/5, 1/10, 2, 3, 4, 5 or 10, and
  above 30%) are also zeroed. One hand-listed event: UA 2016-04-08, the Class C 1:1 share dividend (gap 0.527).
- **Result:** 12 member-days zeroed, listed in `PIT_DATA.md`. Every remaining one-day move above 40% is a dated news
  event (deal, bankruptcy, bank failure, COVID/oil shock).
- **Prices for the engine** are a total-return index built from these returns. Opens are scaled by the same factor,
  and a zeroed day has no gap.
- **Arm S survivors** are the 430 entities matched by ticker to the yfinance survivor panel, with more than 90% of
  days agreeing within 0.5%. The other 32 survivors were not 2016–2023 members under that ticker.
- **Cost if wrong:** a true spin-off day return is set to 0 instead of its small market move, on 12 member-days in
  eight years.
