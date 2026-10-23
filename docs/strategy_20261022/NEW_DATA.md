# C. New-data opportunity (WP-B)

## The finding that sets the priority

Every positive historical result in HyperShift was obtained on a **survivor universe**: 462 current S&P 500 constituents traced back in time. In WP-A:
- the equal-weight survivor benchmark itself shows +3.1%/yr FF alpha (t 2.6);
- restricting the best candidate to point-in-time index members halved its edge;
- its short leg loses money in exactly the names whose failed peers are missing.

**Until the universe problem is fixed, no historical result here can support a capital decision.** Survivorship-free data also unlocks power. With delisted names and point-in-time membership back to 1990, the 1990–2014 history (about 25 years) has never been touched by any strategy here. The minimum detectable IR over 25 years is about 2.8/√25 ≈ 0.56, against about 1.0 with the 7.9 inspected years.

## Candidate sources

Prices and terms were checked on 2026-10-22; vendor terms change, so re-verify before buying.

| Source | Unique information | Point-in-time | Delistings | Licence | Cost | Verdict |
|---|---|---|---|---|---|---|
| **Norgate Data Platinum** (US stocks, from 1990) | Survivorship-free daily prices, **historical index constituents** (S&P 500 membership by date), corporate actions | Membership by date; prices as traded plus adjustment factors | **Included** | Personal use; no redistribution | **$630/yr** (Diamond, from 1950: $787.50/yr); free trial available | **Highest information per dollar.** Fixes the binding problem for every price-based candidate |
| **SEC EDGAR XBRL** (companyfacts, frames, financial-statement data sets) | Reported fundamentals: income statement, balance sheet, EPS; filing timestamps | Usable as first reported via the `filed` date and accession number. The frames API keeps only the latest value, so **do not use frames for backtests** | Companies that filed while listed are included | Public | **Free**; about 1–2 weeks of engineering (concept mapping, as-first-reported logic, unit checks) | **Best value fundamentals.** History is short: XBRL from 2009–2011 |
| **Sharadar** (SF1 fundamentals + SEP prices + S&P 500 table, via Nasdaq Data Link) | Point-in-time fundamentals (`datekey`), prices including delisted, index membership | Vendor-provided | Included (per QuantRocket documentation) | Commercial | **Unverified** (a 2018 source cited from $50/month for individuals) | INVESTIGATE: get a current quote. A single bundle could replace Norgate + EDGAR |
| CRSP / Compustat / I/B/E/S (WRDS) | Gold standard, including analyst revisions | Yes (Compustat Snapshot) | Yes | Institutional | Typically an institutional subscription | DEFER (not realistic for an individual) |
| Free fundamentals APIs (yfinance, FMP and similar) | Current, restated fundamentals | **No** | Partial | Varies | Free or cheap | STOP for backtests: reconstructed fundamentals were not available historically |

## Recommended acquisition strategy

1. **Norgate free trial now.** Check coverage, delisting returns and the membership history against the current panel. If it is adequate, subscribe to Platinum ($630/yr).
2. **Re-run WP-A unchanged on point-in-time Norgate data** for 2016–2023: PRICE3 family, style book, EW and cap-weighted benchmarks. This is the decisive cheap test: it tells us whether the style book's edge survives once failed names are present.
3. **Start EDGAR ingestion only if a fundamental hypothesis is still worth testing after step 2.** Get a Sharadar quote in parallel; if it is under about $1,000/yr, prefer it, because it is less engineering and has longer point-in-time history.

## The one concrete hypothesis this enables

**H-B (earnings-surprise drift, sector-neutral, monthly).** Stocks with high standardised unexpected earnings earn higher factor-adjusted returns over the following month than low-SUE stocks in the same sector. The SUE formula and its availability timing are:

  SUE = (EPS_q − EPS_{q−4}) / std of the last 8 such differences

- **Earnings data:** quarterly diluted EPS as first reported in EDGAR 10-Q/10-K XBRL, taken from the filing date.
- **Trading:** next open after the filing's acceptance date. This is conservative: the earnings press release usually comes earlier, so part of the drift is given up in exchange for no look-ahead.

**Why this hypothesis.** It uses information the price-only program never had (reported earnings), it has a documented economic rationale (post-earnings-announcement drift; the literature to cite must be verified before use), and it is *not* spanned by FF5 + MOM + ST_Rev. That makes incremental value measurable.

**Test design:**
- **Contrast.** Paired net returns of:
  - (a) the style book (or PRICE3) alone;
  - (b) the same book with SUE added as an equal-weight z-score component.
- **Setting.** Point-in-time universe (Norgate), monthly rebalance with buffer, open execution, 10 bp costs plus borrow.
- **Holdout.** 2011–2014 and 2004–2012-equivalent periods are not available for EDGAR (XBRL starts around 2009–2011). The holdout is therefore 2024–2026 (unused by these strategies), plus prospective paper trading.
- **Minimum detectable improvement** (EDGAR-only history 2011–2023, about 13 years): incremental IR ≈ 2.8/√13 ≈ 0.78. Only a large improvement is detectable. With Sharadar's longer point-in-time fundamentals (about 1998–2023, 26 years), the MDE is about 0.55.
- **Decision value.**
  - **SUPPORTED** (incremental net IR ≥ 0.5 with paired CI excluding 0, and FF alpha t ≥ 2): add SUE to the paper-trading spec as a new dated version.
  - **Not supported:** stop fundamental-signal work on large caps and keep the price-only paper trade as the only live experiment.
- **Cost:** about 2–3 weeks of engineering, $0 (EDGAR) or the Sharadar fee, and under 1 hour of CPU.

Sources: [Norgate pricing](https://norgatedata.com/stockmarketpackages.php), [Norgate subscribe](https://norgatedata.com/subscribe/subscribe.php), [SEC EDGAR APIs](https://www.sec.gov/page/sec-api-documentation), [QuantRocket Sharadar](https://www.quantrocket.com/pricing/data/sharadar), [Ken French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
