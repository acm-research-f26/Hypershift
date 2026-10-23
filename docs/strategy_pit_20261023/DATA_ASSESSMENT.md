# C. Survivorship-free data: assessment and validation (2026-10-23)

**Bottom line.**
- **2016–2023: done.** A free point-in-time (PIT) S&P 500 panel *with failed and acquired names* was built and
  validated: Alpaca SIP bars plus fja05680 membership. The usable window is 2016-01-04 to 2023-12-29, enough for
  kill tests, not for confirmation.
- **Earlier history needs paid data and your authorisation.** No historical-investability claim before 2016 can be
  made without it. Nothing was bought and no vendor account was created.

## Requirements and how each source meets them

| Requirement | Alpaca (validated here) | Norgate Platinum | Sharadar (Nasdaq Data Link / QuantRocket) | CRSP / Compustat (WRDS) | yfinance (current panel) |
|---|---|---|---|---|---|
| PIT index membership | No (taken from fja05680, MIT, built from Wikipedia change lists) | **Yes**, historical constituents | **Yes**, S&P 500 additions/removals since 1957 | Yes (via Compustat index files) | No |
| Delisted / acquired names | **Yes**: bars to the last trading day (XLNX, TWTR, ATVI, CTXS, SIVB, FRC, …) | Yes | Yes ("includes active and delisted tickers") | Yes | **No** (survivors only) |
| Terminal / delisting returns | **No**: bars stop at the last listed trade; post-delisting OTC value is absent | Not verified | Not verified | **Yes** (DLRET) | No |
| Identities through renames / ticker reuse | Partly: `asof` maps a ticker to the company holding it then (FB→META, ANTM→ELV, HCP→PEAK, Q→IQV). It splices histories around some mergers (Dow/DuPont chain), and multi-symbol requests can drop symbols | Not verified | Not verified (permaticker) | **Yes** (PERMNO) | No |
| Corporate actions | **Unreliable**: several spin-offs are recorded as reverse splits (TGNA, WRK, UTX, DLPH), which corrupts the `split`/`all` series. Fixed by rebuilding returns from RAW bars (PREREG A1) | Not verified | Not verified | Yes | Splits and dividends only; no spin-off adjustment |
| As-traded vs adjusted | Both (`raw`, `split`, `dividend`, `spin-off`, `all`) | Both | Both | Both | Adjusted plus a separate unadjusted download |
| Classifications | None (sector proxy used) | Not verified | Yes (SIC / sector fields) | Yes (SIC/NAICS; GICS via Compustat) | Current GICS only |
| History | **2016-01-04 onward** | 1990 onward (Diamond: 1950) | Prices 1998; fundamentals 1990 | 1926 onward | Any, survivors |
| Licence | Exchange market-data terms (redistribution terms not verified; data kept in git-ignored `data/pit/`) | Personal use | Professional / non-professional licences | Institutional | Yahoo terms |
| Cost | **$0** with the existing account | **$630/yr** (Diamond $787.50) | Behind a login: **quote needs your account** | Institutional subscription | $0 |
| Trial | n/a | 3 weeks, **2 years of history only**: can check the mechanics, not 1990–2014 | Not found | No | n/a |

Vendor pages read on 2026-10-23:
- [Norgate packages](https://norgatedata.com/stockmarketpackages.php) and [free trial](https://norgatedata.com/freetrial.php)
- [QuantRocket Sharadar data](https://www.quantrocket.com/docs/data/fundamental/sharadar/) and [pricing](https://www.quantrocket.com/pricing/data/sharadar) (prices hidden behind a login)
- [Alpaca bars API](https://docs.alpaca.markets/reference/stockbars) (`asof`, `adjustment` values)

## Validation of the Alpaca PIT panel (details in `PIT_DATA.md`)

| Check | Result |
|---|---|
| Membership intervals 2016–2023 with bars | **695 / 695**, after re-fetching three dropped symbols singly and aliasing CCEP→CCE |
| Member-days with a bar | **99.72%** |
| Members per day | 502–507 (the index holds about 503 securities) |
| Entities | 678, of which 11 are merged across renames (e.g. GEN/NLOK/SYMC, CBS/VIAC/PARA, RTX/UTX) |
| Agreement with yfinance on survivors (E2) | 430 of 462 survivors matched. Median share of days within 0.5%: 1.000; median daily-return correlation 1.000 (5th percentile 0.992) |
| Corporate-action errors (E3) | 12 member-days zeroed (spin-offs misrecorded as reverse splits, AIV/XRX/DD/IR adjustment errors, UA share dividend). Remaining moves above 40% are all dated news events |
| Delistings while a member | 76 entities, mostly cash or stock mergers; two failures (SIVB 2023-03-09, FRC 2023-04-28) |
| Survivorship check (E1) | Equal-weight PIT universe FF alpha **−0.7%/yr (t −0.5)**, 2018–2023. On the survivor panel 2019–2023 it was +3.1% (t 2.6). The survivor subset inside the same PIT data shows +1.5% (t 0.95) |

## Remaining limitations of the free panel

1. **No terminal returns.** Positions are liquidated at the last listed close. A −100% / −30% sensitivity is run on
   the failure delistings. Only two failures occurred among members (SIVB, FRC), so the effect is bounded.
2. **Sector proxy.** No PIT classification is available, so one return-correlation rule assigns all names to GICS
   sectors. It agrees with current GICS on 73% of survivors. Sector neutrality is therefore approximate.
3. **Membership from Wikipedia-derived lists (fja05680).** It is not the official S&P file; daily counts are consistent.
4. **Eight years only.** Six evaluation years after warm-up give a minimum detectable Sharpe of about 1.1. The panel
   can kill a candidate; it cannot confirm one.

## What is blocked, and what it would take

- **Any claim about 1990–2015**, or any test powered to detect a realistic large-cap edge (Sharpe 0.3–0.6 needs
  roughly 22–87 years), needs survivorship-free data before 2016.
- **Cheapest adequate option:** Norgate Platinum, $630/yr. It covers 1990 onward with delistings and historical
  constituents. It does not provide fundamentals, so EDGAR remains the source for those.
- **Alternative:** a Sharadar quote. It covers prices from 1998 plus fundamentals, insider and 8-K events. That
  requires your login.
- **Decision for you:** these purchases are user-gated. `STRATEGY_PIT_REPORT.md` (G/H) says whether the evidence
  justifies one.
