# Historical NASDAQ data and study limitations

The user selected the 100 largest NASDAQ-listed companies, including financial
companies. This is a custom company universe, not the non-financial NASDAQ-100
index. Formation uses the final trading close before each month starts. A
company that first becomes large in December is not included in January on
that basis. Training and validation universes must also be historical.

That custom ranking remains deferred. The user explicitly authorized the
historical official NASDAQ-100 as a fallback when the largest-company ranking
could not be established. That fallback is now active, with monthly snapshots
and every official share class (101–104 securities). It excludes financial
companies. The [charts page](CHARTS.md) records the training/publication status;
the [active protocol](../../config/nasdaq100_historical_monthly.json) is frozen.

## Free-source findings used in the fallback

Alpha Vantage returned historical active and delisted listings. Its quarterly
shares response lacked publication dates; a reporting-period date does not
establish when a share count became known. Full daily and adjusted-daily price
requests returned premium-access messages. This was an entitlement limitation,
not a claim that the daily call allowance had been exhausted. These responses
cannot establish a complete monthly largest-company market-cap ranking.

Tiingo recovered native former-member histories that Yahoo no longer returned,
including XLNX, SGEN, SPLK and ANSS. The [free Starter plan](https://www.tiingo.com/about/pricing)
publishes 50 requests/hour, 1,000/day and 500 unique symbols/month. Its
[symbology guidance](https://www.tiingo.com/documentation/appendix/symbology)
limits retained delisted histories when symbols have been reused. We checked
identities before prices: CA resolved to an unrelated ETF and was rejected;
unidentified DISCA metadata was also rejected. Native WBD was used only during
its verified WBD-era membership. Credentials are read from local environment
variables and are excluded from saved requests, logs and tracked files.

Membership comes from a pinned MIT-licensed [historical event collection](https://github.com/jmccarrell/n100tickers),
with ten year-to-year closure checks and annual Nasdaq announcement cross-checks.
This is a research reconstruction, not an exchange-certified archive of every
as-published announcement. Sources disagree on the exact 2025 SOLS off-cycle
dates. Both yield the same November monthly snapshot; the company's IPO and
40-session warmup prevent its use by the forecasting models during this study.
The monthly basket follows the declared monthly snapshot rather than every
intra-month index change.

Prices combine Yahoo for continuing securities and Tiingo for native former
members. Retrospective split/dividend adjustment is used for total-return marks;
raw close times volume supplies liquidity. Valid issuer renames are mapped
explicitly. Post-acquisition vendor quotes were removed rather than treated as
continued exchange trading. Four test-period acquisitions use primary terms:

| Security | Completion | Economic consideration |
|---|---|---|
| XLNX | 2022-02-14 | [1.7234 AMD shares](https://ir.amd.com/financial-information/sec-filings/content/0000002488-22-000031/amd-20220214.htm) |
| SGEN | 2023-12-14 | [$229 cash](https://www.pfizer.com/news/press-release/press-release-detail/pfizer-completes-acquisition-seagen) |
| SPLK | 2024-03-18 | [$157 cash](https://newsroom.cisco.com/c/r/newsroom/en/us/a/y2024/m03/cisco-completes-acquisition-of-splunk.html) |
| ANSS | 2025-07-17 | [$199.91 cash plus 0.3399 SNPS shares](https://www.sec.gov/Archives/edgar/data/883241/000114036125026139/ef20051970_8k.htm) |

Stock consideration is sold at the completion close with costs; cash becomes
available at that close. Actual brokerage settlement may be later. This is an
explicit settlement approximation, and no settlement-delay stress was run.
No unexplained active selected-stock price gaps remain. Four unique supervised
targets spanning older, unpriced mergers are unobserved and excluded from loss;
their signal-time members remain present. Fold counts are published with the charts.

Before replacement test metrics, the training start was amended to July 2019
because older CA and the pre-reorganization FOX histories could not be safely
recovered. The first test fold has 1.5 training years; subsequent folds have
2.5, 3.5 and four. Earlier prices provide warmup. The 2022–2025 evaluation dates
are unchanged. These previously inspected years remain exploratory.

The new engine applies monthly membership to preprocessing, training, graphs
and forecasts; tests exclude inactive nodes even when layers have biases. It
uses training-only correlations, purged outcomes, validation-only epoch
selection, next-close frozen orders and a continuous cash/share account across
year changes. Arithmetic audits cannot certify vendor revisions or every
historical announcement. The custom ranking contract below remains available
if a more complete dataset becomes accessible.

## Sources checked

| Source | Useful coverage | Unresolved requirement |
|---|---|---|
| [SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Free filing histories and XBRL facts; no API account required | No consolidated daily trading/delisting-return feed or ready monthly NASDAQ market-cap rankings; use actual filing availability, not reporting-period date |
| [Alpha Vantage](https://www.alphavantage.co/documentation/#listing-status) | Historical active/delisted listing queries for dates after 2010 | Need to test exchange history, issuer identities, caps and delisted prices separately; a listing record is not a delisting payout |
| [SimFin](https://www.simfin.com/en/prices/) | Free account advertises five years of fundamentals/charts and API/bulk access | No verified complete historical exchange/delisting coverage; five years does not establish the earlier training universe |
| [FMP](https://site.financialmodelingprep.com/documentation) | Historical cap and delisted-company endpoints documented | Free-tier entitlement, earlier history, complete issuer coverage and share-count publication timing unverified |
| [Massive/Polygon](https://massive.com/docs/rest/stocks/tickers/ticker-overview) | Historical ticker details and exchange/identity fields | Free history is limited; documentation explicitly backdates some SEC information to report period, so historical cap cannot automatically be treated as known-at-the-time |
| [Alpaca](https://docs.alpaca.markets/us/docs/historical-stock-data-1) | Historical prices and separate corporate-action endpoints | Does not establish the requested full historical company-cap ranking from a price feed alone |
| [CRSP](https://www.crsp.org/crsp_pdf/crsp-us-stock-indexes-databases-data-descriptions-guide-crspaccess/) | Security/name/exchange histories, shares and delisting information | Access normally requires a subscription/institution; publication/revision semantics still need review |
| [Sharadar](https://sharadar.com/docs/daily) | Daily cap data and a separate active/delisted price feed | Subscription/access and historical exchange/as-reported timing need review; no verified complete free solution |

Alpha Vantage's [free signup](https://www.alphavantage.co/support/#api-key) is a
reasonable first capability check, not a promise that the whole study can be
finished for free. Its documented standard free limit is 25 requests/day.
`probe_public_nasdaq_sources.py` makes at most four requests by default, saves
the original CSVs, handles access/rate messages and reads the key only from
`ALPHAVANTAGE_API_KEY`. Keep keys in the local environment rather than chat or
tracked files. A successful probe supplies listings only.

No provider is selected to improve the reported return. The requirement is
better historical coverage and timing. A clean study may perform worse.

## Input contract for the ranking module

Supply provider-normalized inputs under `data/nasdaq_monthly_pit/input/`.
Preserve raw downloads, query dates and provider documentation separately. Do
not manufacture availability dates or mark an unreviewed source as complete.

`candidate_snapshots.csv`: one row for every NASDAQ common share/common ADR
eligible at every formation date, including subsequently delisted issuers:

| Column | Meaning |
|---|---|
| formation_date | Actual final exchange session before the target month |
| company_id | Stable issuer identifier; financial companies included |
| security_id | Stable share-class identifier; ticker reuse cannot join different issuers |
| ticker | Symbol valid at formation; never the sole identity key |
| exchange | Historical primary exchange, `XNAS` |
| security_type | `CS` or `ADRC`; no ETF/preferred/warrant/unit rows |
| company_market_cap_usd | Reconciled issuer cap, identical across that company's share-class rows; not today's cap or a sum of repeated issuer caps |
| adv20_usd | Dollar volume observed during the prior 20 sessions; used to pick one share class |
| cap_available_date | Latest publication date of any fact used to calculate cap; a later-filed share count cannot be assigned its earlier report date |
| listing_available_date | Date when the listing/exchange information used was available |
| identity_available_date | Date when the identifier mapping used was available |
| source | Raw source reference for reconciliation |

Dates in this initial contract are day-level. Facts without intraday timestamps
must conservatively become usable on a later session when same-day publication
timing is unknown. Split changes and share classes/ADR ratios must be reconciled
before ranking. The module cannot independently certify these provider facts.

`coverage.csv`: one reviewed inventory per formation date, with
`formation_date,expected_security_count,historical_listing_complete,
missing_company_caps,publication_timing_reviewed,issuer_identity_reviewed`.
The three review flags must be literal booleans. Missing candidate rows or
missing company caps block ranking rather than silently excluding companies.
`calendar.csv` has a `date` column containing actual NASDAQ sessions, sorted
and unique. Synthetic weekday calendars are only for unit tests.

Required formations span December 2016 through November 2025, supporting
monthly universes January 2017 through December 2025. Daily history starts in
2016 for features and momentum warmup. If the available source cannot reach
that far, changing the training window requires a declared protocol amendment
before any replacement test metrics are inspected.

Run the membership gate with:

```powershell
.\.venv\Scripts\python.exe historical_nasdaq_universe.py
```

It writes a new immutable membership file only when the supplied ranking inputs
pass. Prices/corporate actions still need a separate gate: complete daily
raw close/volume, total returns, correctly linked symbol changes and terminal
cash/share consideration for acquisitions/delistings. A disappearing price
cannot be forward-filled forever at its last value. Missing selected securities
must not be replaced with later winners.

## Engine requirements for any future custom ranking

The code uses fixed security slots with dynamic monthly masks. Any custom run needs
monthly membership in every training/scaling/graph/validation/trading step.
Inactive and not-yet-known securities must not influence active forecasts,
including through biased temporal layers or hypergraph edge normalization.
Ticker names and the union of future constituents cannot establish training
eligibility. Known month-end membership may affect a next-close order only
after its underlying facts are available.

Corporate-action settlement and a continuous 2022–2025 account must preserve
cash and holdings across yearly model changes. SPY/QQQ must have exactly the
same holding dates, cash/dividend conventions and cost treatment. Gate and
ledger tests come before retraining. The fallback meets the research-grade
checks above; a future custom-cap dataset still requires its own coverage and
publication-timing review.
