# C. S&P 400 pilot: three Wikipedia revisions and 20 deleted constituents (2026-10-24)

Scripts: `scripts/sp400_membership.py` (revisions; cache `data/sp400/wiki/`, git-ignored) and
`scripts/sp400_pilot_deleted.py` (output `data/sp400/pilot_deleted.json`). No returns were computed.

## A. Historical revisions of "List of S&P 400 companies"

Each entry is the last revision *before* the stated date (MediaWiki API, `rvdir=older`). It is the page as editors
had left it at that time, not today's page and not a retrospective list.

| Target date | Revision used | Constituent table | CIK field | Change table |
|---|---|---|---|---|
| 2016-07-01 | 718663041, **2016-05-04** (58 days old) | 400 rows: Ticker, Company, GICS sector/sub-industry, "SEC Filings" link | **No.** The SEC link is `browse-edgar?CIK=<ticker>`, a ticker, not a CIK | 9 rows only |
| 2020-01-01 | 932594637, 2019-12-27 | 400 rows (Security, Ticker, GICS, SEC link) | No | 214 rows |
| 2023-12-31 | 1191696148, 2023-12-25 | 401 rows (+ headquarters) | No | 431 rows (retrospective log, written over time) |

The column names and order change between revisions, but the tables parse reliably.

## B. Twenty deleted constituents

Removals from the dated change log, 2016–2023, spread across removal reasons. Identity was checked four ways:
- Alpaca `asof` bars (as-of the day before removal);
- the SEC CIK by EDGAR ticker lookup, else the current ticker table;
- a name match against the Wikipedia security name;
- a 10-Q/10-K filed within 200 days before removal, with an implied market cap (raw close × SEC-reported shares) of $0.3–40B.

| Removed | Ticker | Reason class | Alpaca bars | CIK found | Name match | Mid-cap check | Identity |
|---|---|---|---|---|---|---|---|
| 2016-01-19 | ATW | acquired | 2016-01..2017-10 | — (retired ticker) | — | — | UNRESOLVED |
| 2017-03-01 | CBOE | acquired | ✓ | 1374310 | ✓ | $6.3B | OK |
| 2018-05-10 | DBD | acquired | ✓ | 28823 | ✓ | $1.0B | OK |
| 2021-05-07 | PRSP | acquired | 2018-06..2021-05 | — | — | — | UNRESOLVED |
| 2023-09-29 | SYNH | acquired | ✓ to 2023-09-28 | — | — | — | UNRESOLVED |
| 2016-01-05 | TW | promoted to S&P 500 | **1 bar** | 1758730 = **Tradeweb (wrong company: reused ticker)** | ✗ | — | FAIL (caught) |
| 2018-04-04 | MSCI | promoted to S&P 500 | ✓ | 1408198 | ✓ | $13.1B | OK |
| 2023-10-12 | NATI | acquired by an S&P 500 company | ✓ | 935494 | ✓ | $8.0B | OK |
| 2017-03-29 | WETF | to S&P 600 | ✓ | 880631 | ✓ | $1.2B | OK |
| 2019-04-03 | PBI | to S&P 600 | ✓ | 78814 | ✓ | $1.3B | OK |
| 2020-04-27 | MDP | to S&P 600 | ✓ to 2021-12 | — | — | — | UNRESOLVED |
| 2016-04-04 | SUNE | market cap (bankrupt) | ✓ to 2016-04 | 22701 = **SUNation Energy (wrong company: reused ticker)** | ✗ | — | FAIL (caught) |
| 2020-09-21 | PBF | market cap | ✓ | 1534504 | ✓ | n/a | OK |
| 2022-10-03 | EQT | market cap | ✓ | 33213 | ✓ | $15.1B | OK |
| 2023-12-18 | VSTS | market cap | from 2023-10 | 1967649 | ✓ | new registrant | UNRESOLVED |
| 2016-06-24 | FBHS | other | ✓ | 1519751 | ✓ | $8.7B | OK |
| 2023-06-01 | BOH | other | ✓ | 46195 | ✓ | $1.6B | OK |
| 2017-03-20 | DNR | distress | ✓ to 2020-07 | — | — | — | UNRESOLVED |
| 2018-06-18 | ENDP | distress | ✓ to 2022-08 | — | — | — | UNRESOLVED |
| 2021-05-03 | GNW | distress | ✓ | 1276520 | ✓ | $2.2B | OK |

**11 of 20 are verified.** Two ticker lookups return a different, later company (TW→Tradeweb, SUNE→SUNation);
the name check catches both. Seven retired tickers have no SEC match by ticker and would need name search, the
route that produced wrong-company matches in the S&P 500 round (Churchill Downs, Endonovo, Paramount Gold).

## C. Probe result

| Requirement | Status | Evidence |
|---|---|---|
| Historical records exist | **PASS** | Revisions monthly since 2010-12 (12 of 96 months in 2016–2023 have no edit) |
| Page format parses reliably | **PASS** | 396–402 rows every month; column names vary but parse |
| Change logs sufficient to reconstruct membership | **FAIL** | The log lists S&P 400 removals for only **81 of 93** independently dated S&P 500 promotions (87%); the misses cluster in 2016–2019. 26% of logged changes never appear in snapshots within 2 months (`SP400_DATA_FEASIBILITY.md`) |
| Delisted-name prices retrievable | **PASS** (bars) | 20/20 have Alpaca bars |
| Securities mapped to the correct company | **UNRESOLVED** | 11/20 verified; 2 wrong-company ticker reuses; 7 need name search |
| CIK available or reconstructible | **FAIL / UNRESOLVED** | No CIK in any revision; ticker-based reconstruction is error-prone |
