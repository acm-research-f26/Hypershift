# D. Can a point-in-time S&P 400 universe be rebuilt from free data? (2026-10-24)

Method (`scripts/sp400_membership.py`):
- read 97 monthly revisions, 2016-01..2024-01, each the last revision before the 1st of the month;
- compare month-to-month membership with the dated change log of the 2024-01 revision;
- cross-check with an **independent** source: every S&P 500 addition in fja05680 whose company was in the S&P 400
  list the month before must leave the S&P 400 at that date.

No prices or returns were used.

## Results against the gates

| Gate (from `ALTERNATIVES.md`) | Result | Status |
|---|---|---|
| Snapshot within ±5% of 400 names in ≥ 95% of months | 100% of months (396–402 rows) | Passes, but **uninformative**: editors keep the table at 400 rows whether or not it is current |
| Changes reconciled against the source's change log | Only 531 of 981 month-to-month snapshot changes (54%) match a logged change within a month. Of 757 logged additions/removals, 389 appear the same month, 142 one month late, 27 two months late, and **199 (26%) never within 2 months** | **FAIL** |
| **Independent timing check** (S&P 500 promotions, fja05680 dates) | Of 93 promotions, the S&P 400 list dropped the company the same month in 39 cases and one month later in 16. In **38 cases (41%) it lagged 2–26 months** | **FAIL** |
| **Change-log completeness** (same 93 promotions) | The log has the matching S&P 400 removal for 81/93 (87%). Misses are concentrated in 2016–2019 (CNC, HOLX, COO, MAA, ROL, KEYS, JKHY, LW, ATO, WAB, …) | **FAIL** for point-in-time use |
| ≥ 97% of member-days with usable prices | Not built. Pilot: 20/20 removed names have Alpaca bars | Not evaluated (moot) |
| ≥ 90% of member-months mapped to a defensible historical CIK | Pilot: 11/20 verified; no CIK in the source | **UNRESOLVED / at risk** |

## Why this is disqualifying

- **Snapshots lag the true index.** The lag is systematic: companies promoted to the S&P 500, i.e. recent winners,
  stay on the S&P 400 list for months after leaving. A snapshot universe would hold the wrong companies exactly when
  their returns are most unusual.
- **The change log alone cannot replace the snapshots.** It misses 13% of independently verifiable removals, and
  more in 2016–2019. A backward rebuild from the 2023 list would carry those errors into the earliest, most
  consequential years.
- **Identity cannot be anchored to a CIK.** The source has none, and ticker reuse produced 2 wrong-company matches
  in 20.

**Conclusion: accurate monthly point-in-time S&P 400 membership cannot be reconstructed from free Wikipedia data to
the standard of the S&P 500 panel.** Fabricating the missing history is not an option, so the free-data approach
stops here.

## Cheapest trustworthy alternative

| Source | What it provides | Cost | Status |
|---|---|---|---|
| **Norgate Data Platinum** | Historical S&P MidCap 400 (and 500/600) constituents with dates, survivorship-free prices including delisted names, from 1990 (vendor page, checked 2026-10-22) | **$630/yr** (Diamond $787.50) | Needs your authorisation. The 3-week trial carries 2 years of history, enough to validate membership for 2024–2026 but not 2016–2023 |
| S&P DJI index-change press releases (archives, e.g. PR Newswire) | Authoritative effective dates for every change | Free, but scraping 8 years of announcements and the ticker/CIK identity work remain | Possible, not reliable without a known-complete archive |
| CRSP/Compustat (index constituent files) | Gold standard | Institutional | Not realistic |

Even with trustworthy membership, the economic case is weak (`ECONOMIC_FEASIBILITY.md`), so this report does not
recommend buying data **for this signal**.

## Data-quality items not examined, because the membership gate failed

These were deliberately left undone; they are moot without a valid universe:
- mid-cap splits, spin-offs and special dividends;
- raw vs adjusted prices;
- volume adjustment and halts;
- dated SIC;
- full CIK mapping.

The S&P 500 pipeline handles all of them and would be reused if a valid membership source existed.
