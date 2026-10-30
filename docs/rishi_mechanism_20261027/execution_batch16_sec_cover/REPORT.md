# SEC filing-cover identity feasibility

The fixed protocol ../SEC_COVER_IDENTITY_PROTOCOL.md (71572ac) inspected public SEC filing evidence for the 2024 and 2025 first-signal eligible names. Each candidate CIK came from a previously checked, retrospective S&P-study ticker map; each match additionally required a recent original filing before the signal, the exact ticker in a tagged cover-page trading symbol, and a consistent filing header with one SIC. Candidate discovery does not establish historical identity by itself. The SEC notes that submissions API metadata are current; it must not be used as a historical ticker mapping without dated evidence ([SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)).

| First-signal fold | Eligible | Filing-cover matches | No candidate CIK | Multiple candidate CIKs | Ambiguous/missing header SIC |
|---|---:|---:|---:|---:|---:|
| 2024 | 101 | 81 | 18 | 1 | 1 |
| 2025 | 101 | 79 | 20 | 1 | 1 |

All 202 fixed-name receipts exist. An independent pass checked all 160 matched records against the saved original response bytes: header filing date, CIK, accession and SIC; exact tagged ticker; response hashes; and the 400-day maximum age. A separate check confirmed the selected accession was the latest eligible original filing in the cached history. No discrepancies were found. There were no network-error receipts. The issuer candidates that fail or are missing remain unresolved; the full receipts and coverage are available alongside this report, and raw filing/header bytes remain in ignored `results/RISHI_SEC_COVER_20261028/`.

For example, Apple's [2024 annual report](https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm) names AAPL on its cover, and its [filing header](https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/0000320193-24-000123-index-headers.html) identifies CIK 320193, filing date 2024-11-01 and SIC 3571. This demonstrates a dated route to the 2025 snapshot; it is not an assertion that a single filing proves an entire universe.

The finding materially improves feasibility, but a filing confirms its cover on its filing date, not the absence of subsequent ticker or corporate changes. It does not certify that the NDX economic price series belongs to that issuer, that the SIC stayed current, or that index membership was published before the decision. It is not GICS classification. No historical preference, outcome or account performance was examined in this batch. A separately frozen preference analysis must specify how to handle these gaps and the unmatched names. Strict checkpoint reproduction and financial-fitting gates remain unchanged.
