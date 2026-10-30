# Supplemental SEC issuer evidence

The preregistered ../SEC_MISSING_IDENTITY_PROTOCOL.md (297b7bc) examines only the 42 first-signal 2024/2025 names unresolved in the earlier SEC cover audit. The current-vintage SEC ticker list was used solely to discover a candidate CIK; each successful record additionally required a dated original filing before the signal, exact tagged cover ticker and a consistent SEC header SIC. Original matched records were not replaced.

| Fold | Earlier matches | Additional matches | Combined eligible names with filing evidence | Still unresolved |
|---|---:|---:|---:|---:|
| 2024 | 81 | 17 | 98 / 101 | 3 |
| 2025 | 79 | 21 | 100 / 101 | 1 |

In 2024, one remaining ticker was not tagged in the selected cover, one header had ambiguous SIC, and one ticker was absent from the current candidate list. In 2025, the remaining header had ambiguous SIC. The supplement retains all failures in RECEIPTS.csv. An independent audit verified all 38 added matches against the downloaded cover/header bytes, CIK, accession, filing date, exact ticker, SIC, 400-day age limit, submissions-response hashes and latest eligible filing in the recorded submissions data. Four collector tests passed. AUDIT.json records coverage; raw public responses are preserved in ignored `results/RISHI_SEC_COVER_SUPPLEMENT_20261028/`.

Higher coverage materially reduces missing-classification exposure but does not make this a fully point-in-time entity map. The candidate list is from 2026; a ticker match on a prior filing does not prove continuous identity through the signal date or linkage to the NDX price series. SIC may change after filing, and as-published membership remains uncertified. These are SIC/FF12 categories, not GICS sectors. No preference or financial-return comparison was performed using the added records. A separate frozen protocol is required before combining them with Hlabel preferences; earlier partial-coverage results remain intact.
