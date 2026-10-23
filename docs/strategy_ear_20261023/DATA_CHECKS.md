# B. Targeted PIT integrity checks for the EAR test (2026-10-23)

Earlier checks are reused (`../strategy_lazyprices_20261023/INFRA_CHECKS.md`, `PIPELINE.md`, `PRECHECKS.md`). Only
the items below are new or specific to EAR.

| Item | Finding | Action |
|---|---|---|
| **`IR` identity splice** (overnight review I5) | Entity `IR` carries Gardner Denver prices from 2017-05-12 (its IPO) to 2020-03-02, while its membership and CIK are Ingersoll-Rand plc (now TT) | **Excluded:** `IR` is ineligible before 2020-03-03 (642 stock-days; `ear_common.SPLICE`). Fixed before any EAR outcome |
| **Other CIK-change entities** | All 8 entities whose CIK changes inside membership were checked on raw prices: **GOOGL** (Google→Alphabet), **XRX**, **APA** (holding reorganisations) and **JCI** (old Johnson Controls, raw $39.08 on 2016-01-04) are the same economic series; **BKR** and **FTI** have no bars before the merged company exists (ineligible then, no splice); **CB** carries ACE prices under the old Chubb CIK in January 2016 only, before the evaluation window | No other exclusion needed |
| Event ↔ security identity | Item 2.02 events join entities through the dated, SEC-verified CIK segments (filed between 120 days before the segment start and the segment end), as in the filing-text test | — |
| **Sector timing** | The earlier return-correlation proxy used up to 62 later returns, and current SIC is not dated | **EAR uses FF12 from the SIC in the EDGAR header of each CIK's earliest Item 2.02 filing in the window, valid from that filing date; before it, industry 12.** 653 CIKs, none missing; FF12 differs from current SIC for 9 of 648 (`scripts/ear_sic.py`) |
| Corporate actions | Returns are rebuilt from raw bars with checked adjustments; 12 member-days neutralised; the largest remaining member-day move is 62% (SIVB/FRC, genuine) | Reused |
| Exits and delistings | `pit_engine.run_targets`: names leaving the index are sold at the next rebalance and delisted names are liquidated at the last close (`test_run_pit_exits_name_that_left_index`, `test_run_targets_equals_equal_weight_benchmark`) | Reused |
| Next-open execution | The open never uses the same day's close (`test_price_index_open_uses_no_same_day_close`). Signals use closes up to F−1 only | Reused |
| Eligibility vs prices | Eligible = member and 64 consecutive observed sessions at the close of F−1; 99.72% of member-days have a bar | Reused |
| Signal window | `ear.window_ar` returns NaN unless every bar of the window and the day before it is observed (`test_window_ar_requires_all_bars`). 56 of 17,024 events have no EAR | — |

**New targeted tests:** `tests/test_ear.py` (7):
- 8-K/A excluded and repeated filings collapsed;
- filing-date event day;
- disjoint, market-adjusted EAR and control windows;
- missing bars;
- causal event bins (a later extreme event cannot change earlier bins);
- industry-neutral tilt with the 2× cap.
