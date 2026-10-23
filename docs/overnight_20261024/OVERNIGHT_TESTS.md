# SEC data-integrity tests (`tests/test_sec_integrity.py`)

Synthetic fixtures only: no network, no data files, no strategy returns.
- Run: `PYTHONPATH=src .venv/Scripts/python.exe -m pytest -q tests/test_sec_integrity.py`.
- **Result at a14fd5d + this file: 28 passed, 6 strict xfail.**
- Together with `test_edgar.py`, `test_lazyprices.py` and `test_pit.py`: **78 passed, 6 xfailed** (6.7 s).
- **With `patches/C1_xml_declaration_and_refetch.patch` applied** (checked in a scratch copy of a14fd5d, with
  `PYTHONPATH` pointing at it):
  - the 3 C1 xfails turn to XPASS (strict), which proves the fix;
  - all of `test_edgar.py` still passes.
  - After applying the patch, remove the C1 `xfail` marker.

## What already existed (not duplicated)

| File | Already covered |
|---|---|
| `test_edgar.py` | Timestamp inversion on 3 live filings; next-trading-day availability (morning, Friday evening, Saturday, beyond calendar); hidden XBRL, script and numeric-table removal; cosine; `10-K/A` and `10-KT` exclusion with fiscal-year-change no-match; FF12; name matching (11 pairs, strict, accents) |
| `test_lazyprices.py` | Formation days; signal availability and 3-month expiry; within-form bottom quintile; avoidance weights; `run_targets` equals the equal-weight benchmark |
| `test_pit.py` | Exit of names that left the index; delisting liquidation with a terminal return; reused-ticker interval split; clean-returns rules; open price never uses the same day's close |

## New tests and outcomes

| Requirement (task list) | Test(s) | Outcome |
|---|---|---|
| Parses real-world document headers | `test_extract_tokens_without_encoding_declaration` (3 cases); `test_extract_tokens_with_encoding_declaration` (3 cases) | pass; **strict xfail: C1, a CRITICAL verified defect** |
| Not usable before acceptance; an after-close filing cannot trade at the same day's close | `test_filing_on_last_session_before_formation_is_not_traded_at_formation`; `test_exchange_holiday_with_edgar_open`; `test_acceptance_inversion_around_dst_and_year_end` (5 cases) | pass |
| Amendment does not replace the original | `test_amendment_never_becomes_the_current_or_prior_document` | pass |
| Prior cannot be a future filing | `test_prior_must_be_accepted_before_current_never_a_future_filing` | pass |
| Missing comparison gives a defined result | `test_missing_or_degenerate_comparison_gives_no_signal_without_error` (no report date; prior without date; new registrant; 52/53-week year) | pass |
| Latest-filing rule (PREREG Comparison/Validity) | `test_latest_filing_without_comparison_gives_no_signal` | **strict xfail: I3, a spec deviation** |
| Duplicate originals / accessions | `test_duplicate_original_for_same_period_uses_latest_as_prior`; `test_one_accession_shared_by_two_share_classes_gives_one_row_per_entity` | pass (dual-class double rows are documented behaviour, M2) |
| CIK mapping does not splice companies | `test_cik_segments_do_not_splice_predecessor_or_successor_filings`; `test_name_match_rejects_lookalikes` (4 cases); `test_name_match_rejects_reporting_subsidiary` | pass; **strict xfail: M5** (strict matching accepts "DUKE ENERGY CAROLINAS" for "Duke Energy") |
| Errored documents never become signals | `test_errored_rows_never_become_signals` | pass |
| Deterministic signal | `test_text_signal_is_deterministic_and_symmetric`; `test_changer_ties_are_broken_deterministically_by_entity_order` | pass |
| Formatting-only changes do not dominate | `test_markup_only_changes_do_not_change_tokens` (inline tags, hidden XBRL facts, tagged numbers, whitespace and `&#160;` inside a paragraph, appended numeric table); `test_numeric_table_threshold_is_digits_over_letters_plus_digits`; `test_block_whitespace_does_not_change_tokens` | pass; **strict xfail: I1** (whitespace between block elements changes tokens; measured on real 10-Ks in the risk review) |
| Corporate events give valid portfolio returns | `test_delisting_goes_to_cash_and_cash_is_redeployed_at_next_rebalance`; `test_halted_name_keeps_weight_and_targets_scale_to_remaining_nav`; `test_avoidance_weight_is_not_capped_at_1_25x` | pass (the last documents M3) |

## Defects the tests reveal

- **C1 (CRITICAL, verified on real filings).** Documents whose XML declaration names an encoding raise `ValueError`
  in `extract_tokens` (`edgar.py:86-90`). Both Microsoft 10-Ks checked fail: DFIN, FY2022 and FY2023. Fix in
  `patches/`.
- **I3 (IMPORTANT).** `signals_at` falls back to an older filing when the latest has no comparison; PREREG says
  no signal.
- **I1 (IMPORTANT, design).** Block-element whitespace sensitivity. Not a code bug under the frozen wording.
- **M5 (MINOR).** Subsidiary look-alikes pass strict name matching; only the 5 fallback CIKs are exposed.

## Writing the fixtures exposed one more instance of I1

The first version of the numeric-table test failed because `<p>text here</p><table>...` merged "here" with the
table's first word. This is the I1 mechanism again, found by accident. The fixture now separates elements with
newlines.
