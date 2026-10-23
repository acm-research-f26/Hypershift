# Overnight handoff (2026-10-24), draft: being written

**Read first: C1 (CRITICAL) in `OVERNIGHT_RISK_REVIEW.md`.**
- `edgar.extract_tokens` raises `ValueError` on inline-XBRL documents whose XML declaration names an encoding.
  Verified on two Microsoft 10-Ks filed by DFIN (0001564590-22-026876 and 0000950170-23-035122).
- `lazy_signals.py` stores those documents as errors, so their pairs get no signal.
- Fix: `patches/C1_xml_declaration_and_refetch.patch`.
- A plain rerun re-fetches nothing, because errored token files exist. Apply the patch (or delete the errored token
  files), rerun `lazy_signals.py`, then rerun `lazy_checks.py`.
- Count the affected rows first with the one-liner in the review.

The full handoff follows when the remaining tasks are done.
