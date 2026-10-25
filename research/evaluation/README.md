# Audits, verification and publication seals

This directory verifies forecast timing, saved checkpoints, portfolio accounting
and published artifact hashes. An audit belongs to its own experiment and should
not be interpreted as an independent replication on fresh dates.

- `audit_historical_ndx*`, `audit_ndx_alpaca`, `audit_ndx_topk_pseudohuber` and
  `audit_ndx_long_short` verify the corresponding NASDAQ studies.
- `audit_ndx_strategy_comparison_20261020` verifies the original cross-study
  consolidation; the latest workbook audit is described below.
- `audit_saved_models`, `verify_*` and the remaining `audit_*` sources preserve
  checkpoint, hardware, publication and earlier cohort checks.
- [seal_ndx_comparison.py](seal_ndx_comparison.py) verifies frozen sources,
  fit artifacts and publication hashes for the loss/optimizer and reference studies.

From the repository root, verify the existing seals:

```powershell
python -m hypershift run seal_ndx_comparison ndx_loss_optimizer_20261024
python -m hypershift run seal_ndx_comparison ndx_reference_networks_20261024
```

These commands check an existing seal; the same script creates one if absent.
The latest comparison has a read-only audit of the current workbook and charts:

```powershell
python -m hypershift python -c "from update_ndx_new_findings_20261024 import audit; audit(write_outputs=False)"
```

Its success label is `passed_staged_workbook` because the helper shares a staging
validation path; with no workbook argument it checks the actual `docs/` workbook.
It checks 346 main strategies, eight sheets and 634 chart-catalog entries.
See the [test guide](../../tests/README.md),
[published evidence index](../../docs/audit/README.md), and
[layout verification receipt](../../docs/audit/source_layout_20261024.json).
