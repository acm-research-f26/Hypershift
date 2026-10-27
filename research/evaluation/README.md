# Audits, verification and publication seals

This directory verifies forecast timing, saved checkpoints, portfolio accounting
and published artifact hashes. An audit belongs to its own experiment and should
not be interpreted as an independent replication on fresh dates.

- `audit_historical_ndx*`, `audit_ndx_alpaca`, `audit_ndx_topk_pseudohuber` and
  `audit_ndx_long_short` verify the corresponding NASDAQ studies.
- `audit_ndx_strategy_comparison_20261020` verifies the original cross-study
  consolidation; the latest workbook audit is described below.
- Shared hardware and NASDAQ publication verifiers remain here; older pooled
  cohort auditors are in [archive](../../archive/README.md).
- [audit_ndx_drawdown_research.py](audit_ndx_drawdown_research.py) independently
  verifies all 915 drawdown-study metrics and replays 82 selected seed/cost
  accounts. Run with `python -m hypershift run audit_ndx_drawdown_research`.
- [seal_ndx_comparison.py](../../latest/evaluation/seal_ndx_comparison.py) verifies frozen sources,
  fit artifacts and publication hashes for the loss/optimizer and reference studies.

From the repository root, verify the existing seals:

```powershell
python -m hypershift run seal_ndx_comparison ndx_loss_optimizer_20261024
python -m hypershift run seal_ndx_comparison ndx_reference_networks_20261024
```

These commands check an existing seal; the same script creates one if absent.
The October 24 comparison has a read-only audit for its own publication snapshot:

```powershell
python -m hypershift python -c "from update_ndx_new_findings_20261024 import audit; audit(write_outputs=False)"
```

Its success label is `passed_staged_workbook`. It checks 346 main strategies,
eight sheets and 634 chart-catalog entries and predates the drawdown integration.
For the current ten-sheet / 364-strategy / 652-entry master, use the
[integration audit](../../docs/ndx_drawdown_20261026/master_update/AUDIT.json)
and [publication hashes](../../docs/ndx_drawdown_20261026/master_update/PUBLICATION.json).
The local staged integration can be rechecked with
`python -m hypershift run update_ndx_drawdown_master --stage audit`.
See the [test guide](../../tests/README.md),
[published evidence index](../../docs/audit/README.md), and
[layout verification receipt](../../docs/audit/source_layout_20261024.json).
