# Factor-control completion and repair

All 198 declared accounts were simulated and independently checked before the
original runner stopped while aggregating its summary: `DataFrame.style`
resolved to pandas' Styler property rather than the column named `style`.

`finalize_ndx_factor_controls.py` completes the summaries using explicit column
indexing. The original runner remains byte-preserved to match its frozen
protocol. The finalizer recovers return paths from saved NAV ledgers,
reconstructs the declared targets and verifies all 198 paths with independent
value-position accounting. The original metrics and ledgers are hash-preserved.
The repair was recorded after these outcomes, and changes no financial result.

The completed study reports **zero passing hybrid recipes**. Neither this
repair nor the separate scheduling amendment changed any strategy parameters.

From the repository root, the original frozen entry point is
`python -m hypershift run run_ndx_factor_controls --stage evaluate`; it stops on
the documented aggregation error after saving the accounts. Complete that
attempt with `python -m hypershift run finalize_ndx_factor_controls`, then use
`python -m hypershift run run_ndx_factor_controls --stage report`. Completed
destinations refuse overwrite; new experiments require separately declared
destinations/protocols. Source forecasts must be present from the locked
120-fit study.

[Audit](AUDIT.json) · [Aggregation amendment and raw hashes](AGGREGATION_AMENDMENT.json) · [All results](REPORT.md)
