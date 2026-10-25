# Dataset contracts and experiment protocols

This directory stores versioned inputs and settings, not trained models or data
files. Frozen JSON protocols retain their original bytes. Run-specific source,
input, checkpoint and publication locks live under ignored `runs/` directories.

## Historical monthly NASDAQ-100 research

| File | Scope |
| --- | --- |
| [nasdaq100_historical_monthly.json](nasdaq100_historical_monthly.json) | Active historical official monthly membership fallback |
| [nasdaq_monthly_pit.json](nasdaq_monthly_pit.json) | Deferred custom historical 100-largest-company universe |
| [nasdaq100_alpaca_20261019.json](nasdaq100_alpaca_20261019.json) | Matched daily/15-minute regular-session comparison |
| [alpaca_symbol_identity_20261019.json](alpaca_symbol_identity_20261019.json) | Reviewed Alpaca symbol/issuer identities |
| [ndx_ranking_20261020.json](ndx_ranking_20261020.json) | Ranking and benchmark-residual objectives |
| [ndx_broad_research_20261020.json](ndx_broad_research_20261020.json) | Broader features and model families |
| [ndx_corrections_20261020.json](ndx_corrections_20261020.json) | Frozen-core correction and validation rules |
| [ndx_seed_expansion_20261020.json](ndx_seed_expansion_20261020.json) | Fixed seed inventory and initialization sensitivity |
| [ndx_extended_hours_20261024.json](ndx_extended_hours_20261024.json) | Daily, after-hours and premarket information |
| [ndx_extended_hours_top5_20261024.json](ndx_extended_hours_top5_20261024.json) | Earlier unblended AH+PM holding-count test |
| [ndx_loss_optimizer_20261024.json](ndx_loss_optimizer_20261024.json) | Robust losses and optimizer recipes |
| [ndx_reference_assets_20261024.json](ndx_reference_assets_20261024.json) | Declared reference implementations and factor inputs |
| [ndx_reference_networks_20261024.json](ndx_reference_networks_20261024.json) | MASTER, restricted dynamic hypergraph, PRISM-VQ and fixed rank blends |
| [nasdaq_topk_pseudohuber_20261011.json](nasdaq_topk_pseudohuber_20261011.json) | Earlier top-k/loss follow-up |
| [nasdaq_long_short_20261011.json](nasdaq_long_short_20261011.json) | Signed allocations, cash and borrow-cost controls |

[ndx_confirmation_gate_20261024.json](ndx_confirmation_gate_20261024.json) is
**not activated**: it defines requirements for a future frozen candidate and
untouched evaluation. It does not record a completed or live confirmation result.
[source_layout_20261024.json](source_layout_20261024.json) records the byte-preserving
source relocation; it is distinct from scientific input and code locks.

## Earlier research inputs

`modern_source_snapshot.json` preserves the original twelve-name basket.
`NASDAQ_source_cohort.txt` and `NYSE_source_cohort.txt` preserve cohorts from the
pinned Temporal Relational Stock Ranking repository. Their historical selection
and survival biases remain; they are not monthly point-in-time index membership.
`recent_identity_exclusions.json` records issuer mismatches in the earlier modern
extension. Its unavailable names were not replaced.

`robust_experiment_plan.json`, `huber_ranking_experiment_plan.json`,
`qqq_control_amendment.json`, `robust_diagnostics_lock.json`,
`robust_presentation_lock.json` and `random_trials_20261010.json` belong to earlier
cohort research. `largecap_retraining_20261010.json` belongs to the retired
2026-selected study and is not an active benchmark.

Find the matching [training runner](../research/training/README.md) and
[report](../docs/README.md) before using a protocol. Changing a completed protocol
invalidates its original source/input locks; new experiments need separate files.
