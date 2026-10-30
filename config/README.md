# Dataset contracts and experiment protocols

This directory stores versioned inputs and settings, not trained models or data
files. Frozen JSON protocols retain their original bytes. Run-specific source,
input, checkpoint and publication locks live under ignored `runs/` directories.

## Historical monthly NASDAQ-100 research

| File | Scope |
| --- | --- |
| [ndx_tabular_alpha_20261030.json](ndx_tabular_alpha_20261030.json) | Fixed64fit/282account causal shared-feature forest/Ridge experiment; all seeds,controls and costs retained |
| [ndx_portfolio_robustness_20261030_v2.json](ndx_portfolio_robustness_20261030_v2.json) | Fixed 60-fit / 540-account target-averaging, trading-band and shared-scorer study; first attempt preserved, with exact family-first rank aggregation restored |
| [ndx_factor_controls_20261029.json](ndx_factor_controls_20261029.json) | Separate 198-account causal style-factor and fixed neural-blend arm, frozen before new financial results |
| [ndx_rank_retraining_20261029.json](ndx_rank_retraining_20261029.json) | 120 fixed fits, neutral stock-ID initialization, ranking objective, split validation, 15 predefined seeds and 576 accounts |
| [ndx_rank_stability_20261029.json](ndx_rank_stability_20261029.json) | Fixed validation-ranking and allocation grid: 24 recipes / 216 accounts, frozen performance and consistency goals |
| [ndx_retention_replication_20261029.json](ndx_retention_replication_20261029.json) | Pinned tom-shlom retention/QQQ-hedge replication: exact 96-account grid and comparison tolerances |
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
| [ndx_loss_optimizer_20261024.json](../latest/config/ndx_loss_optimizer_20261024.json) | Robust losses and optimizer recipes |
| [ndx_reference_assets_20261024.json](../latest/config/ndx_reference_assets_20261024.json) | Declared reference implementations and factor inputs |
| [ndx_reference_networks_20261024.json](../latest/config/ndx_reference_networks_20261024.json) | MASTER, restricted dynamic hypergraph, PRISM-VQ and fixed rank blends |
| [nasdaq_topk_pseudohuber_20261011.json](nasdaq_topk_pseudohuber_20261011.json) | Earlier top-k/loss follow-up |
| [nasdaq_long_short_20261011.json](nasdaq_long_short_20261011.json) | Signed allocations, cash and borrow-cost controls |

[ndx_confirmation_gate_20261024.json](../latest/config/ndx_confirmation_gate_20261024.json) is
**not activated**: it defines requirements for a future frozen candidate and
untouched evaluation. It does not record a completed or live confirmation result.
[source_layout_20261024.json](source_layout_20261024.json) records the byte-preserving
source relocation; it is distinct from scientific input and code locks.

## Archived protocols and layout history

Unrelated fixed-cohort, random-trial, presentation and retired-universe protocols
are in [archive/studies_20261025/config](../archive/studies_20261025/config/).
They are restored to their old paths only with `python -m hypershift --archived`.
Shared NASDAQ protocols remain here; the latest study protocols live in
[latest/config](../latest/config/README.md).

[study_layout_20261025.json](study_layout_20261025.json) records the latest-study
and archive moves, hashes and runtime aliases. The older `source_layout_20261024.json`
is a historical relocation receipt; its paths precede the current grouping.

Find the matching [training runner](../research/training/README.md) and
[report](../docs/README.md) before using a protocol. Changing a completed protocol
invalidates its original source/input locks; new experiments need separate files.
