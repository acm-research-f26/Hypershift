# Shared research and portfolio machinery

These modules implement panel construction, causal masking, forecast aggregation,
portfolio policies and economic accounting. They are imported by experiment runners.

| Module | Responsibility |
| --- | --- |
| [ndx_portfolio_robustness.py](ndx_portfolio_robustness.py) | Averaged signed targets, causal actual-holdings bands and audited self-financing replay |
| [ndx_rank_prior.py](ndx_rank_prior.py) | Training-only return-rank priors, chronological exponential decay and neutral stock-history shrinkage |
| [ndx_tabular_alpha.py](ndx_tabular_alpha.py) | Causal multi-horizon styles, eligible cross-sectional feature ranks, train-only scaling and date-normalized labels |
| [ndx_beta_risk.py](ndx_beta_risk.py) | Preceding-close basket-beta hedge controls with gross/short-volatility constraints |
| [pit_research_engine.py](pit_research_engine.py) | Historical membership, chronological panels and long-only accounting |
| [ndx_policy_engine.py](ndx_policy_engine.py) | NASDAQ strategy and portfolio policy support |
| [ndx_retention.py](ndx_retention.py) | Causal prior-target retention bonus and continuous QQQ hedge sizing |
| [ndx_rank_stability.py](ndx_rank_stability.py) | Validation-only seed weighting, rank disagreement penalties and broader allocations |
| [ndx_rank_retraining.py](ndx_rank_retraining.py) | Neutral stock-ID initialization, centered ranking loss and purged split-validation checkpoint selection |
| [ndx_seed_attribution.py](ndx_seed_attribution.py) | Independently replayed additive asset-level profit, costs and net-return contributions |
| [ndx_causal_factors.py](ndx_causal_factors.py) | Trailing momentum/risk/liquidity controls and fixed neural blends, with no future-price dependence |
| [long_short_pit.py](long_short_pit.py) | Signed portfolios, cash and borrow-cost accounting |
| [research_engine.py](research_engine.py), [robust_engine.py](robust_engine.py) | Shared and earlier research-engine utilities |
| [training_device.py](training_device.py) | CPU/CUDA selection and reproducibility settings |

See [training](../training/README.md) for entry points and
[tests](../../tests/README.md) for timing, masking, cost and accounting checks.
The [hardware guide](../../docs/TRAINING_HARDWARE.md) describes backend constraints.
Run historical imports through the root launcher, as described in the
[layout guide](../../docs/PROJECT_LAYOUT.md).
