# Training-return priors: fixed shrinkage experiment — October 30, 2026

The fixed **264-account** arm has **0** passing improvement recipes and **0** with SR>1.5 in all five predefined ensembles. No new neural fits; every training prior uses only its fold's observed training labels.

![Model comparison](model_comparison.png)

Fixed chart/table representatives: retention five, averaged cohort targets, no band, base fees and3%borrow. These are not a substitute for the full grid.

| Model | Original five SR | Second five SR | Third five SR | Ten SR | Fifteen SR |
|---|---:|---:|---:|---:|---:|
| 13-week training prior + neural | 1.195 | 1.115 | 1.147 | 1.182 | 1.188 |
| 26-week training prior + neural | 1.366 | 0.998 | 1.237 | 1.217 | 1.245 |
| 52-week training prior + neural | 1.040 | 0.967 | 1.261 | 1.027 | 1.136 |
| Full-history training prior + neural | 0.812 | 0.769 | 0.974 | 0.814 | 0.884 |

## Seed-free controls

| Control | Allocation | Band | Sharpe | CAGR | Max DD |
|---|---|---:|---:|---:|---:|
| prior13 | retain5 | 0.0% | 1.306 | 25.39% | -11.27% |
| prior13 | issuer5 | 0.0% | 1.251 | 24.19% | -11.35% |
| prior26 | retain5 | 0.0% | 1.087 | 21.12% | -12.14% |
| prior26 | issuer5 | 0.0% | 1.082 | 20.54% | -11.32% |
| prior52 | retain5 | 0.0% | 1.112 | 21.63% | -14.39% |
| prior52 | issuer5 | 0.0% | 1.103 | 21.57% | -12.55% |
| prior_full | retain5 | 0.0% | 1.123 | 20.70% | -11.02% |
| prior_full | issuer5 | 0.0% | 0.902 | 17.20% | -11.02% |

Deterministic controls are reported once, not copied into artificial seed ensembles. Neural blended groups overlap and share the same historical outcomes.

## Design and interpretation

The four priors use training-return percentile ranks with13/26/52decision-week exponential half-lives or full training history, plus20neutral0.5pseudo-observations per stock. Every annual prior ends before its prior-year validation window; no validation/test outcome estimates the stock prior. Unseen/unsupported stocks are neutral. All four priors are tried without handpicking names.

Neural blends use50%eligible percentile rank of the training prior plus25%meanTHINKseed ranks and25%mean dynamicseed ranks. The24prior-only controls retain the raw mean-rank score scale used in the older deterministic control. Retention is sensitive to this scale; issuer selection is order-based. All six full-history control cases reproduce the previous completed source metrics.

The arm was frozen while five unused neural seeds trained, before any unused-seed account was scored. Any source-selected candidate must be frozen before their financial results are examined. This deliberate stock-specific historical preference is not a model that generalizes to unseen stock identities.

All account paths match independent dollar-position accounting. Frozen source/input hashes, chronology, gross budgets, eligibility and terminal liquidation remain inherited and checked.

These are disclosed additional development searches after the earlier failed shared/family experiments. No unseen market evaluation or multiple-search-adjusted significance is claimed. Seed consistency alone cannot establish future Sharpe.

[Every ensemble account](METRICS.csv) · [Every recipe/cost summary](CONSISTENCY.csv) · [Seed-free controls](CONTROLS.csv) · [Frozen protocol](PROTOCOL.json) · [Selection](SELECTION.json)

The [current overview](../ndx_seed_robustness_20261030/REPORT.md) adds the separately frozen reserved-seed confirmation, full search history and final model comparison.
