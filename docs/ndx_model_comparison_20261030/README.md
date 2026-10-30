# Daily model-path comparison — October 30, 2026

These figures replot locked daily returns; they do not train, retune, or simulate
new accounts. All paths share 1,003 sessions, January 3, 2022–December 31, 2025.
Net wealth compounds the saved daily returns from $100; drawdowns include the
initial $100 NAV. Sharpe uses a 3% annual risk-free hurdle and 252 sessions.
Modeled trading fees and 3% calendar-day short borrowing are already deducted.

![Combined ten](ten_seed_paths.png)

![Separate cohorts](disjoint_seed_paths.png)

![Combined fifteen](fifteen_seed_paths.png)

## What is being compared

- The old retention strategy is the historically selected recipe, not a fresh
  control selected without these outcomes. Its original/second/ten SR is
  1.731 / 0.648 / 1.381. Old models were not fitted for the third/fifteen groups;
  no missing old paths are extrapolated or replaced with original-five results.
- Four trained representatives expose both successful and failed choices:
  loss-selected five issuers, loss-selected retention five, rank-selected
  retention five, and three-snapshot five issuers. The issuer loss row was the
  observed best minimum across all five ensembles; retention loss was the best
  minimum across the three disjoint groups. These choices are descriptive.
- Stable-consensus / soft ten was the earlier postprocessing study's best
  weaker-half candidate. It uses a 50% hedge ceiling; other displayed accounts
  use 75%. Both ceilings are subject to remaining gross budget and trailing
  volatility limits. That comparator changes portfolio and hedge rules as well
  as model aggregation; it is not an isolated model-only comparison.
- The factor hybrid is the strongest weaker-group hybrid in its fixed study:
  50:50 ranking-checkpoint forecasts and liquid-consensus factor, retention five.
  It is outcome-selected and did not meet the performance target. The separate
  study blended the ranking checkpoint, not the subsequently stronger loss one.
- The training-mean-rank control uses only each fold's training-period stock
  return ranks, with shrinkage. It is deterministic, seed-free, and repeated
  visually for context rather than counted as independent seed evidence.

All neural ensembles blend THINK and dynamic-hypergraph rank forecasts equally.
New models use 12 epochs; the old schedule used up to 30 with early stopping.
Initialization, objective, validation selection, and duration changed together.
Five issuers consolidates share classes; retention five selects securities and
adds 0.02 to previous target selections. Both use the 20% stock-basket volatility
target and the existing weekly execution/cost model. Portfolio rules differ,
so aggregate curves do not isolate architecture or initialization effects.

## Interpretation and limitations

Loss-selected five issuers reduces the three disjoint cohort SR spread to
approximately 0.053, with SR 1.171 / 1.147 / 1.200; ten/fifteen SR is 1.023 / 1.235.
No trained recipe cleared SR>1.5 across every predefined ensemble. Pooled ranks
are not averages of portfolio returns: top-k selection, retention, and risk sizing
are nonlinear. More seeds can therefore change or worsen the account path.

Neutral initialization, ranking-aligned training, validation separation, and
all-cohort reporting improve the design and observed seed stability. They do
not establish unseen-market generalization. All 2022–2025 outcomes have been
repeatedly inspected, and all cohorts share those market realizations. The ten
and fifteen groups also overlap constituent models. No new untouched evaluation,
multiple-search correction, or causal single-change ablation is supplied here.

The plots are a representative overview, not the full experiment grid. All
recipes, costs, failed cases, and controls remain in the original reports:
[retraining](../ndx_rank_retraining_20261029/REPORT.md),
[postprocessing](../ndx_rank_stability_20261029_v2/REPORT.md),
[retention](../ndx_retention_replication_20261029/REPORT.md), and
[factor controls](../ndx_factor_controls_20261029/REPORT.md).

## Verification and reproduction

Every source PATHS.npz and METRICS.csv is checked against its completed-run SHA256
manifest. Date arrays must match exactly. Every performance field for the 32
distinct compared accounts is recomputed from daily returns and checked against
its saved CSV (relative tolerance 1e-10, absolute tolerance 1e-12). No experiment
source, checkpoint, ledger, or completion manifest is modified. The publication
manifest records source hashes, chart hashes, and the plotting script hash.

From the workspace root:

```powershell
.\.venv\Scripts\python.exe research/reporting/plot_ndx_model_comparison_20261030.py
```

[Compared account metrics](COMPARISON_METRICS.csv) · [Provenance](MANIFEST.json)
