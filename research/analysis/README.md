# Comparisons and research diagnostics

These sources analyze saved NASDAQ-100 forecasts and portfolio paths. They cover
strategy comparison, rank depth and loss similarity.

- [run_ndx_prior_robustness.py](run_ndx_prior_robustness.py): all264training-prior
  controls/blends, frozen before reserved-seed account outcomes. The separately
  frozen [confirmation](../evaluation/confirm_ndx_rank_prior.py) improves SR on
  reserved initializations; the original stricter source gate remains failed.
- [run_ndx_beta_robustness.py](run_ndx_beta_robustness.py): all168causal beta-risk
  and lower-volatility controls; all original-risk references reproduce, but
  the alternatives fail the all-ensemble1.5goal.

- [run_ndx_portfolio_robustness_v2.py](run_ndx_portfolio_robustness_v2.py): fixed
  target averaging, actual-holdings bands and controlled shared-THINK retraining.
  The first [attempt](run_ndx_portfolio_robustness.py) is preserved after its
  baseline rank-tie parity gate failed. The conditional
  [family follow-up](run_ndx_family_robustness.py) separates model families and
  selects blend weights using prior-year validation only.

- [run_ndx_factor_controls.py](run_ndx_factor_controls.py): separately frozen
  198-account causal factor/control arm. The frozen runner completes accounts
  before its documented pandas aggregation collision; finish saved attempts with
  [finalize_ndx_factor_controls.py](finalize_ndx_factor_controls.py). See the
  [repair note](../../docs/ndx_factor_controls_20261029/REPRODUCTION_NOTE.md).

- [analyze_ndx_seed_attribution.py](analyze_ndx_seed_attribution.py): descriptive
  stock-level attribution of the original/new ensemble gap, with independent
  accounting reproduction. See the
  [attribution report](../../docs/ndx_seed_attribution_20261029/REPORT.md).

- [run_ndx_rank_stability.py](run_ndx_rank_stability.py): fixed validation
  RankIC/stability weights, consensus penalties and broader allocations. Use
  `--stage freeze|run|report` with an explicit fresh `--name`; the completed run is
  `ndx_rank_stability_20261029_v2`. All 216 accounts are retained; neither
  performance-and-consistency goal passes. See the
  [findings](../../docs/ndx_rank_stability_20261029_v2/FINDINGS.md).

- [run_ndx_retention_replication.py](run_ndx_retention_replication.py): pinned
  `tom-shlom` retention and continuous-QQQ-hedge replication using existing
  `feat-rs` forecasts; stages `source`, `freeze`, `run`, `report`. All 96 source
  accounts and 12 existing baselines reproduce. See the
  [report](../../docs/ndx_retention_replication_20261029/REPORT.md).

- [run_ndx_model_hedge.py](run_ndx_model_hedge.py): eight own-model hedge signals,
  three seed groups and 495 cost/borrow accounts. The preserved runner's no-op
  inference is completed by [finalize_ndx_model_hedge.py](finalize_ndx_model_hedge.py).
  See the [signal study and audit](../../docs/ndx_model_hedge_20261027/REPORT.md).

- [run_ndx_weak_market_short.py](run_ndx_weak_market_short.py): frozen 39-rule
  comparison of weak-trend cash exits, QQQ short switches and QQQ hedges; 531
  seed/cost/borrow accounts, with a separate signed accounting engine and
  independent dollar-position replay. See the [shorting report](../../docs/ndx_weak_market_short_20261027/REPORT.md).

- [run_ndx_matched_controls.py](run_ndx_matched_controls.py): frozen 29-strategy
  comparison of neural selections, QQQ risk controls and issuer momentum; also
  freezes the prospective candidate before new control metrics. Results and
  activation prerequisites are in the [matched-control report](../../docs/ndx_matched_controls_20261027/REPORT.md).

- [compare_ndx_strategies_20261020.py](compare_ndx_strategies_20261020.py): base
  cross-study comparison and paired/block-bootstrap Sharpe analysis.
- [analyze_ndx_rank_depth.py](analyze_ndx_rank_depth.py): holding-count and
  ranks-six-to-ten diagnostics.
- [analyze_ndx_loss_similarity.py](analyze_ndx_loss_similarity.py): similarities
  between trained objective variants.
- Earlier cohort, exchange-geometry and pooled sensitivity sources are preserved
  in [archive](../../archive/README.md).

Use `python -m hypershift list analysis` from the repository root to locate sources.
The latest consolidation extends the original comparison in
[update_ndx_new_findings_20261024.py](../../latest/reporting/update_ndx_new_findings_20261024.py).
The [master charts](../../charts.md), [workbook](../../docs/NASDAQ100_strategy_comparison.xlsx)
and [reporting guide](../reporting/README.md) identify the current complete results.

Historical leader tables are retrospective. Absolute Sharpe intervals and paired
differences answer different questions; neither a selected historical CAGR nor
a high point-estimate Sharpe establishes reliable benchmark superiority.
