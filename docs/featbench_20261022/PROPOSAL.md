# Study FB spec: the user's proposal (pasted 2026-10-22, saved verbatim in substance)

This is the spec Study FB implements (`docs/superpowers/plans/2026-10-22-feature-benchmark.md`). It was pasted into the session by the user after the retrospective (claude.ai doc "HyperShift Research Retrospective"). Literature links and promotional footer omitted.

## Ranking (the user's estimates of research value)

| Rank | Experiment | Effort | Potential value | Tested already? |
|---|---|---|---|---|
| 1 | LightGBM + volume, volatility, and liquidity features | Low | Very high | No |
| 2 | Predict residual returns instead of raw returns | Low | High | Partially |
| 3 | Test different prediction horizons | Very low | High | Not systematically |
| 4 | Cross-sectional feature engineering and interactions | Low | High | Not systematically |
| 5 | Sector-neutral and risk-normalized portfolio construction | Low | High | Partially |
| 6 | Test signal concentration in predictable stock subsets | Low | Medium–high | No clear evidence |
| 7 | Fix learned-hyperedge collapse and test external relationships | Medium | Medium | Attempted but inconclusive |

All historical periods have been inspected: these are exploratory experiments, followed by a prospective test of whatever survives.

## 1. LightGBM + richer market features
- About 15–25 predictors: returns (1, 5, 10, 20, 60-day), reversal (prior 1, 3, 5-day), volatility (5, 20, 60-day realised), volume (relative volume, volume change), liquidity (dollar volume, Amihud), momentum (20–60 day excluding recent days), market exposure (rolling beta), relative strength (return minus sector/index), risk (downside volatility, maximum drawdown).
- Models: Ridge, LightGBM with shallow trees, a simple MLP on identical features. Compare against the existing reversal baseline. Key comparison ΔIC = IC(LightGBM) − IC(Ridge). Also sector/factor-neutral RankIC and cost-adjusted portfolio performance.

## 2. Factor-residual targets
- Targets: A raw forward return; B sector-relative forward return; C factor-residual (market, sector and selected style exposures estimated from historical information, prespecified model). Train Ridge and LightGBM on each; evaluate all on a common factor-neutral ranking metric and a common portfolio procedure.

## 3. Horizons 1, 5, 10, 20 days
- Fixed model and features; purged training/validation boundaries; account for overlapping forward returns. Compare RankIC, factor-neutral RankIC and turnover-adjusted performance. Fewer independent observations at 20 days.

## 4. Cross-sectional feature engineering
- Compare raw, cross-sectional rank, sector-relative, volatility-scaled and cross-sectional standardised representations, with Ridge and LightGBM.

## 5. Portfolio construction on frozen predictions
- A equal-weight top-10; B sector-neutral top/bottom; C volatility-scaled; D beta-neutral long/short; E lower-turnover no-trade buffer (enter top 10, keep until rank 20; not optimised).

## 6. Predictable subsets
- RankIC by size, liquidity, volatility, trading activity, groups assigned with characteristics known before the prediction date; no threshold optimisation; findings are hypotheses only.

## 7. Learned hyperedges
- Stage 1: synthetic planted groups; temperature, non-uniform prototype initialisation, balancing/entropy objective; confirm recovery. Stage 2: no graph vs static vs random vs learned, matched capacity and protocol. Lower priority than features.

## Combined first experiment
- Existing S&P 500 panel plus volume; features price/momentum/volatility/liquidity; Ridge and LightGBM; targets raw and sector-relative; horizons 1, 5, 20; baseline 16-day reversal; primary metric factor-neutral RankIC; secondary raw RankIC, turnover, net Sharpe; walk-forward with purging; mostly CPU.

## Order of work
1. Power analysis and data audit (detectable effect sizes from saved predictions with date-block resampling; volume, adjustment and constituent checks).
2. Simple-model feature benchmark on historical data, strictly exploratory.
3. Prospective validation: freeze one specification and evaluate on observations after the 2026-10-22 freeze, primary comparison predeclared.

## Baseline decomposition (run first)
- Factor-neutral predictive contribution of r(t−1), r(t−2:t−5), r(t−6:t−20), r(t−21:t−60), individually and combined with Ridge (regularisation fixed on training/validation).
