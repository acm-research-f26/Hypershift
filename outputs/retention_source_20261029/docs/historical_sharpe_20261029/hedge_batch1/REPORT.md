# Historical QQQ hedge search

2026-10-29. Protocol `f121483`, code `98bcd6a`. The 96 modeled accounts combine issuer-five or retention-five long targets, three overlapping seed groups, four short caps, 3%/6% annual borrow and 1x/2x costs. The hedge is continuously on, with short weight bounded by the cap, available gross budget, and 20% divided by trailing QQQ volatility. Actual mean short exposure was about 31% in the selected retention accounts; a 75% cap does not mean a 75% short. This post-selection search uses the same extensively inspected market history.

| Retention-five, 3% borrow, 1x cost | QQQ short cap | Sharpe | CAGR | Max drawdown |
|---|---:|---:|---:|---:|
| Original five | 0% | 1.383 | 33.71% | -18.99% |
| Original five | 75% | 1.731 | 33.11% | -13.36% |
| New five | 0% | 0.597 | 13.98% | -17.19% |
| New five | 75% | 0.648 | 12.31% | -17.11% |
| Ten seed | 0% | 1.158 | 28.22% | -17.81% |
| Ten seed | 75% | 1.381 | 26.35% | -13.79% |

The best observed result was 1.731, 33.11% CAGR, -13.36% drawdown, for the original-five retention book with 75% cap and base assumptions. At 6% borrow and 2x cost it was 1.570 / 29.85% / -15.15%. It is an always-on hedge, not a model timing prediction, and is not a confirmed or prospective performance estimate. Short availability, margin, recall, actual borrow and execution impact are not established; short proceeds and idle cash earn zero, and gross exposure can drift above its target. The source hedge's original daily paths are absent, so published parity is on metrics, not direct day-by-day source files.

Six published issuer baseline/continuous-hedge aggregate metrics matched within 8.33e-17. A separate value-position implementation compared with all 96 signed-share daily paths within 8.89e-16. A separate arithmetic check verified 96 NAV paths/Sharpe/CAGR/drawdown, gross targets and 384 calendar returns within 1.55e-14. Two target-construction unit tests passed. All metrics including turnover and costs, annual returns, targets and ledgers are retained in this study and `results/HISTORICAL_SHARPE_HEDGE_20261029/`. Verification does not independently certify historical input vintage or real broker execution.
