# Selection and turnover search

2026–10–29: 72 accounts, fixed protocol `5a58312`, code `daf205d`. All six equal-five baseline daily paths exactly reproduce the original saved paths before the new variants were simulated. This search uses the same already-inspected 2022–2025 history and was designed after earlier experiments; values are selected historical simulations, not confirmation. Five and ten holdings here mean securities, not distinct issuers.

| Seed group | Five-stock vol20 rule | Sharpe | CAGR | Max drawdown | Annual turnover |
|---|---|---:|---:|---:|---:|
| Original five | Equal | 1.199 | 28.96% | -17.78% | 21.41 |
| Original five | Retention bonus | 1.383 | 33.71% | -18.99% | 17.13 |
| New five | Equal | 0.678 | 15.80% | -18.34% | 27.73 |
| New five | Retention bonus | 0.597 | 13.98% | -17.19% | 19.22 |
| Ten seeds | Equal | 1.080 | 26.00% | -21.69% | 22.19 |
| Ten seeds | Retention bonus | 1.158 | 28.22% | -17.81% | 15.95 |

Retention adds 0.02 to the score of stocks in the previous decision's target before choosing the top five. It is a target-based approximation, not a guarantee that actual positions persist. This is the highest Sharpe among the 72 accounts (1.383 original five); the new-five result weakens, so the increase is not consistent across these overlapping seed groups. Ten-seed three-stock confidence weighting produced 1.161 Sharpe / 29.45% CAGR / -20.88% drawdown, the highest ten-seed value in this search. These are different baskets with different exposures and transaction paths; lower turnover does not by itself establish the reason for the improvement. No claim of statistical superiority or future performance follows. Full trial metrics and calendar returns are in the CSVs; all 72 paths, target schedules and ledgers are under `results/HISTORICAL_SHARPE_SELECTION_20261029/`.

Independent verifier `research/analysis/rishi_historical_selection_verify.py` reconstructed 7,236 target rows, all 72 account-path Sharpes, CAGRs, drawdowns and NAVs, and 288 calendar returns, maximum numerical error 2.22e-14. Two focused tests checked ties, eligibility, target retention and temporal dependence. Verification does not independently reconstruct every fill, the provider's data vintage, or actual trading costs.
