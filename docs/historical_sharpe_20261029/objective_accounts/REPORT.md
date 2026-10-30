# Matched financial training-objective accounts

October 29. Training protocol `f934d0f`, training code `3a63017`; evaluation protocol `27a152b`, evaluation code `e357cf3`. All 48 new fits completed and hashes were locked before test scoring. 84 source-engine modeled accounts use four fixed seeds, three newly trained parent objectives, parent-only or 50:50 blend with matched published dynamic seeds, and issuer-five/vol20 or equal-ten/full. Checkpoints selected by validation MSE for every training objective. The small four-seed ensemble is not directly comparable to Rishi's ten-seed headline. History was extensively inspected before this experiment; no future claim.

| Four-seed 50:50 blend, five-issuer vol20 | Sharpe | CAGR | Max drawdown | Annual turnover |
|---|---:|---:|---:|---:|
| MSE | 0.790 | 18.20% | -21.36% | 21.98 |
| Pairwise ranking | 0.812 | 18.98% | -23.44% | 18.76 |
| Portfolio surrogate | 0.515 | 12.21% | -23.86% | 18.23 |

The ranking objective's point improvement here is small and has a deeper drawdown; the portfolio surrogate lowers turnover but performs worse on this selected four-seed account. These are modeled costs and post-search point results, not evidence of general superiority. Across all 84 accounts the largest Sharpe was 0.964 (single seed 19, pairwise blend, issuer-five) and is a selected ceiling, not a robust ensemble. All seed/group, allocation and year results are in METRICS.csv and CALENDAR.csv. Portfolio training uses label-valid masks and resets turnover at each eight-origin minibatch; those are training-specific approximations, not observed broker execution or an ex-ante label mask. Common validation MSE may favor MSE-calibrated scores and is not a ranking-optimized selection rule. Checkpoint reproduction and vendor-vintage limitations remain.

The independent verifier rebuilt 16,884 target rows and verified 84 NAV/Sharpe/CAGR/drawdown and 336 annual returns, max error 1.18e-14. Its first pass exposed float32-vs-float64 percentile arithmetic creating artificial tie breaks in the verifier; casting ranks to float64 matched the source rank precision, then the independent reconstruction passed. It does not independently reconstruct every share fill or certify data availability. Two rank/tie/mask tests passed.
