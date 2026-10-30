# Seed robustness: first account results

October 29. Protocol 1cc46c6, replay code 53d8639. All 78 accounts completed (ten individual matched parent/dynamic seed pairs plus three ensembles, six fixed recipes). Nine prior hedge reference paths match to 1e-10; every path matches separate value-position accounting to 1e-12. An independent arithmetic calculation verified 78 Sharpes, CAGRs, drawdowns, NAVs and 312 annual returns to maximum error 1.38e-14. Targets and deeper attribution remain to be independently examined.

| Leading recipe | Minimum individual-seed Sharpe | Median | Maximum |
|---|---:|---:|---:|
| Retention + QQQ cap75 (1.731 original-five ensemble) | -0.169 | 0.460 | 0.838 |
| Retention + QQQ cap50 (1.724 original-five ensemble) | -0.160 | 0.442 | 0.819 |
| Issuer-five + QQQ cap75 (1.466 original-five ensemble) | -0.320 | 0.449 | 0.954 |

The ensemble combines rankings before choosing stocks; it is not an average of individual accounts. These results suggest ensemble selection is essential to investigate rather than treating the leading Sharpe as typical of a single seed. This is not yet proof of the economic signal or of the causal value of averaging. All seeds use the same repeatedly inspected history. Same-seed retention/hedge controls, calendar returns, costs, turnover and exposures are retained in CSVs and detailed ledgers under results/HISTORICAL_SEED_INVESTIGATION_20261029. Next: independent target reconstruction, rank/portfolio agreement and persistence, issuer/year contributions and fee/borrow accounting, style and market exposure. No trained-network interventions or refits performed.
