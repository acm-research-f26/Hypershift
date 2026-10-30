# Traded selection categories

Frozen protocol: ../TRADED_BOUNDARY_PROTOCOL.md. Two ten-seed full-exposure blend books, 2022–2025, use reconstructed economic fills. Categories identify whether each blend target was also in the parent and/or dynamic standalone top-k at that decision. Price P&L follows the previous held category; normal fill fees follow the new target category, with exiting positions' fees assigned to `exit_cost`. Terminal and corporate-action fees follow the prior category. This convention partitions the existing account; it is not the performance of portfolios that omit or add a category.

| Book | Both: net contribution / mean exposure | Parent only | Dynamic only | Neither | Exit costs |
|---|---:|---:|---:|---:|---:|
| Top 5 | 12.96 pp / 21.78% | 12.45 pp / 37.72% | 6.64 pp / 24.10% | 2.48 pp / 16.12% | -1.07 pp |
| Top 10 | 15.42 pp / 29.20% | 4.38 pp / 33.76% | 7.84 pp / 28.21% | -0.63 pp / 8.62% | -0.85 pp |

Net figures are annualized arithmetic daily contributions, not CAGR or fractions of Sharpe. Residual held-outside-target contribution is zero in these paths. Category daily totals reproduce the existing account paths to maximum absolute error 1.66e-15 (top5) and 2.78e-15 (top10); issuer, calendar, cost and exposure partitions are retained in the CSVs. The top10 overlap has substantial positive realized contribution, while its neither-standalone group is negative under this accounting convention. This does not identify which model learned the relevant information, nor the effect of removing either model. Unheld candidates cannot have realized portfolio P&L. Source adjusted-close units are economic accounting units, not certified broker fills; the sample is selected development history.

Outputs: CATEGORY_ATTRIBUTION.csv, CATEGORY_ISSUERS.csv, CALENDAR_ATTRIBUTION.csv, RECONCILIATION.csv and RESULTS.json; daily Unicode-axis arrays are in the ignored results directory. Input and original run-code hashes are in RESULTS.json.
