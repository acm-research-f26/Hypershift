# Fixed blend-weight search

Exploratory 2022–2025 history; plan committed `7b9325f` and code `86aa9c4` before outcomes. Five fixed dynamic-family weights, three seed groups, two policies: 30 accounts, of which six are the existing 50:50 baseline. Source scoring and share-account costs apply. Each group's two midpoint baselines were checked before the other weights in that group; all six match saved daily returns exactly. Baselines for later groups were not checked before earlier groups' alternatives, despite the plan's intended global ordering; all choices were frozen first and all parity checks passed.

| Group | Portfolio | Best dynamic weight in grid | Best Sharpe / CAGR / max drawdown | 50:50 Sharpe |
|---|---|---:|---:|---:|
| Original five | Issuer-five, vol20 | 50% | 1.178 / 28.57% / -20.56% | 1.178 |
| New five | Issuer-five, vol20 | 75% | 0.714 / 16.38% / -21.38% | 0.677 |
| Ten seeds | Issuer-five, vol20 | 50% | 1.112 / 26.81% / -21.69% | 1.112 |
| Original five | Equal ten, full | 50% | 0.848 / 27.79% / -35.04% | 0.848 |
| New five | Equal ten, full | 75% | 0.533 / 15.40% / -30.05% | 0.467 |
| Ten seeds | Equal ten, full | 50% | 0.755 / 23.92% / -34.88% | 0.755 |

The 75% dynamic weight improves the weaker new-five group's historical point result, but does not improve either ten-seed setup in this grid. The two five-seed groups overlap the ten-seed ensemble and share one market history; these are not independent replications. The grid was run on already searched years; a best variant is not evidence of future improvement. Full metrics including turnover are in METRICS.csv and year-by-year returns in CALENDAR.csv. Source forecasts, eligibility, valuation and execution limitations remain.

Verification: two unit tests passed; six source baseline paths have zero maximum error. A separate calculation recomputed 30 Sharpes, CAGRs, drawdowns, NAV paths and 120 annual returns with maximum error 1.55e-14. It did not independently recreate all share orders or targets. PATHS.npz stores its dates as a Python object array due to pandas conversion; loading it requires `allow_pickle=True` and should be done only for this locally generated trusted artifact. No original data or outputs were overwritten.
