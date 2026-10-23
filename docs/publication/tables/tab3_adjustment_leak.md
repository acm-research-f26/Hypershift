# Table 3. Back-adjustment term and components as scores (462 stocks, 2019–2023 signals; exploratory)

| Score (all scored against 5-day factor-neutral returns, 2019–2023) | Rank IC pooled | 95% CI |
|---|---|---|
| Trailing distribution yield TY (known at t) | -0.0086 | [-0.0203, +0.0031] |
| L = log(AdjClose/Close), full ex-post term | +0.0189 | [+0.0080, +0.0298] |
| L residualised on TY (regression residual) | +0.0305 | [+0.0242, +0.0373] |
| L_div: future distributions at today's price | +0.0096 | [-0.0015, +0.0210] |
| L_path: future price-path component | +0.0935 | [+0.0857, +0.1014] |
| L(end): distributions after 2024-01-11 | +0.0171 | [+0.0069, +0.0273] |

Sources: `data/audit/yf_unadjusted/unadjusted.npz`, FB saved labels; components from `scripts/pub_leak_mechanism.py`.
