# How much of the coverage-control gap is modeled fees?

Higher trading fees account for **2.72 percentage points per year** of the blend's **22.10-point annual arithmetic net-return difference** over the matched linear coverage control. The remaining **19.39 points** are a difference in accounting gross contributions on the positions and capital paths that actually occurred. This does not establish what a fee-free strategy would have earned, or why its stock selections differed.

The comparison uses the four saved batch34 ledgers without rerunning trades. Each day's fee drag is its fee divided by that account's own preceding NAV; gross contribution is the recorded net return plus that drag. The arithmetic identity is exact: `net difference = gross contribution difference − fee-drag difference`. Values below are 252 times the daily mean, in percentage points per year, not CAGR or fractions of Sharpe.

| Comparison | Net difference | Gross contribution difference | Original minus control fee drag |
|---|---:|---:|---:|
| E minus its coverage6 control | +22.10 | +19.39 | -2.72 |
| E_Hlabel minus its coverage6 control | +21.61 | +17.89 | -3.72 |

For E, the recorded annualized fee drag is **1.82 pp**, versus **4.54 pp** for its control; the corresponding gross contributions are **27.99 pp** and **8.60 pp**. The E gross-contribution difference has a block-21 95% interval of **[+7.72, +31.49] pp**, and the fee-drag difference is **[-3.00, -2.44] pp**. Blocks 5 and 63 and both comparisons are in `CONTRASTS.csv`. These are exploratory intervals on previously examined history, not search-adjusted confirmation or a new primary gate.

The gross accounting view retains the effects that past fees had on cash, quantities, fills and subsequent capital. It is **not** a zero-fee replay or an equal-risk portfolio; different names, exposures, timing and turnover contribute to the gross difference. The controls inherit each original model's industry schedule and position-size distribution. Financial returns and reconstructed economic fills remain subject to the earlier vintage, identity and execution limitations.

Protocol e9f4a68 and code d4ce2e1 were fixed after batch34 results and before this decomposition. Four source ledgers and all **18** paired net/gross/fee intervals were independently reconstructed to maximum error 7.11e-15. Two focused tests passed. Daily components, all account and calendar rows, source hashes and the complete interval table are in this folder. No new trading account, model inference, financial fitting, protected-window scoring or gate change occurred.
