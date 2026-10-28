# Orders and holdings behind the twelve headline paths

2026-10-27. Protocol `8699830`, implementation `8c54622`. All twelve workbook paths reproduce from traced operations, including the original-five 1.199-Sharpe leader. Maximum daily-return error is below 1e-16, cash error below $1.40e-9, share error exactly zero and normalized NAV error below 1.40e-15. Independently accumulated quantities and cash flows reconstruct NAV; per-stock gross mark P&L minus fees reconstructs every daily net return. Acquisition principal becomes cash and share removal, not new profit.

The ten stock-model accounts retain submitted desired/capped quantities, target weights, decision/cutoff/fill clocks, actual fills, positions, marks, fees and liquidation records. No participation cap binds and no missing fill occurs in these particular $1 million paths. Cash scaling occurs in some per-security fills; it remains part of the reconstructed policy. Synthetic tests separately exercise binding caps, settlement, unavailable fills and terminal costs.

SPY and QQQ have a different source contract: a first-close dollar-budget purchase and terminal liquidation. Their two fills, daily units and first/last costs reconcile. There is no separate frozen-stock-share submission event for these benchmarks; their orders CSV is empty by design. Do not infer identical auction instructions across the stock model and ETF benchmark.

## Concentration of the highest-Sharpe path

Original five seeds, original top five, 20% volatility target:

| Issuer | Annual arithmetic net contribution |
|---|---:|
| Alphabet, both classes combined |9.68 pp|
| NVIDIA |8.33 pp|
| Broadcom |2.81 pp|
| Total account |27.53%|

Those three issuers contribute 75.64% of the account's total net arithmetic mean return. For the ten-seed version, NVIDIA/Alphabet/Broadcom contribute 74.15%; the corresponding full-exposure account's fraction is 64.74%. These are fractions of additive net arithmetic contributions, not fractions of CAGR or Sharpe explained. The complete stock and issuer tables retain every name, fees and holding days; the top-1/3/5 summary retains all other contributions as well.

This confirms a concentrated historical economic mechanism, without proving that stable learned identity weights caused the concentration. No stock deletion strategy was selected or tested.

## Limits and files

These are reconstructed canonical-model records, not broker or exchange records. Source adjusted closes approximate reinvested total returns, so traced units are units of the source economic-price series; they are not certified raw broker share counts. Raw-close dollar volume supplies liquidity. Acquisition settlement and stock-consideration liquidation retain the source approximations. No independent market execution policy or historical vendor latency was validated by this accounting identity.

Small files: STOCK_ATTRIBUTION.csv, ISSUER_ATTRIBUTION.csv, CONCENTRATION.csv, RECONCILIATION.csv and RESULTS.json. Larger archives in results/RISHI_MECHANISM_LEADER_TRACE_20261027 contain Unicode date/security axes, units, economic marks, cash/NAV, fees, gross dollar P&L, net contributions and order/fill records. Three tests passed before real-path tracing. Independent review found no accounting defect.
