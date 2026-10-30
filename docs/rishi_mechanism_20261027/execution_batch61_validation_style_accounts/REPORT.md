# Do simple style rankings reproduce the historical accounts?

The simple controls tell different stories for the risk-managed five-issuer book and the full-exposure ten-stock book. In the ten-seed comparison at base modeled costs:

| Selection | Five issuers, 20% vol target: CAGR / Sharpe / max drawdown | Ten securities, full exposure: CAGR / Sharpe / max drawdown |
|---|---|---|
| Changing-score model | 26.81% / 1.112 / -21.69% | 23.92% / 0.755 / -34.88% |
| Fixed validation preference | 26.71% / 1.080 / -25.44% | 17.27% / 0.597 / -35.52% |
| Pretest liquidity ranking | 19.99% / 0.833 / -27.63% | 25.02% / 0.762 / -46.05% |
| Pretest QQQ-beta ranking | 21.01% / 0.832 / -21.57% | 27.32% / 0.719 / -48.36% |
| Equal liquidity/beta rank blend | 24.69% / 0.988 / -27.79% | 18.47% / 0.558 / -53.93% |

The pretest liquidity ranking's high target overlap with fixed validation preference did **not** reproduce its five-issuer risk-managed historical point result, although the liquidity top-ten book had a comparable Sharpe and a deeper drawdown than the model top-ten book. For five issuers, the changing model minus liquidity annual arithmetic net-mean difference is **+5.46 pp [-2.33, +13.72]**; fixed validation minus liquidity is **+5.51 pp [+0.98, +10.18]**. For ten securities, model minus liquidity is **-1.35 pp [-14.05, +11.51]**. These are 21-session-block pointwise 95% intervals. The one positive lower bound among the 36 base-block comparisons is **unadjusted and selected after extensive prior research**, not evidence of a confirmed discovery; most intervals are wide and none of these figures establishes economic equivalence.

The simple controls were chosen after earlier style associations; their annual scores use a common support requiring all four pretest characteristics, with a median fallback for unsupported eligible securities. They are not independently optimized or trained, but they are not an untouched test either. Volatility targeting is recomputed separately for each basket. Different holdings, exposures, turnover, costs and historical market movements confound performance; the comparison does not establish which inputs the neural model learned, explain all returns or validate a future strategy. Seed groups share the same market history and the simple-control paths are duplicated across groups.

The [protocol](../VALIDATION_SIMPLE_STYLE_ACCOUNTS_PROTOCOL_20261029.md) and implementation were committed as `62d4839` and `0b84d1f` before these outcomes. The batch contains **30** account rows and **108** paired interval rows; all 12 reference paths exactly reproduce prior daily returns (maximum normalized NAV error 4.66e-16). An independently written checker reconstructed **3,618** simple target rows, all 30 account NAV/metrics and target exposures, and all 108 paired mean intervals, maximum discrepancy 1.55e-14. SR/CAGR intervals are source-bootstrap calculations not separately reconstructed here. All account rows, calendar returns, model/validation comparisons, costs and hashes are retained. No model fitting, checkpoint intervention, protected scoring or prospective activation occurred; the existing research gates remain unchanged.
