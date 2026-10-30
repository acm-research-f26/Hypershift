# Validation versus liquidity: cost sensitivity

2026-10-29. Exploratory, post-selection analysis on repeatedly inspected 2022–2025 NDX history. Protocol `VALIDATION_LIQUIDITY_COST_PROTOCOL_20261029.md` was committed as `fba6af7` and replay code as `7f3462b` before these outcomes. The fixed validation preferences use the checkpoint-selection validation fold; they are not a training-label or trained-model intervention.

Sixteen accounts replay the previously saved fixed-validation and simple liquidity targets under four modeled cost multipliers. All four 1x daily return paths exactly reproduce batch 61; maximum relative NAV discrepancy 4.66e-16. The 0x case is optimistic and 2x/4x are hypothetical stress scenarios, not calibrated execution estimates. Accounts dynamically rebalance, so costs can affect subsequent positions and exposures.

| Allocation | Cost multiplier | Fixed CAGR / Sharpe | Liquidity CAGR / Sharpe | Fixed minus liquidity annual mean, pp (21-day block 95% interval) |
|---|---:|---:|---:|---:|
| Five issuers, 20% volatility target | 0 | 27.12% / 1.095 | 20.33% / 0.846 | +5.55 [0.98, 10.19] |
| Five issuers, 20% volatility target | 2 | 26.30% / 1.065 | 19.65% / 0.819 | +5.48 [0.89, 10.11] |
| Five issuers, 20% volatility target | 4 | 25.49% / 1.034 | 18.97% / 0.792 | +5.40 [0.81, 10.04] |
| Ten equal holdings, full exposure | 0 | 17.52% / 0.604 | 25.21% / 0.766 | -7.46 [-19.53, 4.35] |
| Ten equal holdings, full exposure | 2 | 17.03% / 0.590 | 24.83% / 0.757 | -7.58 [-19.70, 4.24] |
| Ten equal holdings, full exposure | 4 | 16.54% / 0.575 | 24.46% / 0.748 | -7.70 [-19.87, 4.14] |

Block-length sensitivity matters: for five issuers at 2x and 4x cost, the 5-day-block intervals are [-0.06, 11.31] and [-0.17, 11.22] percentage points per year, respectively, and include zero; the 0x interval is [0.04, 11.37]. The corresponding 63-day-block intervals are [1.88, 9.22], [1.81, 9.15] and, at 0x, [1.97, 9.28]. All ten-holding intervals include zero at all three block lengths. These are sensitivity checks, not independent replications; no block length should be selected after observing the outcomes.

The contrast is already present at zero modeled costs and changes little across these multipliers. That supports the narrow accounting interpretation that modeled trading fees alone do not explain the historical differences. The two allocation policies choose different baskets and exposures; this does not isolate a causal learned signal or establish future performance. Intervals are pointwise and unadjusted after extensive exploration; different cost levels and block lengths share the same market history, and no multiplicity-adjusted claim is made. The risk-targeted result also depends on each basket's own estimated volatility. Calendar returns, all block sizes, account metrics and reference checks are in the adjacent CSVs; paths and ledgers are in `results/RISHI_VALIDATION_LIQUIDITY_COST_20261029/`.

Independent verifier `research/analysis/rishi_validation_liquidity_cost_verify.py` checked 16 account NAV paths, three performance metrics and causal target exposures plus 18 paired annual-mean bootstrap intervals, maximum error 1.24e-14. It does not independently verify Sharpe/CAGR bootstrap intervals or execution realism. Saved target reconstruction was separately verified in batch 61. Source-vintage and security-identity limitations remain; the selected-checkpoint numerical gate remains unresolved.
