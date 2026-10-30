# Causal factor controls and neural blends — October29,2026

Completed198fixedaccounts; **0 hybrid recipes have SR>1.5 in all five groups** at base costs. This arm was specified during fitting, before any new account metrics. It does not alter the separate120-fit/576-account study. These remain selected development results on spent2022–2025history.

| Style | Neural weight | Allocation | Original SR | New SR | Fresh SR | Ten SR | Fifteen SR | Pure-factor SR | Minimum increment over pure | Goal |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| liquid_consensus | 50% | retain5 | 0.751 | 0.837 | 0.911 | 0.762 | 0.891 | 0.340 | 0.411 | False |
| liquid_consensus | 50% | issuer5 | 0.822 | 0.801 | 0.681 | 0.719 | 0.712 | 0.207 | 0.474 | False |
| liquid_consensus | 25% | retain5 | 0.521 | 0.527 | 0.538 | 0.557 | 0.525 | 0.340 | 0.181 | False |
| liquid_consensus | 25% | issuer5 | 0.582 | 0.417 | 0.532 | 0.521 | 0.512 | 0.207 | 0.211 | False |
| momentum252_21_risk | 25% | retain5 | 0.587 | 0.367 | 0.470 | 0.400 | 0.413 | 0.197 | 0.170 | False |
| momentum252_21_risk | 25% | issuer5 | 0.529 | 0.363 | 0.340 | 0.429 | 0.454 | 0.036 | 0.303 | False |
| momentum252_21_risk | 50% | retain5 | 0.332 | 0.578 | 0.405 | 0.398 | 0.335 | 0.197 | 0.135 | False |
| momentum252_21_risk | 50% | issuer5 | 0.310 | 0.704 | 0.630 | 0.425 | 0.415 | 0.036 | 0.274 | False |
| momentum63_risk | 50% | issuer5 | 0.188 | 0.061 | 0.064 | 0.108 | 0.160 | -0.212 | 0.273 | False |
| momentum63_risk | 50% | retain5 | 0.266 | 0.253 | 0.045 | 0.172 | 0.206 | -0.082 | 0.127 | False |
| momentum63_risk | 25% | retain5 | 0.113 | 0.174 | -0.053 | 0.015 | 0.151 | -0.082 | 0.029 | False |
| momentum63_risk | 25% | issuer5 | -0.135 | 0.085 | -0.290 | 0.071 | -0.100 | -0.212 | -0.078 | False |

## Interpretation

Momentum63/20-dayvol and12-minus-1momentum/60-dayvol use trailing marks through each decision origin. A third score averages their percentiles with liquidity(logADV). Missing factor history is neutral0.5; current stock eligibility is unchanged. All hybrid scores use the ranking-selected neural forecasts only, with fixed25%/50% neural weight and fixed75%/50% factor weight. Pure factors are evaluated once each, not duplicated across seed groups. All accounts keep the20%basket-volatility target and existing continuousQQQ cap75 execution/costs. Higher hybrid seed consistency can come from the shared deterministic component. Compare each row to its pure-factor control before crediting neural training; a negative minimum increment means at least one neural ensemble underperformed the matching pure factor.

All198paths pass independent dollar-position replay; all results/cost stresses are retained. No outperformance significance or untouched future evidence is claimed. The controls are motivated by [Jegadeesh/Titman](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x) and [Moreira/Muir](https://www.nber.org/papers/w22208); the specific features/holding rules here do not replicate those papers.

[Allmetrics](METRICS.csv) · [Allcostregimes](CONSISTENCY.csv) · [Audit](AUDIT.json) · [Protocol](PROTOCOL.json)
