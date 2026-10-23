# A. HyperShift: authoritative status (2026-10-24)

## Completed strategy decisions

| Strategy (universe, period) | Decision | Key result | Report |
|---|---|---|---|
| THINK / hypergraph and price-only ML (all studies) | STOP | No leak-free advantage | `docs/phase2/REPORT_CLOSURE.md`, `docs/audit_20261022/FINAL_DECISION.md` |
| PRICE3 and price-only tilts (S&P 500 survivors) | STOP | Active IR −0.74 | `docs/strategy_20261022/STRATEGY_REPORT.md` |
| Style-only LightGBM book (PIT S&P 500, 2018–23) | STOP | Net Sharpe +0.11 [−0.61, +0.90] | `docs/strategy_pit_20261023/STRATEGY_PIT_REPORT.md` |
| Filing-text changer avoidance (PIT S&P 500, 2016–23) | STOP | Active IR −0.01 | `docs/strategy_lazyprices_20261023/REPORT.md` |
| EAR overlay (PIT S&P 500, 2016–23) | STOP | Active IR −0.47; no event information beyond its control | `docs/strategy_ear_20261023/STRATEGY_DECISION.md` |
| Stable reporters | ARCHIVED | Selected diagnostic; no independent evidence | `docs/strategy_ear_20261023/ALTERNATIVES.md` |
| **S&P 400 filing-text expansion (free data)** | **STOP: free approach not viable** | Membership fails timing and completeness checks; economics marginal | this folder |

## Data and constraints

- **Explored:** 2016–2023, on the PIT S&P 500 panel (Alpaca + fja05680) and EDGAR.
- **Protected:** 2024-01..2026-09 has not been released for any strategy.
- **Untouched:** pre-2016 history needs paid data; none has been authorised.
- **Capital:** none authorised.
- **Running:** nothing (2026-10-24).

## Open decision for you

Whether to authorise long-history, survivorship-free data with dated index membership (Norgate Platinum, $630/yr),
or to pause strategy research. See `ALTERNATIVE_RANKING.md`.
