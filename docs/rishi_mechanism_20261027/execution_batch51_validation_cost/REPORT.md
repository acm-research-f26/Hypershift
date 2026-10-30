# Cost sensitivity of fixed validation preferences

The changing-score ten-seed issuer-five account trades substantially more than the fixed-validation account, and its historical advantage shifts as modeled trading costs rise. Each row below reruns both accounts with its own cash and position path under the same 20% basket-volatility rule.

| Modeled cost | Model CAGR / Sharpe | Fixed-validation CAGR / Sharpe | Model minus fixed annual net mean, 95% interval |
|---|---|---|---|
| Zero | 28.92% / 1.192 | 27.12% / 1.095 | +1.27 pp [-5.54, +8.32] |
| Base 1× | 26.81% / 1.112 | 26.71% / 1.080 | -0.05 pp [-6.77, +6.99] |
| 2× | 24.74% / 1.031 | 26.30% / 1.065 | -1.37 pp [-8.16, +5.69] |
| 4× | 20.70% / 0.870 | 25.49% / 1.034 | -4.01 pp [-10.83, +3.08] |

Intervals are 21-session stationary-bootstrap, pointwise, and the base row is inherited from the preceding comparison with its own fixed bootstrap seed. All four intervals include zero and extend beyond the ±2 pp economic margin: neither superiority nor equivalence is established. Zero cost is an optimistic scenario; 2×/4× are sensitivity assumptions, not measured broker costs. Different stock selections and resulting volatility estimates remain confounded. Validation selected the checkpoints, so this is not training Hpred or a causal ablation. All results concern repeatedly inspected 2022–2025 development history.

Protocol `c2e3673` and implementation `a6a2f9d` were fixed before these outcomes. The batch contains **48** account replays, including 12 base-cost paths that reproduce the previous saved daily returns exactly (maximum normalized NAV error 4.66e-16). A separate verifier checked all 48 NAV paths, metrics and target exposures, plus all **54** new paired mean intervals, maximum error 1.60e-14. SR and CAGR interval columns use the source bootstrap and were not independently rebuilt here. `METRICS.csv`, `CONTRASTS.csv`, `CALENDAR.csv` and `BASE_REPLAY.csv` retain all fixed groups and scenarios. No neural fitting, checkpoint intervention, protected scoring or prospective activation occurred; existing gates remain unchanged.
