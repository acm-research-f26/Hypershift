# Study FB power analysis

Source: v3 deterministic baselines `ridge` and `reversal` (seed 7), folds 2022/2023, saved `daily.npz` (`results/SCREEN_SP500_20261019_v3`). Paired daily ΔRankIC = ridge − reversal. SE of the fold mean from a stationary bootstrap (block 10, B 5000). MDE = 2.8 × SE (80% power, two-sided α 0.05). Deterministic models have no seed variance, so date variance is the whole uncertainty.

| Fold | Metric | Days | Daily SD | Bootstrap SE of mean | iid SE |
|---|---|---|---|---|---|
| 2022 | RankIC | 246 | 0.2306 | 0.0304 | 0.0147 |
| 2022 | neutral_RankIC | 246 | 0.1587 | 0.0201 | 0.0101 |
| 2023 | RankIC | 245 | 0.2036 | 0.0283 | 0.0130 |
| 2023 | neutral_RankIC | 245 | 0.1247 | 0.0128 | 0.0080 |

## Minimum detectable effects

| Design | SE (neutral RankIC) | MDE (neutral) | MDE (raw) |
|---|---|---|---|
| 1 fold (~250 days) | 0.0164 | 0.0460 | 0.0821 |
| 2 folds | 0.0116 | 0.0325 | 0.0581 |
| 5 folds 2019–2023 (pooled mean of fold means) | 0.0073 | 0.0206 | 0.0367 |

## Prospective test length

| Prospective days | MDE (neutral, scaled from the fold SE by √(days)) |
|---|---|
| 60 | 0.0930 |
| 125 | 0.0644 |
| 250 | 0.0455 |
| 500 | 0.0322 |

Days needed for MDE 0.01: **5187**; for MDE 0.02: **1297** (neutral RankIC, paired contrast of two deterministic scorers; a contrast against a weaker or less correlated comparator has a larger SD).

Caveat: estimated from one contrast (ridge vs reversal) in two years; contrasts between highly correlated models have smaller SD and smaller MDE.
