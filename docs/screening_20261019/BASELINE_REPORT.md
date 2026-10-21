# Baseline report — v3 broad screen (S&P 500 panel, folds 2022/2023)

Exploratory. Every number comes from `v3_SCREENING_TABLES.md` and `v3_TRADING_SENSITIVITY.csv`, which are built from `results/SCREEN_SP500_20261019_v3` (171/171 runs, 0 failed).

## Common baseline

`static` is the parent of every A–I comparison except E gated, whose parent is `mix`.
- **Model:** Euclidean temporal convolution, then hypergraph attention, then a linear head. Hyperedges are 16 KMeans clusters of training-period return correlations plus singleton self edges. Hidden 16, sequence 16.
- **Target and training:** 5-day forward returns, masked MSE, Adam lr 1e-3 with weight decay 0. At most 40 epochs, patience 8. The checkpoint is selected on validation RankIC.
- **Data:** stride-1 training signals (1,443 dates for 2022, 1,695 for 2023), 458–462 eligible stocks.
- **Seeds:** 7, 19, 42.

The same data, folds, eligibility and selection rule apply to every arm. Details are in `DATA_AUDIT.md` and the v3 amendment in `RESEARCH_SCREENING_PLAN.md`.

## Ranking quality (pooled = equal-weight mean of the 2022 and 2023 fold means)

| Arm | What it is | Runs | RankIC 2022 | RankIC 2023 | Pooled RankIC | Neutral RankIC (pooled) | NDCG@10 excess | spread10 |
|---|---|---|---|---|---|---|---|---|
| static | common baseline (Euclidean + correlation hyperedges) | 6 | −0.0070 | +0.0213 | 0.0071 | −0.0006 | 0.0147 | 0.0013 |
| hyp | static with hyperbolic layers (THINK-style) | 6 | +0.0047 | +0.0190 | 0.0119 | +0.0087 | 0.0089 | 0.0010 |
| none | static without hyperedges (temporal only) | 6 | +0.0055 | +0.0172 | 0.0113 | +0.0010 | 0.0151 | 0.0010 |
| mlp | per-stock MLP on window features | 6 | +0.0068 | +0.0154 | 0.0111 | −0.0000 | 0.0113 | 0.0008 |
| reversal | −16-day return | 2 | +0.0153 | +0.0057 | 0.0105 | +0.0058 | 0.0120 | −0.0010 |
| ridge | Ridge on 4/16/64-day window means + last value | 2 | −0.0082 | +0.0209 | 0.0064 | +0.0045 | 0.0004 | 0.0017 |
| momentum | +16-day return | 2 | −0.0153 | −0.0057 | −0.0105 | −0.0058 | 0.0143 | 0.0010 |
| random | Gaussian scores (3 seeds) | 6 | −0.0013 | +0.0008 | −0.0002 | −0.0005 | −0.0006 | 0.0001 |

"Neutral" RankIC is computed on returns after removing trailing 126-session beta × market and then demeaning by GICS sector. NDCG excess is measured over random ordering. Ridge, momentum and reversal are deterministic, so each has one run per fold.

**Reading.**
- Every baseline's RankIC lies within ±0.022 in each fold. Pooled RankIC lies within ±0.012.
- `static` is the weakest neural baseline in 2022 and roughly tied in 2023.
- `static` is **not separable** from the no-graph model, the MLP, reversal or Ridge: every paired CI includes 0 (see `v3_SCREENING_TABLES.md`, controls table).
- Factor-neutral ranking is weaker still: |neutral RankIC| ≤ 0.013 for these baselines in every fold, and pooled |neutral RankIC| ≤ 0.009 for every v3 arm. Most of the raw RankIC is market, beta or sector exposure.
- Predictions correlate 0.8–0.9 with the day's market move for every neural arm.

## Trading sensitivity (seed-mean net Sharpe; daily top-k, cost per unit turnover)

| Arm | 2022 k10 0bp | 2022 k10 10bp | 2022 k10 25bp | 2023 k10 0bp | 2023 k10 10bp | 2023 k10 25bp | CAGR 2022 / 2023 (k10, 10bp) | Max DD 2022 / 2023 | Turnover |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.04 | −0.13 | −0.39 | 1.41 | 1.12 | 0.68 | −11.3% / +26.7% | −26% / −16% | 0.26 / 0.27 |
| hyp | 0.13 | −0.04 | −0.29 | 1.09 | 0.72 | 0.17 | −8.3% / +14.8% | −25% / −19% | 0.25 / 0.33 |
| none | 0.05 | −0.15 | −0.44 | 1.60 | 1.30 | 0.85 | −11.2% / +32.0% | −22% / −17% | 0.28 / 0.28 |
| mlp | 0.02 | −0.17 | −0.45 | 1.62 | 1.28 | 0.77 | −12.4% / +30.9% | −24% / −18% | 0.27 / 0.31 |
| reversal | 0.16 | −0.00 | −0.25 | 1.37 | 1.11 | 0.71 | −6.9% / +25.2% | −25% / −13% | 0.24 / 0.24 |
| ridge | −0.28 | −0.43 | −0.65 | 1.53 | 1.28 | 0.91 | −22.2% / +33.9% | −32% / −24% | 0.24 / 0.25 |
| momentum | −0.01 | −0.22 | −0.54 | 2.05 | 1.79 | 1.40 | −10.4% / +44.6% | −25% / −19% | 0.25 / 0.23 |
| random | −0.07 | −0.49 | −1.11 | 1.01 | 0.32 | −0.71 | −13.1% / +3.7% | −23% / −18% | 0.39 / 0.39 |

Top-10 at 10 bp is the predeclared default. Top-5 and top-20 are in `v3_TRADING_SENSITIVITY.csv`.

**Reading.**
- 2022 is a down year: every long-only top-k basket loses after costs.
- In 2023 every non-random baseline earns net Sharpe ≈ 0.7–1.8. That level reflects a strong large-cap year and **survivorship bias**: 2026 constituents are traced back to 2015.
- Momentum has the best 2023 Sharpe despite a *negative* RankIC. Basket Sharpe and cross-sectional ranking measure different things, and one year of baskets cannot separate skill from exposure (compare Phase 1.5a).
- The trading backtest is optimistic: entry at the signal-day close, no delisting payouts, and costs only as a turnover multiple.

## Use in the screen

- `static` was the predeclared parent because it is the minimal THINK-style model with hyperedges. It is a weak and noisy parent: its seed-to-seed per-fold RankIC spread is ±0.01–0.02.
- Any arm "beating static" by less than that spread is not evidence of a mechanism. For this reason the deep tests use 10 seeds and 3 folds (see `DEEP_ABLATIONS.md`).
- The baselines also show that the no-graph model, the MLP and reversal(16) are the realistic bars, not static alone.
