# Study FB — simple models, richer features, targets and horizons (2026-10-22)

**Exploratory.** Every year scored here (2019–2023) has been inspected before in this repository. The design, models, hyperparameters, families and decision rule were predeclared in `docs/superpowers/plans/2026-10-22-feature-benchmark.md` (commit 6737b62) before any FB number existed. 2024+ was not touched (spent for Study J only). Spec: `PROPOSAL.md`.

DRAFT — sections are filled as tasks finish. Tables: `fb_TABLES.md`, `fb_PAIRED.csv`, `fb_PORTFOLIO.md`, `fb_SUBGROUP.md`; audit and power: `DATA_AUDIT.md`, `POWER.md`.

## Answer so far

- **Power first:** a paired neutral-RankIC contrast between two deterministic scorers has MDE ≈ 0.021 over 5 folds and ≈ 0.046 in one year (`POWER.md`). Every effect below is smaller than that, so NO EVIDENCE is the expected outcome even for a real effect of the size seen.
- **#1 LightGBM + volume/volatility/liquidity features: no gain.** LightGBM − Ridge +0.0036 (Holm 0.575); neither beats 16-day reversal (Ridge −0.0120, LightGBM −0.0084). Adding the volume/liquidity features *lowered* LightGBM in all 5 folds (−0.0088, Holm 0.181).
- **#2 Targets: the one consistent direction, still below the bar.** Training on sector-relative or factor-residual targets raised neutral RankIC in 18 of 20 fold × contrast cells; LightGBM residual − raw +0.0140 [+0.0018, +0.0261], 5/5 folds, Holm 0.065 (NO EVIDENCE under the predeclared rule). `lgbm:rich:residual` has the highest neutral RankIC of all arms, +0.0127 [+0.0029, +0.0228], positive in every fold (descriptive).
- **#3 Horizons:** no model beats reversal16 at 1, 5, 10 or 20 days; at 10 and 20 days both models are clearly below it (descriptive).
- **Lag decomposition:** only the 1-day block has a CI above zero (+0.0107 [+0.0032, +0.0185], descriptive); a Ridge on the four lag blocks is +0.0044 over reversal16 (Holm 0.348).
- **#4 Representations:** nothing beats per-date z-scores after Holm (best Ridge sector-z +0.0063, 5/5 folds, Holm 0.155).
- **#5 Portfolios:** long-only baskets (Sharpe 0.48–0.95, beta 1.3–1.6) are market-like next to hold-all 0.86; every market-neutral rule is at or below zero.
- **#6 Subgroups:** no consistent pattern (1 of 16 high−low contrasts has a CI excluding 0).
