# Study FB — simple models, richer features, targets and horizons (2026-10-22)

**Status: exploratory.** Every year scored here, 2019–2023, had been inspected before. The plan predeclared the design, models, fixed hyperparameters, families and decision rule (`docs/superpowers/plans/2026-10-22-feature-benchmark.md`, commit 6737b62) before any FB number existed. Nothing was scored on 2024+, which Study J spent. The spec is the user's proposal (`PROPOSAL.md`).

Files:
- Tables: `fb_TABLES.md`, `fb_PAIRED.csv`, `fb_PORTFOLIO.md`, `fb_SUBGROUP.md`, `fb_HYPEREDGE.md`.
- Audit and power: `DATA_AUDIT.md`, `POWER.md`.
- Freeze: `FREEZE_MANIFEST.json`, `PROSPECTIVE_PROTOCOL.md`.
- Runs: `results/FEATBENCH_SP500_20261022` (170 runs: 32 arms × 5 folds plus LightGBM seeds 19/42; 0 failed).

## Answer

**Richer features and stronger simple models did not beat 16-day reversal. Changing the target was the one direction that improved factor-neutral ranking consistently, and it still missed the predeclared bar.** The effects involved are smaller than this design can detect.

- **Power:** a paired factor-neutral RankIC contrast has a minimum detectable effect (MDE) of about 0.021 over 5 folds, and 0.046 in a single year. Every predeclared contrast below is smaller than 0.021. The only larger differences are descriptive horizon comparisons against reversal16 at 10 and 20 days (−0.016 to −0.031).
- **Frozen for a prospective test:** `lgbm:rich:residual`, scored only after 2026-10-22. It needs at least 250 days, and at that length only a large effect is detectable.

## Setup

- **Data:** 462 current S&P 500 stocks with daily adjusted OHLCV from yfinance, aligned to the screening panel (`DATA_AUDIT.md`). There are no fundamentals and no shares outstanding, so "size" means 20-day dollar volume.
- **Features:** 20 features (`src/hypershift/featbench/data.py`), with code tested for no look-ahead (but see the dividend caveat): 1–60-day returns, 20–60 momentum, 5/20/60-day volatility, downside volatility, 60-day drawdown, beta, log dollar volume, relative volume, volume change, Amihud illiquidity, high–low range, and sector-relative 5- and 20-day returns.
- **Folds:** 2019–2023 walk-forward. Each trains on labels maturing before Y−1, validates on Y−1 and evaluates Y. Purging is horizon-aware.
- **Models:**
  - Ridge, alpha chosen on validation from {0.1, …, 1000};
  - LightGBM: 15 leaves, depth 4, lr 0.05, rounds chosen on validation;
  - MLP (32, 16), 3 seeds averaged;
  - fixed lag-block scores.
- **Primary metric:** daily factor-neutral RankIC, i.e. the forward return minus beta × market, demeaned within sector.
- **Evaluator check:** `block:reversal16` reproduces v3 `reversal` exactly (test).
- **Statistics:** stationary bootstrap per fold with block max(10, 2h); pooled = mean of 5 fold means; Holm within each family. SUPPORTED requires Holm p < .05, a CI that excludes 0, and ≥ 4/5 fold means with the pooled sign.

## Results by proposal item (pooled factor-neutral RankIC, 2019–2023)

| # | Question | Key result | Verdict |
|---|---|---|---|
| — | Power | MDE 0.021 (5 folds), 0.046 (1 year); 1,300 days for 0.02 | — |
| 1 | LightGBM + volume/volatility/liquidity | LightGBM − Ridge +0.0036 (Holm 0.575). Ridge − reversal16 −0.0120 (0.276). LightGBM − reversal16 −0.0084 (0.394). MLP − Ridge +0.0101 (0.292) | NO EVIDENCE |
| 1b | Do volume/liquidity features add anything? | lgbm rich − lgbm price-only −0.0088, **0/5 folds positive** (Holm 0.181) | NO EVIDENCE (nominally harmful) |
| 2 | Sector-relative / factor-residual targets | All 4 contrasts positive in 18/20 fold cells. LightGBM residual − raw **+0.0140 [+0.0018, +0.0261], 5/5 folds**, Holm 0.065. Ridge sector − raw +0.0075, Holm 0.065 | NO EVIDENCE (closest to the bar) |
| 3 | Horizons 1/5/10/20 | No model beats reversal16 at any horizon. At 10 and 20 days both models are below it (e.g. Ridge −0.031 at h20, CI excludes 0) | descriptive |
| — | Lag decomposition | 1-day block +0.0107 [+0.0032, +0.0185] (the only single block with CI > 0); 2–5d +0.006, 6–20d +0.007, 21–60d −0.007; Ridge on the four blocks − reversal16 +0.0044 (Holm 0.348) | descriptive / NO EVIDENCE |
| 4 | Cross-sectional representations | None beats per-date z-scores after Holm; best Ridge sector-z +0.0063, 5/5 folds, Holm 0.155 | NO EVIDENCE |
| 5 | Portfolio construction (frozen predictions) | Long-only rules: pooled net Sharpe 0.48–0.95 with beta 1.3–1.6, vs hold-all 0.86 (no costs). Every market-neutral rule (sector long/short, beta-neutral long/short) is −0.80 to −0.00 | descriptive: no neutral tradable signal |
| 6 | Predictable subsets | 1 of 16 high-minus-low tercile contrasts has a CI excluding 0 (Ridge, Amihud). No consistent pattern | descriptive, hypothesis only |
| 7 | Learned hyperedges | The fix (temperature, cluster prior, entropy balance) failed its synthetic test (ARI 0.02); reverted. The v3 soft arm does leave uniform on synthetic group data, so its real-data collapse is consistent with no group signal | negative engineering result |

**Best arm:** `lgbm:rich:residual` has neutral RankIC +0.0127 [+0.0029, +0.0228] and is positive in every fold. Against reversal16 it is +0.0056 [−0.0069, +0.0184] (p 0.39). Against Ridge on raw targets it is +0.0176 [+0.0023, +0.0325] (descriptive, not a predeclared contrast).

## What this means

1. **Model class and features are not the bottleneck.** Ridge, LightGBM and an MLP on 20 conventional features land where a one-line reversal rule does. Volume and liquidity information did not help the tree model.
2. **The target is the lever.** Training against sector- or factor-residual returns improved factor-neutral ranking in almost every fold, for both models. This fits the retrospective's finding that raw-return models learn market and sector exposure. The effect is about 0.01 and inside the noise of a 5-fold design.
3. **Ranking gains do not survive market-neutral portfolios.** Every long-only basket earns roughly what holding the market does, with beta above 1.3. Sector- and beta-neutral long/short versions earn nothing.
4. **Among single lag blocks, only the previous-day return has a CI above zero** (+0.0107). 16-day reversal itself crosses zero, and lags beyond 5 days add nothing. Of the fitted models, only `lgbm:rich:residual` also has a CI above zero.

## Caveats

- **Exploratory:** all five years were inspected before, and the freeze selected the best of 17 arms, so prospective performance is expected to be lower.
- **Dividend look-ahead in dollar-volume levels (found by the final review):** prices are yfinance `auto_adjust` closes, back-adjusted for dividends paid up to 2026. So the *level* features `log_dvol_20` and `amihud_20` carry a future cumulative dividend factor (about −0.4 log over 7 years for a 6% yielder). Ratio and return features are unaffected. The frozen spec uses `log_dvol_20` as a feature and in its residual-target regression, and the subgroup terciles use it. No verdict depends on it, since volume features nominally hurt. The causal unit test checks the code, not the vendor's adjustment. The prospective test inherits the same convention.
- **Survivorship and labels:** current S&P 500 constituents with current GICS sectors. Volume is not verified for split adjustment.
- **Thin hyperparameter search:** a single fixed LightGBM configuration. Early stopping often picked 10–80 rounds because validation neutral RankIC barely rises above 0. A tuned GBM could differ, but the plan deliberately kept tuning equal to Ridge's.
- **Approximate portfolio simulator:** it ignores drift. For the equal-weight rule it matches the screening backtest to within 0.05 Sharpe.
- **Data limits:** no fundamentals, no point-in-time membership, no delisting returns.

## Next

- **Prospective test (`PROSPECTIVE_PROTOCOL.md`):** fetch data after 2026-10-22, run `scripts/featbench_prospective.py` once at least 250 post-freeze days exist, and report INTERIM before that.
- **If more exploratory work is wanted:** the evidence points to targets and neutral evaluation, not models. Two candidates:
  - a factor-residual target combined with reversal-type features at the 1-day lag;
  - an identity-based learned-hyperedge design, which must pass the synthetic test in `fb_HYPEREDGE.md` first.
