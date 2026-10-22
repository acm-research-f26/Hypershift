# Study FB prospective protocol (predeclared 2026-10-22, before any post-freeze data)

This is the confirmatory step for Study FB. Everything before it is exploratory. Machine-readable version: `FREEZE_MANIFEST.json`. Code: `src/hypershift/featbench/prospective.py`, `scripts/featbench_prospective.py`, with source hashes recorded in the manifest.

## What is frozen

- **Specification:** `lgbm:rich:residual`. LightGBM on the 20 Study FB features (per-date z-scores), trained on 5-day factor-residual returns, with 50 boosting rounds and the other parameters as in the plan.
  - **Why this one:** the predeclared rule picks the highest pooled neutral RankIC over 2019–2023 among the h = 5 Ridge/LightGBM arms, which gives +0.0127 against +0.0115 for `ridge:lags`.
  - **Expect a lower value prospectively.** It is the best of 17 arms on years we have inspected, so selection bias favours it.
- **Comparators:**
  - `ridge_rich`: Ridge on the same features, raw target, alpha 1000.
  - `reversal16`: minus the 16-day return.
  - `random`.
- **Hyperparameters** are the medians of the values selected in 2019–2023. The retrain has no validation step, so no FB metric is ever computed before the freeze.

## Data

- **Fetch:** extend `data/fresh/sp500_daily` past 2026-10-22 with the existing pipeline (`scripts/fetch_fresh.py --source yf --kind daily`, then the panel and OHLCV builders used for v3 and J), keeping the same 462-stock universe and format.
- **Train** on every signal whose 5-day label matures on or before 2026-10-22.
- **Score** only signal dates after 2026-10-22 whose labels have matured. The code refuses earlier dates.

## Primary test (run once)

- **Contrast:** frozen − reversal16 on daily factor-neutral RankIC (5-day return minus beta × market, sector-demeaned).
- **Test:** two-sided stationary bootstrap, block 10, B = 5000.
- **SUPPORTED** if p < 0.05 and the 95% CI excludes 0. Otherwise NO EVIDENCE.
- **When:** once, after at least **250** post-freeze signal days (about 2027-10). Earlier runs are labelled INTERIM and are descriptive only.
- **Detectable effect** at 250 days is about **0.046** neutral RankIC (`POWER.md`). The development effect of the frozen spec over reversal16 was about +0.006, so a NO EVIDENCE outcome is likely even if the effect is real. Detecting 0.02 needs about 1,300 days. The test can therefore rule in only a large effect; it cannot rule out a small one.
- **Secondary (descriptive):** frozen vs 0; frozen − ridge_rich; raw RankIC.

## What may not change after this commit

The spec, the parameters, the comparators, the primary contrast, the rule, the minimum days and the universe. A change requires a new, dated protocol and resets the clock.
