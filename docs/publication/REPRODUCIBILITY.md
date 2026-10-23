# E. Reproducibility checklist

## Code

- [x] **Repository:** `src/hypershift/` (model, data, evaluation, screening, featbench, fbaudit) and `scripts/` (one script per analysis).
- [x] **Tests:** `pytest -q` passes (468 tests at the closure review), including:
  - leak and guard tests (`test_freeze_hashes_unchanged`, `test_lf_*`, `test_audit_scripts_parse`);
  - the evaluator anchor test (FB reversal16 reproduces the v3 screening scorer);
  - recompute tests.
- [x] **Headline-number verifier:** `scripts/pub_verify.py`, with output in `docs/publication/CLAIMS_VERIFICATION.md`. 18 of 18 recomputed values match.
- [ ] **Plotting scripts** listed in `FIGURES_TABLES.md` (`pubfig_*`) are not written yet.

## Data (and what can be released)

| Data | Source | Licence / release | Release plan |
|---|---|---|---|
| RSR NYSE/NASDAQ prices and relations | Feng et al. public repository | Public research data, released by the authors | Point to the source with a download script (`scripts/download_rsr.sh`) |
| THINK paper PDF | IEEE | Copyrighted | Do not redistribute; cite only |
| S&P 500 daily OHLCV | yfinance (Yahoo Finance) | Yahoo terms restrict redistribution | Release the fetch script and ticker list, not the prices. Record the fetch date: adjusted values change with each fetch (§6 of the manuscript) |
| Unadjusted closes | yfinance (`auto_adjust=False`) | Same | Same; sha256 of the local file is recorded in the audit docs |
| Alpaca bars 2018–2023 | Alpaca Markets API | Subject to Alpaca terms | Release the script, not the data |
| S&P 500 membership history | github.com/fja05680/sp500 | Check the repository licence before redistribution | Link, with sha256 recorded in A10 |
| Derived outputs (predictions, daily metric series) | This project | Our own | Release where the licence permits derived statistics (daily IC series, metrics). Predictions are arrays of model scores, but check vendor terms on derived data |

## Manifests and seeds

- [x] FB freeze manifest with code sha256 (`docs/featbench_20261022/FREEZE_MANIFEST.json`); v1-LF hash log (`PROSPECTIVE_ADDENDUM.md`, last row).
- [x] **Seeds:**
  - THINK runs: 0–24 / 0–9;
  - FB: 7 (plus 19 and 42 for LightGBM);
  - matched 2×2: 7, 19, 42;
  - synthetic: fixed generator seeds 0–2.
- [x] LightGBM uses `deterministic=True`. Refits reproduced saved series to 3e-9.
- [ ] **Environment lock:** record exact package versions (`pip freeze`) and the GPU/CPU used for THINK runs. Mixed laptop/Kaggle hardware is documented in the tracker.

## Evaluation definitions (state them in the paper's appendix)

- [x] Sharpe: daily top-5 equal-weight, mean/std × √252, std with ddof 0, no risk-free rate, no costs.
- [x] Leak-free epoch selection is on validation Sharpe.
- [x] Neutral RankIC: 5-day return minus beta × equal-weight market, sector-demeaned; per-date Spearman, at least 20 stocks.
- [x] Statistics: stationary bootstrap (block 10, B 5000), pooled as the mean of fold means, Holm within families.
- [x] Book simulator: drift, costed hedge, open1/close1 execution, linear costs, buffer rule.

## Documented limitations (must appear in the paper)

- [x] Development years 2015–2023 were all inspected, and the 2024+ data used once.
- [x] Survivor-conditioned universes; current GICS sectors.
- [x] No borrow, financing or impact costs.
- [x] One synthetic design for learner power.
- [x] THINK settings are inferred; the authors' code is unavailable.
- [x] Errata are listed (BC review, audit corrections).

## Before external release

- [ ] Remove or replace any file that contains redistributable-restricted raw prices from the public repository.
- [ ] Add a top-level `REPRODUCE.md` that maps each table and figure to its command.
- [ ] Run `scripts/pub_verify.py` from a clean clone with the data scripts.

> **Superseded in part (2026-10-22):** see `REPRODUCE.md` for the environment record, the clean-clone procedure and the release audit. That audit found the THINK PDF and the AGPL-derived arrays in git.
