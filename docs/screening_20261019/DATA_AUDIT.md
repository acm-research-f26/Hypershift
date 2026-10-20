# Data audit — HyperShift broad screen (2026-10-19)

Exploratory study. No dataset here reproduces Rishi's Nasdaq-100 reconstruction; results are **not comparable** to `feat-rs` metrics.

## 1. What exists locally (inspected before any download; nothing was downloaded)

| Source | Universe | Period | Fields | Used? |
|---|---|---|---|---|
| RSR NASDAQ (`data/raw/rsr/data`) | 1,026 qualified stocks | 2013-01-02 → 2017-12-08 | MA5/10/20/30 + close (RSR preprocessing) | **v2 screen** (underpowered; diagnostic only) |
| RSR NYSE | 1,737 stocks | same | same | not used (same short calendar) |
| Alpaca post-2017 (`data/raw/alpaca_post2017`) | 1,647 identity-pass RSR stocks | 2016 → 2023 | OHLCV, unadjusted | **rejected** (integrity gate, §4) |
| Yahoo post-2017 (`data/raw/yahoo_post2017`) | 1,123 of 1,856 RSR NYSE symbols with data | 2014-11 → 2023 | OHLCV, adjusted | not used (survivors-only subset of RSR NYSE; no sector labels) |
| S&P 500 daily (`data/fresh/sp500_daily`, yfinance) | 462 current constituents, GICS sector/sub-industry | 2015-01-02 → 2026-08-31 | OHLCV, split/dividend-adjusted | **v3 primary screen** |
| S&P 500 hourly (`data/fresh/sp500_1h_*`) | ~460 | 2018 → 2026 | hourly bars | not used (daily study) |
| Rishi's Nasdaq-100 data / checkpoints | — | — | — | **unavailable** |

## 2. Primary panel (v3): `data/fresh/sp500_daily/screen_sp500_2015_2026.npz`

- Built by `scripts/prepare_screen_sp500.py` from `panel.npz` (sha256 recorded in `SP500_INPUT_MANIFEST.json`, together with `bars.csv.gz`, `tickers.txt` and `universe.csv`). Assembled panel sha256 `4b0d10dc67ba…`.
- **Shape:** 462 stocks × 2,932 sessions × 5 features. The features are MA5, MA10, MA20, MA30 and close, all from adjusted closes. They are the same five features as RSR/THINK.
- **Normalisation:** the stored features are divided by a per-stock train-period max close. The screening loader then turns each 64-session window into ratios against the signal-day close and divides by **training-only** feature standard deviations. The per-stock scalar cancels exactly, so absolute normalised prices never enter a model.
- **Missingness:** the observed fraction is 0.9997. Gaps are concentrated in 2015, where a few constituents have short histories. Every arm requires 64 consecutive observed sessions, and labels require the whole 5-session path to be observed.
- **Labels:** close-to-close return over the next 5 sessions. Max |h5| = 1.18; 5 labels exceed 100% (NCLH 2020-03-18; ECHO 2025-08-20..25, the EchoStar spectrum sale); 130 exceed 50%, almost all in the March–April 2020 crash. These are real events, so no clipping or repair was applied.
- **Sectors:** current GICS sectors, 11 groups. Sizes: CommSvc 19, ConsDisc 44, Staples 32, Energy 20, Financials 73, Health 57, Industrials 71, IT 66, Materials 23, RealEstate 28, Utilities 29.

### Folds (v3)

| Fold | Train signals (label matured before) | Validation (checkpoint selection) | Evaluation (daily) | Eligible stocks |
|---|---|---|---|---|
| 2022 | 1,443 dates (2020-12-31) | 247 dates of 2021 | 246 dates of 2022 (bear year) | 458–462 |
| 2023 | 1,695 dates (2021-12-31) | 246 dates of 2022 | 245 dates of 2023 (bull year) | 458–462 |

**2024-01-01 onward is reserved** from scoring and selection. It is not a fresh holdout, because an earlier POC on this panel tested 2024-05 onward.

## 3. Causality checks (enforced in code; unit-tested in `tests/test_screening.py`)

- The panel is truncated at the end of the evaluation year **before** any feature, label, statistic or relation is built.
- Training labels mature strictly before the validation year, and validation labels mature before the evaluation year (`test_split_labels_mature_before_boundary`, `test_train_stride_one_keeps_maturity_and_adds_signals`).
- A label at t uses closes t..t+5 and is marked invalid if any close on that path is missing (`test_five_session_target_and_path_mask`).
- Input scalers and the target scale come from training signals only. The 16 correlation clusters come from KMeans on training-period daily returns only.
- Trailing beta uses ≤126 observed daily returns up to and including the signal day. Residual targets use the realised future market and sector return only inside matured labels.
- The rolling-correlation graph uses only the 64-session input window.
- Eligibility to trade uses input history only, never future label availability.

## 4. Rejected: Alpaca 2016–2023

The h5 MSE was dominated by unadjusted reverse splits: SPN apparent 5-session gains of 31.4–32.7×, and CUZ 2.93–3.17× in 2019. SEC filings confirm corporate actions on those dates. Guessing split factors was ruled out, so the panel was rejected, and its pilot outputs (`results/SCREEN_20261019_v1`) are excluded.

## 5. Underpowered: RSR NASDAQ v2

- Fold 2015 has **37** training dates (2013 only, stride 5). Fold 2016 has **88**.
- In completed fits, the epoch selected by validation MSE is 0–3 for almost every graph arm, and the validation prediction std is 1–7% of the target std, so these models are close to initialisation.
- v2 is retained as a diagnostic of the *training-data/budget* failure mode (see `DIAGNOSTIC_REPORT.md`), not as mechanism evidence.

## 6. Biases and limits

- **Survivorship bias (strong):** 2026 constituents are traced back to 2015. Absolute returns, Sharpe and momentum-type signals are inflated; comparisons between arms on the same panel stay like-for-like.
- **Sector labels are current**, not point-in-time.
- **No delisting returns.** Held positions with missing marks are counted and given a zero return.
- **Trading is optimistic:** entry and marking at the signal-day close, and 10 bp one-way costs, with no impact or borrow modelling.
- **Re-inspected years:** the project has looked at these years before, so nothing here is confirmatory.

## 7. Foundation-model provenance (Experiment J)

- The S&P 500 bars contain OHLCV, so Kronos' input requirement **is** met on this panel (it was not on RSR, which has only MA/close).
- Kronos (arXiv 2508.02739, Appendix D, "Forecasting Task Setup") states that "The pre-training data for Kronos extends up to June 2024", and lists NASDAQ (XNAS) among its in-distribution exchanges.
- The 2022 and 2023 evaluation years therefore fall **inside** Kronos' pretraining window, so frozen Kronos features would be contaminated on the main folds. A leak-free J test needs evaluation dates ≥ 2024-07, which lie inside this study's reserved period.
