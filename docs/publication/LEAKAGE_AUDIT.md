# Audit of "leak-free" claims

**Rule adopted for the revised manuscript.** The word "leak-free" is removed. Each experiment is described by which of the five information-timing properties below it satisfies:

| Code | Property | "Satisfied" means |
|---|---|---|
| **E** | Epoch / model selection | Selection uses validation data only, never the test year |
| **N** | Normalisation | Any scaling uses statistics from the training period only |
| **F** | Features | Every input at t is computable from data available at t (in particular, no level feature from back-adjusted prices) |
| **Y** | Labels | Labels may use post-t data by design, as legitimate ex-post construction; what matters is that features and labels do not overlap in time |
| **U** | Universe and classification | Eligibility and sector labels at t use only information available at t (no survivorship, no current-classification look-back) |

## Status by experiment

| Experiment | E | N | F | Y | U | Correct description |
|---|---|---|---|---|---|---|
| R5_g2, original THINK reproduction (2017) [T1] | ✓ validation-selected (test-selected reported only as a diagnostic) | ✗ full-series max normalisation (`norm: paper`) | **Unknown.** RSR prices come from the authors' files; their adjustment convention is not documented by us. Features are ratios of normalised closes and moving averages | ✓ | ✗ RSR universe = stocks trading through 2013–2017 (survivor-conditioned) | "validation-selected, full-series normalisation, survivor universe" |
| R5_f, corrected reproduction (2017) [T3] | ✓ | ✓ for `norm=train` / ✗ for `norm=paper` (both reported) | Unknown (as above) | ✓ | ✗ (as above) | "validation-selected; train-period (or full-series) normalisation; survivor universe" |
| Phase 1.5b/c, NYSE 2018–2023 [T6] | ✓ frozen weights (1.5b); annual retraining with validation (1.5c) | ✓ train-period | ✓ Alpaca split-only adjustment (no dividend back-adjustment); returns exclude dividends | ✓ | Partly: RSR universe chosen on 2013–2017 availability; masked after delisting, so no survivorship *after* 2017 within that universe | "frozen / walk-forward, split-adjusted prices, RSR-defined universe" |
| Research screen v3, Studies X and J (S&P 500) | ✓ | ✓ per-date or train statistics | ✓ return-type features | ✓ | ✗ current constituents and current GICS | "validation-selected, causal return features, survivor universe" |
| Study FB as run (2019–2023) | ✓ | ✓ per-date cs_z | ✗ `log_dvol_20` and `amihud_20` from back-adjusted levels | ✓ | ✗ | "contains adjusted-level features (contaminated)" |
| Audit refits, "corrected" (A5, A2c) | ✓ | ✓ | ✓ unadjusted close × volume (fallback to adjusted for 2 stocks with no consistent raw series) | ✓ | ✗ | "adjustment-corrected features, survivor universe" |
| Matched 2×2 (A11), economics (A8, A13) | ✓ (fixed rounds, no selection) | ✓ | ✓ as corrected | ✓ | ✗ | as above |
| Prospective v1 | ✓ frozen before data | ✓ | ✗ adjusted levels (vendor convention retained) | ✓ | ✓ for post-freeze dates (universe fixed at the freeze; nothing is selected on post-freeze survival) | "frozen; contains adjusted-level features" |
| Prospective v1-LF | ✓ | ✓ | ✓ unadjusted levels; missing raw → ineligible | ✓ | ✓ for post-freeze dates | "frozen; adjustment-corrected" (renamed from "leak-free") |

## Consequences for wording

- **Rename.** "Leak-free Sharpe" becomes "validation-selected Sharpe". Where the run used full-series normalisation, say so in the same sentence.
- **Rename.** "Leak-free features / refits" becomes "adjustment-corrected features / refits".
- **Rename.** "v1-LF" is described as the "adjustment-corrected" prospective run. The file and code names are kept for traceability.
- **Survivorship (U) is never satisfied** for the S&P 500 development studies and the RSR universes. Every development result is therefore conditional on that universe.
- **The RSR feature adjustment convention is unknown.** State it as a limitation.
