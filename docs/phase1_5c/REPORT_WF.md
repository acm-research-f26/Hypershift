# Phase 1.5c: walk-forward (annual retraining) results, THINK (HH)

2026-10-04. Spec: `docs/phase1_5c/SPEC.md` (predeclared). Analysis: `scripts/wf_analysis.py`; tables `docs/phase1_5c/wf_tables.md` (T1-T8); numbers `docs/phase1_5c/wf_results.json`. Data: Alpaca panel (Phase 1.5b, A2 convention), test years 2019-2023, 5 seeds, HH only. **EH is pending** (kernels `wfe-*` queued 2026-10-04), so the HH vs EH test is not yet run.

## Executive summary

1. **Sharpe vs the market: level, not above.**
   - Pooled 2019-2023 HH Sharpe is 0.57 to 1.18 per seed (mean 0.70). The seed-averaged daily series gives 0.80, 95% CI −0.04 to 1.67.
   - Hold-all on the same stocks and days gives 0.69 (CI −0.21 to 1.60). Every seed's excess-over-hold-all CI includes 0.
   - Net of 25 bp per side, 4 of 5 seeds are negative.
2. **Learned ranking: NO EVIDENCE.**
   - Intersection-union p over seeds: random top-5 0.285, beta-matched 0.323, industry-matched 0.599, label permutation 0.297. Every Holm-adjusted p is 1.0.
   - Mean IC is 0.0018.
   - Only seed 4 beats the nulls; the predeclared rule needs every seed.
3. **Did retraining help vs the frozen 1.5b model? Not detectably.**
   - Same 2019-2023 days: walk-forward 0.80 vs frozen 0.67. Contrast +0.13, 95% CI −0.30 to +0.60, p_boot 0.61.
   - Per-seed signs flip (−0.51, +0.42, −0.50, +0.14, +0.76), and IC is no higher (0.0018 vs 0.0025).

**Bottom line:** retraining every year does not make this THINK reimplementation work. It tracks a high-beta version of the market. This matches the prior stated in the spec (5-10% chance of detectable skill).

## Verification

- 25 runs, all with `metrics.json` and `best_state.pt`, none failed.
- Test dates are exactly each calendar year: 252/253/252/251/250 days, 1,258 total, all within the 1.5b calendar. Test `gt` and `mask` equal the panel and the frozen run on shared days.
- The stable-tie top-5 recomputed from `test_pred` reproduces every saved `test_daily` to 1e-6.
- No test-year target is in train or validation. This is enforced by the split code and `tests/test_alpaca_wf.py`; the per-run config stores only `wf_test_year`.

## Details

| | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|
| HH seed-average Sharpe | −0.40 | 1.59 | 1.85 | 0.34 | 0.51 |
| Hold-all Sharpe | 1.88 | 0.43 | 1.68 | −0.53 | 0.82 |

- **Basket beta** is about 1.25 vs a universe beta of 1.02. The baskets are high-beta: volatility selection, not directional skill. That explains why HH beats hold-all in the 2020 rebound and loses in 2019.
- **Epoch selection is erratic** (selected epochs range from 1-5 up to 87). In 2022, validation Sharpe was 1.8-2.4 while test Sharpe was −0.19 to 0.47, which is consistent with selection acting on noise.
- **Net of costs (10 bp per side):** 0.05 to 0.75 by seed.

## Method notes and limitations

- **Nulls** were drawn per yearly window and concatenated by date, because each year is a different model. Beta comes from each window's own training period. This differs from 1.5b, which used one model and one beta.
- **Alpaca data limits** (from 1.5b):
  - no delisting returns: names are masked after their last bar
  - A2 split convention is approximate
  - MA fill rule is INFERRED
  - symbol-keyed identity, with a reused-symbol gap rule that uses future information for masking only
  - history only from 2016, so the 2019 window trains on about 2 years
- **Graph:** frozen at 2017 relations in every window, by design.
- **Arms:** HH only. HH vs EH (the paper's comparison) is reported when the EH kernels merge.
