# Phase 1.5c: walk-forward (annual retraining) test, predeclared spec

Written 2026-10-04 by the Claude Code orchestrator, before any training or scoring. The user asked for it ("better to be exhaustive"). Prior expectation, stated in advance: 5-10% chance of detectable ranking skill (2017, the freshly trained case, showed none).

## Question

Does THINK show ranking skill or Sharpe above hold-all when it is retrained every year on the most recent data? The frozen test (Phase 1.5b) only answered this for a model trained once.

## Design (fixed)

- **Data:** the Alpaca panel only (Phase 1.5b, `adjustment=split`, convention A2, the same 1,737-node order, the 1,647 identity-pass nodes, masked after their last bar, MA fill INFERRED as in 1.5b), 2016-01-04..2023-12-29. No RSR splicing. Inputs are relative (scale-invariant), so normalisation scale is irrelevant.
- **Graph:** frozen RSR graph v2 (2017 relations) in every window. Stale by design; a per-year graph rebuild is out of scope (no point-in-time Wikidata).
- **Windows (expanding; target-date splits as in RSR):**

  | test year | train targets | validation targets (epoch selection) |
  |---|---|---|
  | 2019 | first valid target in 2016 .. 2017-12-29 | 2018 |
  | 2020 | .. 2018-12-31 | 2019 |
  | 2021 | .. 2019-12-31 | 2020 |
  | 2022 | .. 2020-12-31 | 2021 |
  | 2023 | .. 2021-12-31 | 2022 |

  2018 is excluded as a test year: it would have only about 200 training windows.
- **Model and settings:** identical to `R5_f3_alpha0_train` (the R5_f2 alpha=0 config: relative inputs, `weight_decay=0`, `alpha=0`, 100 epochs, seq 16, batch_days 8, lr 1e-3), selection on validation Sharpe, `save_weights=true`.
- **Arms and seeds:** HH (THINK) seeds 0-4 first. EH (paper's Euclidean arm) seeds 0-4 when GPU quota allows (next quota week if needed). Analysis runs on whatever is complete; INSUFFICIENT SEEDS is reported honestly.
- **Leakage rules:**
  - No window sees its test year in training or selection.
  - Test-year outputs are scored once per run.
  - Nothing (epochs, settings, seeds, years) is changed after test-year scoring.
- **Primary statistics** (same as 1.5b, for comparability):
  - Per seed, the pooled Sharpe on the concatenated daily top-5 returns of 2019-2023 (never the mean of annual Sharpes).
  - Formal family, Holm: HH vs random daily top-5, beta-matched (beta from each window's own training period), industry-matched, and label-permutation nulls, intersection-union over seeds; HH vs EH when EH exists.
  - `B_NULL 2000`, `B_PERM 500`, `N_BOOT 5000`, stationary block 10.
- **Descriptive:**
  - annual Sharpe; hold-all on the same mask; net of 5/10/25 bp
  - IC, NDCG@5, decile hits
  - **comparison with the frozen 1.5b model on the same 2019-2023 days**
- **Compute estimate:** about 4 GPU-h per seed per arm (training windows grow from about 430 to about 1,440), so about 21 GPU-h for HH. Kernels stay under 12 h each, 2 run concurrently, and launch through `kaggle/queue.txt`.
