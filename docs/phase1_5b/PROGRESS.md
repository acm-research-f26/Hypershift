# Phase 1.5b progress (one line per finished step)

- A1 done: RunConfig.save_weights (default False) + train_one_run writes best_state.pt / epoch_preds/{val,test}_eNNN.npy; tests/test_save_weights.py (3 tests) pass with tests/test_loop.py.
- A2 done: presets r5f3h, r5f3h-s, r5f3e, r5f3e-s in kaggle/run_kaggle.py, launch.sh, README; config verified identical to R5_f2_alpha0_train HH/EH except exp and save_weights.
- A3 prepared: kaggle/queue.txt = r5f3h-s, r5f3e-s, r5f3h, r5f3e; driver dry-run after --reset only wants r5f3h-s (old kernels/analysis flagged done, not relaunched).
- B1 done: docs/phase1_5b/DATA_COMPATIBILITY.md spec fixed (sections 1-8) before any overlap statistic. Smoke kernels r5f3h-s, r5f3e-s launched via driver (RUNNING at 00:44).
- B2 done: src/hypershift/data/post2017.py + tests/test_post2017_data.py (15 tests) + scripts/post2017_data_audit.py; Yahoo download (1123/1856 symbols) and audit run; outputs docs/phase1_5b/post2017_audit_{results.json,tickers.csv}.
- B3 done: gate verdict in DATA_COMPATIBILITY.md section 9 = PASS exploratory/survivor-biased (pooled R5 share 0.9705; 967/1737 nodes; condition: use genuine-split-adjusted close, not raw).
- A3 done: smokes r5f3h-s/r5f3e-s were no-ops (est_min guard, fixed a16e21b); r5f3e-s2 smoke verified on Kaggle (best_state.pt + epoch_preds; CPU reload matches test_pred to 6e-8). Full kernels hypershift-run-r5f3h (launched 00:49) and hypershift-run-r5f3e (01:05) RUNNING; driver merges them. Expect ~3.5 h each.
- B4 done: Alpaca second source. Amendment A2 (648e517) fixed before any statistic; src/hypershift/data/alpaca.py, tests/test_alpaca_data.py, scripts/alpaca_data_audit.py; results in DATA_COMPATIBILITY.md section 10. Identity 1647/1737 (pooled R5 split 0.977, raw 0.993, all 0.655); eligible 1636 on 2018-01-02 -> 1229 at 2023 end (418 ended); R8: Alpaca-eligible 1.964 vs full 1.977 vs survivors-only 2.520. Verdict PASS; recommended 2018-2023 source = Alpaca (split). No 2018+ return computed.
