# Phase 1.5b progress (one line per finished step)

- A1 done: RunConfig.save_weights (default False) + train_one_run writes best_state.pt / epoch_preds/{val,test}_eNNN.npy; tests/test_save_weights.py (3 tests) pass with tests/test_loop.py.
- A2 done: presets r5f3h, r5f3h-s, r5f3e, r5f3e-s in kaggle/run_kaggle.py, launch.sh, README; config verified identical to R5_f2_alpha0_train HH/EH except exp and save_weights.
- A3 prepared: kaggle/queue.txt = r5f3h-s, r5f3e-s, r5f3h, r5f3e; driver dry-run after --reset only wants r5f3h-s (old kernels/analysis flagged done, not relaunched).
