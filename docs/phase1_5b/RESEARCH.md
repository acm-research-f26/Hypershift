# Post-2017 feasibility research (2026-10-03)

This note records read-only findings from three independent subagent reviews of data compatibility, checkpoint/resume, and statistical evaluation. No post-2017 inference or training was run.

## Data

- The historical RSR NYSE panel has 1,737 ordered ticker nodes and ends 2017-12-08 (docs/phase1_5/C_data_graph.md; src/hypershift/data/rsr.py). Its five channels are MA5/10/20/30 and close; norm=train uses the pre-validation training maximum.
- The local data/fresh/sp500_daily panel has 462 names and 2,932 dates (2015-01-02 through 2026-08-31), including 1,509 dates in 2018-2023. Only 282 tickers exactly match the RSR NYSE roster. The other daily panel starts in late 2023. These counts came from local panel inspection.
- The fresh path starts from current S&P 500 members, filters on whole-period missingness, uses Yahoo auto_adjust=True, computes its own moving averages and chooses its own 60%-timeline normalization (src/hypershift/data/fresh.py:20-39,103-130; scripts/fetch_fresh.py:27-39). The graph is not keyed to this reduced/sorted panel.
- On 741 overlapping 2015-2017 dates for the 282 ticker matches, the fresh-to-RSR close ratio is usually not constant: median standard deviation of log ratio 0.0196; only 29/282 are below 0.001. Simple price rescaling cannot splice these panels. Ticker equality alone also does not prove security identity across renames/reuse.
- Result: no credible full-universe 2018-2023 test can be run on local data as-is. A current-survivor subset would answer a different, biased question and must be labeled that way.

A possible source is CRSP US Stock Databases, whose official documentation describes daily data, permanent security IDs, corporate actions and delisting information (https://www.crsp.org/research/). Access is not established; the user is unsure. Mapping the RSR ticker roster to permanent IDs still requires a checked identity crosswalk.

## Training artifact and cost

- The original results contain predictions, daily returns, metrics and histories but no model weights. train_one_run in src/hypershift/train/loop.py:187-271 has no state_dict save. A later-year frozen inference requires a new replication.
- The historical alpha=0 HH runs each used 100 epochs; selected epochs were 8, 32, 0, 1 and 2. The follow-up proposal's five x 40 epochs changes the training/selection horizon. At observed 23-24 seconds per epoch, five x 100 is about 3.3 serial-equivalent GPU-hours before smoke/startup/download overhead.
- An exact resume must preserve latest model and optimizer, best-validation weights/score, patience, epoch and history, shuffled train offsets, and Python/NumPy/Torch RNG states. Kaggle's existing prior-output extraction and local merge discard incomplete run folders (kaggle/run_kaggle.py:128-155; scripts/overnight/merge_zip.py:20-32); both need changes before cross-session resume works.

## Evaluation rule

- Freeze pre-2017-validation-selected weights and manifests before seeing 2018-2023 returns. Use the original ordered graph and inference semantics. The later-year data cannot choose an epoch, seed subset or setting.
- Report per-seed and predeclared seed-ensemble Sharpe on the concatenated daily 2018-2023 series, plus annual and leave-one-year-out diagnostics. Do not average annual Sharpes or treat seeds sharing the same days as independent observations.
- Compare on identical dates with hold-all and random/matched baskets, use paired trading-day block uncertainty and 5/10/25 bp per side cost sensitivity. The four 2017-style nulls address ranking skill separately from Sharpe persistence.
- Existing sharpe_contrast_ci truncates to the last common length rather than joining on date (src/hypershift/eval/stats.py:42-52). New analysis must align by explicit dates.

## Decision

The CPU data-compatibility gate is worth doing first. Do not launch the replication until historical security identities, delisting/corporate-action treatment, compatible bars/features, and graph order are verified. The implementation plan is docs/superpowers/plans/2026-10-03-post2017-frozen-sharpe.md.

