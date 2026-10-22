# SDD ledger — plan: docs/superpowers/plans/2026-10-22-feature-benchmark.md
Setup: Ruling: no worktree; work on tom-shlom in place — user's standing practice (commit locally on tom-shlom, never push); branch is not main — cost if wrong: FB commits interleave with other sessions' history.
Setup: Spec = user's pasted proposal (session only) + retrospective doc section H; no repo spec file — rulings provisional against those.
Pre-flight: T2 produces load_fb/target/transform/neutral_rankic/FEATURES -> T3 consumes same names: consistent.
Pre-flight: T3 produces CLI arm grammar model:features:target:horizon:repr; T4-T6 use it; block arms use 'block:<name>' with defaults — Ruling: parse missing fields as defaults (raw,5,cs_z) — cost: none.
Pre-flight: T7 consumes predictions.npz from T4 runs (lgbm/ridge/reversal16): consistent.
Pre-flight: T9 selection consumes pooled neutral RankIC from T4-T6 analysis: consistent.
Task 1: complete (commits 4ca69f6..6737b62, tests: scripts/featbench_audit.py ran; audit + power written; no unit tests in Task 1)
Task 2: Ruling: mdd_60 = close_t / max(close over last 60 sessions) - 1 (drawdown from the 60-day high), not the min of the running drawdown path — simpler, causal, same intent — cost if wrong: one feature differs in definition.
Task 2: Ruling: amihud_20 kept unlogged (heavy-tailed, sd 21k); cs_z clips at +-5 and raw clips at +-10, trees are invariant — cost if wrong: Ridge raw-repr arm sees a clipped outlier feature.
Task 2: Ruling: residual target = per-date OLS on sector dummies (span the intercept) + standardised beta, log_dvol_20, ret_60, vol_60; days with < 30 eligible stocks get 0 — cost: none expected.
Task 2: complete (commits 6737b62..627add2, tests: pytest tests/test_featbench.py → 10 passed)
Task 3: Ruling: run folder = arm with ':' replaced by '__' (Windows paths forbid ':') — cost: none.
Task 3: Ruling: smoke guard checks the root folder's basename for 'smoke' (pytest tmp dirs contain the test name) — cost: none.
Task 3: Ruling: raw/neutral/style RankIC all via screening rank_metrics (same evaluator as v3; anchor test proves identity on reversal 2022/2023) — cost: none.
Task 3: complete (commits 627add2..63477c1, tests: pytest tests/test_featbench.py → 14 passed incl. slow v3 anchor)
Task 4: partial (commits 63477c1..47eabbd): E1+E2 ridge/lgbm/seeds done and analysed; mlp:rich running in background (bue5d2t5l). Hand check lgbm:rich 2021 neutral RankIC -0.016979 == stored. test_paired_matches_manual passes.
Task 5: complete (commits 47eabbd..08205d5, tests: analysis rerun; targets family + horizons table generated; no new code)
Task 4: complete (MLP finished; analysed in 639f235)
Task 6: complete (commits ..639f235, tests: analysis rerun; representations family generated)
Task 7: Ruling: lgbm:rich:residual added to the portfolio/subgroup arms post hoc (highest neutral RankIC); labelled post hoc in the doc — cost if wrong: one extra descriptive row.
Task 7: Ruling: simulator book = mean of last 5 sleeve weights with no drift (approximation); validated against trading() for rule A (|diff| <= 0.05 Sharpe in all 20 fold-arm cells) — cost: small Sharpe bias.
Task 7: Ruling: hold-all benchmark row (equal weight eligible, no cost) added for reading the long-only rules — cost: none.
Task 7: complete (commits ..dea8109, tests: pytest tests/test_featbench.py -m 'not slow' → 18 passed)
Task 8: Ruling: synthetic test redesigned twice (random-walk factor let stocks denoise from own history; fixed 16-day set allowed memorisation) — fresh batches + white-noise factor make pooling necessary — cost: none.
Task 8: Ruling: soft_t (temperature 0.1, noisy cluster prior, entropy-balance 0.1) FAILED test_soft_t_recovers_planted_groups (ARI 0.02 vs needed > 0.5); per plan time-box, code + test reverted (git checkout), diagnostic kept as scripts/featbench_soft_diag.py and docs/featbench_20261022/fb_HYPEREDGE.md; no screening source changed, so v3/X/J config identities are untouched — cost if wrong: the fix idea is lost from code (documented in the doc).
Task 8: complete as a negative engineering result (commit a9d2221; no passing test claimed)
Task 9: Ruling: selection = lgbm:rich:residual (+0.0127) over ridge:lags (+0.0115), gap 0.0012 > 0.001 tie margin; frozen rounds 50 = median(40,20,300,90,50); comparator ridge alpha 1000 — winner-of-17 selection bias disclosed — cost if wrong: prospective spec is not the best true model.
Task 9: Ruling: prospective min days 250 (MDE ~0.046) with INTERIM label before that; primary = frozen - reversal16 — cost: low power disclosed.
Task 9: Ruling: test_score_runs_on_synthetic_post_freeze added (end-to-end score path on synthetic post-freeze dates; real end-to-end impossible before new data) — cost: none.
Task 9: complete (commits ..38612d2, tests: pytest tests/test_featbench.py -m 'not slow' → 22 passed)
Task 10: complete (commits 38612d2..06d8aa9, tests: full pytest -q → 440 passed in 564 s; hand checks: lgbm:rich 2021 neutral -0.016979 == stored; lgbm residual-raw 2020 0.01342 == table)
Final review: subagent (opus) on review-4ca69f6..06d8aa9.diff; no Critical, 1 Important, 6 Minor.
Final: fixed rule D unbounded leverage — test_beta_ls_gross_bounded_with_low_beta_shorts RED (gross 25.0) → GREEN, suite tests/test_featbench.py 24/24 (f92fbbc)
Final: Ruling: re-graded reviewer Minor 2 (report text: 'every effect under 0.021', 'reversal only CI>0') to Important — misstatements in the deliverable — fixed in text (f92fbbc) — cost if wrong: none.
Final: Ruling: re-graded reviewer Minor 1 (dividend back-adjustment leaks future factor into log_dvol_20/amihud_20 levels) to Important-for-disclosure; disclosed as a report caveat, not fixed (needs unadjusted volume/price or ratio-only features; frozen spec keeps the convention) — cost if wrong: frozen spec carries a small look-ahead in two level features during development scoring.
Final: minor (deferred): prospective load_range crashes if the extended panel ends in the first days of a year (load_panel fold coverage) — timing crash only.
Final: minor (deferred): feature cache key omits screening/data.py hash.
Final: minor (deferred): simulator ramps in over the first 4 days of each fold and charges day-1 turnover.
Final: minor (deferred): style_neutral_RankIC is self-scored for residual-target arms (not cited in the report).
