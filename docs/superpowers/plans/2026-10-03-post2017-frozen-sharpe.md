# Post-2017 Frozen THINK Sharpe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Test whether a new validation-selected replication of the 2017 R5_f2_alpha0_train/HH strategy retains its daily top-5 Sharpe in 2018-2023, and whether any return exceeds exposure and chance explanations.

**Architecture:** Validate a point-in-time extension of the original 1,737-node NYSE panel before spending GPU time. Train five new 100-epoch seeds with resumable checkpoints, freeze validation-selected weights, then run one locked inference pass on 2018-2023. Analyze by trading date, with per-year diagnostics and pooled daily returns.

**Tech Stack:** Python, NumPy, pandas, PyTorch, pytest; existing RSR loader, THINK loop/evaluator, Kaggle flow and forensic nulls.

**Spec:** docs/phase1_5a/REPORT_2017.md and docs/phase1_5a/PROPOSAL_followups.md, narrowed to the user's post-2017 question. The proposal's 40-epoch estimate changes the historical 100-epoch training horizon; this plan uses 100.

## Revision 2026-10-04 (Claude review; overrides the tasks below where they conflict)

R1. **Training runs in parallel with the data gate.** Training and checkpoint selection use only pre-2017 RSR data. The weights and the freeze manifest are hashed and committed **before any 2018+ return is scored**, so a parallel run cannot leak. The cost is about 7 GPU-h, and the run is useful even if the data gate fails: it gives the 2017 replication, the per-epoch trajectory Phase 1.5a could not recover, and HH-vs-EH with saved weights. The "No GPU training until the data gate passes" constraint is replaced by: **no 2018+ inference or scoring until the data gate passes and the freeze manifest is committed.** The data audit may count coverage and availability for 2018+, but must not compute any strategy, portfolio or model return for 2018+.

R2. **Task 2 is reduced to an opt-in save; cross-kernel resume is dropped.**
- 5 seeds × 100 epochs at about 24 s/epoch is about 3.3 h per arm. That fits one 12 h Kaggle session, and the existing `metrics.json` skip limits a crash to losing one seed.
- A bit-exact interrupted-vs-uninterrupted test is not achievable on GPU (scatter/index_add atomics are nondeterministic).
- New `RunConfig` flag `save_weights: bool = False`. When true, `train_one_run` writes `best_state.pt` (the state_dict at each validation improvement) and `epoch_preds/val_eNNN.npy` / `test_eNNN.npy` (float32 [N,D]) every epoch.
- Defaults are unchanged and existing run folders are untouched.
- Tests:
  - flag off leaves the folder file set unchanged
  - flag on: `best_state.pt` loaded into a fresh model reproduces the saved `test_pred.npy` on CPU (atol 1e-6)
  - the per-epoch file count equals `epochs_run`

R3. **Add the EH arm** (the paper's Euclidean arm, TConv+DHHAN, Table II p. 852), same seeds and settings. HH-vs-EH over 2018-2023 is a predeclared formal comparison: paired by seed, Wilcoxon plus a stationary block bootstrap on the date-joined daily Sharpe contrast, verdict word per CLAUDE.md. Experiment names: `R5_f3_alpha0_train` (labels HH and EH).

R4. **Relative inputs are scale-invariant** (`loop.py:58-62` divides each window by its last close). The pre-2017 price scale is therefore not a compatibility requirement. Required instead:
- prices consistent within every 16-day window and MA30 lookback (no unadjusted split inside a window)
- the RSR return definition (close-to-close price return)
- RSR's missing-value fill semantics (fill 1.1, mask keyed on close)

R5. **The price convention is decided by a predeclared overlap test** on 2015-01-02..2017-12-08:
- Candidates: (a) raw close, (b) split-adjusted close, (c) split+dividend-adjusted close.
- The winner is the candidate with the highest share of matched stock-days where |r_new − r_RSR| < 1e-4. The pass threshold is ≥ 0.95 for that candidate.
- Every 2018+ input, including the warm-up of at least 46 trading days before 2018-01-02, comes from the new source only. Never splice RSR and new-source prices.

R6. **Primary question = ranking skill; Sharpe persistence is secondary.** Phase 1.5a found 2017's Sharpe is explained by high-beta exposure and luck, so the 2018-2023 raw Sharpe will mostly track beta (2018 Q4, 2020, 2022). Primary formal family, Holm-corrected:
- the four 2017-style nulls (random daily top-5, beta-matched, industry-matched, label permutation) on HH
- HH vs EH

Sharpe level, annual and leave-one-year-out results are descriptive.

R7. **The 2017 replication check uses a predeclared tolerance, not equality.** Pass if the new mean HH test Sharpe over the 5 seeds lies inside [min, max] of the historical seeds widened by the historical seed SD. Inference proceeds whether it passes or fails; the report states the result.

R8. **Data source ladder:**
- Confirmatory: CRSP via WRDS, if the user obtains access.
- Until then: a Yahoo pilot over all 1,737 RSR tickers (not only current S&P members). Report per-year coverage, label it **exploratory, survivor-biased**, and quantify the bias. For example, compare 2017-test Sharpe on the covered subset against the full universe: same days, saved predictions, no new data needed.
- Ticker identity checks: name and price continuity over the 2015-2017 overlap (R5). Reject a ticker whose overlap fails.

R9. **Report eligible-name counts per year.** The frozen 1,737-node universe only shrinks: delisted names are masked, and new listings are excluded by design.

## Global Constraints

- The original 2017 weights do not exist. The test concerns a new replication, not the historical run or the authors' implementation.
- Primary arm: HH; alpha=0; norm=train; input_mode=relative; weight_decay=0; original 1,737-node NYSE graph v2; seeds 0-4; 100 epochs; original validation-Sharpe checkpoint selection. Copy every other setting from recorded R5_f2_alpha0_train/HH configs and hash resolved configs.
- Keep original pre-2017 train and validation periods. Neither 2017 nor 2018-2023 returns may affect training, checkpoint selection, settings or seed inclusion. The 2017 result is a replication check; 2018-2023 is the locked holdout.
- Preserve ticker order, node indices, MA5/10/20/30/close feature order, target timing, top-5 equal weighting, daily rebalance, tie rule and Sharpe mean/std(ddof=0) times sqrt(252).
- Never write into results/R5_*. A new experiment owns its artifacts. metrics.json remains the completed-run marker.
- The local data/fresh/sp500_daily is not the primary extension: it follows current S&P membership, adjusted Yahoo prices and different filtering. Only 282 tickers exactly overlap the 1,737-name NYSE roster; historical close ratios drift on most matches. It can only support a clearly labeled survivor-only pilot.
- No GPU training until the data gate passes. If the overnight driver is active, use kaggle/queue.txt and a completed smoke kernel for launches.
- Walk-forward retraining and geometry controls remain separate decisions.

## Review Focus

- Reused ticker or merger: a mapping must reject spliced securities.
- Delisting: include terminal return and trading status; never invent a zero or backfilled return.
- Revised adjusted price: overlap and causal-feature tests must catch changes in frozen-era inputs.
- Mid-seed kernel stop: a new session must import its partial checkpoint and match uninterrupted training.
- Missing date in one return series: comparisons must fail explicit date alignment, never align array tails.

---

### Task 1: Point-in-time data compatibility gate (CPU)

**Files:** Create docs/phase1_5b/DATA_COMPATIBILITY.md, scripts/post2017_data_audit.py, tests/test_post2017_data.py; add a narrow loader under src/hypershift/data/ only after its contract is fixed.

**Interfaces:** Accepted loader returns MarketData in original RSR ticker order plus an immutable manifest recording source/as-of, security identity map, corporate-action and delisting convention, graph hash, training-era normalization and dates.

- [ ] Inventory original ticker identities, graph v2 and train normalization. Record annual coverage and why the existing S&P panel is not the primary source.
- [ ] Check available data access first (the user is unsure). CRSP/WRDS is a candidate because its official US Stock Databases documentation describes permanent security IDs, daily data, corporate actions and delisting information; an equivalent source is acceptable. Obtain legally usable bars for the original NYSE identities through 2023. If unavailable, stop here and report the blocker before GPU work.
- [ ] Write DATA_COMPATIBILITY.md to fix identity, close/adjusted-close, dividends/splits, terminal returns, eligibility, feature construction, pre-2017 scale, calendar and the primary 2018-2023 window before scoring returns.
- [ ] Write synthetic tests for ticker reuse, delisting, no future-data effect on earlier features, exact ticker/graph order, fixed train scale and missing-day masking. Run .venv/Scripts/python.exe -m pytest tests/test_post2017_data.py -q and confirm they fail before implementation.
- [ ] Implement loader and audit. Reconstruct 2015-2017 overlap and compare returns and all five features with RSR. Document irreducible differences. Pass only if identity, return semantics, causal features and graph mapping are verified.

### Task 2: Checkpoints and cross-kernel resume

**Files:** Modify src/hypershift/train/loop.py, kaggle/run_kaggle.py, the local partial-run merge path, and add tests/test_checkpoint_resume.py plus a new Kaggle preset. Existing run folders stay untouched.

**Interfaces:** An opt-in run writes an atomic latest checkpoint with model/optimizer, next epoch, best validation score/epoch/weights, patience, history, training-offset order and Python/NumPy/Torch CPU/CUDA random states; best-validation weights, per-epoch predictions/logs, and config/data/graph hashes.

- [ ] Write failing synthetic interrupted-versus-uninterrupted tests for identical final weights, best epoch, history and predictions. Cover partial zip import and existing metrics.json completion skip.
- [ ] Add epoch-boundary checkpoint save/load and append/reconcile history. Refuse resumes with mismatched config, data or graph hashes.
- [ ] Change Kaggle prior-result extraction and local merge to carry incomplete checkpoint folders while protecting complete runs; verify the final zip retains partial artifacts.
- [ ] Run targeted tests and tests/test_loop.py. Prove a one-epoch smoke can be interrupted and resumed across two sessions before a full launch.

### Task 3: Five-seed replication and freeze

**Files:** A fresh results experiment, a Kaggle preset/smoke, and docs/phase1_5b/FREEZE_MANIFEST.md.

**Interfaces:** Five completed 100-epoch runs, each with best-validation weights; a frozen manifest of selected epochs and config/data/graph/checkpoint hashes.

- [ ] Run one-epoch HH smoke and check GPU memory, epoch progress and artifact integrity.
- [ ] Run seeds 0-4 for 100 epochs through the queue. At observed 23-24 seconds per epoch, allow about 3.3 serial-equivalent GPU-hours of model time plus smoke, startup and download overhead; Kaggle session-quota use depends on measured parallelism. Reassess if actual rate materially differs.
- [ ] Verify five completed runs, exact resolved configs, no failed seeds, correct resume and selection solely on pre-2017 validation Sharpe. Report 2017 replication metrics only as a sanity check.
- [ ] Freeze selected weights and all manifests before later-year inference. No later-year result may change checkpoints or exclude a seed.

### Task 4: Locked 2018-2023 inference

**Files:** Create scripts/eval_post2017.py and tests/test_post2017_eval.py; write date-keyed arrays/tables to separate results/post2017_frozen/.

**Interfaces:** Load only best-validation weights. Use existing predict_split/evaluate_all semantics with the accepted panel and frozen graph. Emit date, seed, predictions, eligibility, targets, chosen names and daily returns.

- [ ] Write failing tests proving that future returns cannot change checkpoint choice; permuted tickers, missing dates and fictitious delisting returns are rejected.
- [ ] Run one inference pass for 2018-01-01 through 2023-12-31 bounds, using actual trading dates and late-2017 feature warm-up. Preserve top-5 rule, and record eligible-name count and ties daily.
- [ ] Check hand-calculated stock-day and portfolio-day examples against raw bars; run targeted tests and freeze inference output hashes.

### Task 5: Persistence and ranking evidence (CPU)

**Files:** Create scripts/post2017_analysis.py, tests/test_post2017_analysis.py and docs/phase1_5b/REPORT_POST2017.md.

**Interfaces:** Date-keyed artifacts only. Resample common trading-day blocks across seeds and baselines. The five seeds are repeated training runs, not five independent market samples.

- [ ] Write tests for exact date joins, pooled Sharpe from concatenated daily returns rather than mean annual Sharpes, common 10-day stationary-block bootstrap, cost accounting and deterministic null draws.
- [ ] Report per-seed pooled 2018-2023 gross Sharpe, day-block 95% intervals, annual Sharpe and day counts, and a predeclared average of the five daily seed-return series. Include leave-one-year-out pooled Sharpe and 2017 versus later-year estimates; a non-significant decline is not proof of equality.
- [ ] Compare on identical dates and eligibility masks with equal-weight hold-all, fixed random baskets and random daily top-5. Report daily excess over hold-all with paired-block intervals. Show gross and 5/10/25 bp-per-side net Sharpe under the existing turnover convention.
- [ ] Separately test ranking skill using the four predeclared 2017-style nulls: random daily top-5, training-beta matched, industry matched and label permutation. Use empirical upper-tail p values, common date blocks and Holm correction. Report IC, NDCG@5, decile spread, beta/industry exposure, ties and concentration as diagnostics.
- [ ] Report two distinct conclusions: measured Sharpe persistence and evidence, if any, of learned ranking. Document the data population and remaining source limitations.
- [ ] Run targeted tests and the full suite; verify frozen weights, configs and input/output hashes remained unchanged.

## Decision after the report

If data compatibility fails, stop before GPU training. If the frozen model loses Sharpe, that answers the frozen-model question but does not rule out a periodically retrained strategy. If Sharpe remains high but matched nulls or costs explain it, report persistence without a skill claim. Consider the separate approximately 20 GPU-hour walk-forward experiment only if its adaptive-strategy question remains worth asking.
