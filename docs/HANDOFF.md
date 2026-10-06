# HANDOFF: live state for Claude Code and Codex

**Last updated: 2026-10-05 by Claude Code.** Phase 2 closure done: see `docs/phase2/REPORT_CLOSURE.md`. Nothing running.

## Read in this order

1. This file.
2. `CLAUDE.md`: environment, commands, architecture, pitfalls. It applies to Codex too.
3. The active phase's plan and progress log (see **Running now**).
4. Background, only if needed:
   - `docs/PHASE1_TRACKER.md` (full record and verdicts)
   - `docs/phase1_5/PHASE1_5_SUMMARY.md`
   - `docs/phase1_5a/REPORT_2017.md`

The old long-form handoff (2026-09-28, pre-weight-decay fix, historical only) is `docs/phase1/HANDOFF_2026-09-28.md`.

## Where we are

- **Phase 1 (reproduce THINK, ICDM 2022): done.**
  - Final ruling (2026-10-03): the paper's ranking advantage is **not reproduced under validation selection** in this inferred-settings reimplementation.
  - Corrected full-NYSE THINK vs TConv+DHHAN (EH) is NO EVIDENCE (10 seeds); R8_f baselines are INSUFFICIENT SEEDS.
- **Phase 1.5 (fidelity audit): done.** Root cause of the old collapse: coupled `weight_decay=5e-4`. Fix: `weight_decay=0` plus `input_mode=relative` (`docs/phase1_5/F_learnability.md`).
- **Phase 1.5a (2017 Sharpe≈2 forensics): done.**
  - The alpha=0 THINK 2017 Sharpe of about 2 is not distinguishable from random daily top-5 baskets in a strong year: hold-all 1.53, high-beta baskets.
  - NO EVIDENCE of learned ranking. Power is low, so D (top-tail skill) is unsupported, not excluded.
  - Report: `docs/phase1_5a/REPORT_2017.md`.
- **Phase 1.5b (post-2017 test): in progress.** It tests a new replication on 2018-2023.
  - Plan: `docs/superpowers/plans/2026-10-03-post2017-frozen-sharpe.md`; its **Revision R1-R9 section overrides the tasks**.
  - Progress log: `docs/phase1_5b/PROGRESS.md`. Research notes: `docs/phase1_5b/RESEARCH.md`.
- **Phase 2 (exhaustiveness closure): in progress** (`docs/phase2/SPEC.md`, log `docs/phase2/PROGRESS.md`). W2 STHGCN blocked (data gone). W1 done: the authors' RSR-I code reproduces the RSR paper's NYSE MSE/MRR/IRR under the predeclared rule (`scripts/rsr_paper_compare.py`). W3 done: `scripts/protocol_matrix.py` -> `docs/phase2/protocol_matrix.md` (88 cells; leaky cells marked). W4 closure report done. W6 (author code study, `docs/phase2/W6_AUTHOR_CODE.md`): sibling repos only print test every epoch (no selection), Sharpe x15.87, last-day NDCG bug, fixed seed; THINK repo empty; authors' delta sampler gives NASDAQ 1.0 (match), NYSE 1.0 (paper 0.5). W5 done (Table I hyperbolicity grid): `docs/phase2/W5_DELTA_RESULTS.md`; exact delta_hg of our App. B graph is 1.5 on both markets, delta_rel matches only one fragile setting; Table I not robustly reproduced.

## Running now

- **Phase 2 W2 (STHGCN official code): BLOCKED, nothing running.** Data Drive ids are 404 and `hypergraph.npy` is not published; see `docs/phase2/W2_STHGCN.md`. Needs a copy of the data from the user to proceed.
- **Nothing running on Kaggle.** Phase 1.5c is done (HH and EH, 25 runs each). Laptop GPU queue is separate (see its own log).
- **Phase 1.5c HH results (`docs/phase1_5c/wf_tables.md`, `wf_results.json`, script `scripts/wf_analysis.py`; REPORT_WF.md text was not written by the worker, see PROGRESS):** pooled 2019-2023 HH Sharpe per seed 0.574/0.657/0.535/0.564/1.181 (mean 0.702, seed-avg 0.800, CI -0.04 to 1.67); hold-all 0.693. Ranking skill NO EVIDENCE (IUT p F1 0.285, F2 0.323, F3 0.599, F4 0.297; Holm 1.0; only seed 4 beats the nulls). IC mean 0.0018. WF vs frozen 1.5b on the same days: seed-avg +0.127 Sharpe (CI -0.30 to 0.60, p_boot 0.61), not distinguishable. EH (paper's Euclidean arm): pooled Sharpe 0.457/0.794/0.453/0.273/0.588 (mean 0.513, seed-avg 0.587), IC about -0.003. F5 HH vs EH: Wilcoxon p 0.3125, seed-avg contrast +0.213 (CI -0.19 to +0.60), Holm 1.0, verdict INSUFFICIENT SEEDS (descriptively NO EVIDENCE). Report: `docs/phase1_5c/REPORT_WF.md`.

- **Task 5 done:** `docs/phase1_5b/REPORT_POST2017.md` (tables `post2017_tables.md`, `post2017_results.json`). Conclusions: Sharpe did not persist (HH pooled 2018-2023 mean 0.43, seed-avg 0.45, hold-all 0.51, EH 0.16); ranking skill NO EVIDENCE (Holm-adj p >= 0.625; F1-F4 IUT p 0.73-0.87); HH vs EH INSUFFICIENT SEEDS (seed-avg contrast +0.27, CI -0.13 to 0.69). Phase 1.5b complete; next is the user decision on CRSP / walk-forward retraining.
- **Task 4 done (locked 2018-2023 pass):** outputs `results/post2017_frozen/` (hashes in FREEZE_MANIFEST addendum A). Gross top-5 Sharpe HH seeds 0-4: 0.85/0.10/0.80/0.23/0.19 (mean 0.43); EH mean 0.16. Task 5 (analysis, REPORT_POST2017.md) next.
- **Task 3 done:** r5f3 merged (10 runs verified). `docs/phase1_5b/FREEZE_MANIFEST.md` committed. R7 PASS (new HH mean 2017 SR 2.200 in [1.771, 2.254]); EH new 1.463 vs historical 2.049 (descriptive). Task 4 (locked 2018-2023 inference) and Task 5 follow.

- **No agent is active** (2026-10-04 ~01:40). The Phase 1.5b worker finished Track A and Track B; see `docs/phase1_5b/PROGRESS.md`.
- **Kaggle** (driver fetches, merges, pushes):
  - `hypershift-run-r5f3h` (launched 00:49) and `hypershift-run-r5f3e` (launched 01:05), each HH/EH × 5 seeds × 100 epochs, exp `R5_f3_alpha0_train`, `save_weights=true`. ETA about 04:30-05:00 (unmeasured).
  - A real smoke (`r5f3e-s2`) verified on Kaggle that `best_state.pt` reloads to the saved test predictions (5.6e-8).
  - About 340 MB per run, about 1.7 GB per arm zip.
- **Overnight driver** (`Hypershift_overnight`, every 30 min, log `results/logs/overnight.log`): armed for the r5f3 kernels.
- **Codex fallback** (`Hypershift_codex_fallback`): idle. It covers only Phase 1.5a, which is done.
- **Known gap:** the launch gate checks that the smoke is COMPLETE, not that it produced a `metrics.json`. The original r5f3 smokes ran nothing; that was fixed in `a16e21b`.

## Data gate result (Track B, `docs/phase1_5b/DATA_COMPATIBILITY.md`)

- **Yahoo: PASS as an exploratory, survivor-biased pilot only.**
  - 967 of 1,737 nodes pass identity: 614 have no data, 20 are short, 136 fail overlap (cause UNKNOWN). Those 967 are eligible every day 2018-2023, with zero attrition (survivors).
  - The calendar matches RSR exactly.
  - Survivor bias (R8, 2017): top-5 Sharpe 1.977 → 2.249 on the subset; hold-all 1.531 → 1.762.
- **Alpaca (A2, DATA_COMPATIBILITY.md section 10): PASS, recommended 2018-2023 source (adjustment=split).**
  - 1,647 of 1,737 nodes pass identity (Yahoo 967); pooled overlap share 0.977 (2016-2017 only).
  - Real attrition: 1,636 eligible on 2018-01-02, 1,229 alive at end of 2023 (418 ended).
  - Reverse survivor check: Alpaca-eligible 2017 top-5 Sharpe 1.964 vs full 1.977; survivors-only 2.520.
  - Limits: no delisting returns, history from 2016-01-04, `split` is 2-decimal rounded and adjusts some spin-offs.
- **Price convention: Amendment A1 = (d), adjust genuine splits only.** It was decided on 2015-2017 overlap evidence only, before any 2018+ scoring. RSR is not dividend-adjusted.
- **MA with a missing close inside the window:** RSR's rule is UNKNOWN; the pilot forward-fills (INFERRED).
- No 2018+ return or inference has been computed.

- **Decision (orchestrator, 2026-10-04): the 2018-2023 test uses Alpaca, `adjustment=split`, 1,647 identity-pass nodes, masked after the last bar.** Yahoo is kept only as a survivor-bias reference.

## Next steps

0. **Phase 1.5c is done.** Next: user decision on Phase 2 (diagnosis) and on CRSP/WRDS access.

## (done) Phase 1.5b Tasks 3-5

1. When r5f3h and r5f3e are COMPLETE and merged (driver):
   - verify 10 runs, each with `best_state.pt` and `epoch_preds/`
   - write the freeze manifest (`docs/phase1_5b/FREEZE_MANIFEST.md`: selected epochs, config/data/graph/weight sha256)
   - run the predeclared 2017 replication check (R7)
2. If the data gate passed: one locked 2018-2023 inference pass (plan Task 4), then the analysis (Task 5, skill-first per R6). The Yahoo source is labelled exploratory and survivor-biased.
3. The per-epoch predictions can also fill Phase 1.5a's "not recoverable" trajectory items (optional).

## Open decisions (user)

- **CRSP/WRDS access?** It is needed for a confirmatory, survivor-free 2018-2023 test. Until then the Yahoo pilot is exploratory only.
- **Alpaca market data** (suggested by the user 2026-10-04), a candidate second source after the Yahoo audit.
  - Its bars API has `adjustment=raw|split|dividend|spin-off|all`, which maps directly onto the R5 convention candidates.
  - Its `asof` parameter resolves symbol renames to the underlying entity.
  - The docs do not state how far back history goes, whether delisted symbols are covered, or plan limits for SIP history. All three must be probed empirically.
  - It needs a free account. Keys go in user env vars `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`; never commit or paste them.
  - Next step: rerun the Track B audit (coverage incl. delisted names, R5 overlap test on the 2016-2017 overlap) with Alpaca as the source, then pick the source by those numbers.

## How Claude and Codex work together

- **Roles.**
  - Claude Code orchestrates: plans, delegates to Sonnet workers, reviews.
  - Codex takes over when Claude is out of credits, and can run independent second-opinion reviews (`codex exec review` or the `codex:rescue` agent).
  - Only one agent edits at a time. Before starting, check **Running now**; if another agent is listed as active, do not touch its files.
- **Unit of work.** One plan task at a time, from the active plan. When finished:
  - append `T<id> done <date> by <claude|codex>: <result>` to that phase's `PROGRESS.md`
  - update this file
  - commit
- **Git.**
  - Branch `tom-shlom`. Commit only the files you changed (pathspec commits). Never stage `results/POC_*`, logs, or other agents' work in progress. Pushing is allowed.
  - If Codex's sandbox blocks `.git`, leave the changes uncommitted and say so in **Running now**: the driver, the fallback or Claude will commit them.

## Rules (non-negotiable)

- **Paper source of truth: `docs/paper/icdm22-think.pdf`** (pp. 849-854). Cite the page and section/equation/table. Anything not stated is `INFERRED (not in paper)` or `UNKNOWN`. Never guess.
- **GPU work goes on Kaggle.** Launch through `kaggle/queue.txt` while the driver is active. Every full preset needs its `-s` smoke kernel COMPLETE first.
- **The laptop is CPU only:** prefix Python with `CUDA_VISIBLE_DEVICES=-1` and call `.venv/Scripts/python.exe` directly.
- Never write into an existing `results/<exp>/` folder (especially `R5_*`, `R8_*`, `POC_*`); use new exp names.
- Predeclare test grids, k, thresholds and nulls before looking at returns. Verdict words are exactly STRONG / SEED-ROBUST ONLY / NO EVIDENCE / INSUFFICIENT SEEDS.
- Tests: `.venv/Scripts/python.exe -m pytest -q` (about 1 min).
