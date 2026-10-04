# HANDOFF: live state for Claude Code and Codex

**Last updated: 2026-10-04 ~01:40 by Claude Code.** Whoever finishes a step (Claude or Codex) rewrites **Running now**, **Next steps** and **Open decisions**, and changes the date line. Keep this file under about 120 lines; history belongs in the git log, phase docs and PROGRESS files. If this file looks stale, trust `git log --oneline -20` and the newest `docs/phase*/PROGRESS.md`.

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
- **Phase 2 (diagnosis): not started.**

## Running now

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

## Next steps (Phase 1.5b)

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
