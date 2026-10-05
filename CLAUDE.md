# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`hypershift` is a from-scratch reimplementation of **THINK: Temporal Hypergraph Hyperbolic Network** (Agarwal, Sawhney et al., ICDM 2022). The official repo is empty. The code attempts to reproduce the paper's NYSE/NASDAQ stock-ranking results and runs ablations:
- hyperbolic vs Euclidean
- hyperedges vs pairwise vs none
- grouping source
- universe size
- daily vs hourly data

The spec is `docs/superpowers/plans/2026-09-27-think-reproduction.md`:
- Part 0: paper primer with every equation
- Part 1: decision tree D0–D19
- Tasks 1–18

Read Part 0 before changing any model math. Decisions and results made while executing the plan are logged in `.superpowers/sdd/2026-09-27-think-reproduction/progress.md`. That file is git-ignored; the git log is the durable record.

## Study status

Live state and next steps: `docs/HANDOFF.md` (keep it current).


- **Phase 1 (reproduce): done.** `docs/PHASE1_TRACKER.md` is 26/26. Verdict: **not reproduced under validation selection**. This is an inferred-settings reimplementation that has not shown the paper's advantage. It is not a claim about the authors' work. Many details are INFERRED (see the PA/U entries in the tracker and `docs/phase1/paper_audit.md`).
- **Phase 1.5: fidelity audit** after an external review. Ongoing; notes in `docs/phase1_5/`. Part F found the weight-decay collapse (see Conventions). The corrected full-NYSE reruns `R5_f_*` (r5f) and `R8_f_*` (r8f) are analysed in `docs/phase1_5/F_r5f_results.md`: THINK vs TConv+DHHAN is NO EVIDENCE under both norms (10 seeds), R8_f baselines INSUFFICIENT SEEDS (5 seeds). Final ruling: the paper ranking advantage is not reproduced under validation selection in this inferred-settings reimplementation; corrected THINK vs EH is NO EVIDENCE, and the 2017 Sharpe near 2 is NO EVIDENCE of learned ranking. See docs/PHASE1_TRACKER.md and docs/phase1_5a/REPORT_2017.md. Also done (analysed in `F_p1f_results.md`, `F_r5f2_results.md`): `p1f` (small-scale Phase 1 arms rerun with the F fix; the R8-small and R7 reruns also switch level to relative inputs) and `r5f2` (full NYSE `alpha=0` and `spatial_residual`, `norm=train`, 5 seeds, INSUFFICIENT SEEDS). `r8f-top` was dropped (r8f finished all 20 runs). Review: `docs/phase1_5/REVIEW_overnight.md` (pass 1 and pass 2). Resume notes: the RESUME HERE section of `docs/phase1_5/F_learnability.md`.
- **Phase 1.5a (2017 Sharpe≈2 forensics): done.** Sharpe≈2 is not distinguishable from random daily top-5 baskets in a strong year (hold-all 1.53; high-beta baskets); NO EVIDENCE of learned ranking. See `docs/phase1_5a/REPORT_2017.md`, follow-ups in `docs/phase1_5a/PROPOSAL_followups.md`.
- **Phase 1.5b (post-2017 frozen test): done.** A new replication of R5_f2 alpha=0 (exp `R5_f3_alpha0_train`, weights saved, frozen) scored once on 2018-2023 Alpaca data: HH pooled Sharpe about 0.43 vs hold-all 0.51; NO EVIDENCE of learned ranking (Holm p ≥ 0.6); HH vs EH INSUFFICIENT SEEDS. See `docs/phase1_5b/REPORT_POST2017.md`.
- **Phase 1.5c (walk-forward annual retraining, 2019-2023): done.** HH pooled Sharpe 0.70 (seed mean) vs hold-all 0.69; ranking skill NO EVIDENCE (Holm p = 1.0); not distinguishable from the frozen 1.5b model. EH (paper's Euclidean arm) 0.51; HH vs EH (F5) Wilcoxon p 0.3125, seed-avg contrast +0.21 (CI -0.19 to +0.60), verdict INSUFFICIENT SEEDS (5 seeds; descriptively NO EVIDENCE). Spec `docs/phase1_5c/SPEC.md`, tables `docs/phase1_5c/wf_tables.md`.
- **Phase 2 (exhaustiveness closure): done.** `docs/phase2/REPORT_CLOSURE.md`: the THINK advantage cannot be recreated under any leak-free evaluation; the paper values appear only under test-epoch selection, annualised Sharpe or buggy NDCG; the ordering is not robust (identical-config replications flip THINK vs EH). RSR authors' code reproduces its own paper metric but shows no leak-free skill; STHGCN official code is not runnable (data deleted).
- **Paper source of truth: `docs/paper/icdm22-think.pdf`** (pp. 849–854, including the appendix on p. 854). Cite the page and section/equation/table. Anything the PDF doesn't state is labelled `INFERRED (not in paper)` or `UNKNOWN`; never guess.

## Docs map

- `docs/PHASE1_TRACKER.md`: running record, verdict, all R/A/G/C/PA/U entries.
- `docs/phase1/`: `HANDOFF_2026-10-01.md` (what is running, next steps), `paper_audit.md`, `R5_full_nyse_g2.md`, `R8_baselines_results.md` (+ `R8_baselines.md`), `g2_small_results.md`, `A10_G5_full_nyse.md`, `fig3_degree_reconcile.md`, `R1`–`R3`, `R7`.
- `docs/phase1_5/`: fidelity-audit notes.
- `docs/phase1_5/PHASE1_5_SUMMARY.md`: consolidated Phase 1.5 status (start here). Parts:
  - `A_evaluator.md`: evaluator, target alignment, normalization leak; ties and the Sharpe constant.
  - `B_model.md`: tensor-level model audit against eq. 6-17, departures ranked, new attention switches.
  - `C_data_graph.md`: data and Appendix B graph audit, independent rebuild.
  - `D_known_signal.md` and `D_resolved_configs.md`: planted-signal learning test, and the fully resolved Phase 1 configs and repo-vs-authors defaults.
  - `E_rsr_original.md`: the authors' original RSR-I code scored with our evaluator.
  - `F_learnability.md`: why THINK did not learn the planted signal (weight decay) and the config that does.
- **`docs/HANDOFF.md`: live state (what is running, next steps, open decisions, Claude/Codex protocol). Read it first; update it at every milestone.** `AGENTS.md` points Codex to it. Old long-form handoff: `docs/phase1/HANDOFF_2026-09-28.md`; POC write-up: `docs/POC_PRESENTATION.md`.

## Environment (Windows, Git Bash)

- Venv: `.venv` in the repo. **Never create a venv under `%TEMP%`**: Windows Application Control blocks pandas DLLs there.
- In the Bash tool, **call `.venv/Scripts/python.exe` directly**. Running `source .venv/Scripts/activate` breaks `PATH` (`head`/`uname` not found). If that happens, run `export PATH="/usr/bin:/mingw64/bin:$PATH"`.
- Torch uses a CUDA wheel matched to the driver (currently cu130). The GPU is an RTX 3050 with 4 GB.
- **CPU-only jobs:** set `CUDA_VISIBLE_DEVICES=-1`. An empty value does not disable CUDA here.
- **Smart App Control** blocks `kaggle.exe` and sometimes venv DLLs (scipy/sklearn) at import with "An Application Control policy has blocked this file". Retry the command.

## Compute

- **Kaggle is the default for heavy runs** (full-NYSE, clique, baselines). See `kaggle/README.md`. `kaggle/launch.sh <preset>` uploads and starts, `kaggle/fetch.sh` downloads and `kaggle/merge_results.sh` merges without overwriting. Set `KAGGLE_USER=tomphamdustry`; the CLI runs from a private venv: `kaggle/.venv-kaggle/Scripts/python.exe -m kaggle.cli`. Presets are `SESSION` keys in `kaggle/run_kaggle.py` (`COMMANDS`; numeric 1–11 plus the named `r8f-top`, `p1f`, `r5f2` and their `-s` smoke twins); the list and what each runs is in `kaggle/README.md` and the `launch.sh` header. A full named preset refuses to start until its smoke kernel `hypershift-run-<preset>-s` is COMPLETE.
- **Analysis on Kaggle (CPU, no GPU quota).** `kaggle/launch_analysis.sh [smoke]` pushes kernel `hypershift-run-r5f-an` (code dataset `hypershift-code-an`, mounts the r5f/r8f kernel outputs as `kernel_sources`, runs `scripts/r5f_analysis.py`); `kaggle/fetch_analysis.sh r5f-an F_r5f --install` downloads the md/json/png and copies them to `docs/phase1_5/` and `docs/figures/`. Source kernels must be COMPLETE. Presets `p1f` and `r5f2` run their analysis in-kernel (Cell 7b, `scripts/p1f_analysis.py` / `r5f_analysis.py`) and put the md in the results zip.
- **`kaggle/queue.txt`** is the launch queue read by the overnight driver: one `<preset> [tag]` per line, `#` comments, a launched line is removed. It did not exist at the time of writing; the driver waits for it.
- **The laptop (RTX 3050, 4 GB) is for small jobs.** Queues live in `scripts/queues/*.sh`; launch with `bash scripts/queues/launch.sh <name>` (Task Scheduler, runs on battery).
- Queues die on sign-out, restart, sleep or a console Ctrl+C. Relaunch them; finished runs are skipped.
- Close games, Edge and Copilot before training: VRAM spill to shared RAM makes epochs about 10× slower.
- **Launch checklist:** 1-epoch smoke test of each job type first. About 5 min after launch, confirm the process survived and epochs advance at the expected s/epoch (THINK full NYSE ≈ 21 s/epoch) and that no other app holds GPU memory (`nvidia-smi`).

### Overnight automation

- Task Scheduler task `Hypershift_overnight` runs `bash scripts/overnight/driver.sh` every 30 min (CPU only, no Claude tokens). Each tick, idempotently:
  1. polls Kaggle and fetches and merges finished kernels (`merge_results.sh`, fallback `scripts/overnight/merge_zip.py`);
  2. launches the analysis kernel once r5f and r8f are both finished;
  3. launches the next preset from `kaggle/queue.txt` (at most one per tick; `r8f-top` waits for r8f);
  4. runs `codex exec -m gpt-5.5 -s workspace-write` to update `docs/PHASE1_TRACKER.md` and `docs/phase1_5/PHASE1_5_SUMMARY.md`; Codex's sandbox blocks `.git`, so the driver commits for it;
  5. pushes `tom-shlom`.
- Log `results/logs/overnight.log`; state `scripts/overnight/state.json` (helper `state.py`). `bash scripts/overnight/driver.sh --dry-run` shows what a tick would do; `--reset` clears the `done` flag. `bash scripts/overnight/install.sh` / `uninstall.sh` register and remove the task.
- **Rule:** while the driver is active, do not launch Kaggle work outside `kaggle/queue.txt`; edit the queue instead. When resuming, read the tail of `results/logs/overnight.log` first. The driver needs the user logged on and the laptop awake.

## Commands

```bash
PY=.venv/Scripts/python.exe
$PY -m pytest -q                                         # full suite (~1 min)
$PY -m pytest tests/test_loop.py::test_window_offsets_no_leak -v   # single test
$PY -m pytest -m data                                     # tests needing real RSR data
bash scripts/download_rsr.sh                              # RSR data -> data/raw/rsr/data (sparse clone + relation.tar.gz)

$PY -m hypershift.run --config configs/think_nyse.yaml --seeds 0-4 --set exp=manual epochs=10
$PY scripts/run_grid.py E1_main E2_geometry --dry-run     # list configs of named experiments
$PY scripts/run_grid.py E5_structure --labels HH_clique --seeds 0-14 --set batch_days=8
$PY scripts/aggregate.py [--select-tuning] [--select-attn] [--costs]   # -> results/tables/
$PY scripts/plots.py                                      # -> results/figures/
$PY scripts/r5f_analysis.py [--root DIR] [--r5-prefix R5_f] [--r8-prefix R8_f] [--norms paper train] [--draws 10] [--boot 5000] [--fig PNG] [--note TEXT]   # Phase 1.5 F, full NYSE
$PY scripts/p1f_analysis.py [--root DIR] [--suffix _f] [--summarize]   # Phase 1.5 F, small scale (preset p1f)
$PY scripts/tiebreak_report.py EXP [EXP ...] [--draws 20]              # random tie-break Sharpe column
bash kaggle/launch_analysis.sh [smoke] | bash kaggle/fetch_analysis.sh r5f-an F_r5f --install   # Kaggle CPU analysis kernel
bash scripts/overnight/driver.sh [--dry-run|--reset]                   # one overnight tick
$PY scripts/baselines.py | hyperbolicity.py | time_budget.py
$PY scripts/fetch_fresh.py --source yf --kind daily --start 2015-01-01 --end 2026-09-01 --name sp500_daily
$PY scripts/poc_sectors.py run|tune|tune-select|summarize [--variant V] [--arms ...] [--input-mode relative] [--use-tuned] [--seeds 0-9] [--set K=V ...] [--dry-run]
$PY scripts/run_pygt.py --dataset chickenpox --protocol leakfree --arm <arm>   # R1-R3 (PyG-T datasets)
$PY scripts/run_clf.py --exp E11_clf_g2 [--dry-run] [--seeds N]               # R7 NASDAQ 3-class F1
$PY scripts/r5_g2_analysis.py [--fig] | r8_analysis.py [--fig] | a10_g5_analysis.py   # Phase 1 full-NYSE tables (CPU)
$PY -m hypershift.run ... --set model=rsr_i|sthgcn         # R8 baselines (RunConfig.model)
```

Run scripts from the repo root. Paths like `data/raw/rsr/data`, `configs/*.yaml` and `results/tuned.json` are relative to it.

## Architecture

The pipeline for one run is `train_one_run(cfg)` in `src/hypershift/train/loop.py`:

1. **Data.** `load_market` → `MarketData` (`data/rsr.py`, or `data/fresh.py` for `market: FRESH`).
   - Arrays are `[N, T, C]` features, plus `mask`, `gt` (the return from t−1 to t) and `base_price`.
   - The split indices are `valid_index` and `test_index` (RSR: 756 / 1008 / T = 1245).
2. **Hypergraph.** `base_hypergraph` → `Hypergraph` (`data/hypergraph.py`).
   - Industry hyperedges, plus Wikidata hyperedges, deduplicated. Cache **v2** (`de20f8e`, paper App. B, p. 854): first-order relations are stars per source×relation, second-order relations are pairs.
   - NYSE: 4350 hyperedges, max node degree 114. The 309-stock POC graph has 558. NASDAQ: 1066. **Results run before `de20f8e` used the old graph** (NYSE 312 edges, max degree 37); don't mix them.
   - The build is cached on disk under `data/raw/rsr/data/hypergraph_cache/`.
   - The largest RSR "industry" (500 stocks) is really the `n/a` bucket: stocks with no industry label.
3. **`prepare`** applies, in order: universe subset → decomposition → clique expansion → hub dropping → optional label shuffle.
4. **Windows.** `window_offsets` and `gather_batch` build leak-free windows:
   - train targets are always `< valid_index`
   - the target day is `offset + seq`, the day after the last input day
   - `apply_input_mode` (`level` | `relative`) transforms the inputs only
5. **Model.** `models/think.py` computes `log0(TConv2(DHHAN(TConv1(exp0(X)))))`.
   - Switches: `temporal`/`spatial` ∈ {hyp, euc}, `structure` ∈ {hyper, clique, none}. Arm labels are temporal+spatial: HH = THINK, **EH (TCONV+DHHAN) = the paper's Euclidean arm** (p. 852 Sec. V.A, Table II), EE and HE are extra ablations. Also `model` ∈ {think, rsr_i, sthgcn} (R8 baselines).
   - Attention: `attn_score=eq14` is the default (paper eq. 14 as read on p. 851). ⊙ as a plain product and the softmax are INFERRED (not in paper). Switches `attn_odot` (product | mobius) and `attn_norm` (softmax | none | sum) in `config.py` test those readings; `attn_dist` (mult | neg | off) controls the distance term.
   - Hyperbolic math lives in `geometry/poincare.py` (c = 1, with projection and clamps).
   - Layers are in `models/layers.py` (HNN++ Poincaré FC, β-concat) and `models/attention.py` (gyromidpoint, distance-aware attention, segment softmax via `scatter_reduce`/`index_add`; no PyG).
   - A clique is represented as 2-node hyperedges in the same attention layer.
6. **Loss.** `train/loss.py` is masked MSE plus α × a pairwise ranking hinge (the STHAN-SR objective).
7. **Evaluation.** Each epoch evaluates val and test with `eval/metrics.evaluate_all`.
   - Sharpe = `mean/std(daily top-5 return) * sqrt(252)`, with no risk-free rate and no costs. The paper's formula differs (PA4 in the tracker), so numbers are not directly comparable; the Phase 1 docs also report unannualized columns (ours / √252).
   - The epoch is **selected on validation Sharpe**.
   - `test_oracle_sr`, the max over epochs of test Sharpe, reproduces the paper's protocol and is a diagnostic only.

**Run-folder contract.** Each run writes to `results/<exp>/<label>/seed_<k>/`:
- `config.json`
- `history.jsonl` (per epoch)
- `metrics.json`
- `test_{pred,gt,mask,daily}.npy`
- `failed.json` (only on failure)

A run is skipped if `metrics.json` exists, so everything is resumable. `aggregate.py`, `plots.py` and `poc_sectors.py summarize` read only these files.

**Experiments.** `experiments/grid.py` defines every experiment (E0–E10, E_tune, E_attn) as lists of `RunConfig`, plus `FAMILIES`/`INTERACTIONS`, the comparisons used for verdicts.
- Config precedence in `_geo`: `GEOMS` < `results/tuned.json` < `configs/global.yaml` (compute-budget overrides) < `configs/chosen.yaml` (attention choice) < explicit kwargs.
- Stats (`eval/stats.py`): paired Wilcoxon by seed, Holm correction per family, and a stationary block bootstrap on Sharpe contrasts.
- Verdict words are exactly STRONG / SEED-ROBUST ONLY / NO EVIDENCE / INSUFFICIENT SEEDS.

## Conventions and pitfalls

- `norm: train` (the default) divides by the max close over the training period. `norm: paper` uses the full-series max, which is look-ahead; it is kept only to match the paper.
- With `input_mode: level`, inputs sit near the Poincaré ball boundary (radius ~0.95), where hyperbolic layers degrade. `relative` fixes this.
- Clique arms are ~50× slower than hypergraph arms (145k pairs on NYSE). They use `micro_batch_days: 1`, which gives identical gradients with less memory.
- `run_grid.py --seeds` replaces every selected label's seed list. Use `--labels` to scope it.
- Never tune only one arm. Every compared variant gets the same tuning budget and the same seeds.
- NASDAQ has T = 1245: the raw files have 1246 rows, and the loader drops the last one (`parse_eod(drop_last=True)`, 2017-12-11). It is **not** all missing: 474 of the 1026 listed stocks are missing there, 552 have real values (Phase 1.5 C; rechecked on the raw files).
- **Never train with coupled `weight_decay=5e-4` (the old default).** With Adam at `lr 1e-3` the decay gradient is 10-40x the loss gradient and shrinks the stacked hyperbolic layers to a constant output; planted-signal recovery is 26% of oracle. Use `weight_decay=0` + `input_mode=relative` (63-84% THINK, Phase 1.5 F, `docs/phase1_5/F_learnability.md`). All Phase 1 nulls used the collapsed setup; verdict pending the `R5_f_*` rerun.
- **Prediction ties:** the evaluator breaks top-5 ties by lowest index. Collapsed models tie a lot (THINK: 73 of 237 test days). `scripts/tiebreak_report.py` gives a random tie-break column; the default is unchanged.
