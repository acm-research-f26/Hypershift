# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`hypershift` is a from-scratch reimplementation of **THINK: Temporal Hypergraph Hyperbolic Network** (Agarwal, Sawhney et al., ICDM 2022). The official repo is empty. The code reproduces the paper's NYSE/NASDAQ stock-ranking results and runs ablations:
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

## Environment (Windows, Git Bash)

- Venv: `.venv` in the repo. **Never create a venv under `%TEMP%`**: Windows Application Control blocks pandas DLLs there.
- In the Bash tool, **call `.venv/Scripts/python.exe` directly**. Running `source .venv/Scripts/activate` breaks `PATH` (`head`/`uname` not found). If that happens, run `export PATH="/usr/bin:/mingw64/bin:$PATH"`.
- Torch uses a CUDA wheel matched to the driver (currently cu130). The GPU is an RTX 3050 with 4 GB.

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
$PY scripts/baselines.py | hyperbolicity.py | time_budget.py
$PY scripts/fetch_fresh.py --source yf --kind daily --start 2015-01-01 --end 2026-09-01 --name sp500_daily
$PY scripts/poc_sectors.py run|tune|tune-select|summarize [--variant V] [--input-mode relative] [--use-tuned] [--seeds 0-9]
```

Run scripts from the repo root. Paths like `data/raw/rsr/data`, `configs/*.yaml` and `results/tuned.json` are relative to it.

## Architecture

The pipeline for one run is `train_one_run(cfg)` in `src/hypershift/train/loop.py`:

1. **Data.** `load_market` → `MarketData` (`data/rsr.py`, or `data/fresh.py` for `market: FRESH`).
   - Arrays are `[N, T, C]` features, plus `mask`, `gt` (the return from t−1 to t) and `base_price`.
   - The split indices are `valid_index` and `test_index` (RSR: 756 / 1008 / T = 1245).
2. **Hypergraph.** `base_hypergraph` → `Hypergraph` (`data/hypergraph.py`).
   - Industry hyperedges, plus Wikidata "star" hyperedges, deduplicated.
   - The build is cached on disk under `data/raw/rsr/data/hypergraph_cache/`.
   - The largest RSR "industry" (500 stocks) is really the `n/a` bucket: stocks with no industry label.
3. **`prepare`** applies, in order: universe subset → decomposition → clique expansion → hub dropping → optional label shuffle.
4. **Windows.** `window_offsets` and `gather_batch` build leak-free windows:
   - train targets are always `< valid_index`
   - the target day is `offset + seq`, the day after the last input day
   - `apply_input_mode` (`level` | `relative`) transforms the inputs only
5. **Model.** `models/think.py` computes `log0(TConv2(DHHAN(TConv1(exp0(X)))))`.
   - Switches: `temporal`/`spatial` ∈ {hyp, euc}, `structure` ∈ {hyper, clique, none}.
   - Hyperbolic math lives in `geometry/poincare.py` (c = 1, with projection and clamps).
   - Layers are in `models/layers.py` (HNN++ Poincaré FC, β-concat) and `models/attention.py` (gyromidpoint, distance-aware attention, segment softmax via `scatter_reduce`/`index_add`; no PyG).
   - A clique is represented as 2-node hyperedges in the same attention layer.
6. **Loss.** `train/loss.py` is masked MSE plus α × a pairwise ranking hinge (the STHAN-SR objective).
7. **Evaluation.** Each epoch evaluates val and test with `eval/metrics.evaluate_all`.
   - Sharpe = `mean/std(daily top-5 return) * sqrt(252)`, with no risk-free rate and no costs. This matches the authors' code.
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
- NASDAQ has T = 1245: the raw files have 1246 rows, and the last row, which is all missing, is dropped.
