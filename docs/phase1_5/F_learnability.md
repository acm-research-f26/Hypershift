# F: why THINK did not learn the planted signal, and the config that does

*Phase 1.5 F, 2026-10-02. Code: `scripts/f_diag.py` (CPU probe), `scripts/f_analysis.py` (tables), switches in `src/hypershift/{config,models,train}`, `tests/test_f_switches.py` (9 tests; full suite 291 passed), Kaggle presets 8/8s (stage 1, kernel `hypershift-run-f8`, 171 runs, 1.8 h), 9 (stage 2, `hypershift-run-f9`, 120 runs) and 10/10s (full NYSE rerun, `hypershift-run-r5f`, running). Smoke tests: `hypershift-run-f-smoke`, `hypershift-run-r5f-smoke` (1 epoch each job type, OK). Raw runs `results/f_*` (git-ignored). Benchmark and metrics as in `D_known_signal.md`: 309 stocks, real g2 graph, test IC at the validation-selected epoch, IC/oracle = fraction of the planted signal recovered. Baseline = the D runs (wd 5e-4, lr 1e-3, alpha 1, same code path, seeds 0-2 / 0-4).*

## Verdict

**Root cause (confirmed): coupled L2 weight decay 5e-4 in Adam, with `lr 1e-3`, destroys the model because the loss gradient is far smaller than the decay gradient.** Setting `weight_decay=0` alone raises THINK (HH_hyper) from 26% to 63-84% of the oracle IC and EH from 32% to 72-77% (`high` / `group`, 5 seeds). No other single factor does as much. Level inputs remain a separate, unresolved failure (below).

## 1. Mechanism (CPU probe, `scripts/f_diag.py`, `high`, seed 0, batch 8; a diagnostic, not a training run)

- At init in `relative` mode the loss gradient norm per hyperbolic `z` is 1e-5 to 1e-4 (the planted signal explains only 2-8% of return variance), while the coupled L2 term `5e-4*|w|` is about 3.6e-4: weight decay is 10-40x larger. Adam normalises the combined gradient, so each weight moves about `lr*sign(w)` toward 0 per step, whatever the size of `wd`.
- `PoincareLinear` output is proportional to `|z|` (HNN++ form `v = 2|z| asinh(...)`), so three stacked layers shrink multiplicatively. HH_hyper, wd 5e-4, 200 steps: `|z|` (tconv1, tconv2, DHHAN fc) 0.71/0.71/0.71 -> 0.006/0.04/0.000, prediction sd 2.5e-4 -> 4e-12 (this is D's "1e-8 to 1e-12 collapse"). wd 0, same steps: `|z|` 0.70/0.48/0.71, pred sd stays about 3e-5, batch IC rises to about +0.2.
- EE_hyper (no hyperbolic layers) shows the same pattern more mildly (wd 5e-4: pred sd 2e-4 -> 5e-5, batch IC +0.12; wd 0: 2.5e-4, +0.19).
- Secondary: the HNN++-style init shrinks activations about 10x per layer (relative mode: 0.085 -> 0.024 -> 0.0016 -> 0.0002), so initial pred sd (2e-4) is 50x below the return sd. `init_gain=4` alone does not fix wd (below) and adds nothing once wd is 0.
- Gradient clipping (1.0) never fires in relative mode (pre-clip norm 4e-5 to 2e-3), so it is irrelevant here; it was dropped from the grid.
- DHHAN attention vector `a` receives gradient about 2e-10 at init (near-origin distances), so attention is effectively a uniform mean aggregator early in training (not measured further).

## 2. Stage 1: one factor at a time (Kaggle f8; HH_hyper and EH_hyper; relative inputs; 3 seeds; 30 epochs; IC / oracle IC)

Base = D baseline (wd 5e-4, lr 1e-3, alpha 1, batch_days 8). Rows marked `+wd0` were run on top of `weight_decay=0` because wd 0 was already established as the dominant factor (an economy of runs, not a full factorial).

| config (change vs baseline) | HH high | HH group | EH high | EH group | fidelity label |
|---|---|---|---|---|---|
| D baseline (wd 5e-4) | 0.26 | 0.20 | 0.31 | 0.31 | repo default; all values INFERRED (paper states no hyperparameters) |
| **wd 0** | **0.71** | **0.76** | **0.75** | **0.83** | INFERRED (paper silent on optimizer / wd); not a departure from the paper |
| decoupled wd (AdamW, 5e-4) | 0.68 | 0.79 | 0.74 | 0.77 | INFERRED (same effect as wd 0: 5e-4 x lr is negligible) |
| init_gain 4 (wd 5e-4) | 0.43 | 0.31 | 0.47 | 0.48 | INFERRED (init not in paper); not sufficient alone |
| head_scale 50, learnable, no wd (wd 5e-4 elsewhere) | 0.42 | 0.49 | 0.67 | 0.83 | DEPARTURE (adds a scalar the paper lacks); helps EH, little for HH |
| +wd0 lr 3e-3 | 0.24 | 0.29 | 0.75 | 0.85 | INFERRED; hurts HH |
| +wd0 lr 1e-2 | 0.03 | 0.03 | 0.74 | 0.64 | INFERRED; kills HH |
| +wd0 alpha 0 (MSE only) | 0.77 | 0.98 | 0.77 | 0.97 | INFERRED (loss is not in the paper) |
| +wd0 alpha 10 | 0.42 | 0.55 | 0.48 | 0.30 | INFERRED; large alpha hurts here |
| +wd0 spatial_residual | **0.97** | 0.81 | **0.96** | 0.76 | **DEPARTURE** from eq. 15 (self path added) |
| +wd0 batch_days 1 | 0.66 | 0.35 | 0.74 | 0.77 | INFERRED; not better than 8 |
| +wd0 init_gain 4 | 0.72 | 0.78 | 0.74 | 0.76 | INFERRED; no gain once wd is 0 |
| +wd0 input_std (relative, train stats) x0.3 | 0.57 | 0.71 | 0.73 | 0.65 | DEPARTURE (standardisation not in paper); slightly worse |

Level inputs (`high` only; 3 seeds; wd 0; D level baseline in brackets): HH 0.24 (0.07), EH 0.10 (0.00), EE_none 0.05 (0.11). Level + `input_std x0.3`: HH 0.26, EH 0.26, EE_none 0.11. Level + `target=price` (output read as a price, `(pred - base)/base`): HH -0.02, EH -0.02, EE_none -0.16, i.e. worse. **No level-mode config reaches 50% of the oracle**; relative inputs are needed (in level mode the 1% day-to-day differences must be read out of O(1) offsets; EE_none fails with or without wd).

## 3. Stage 2: all arms incl. no-graph, 5 seeds, 30 epochs, relative inputs (Kaggle f9)

IC (IC / oracle IC). Oracle IC: high 0.279, group 0.146. c1 = wd 0; c2 = wd 0 + alpha 0; c3 = wd 0 + spatial_residual; c4 = wd 0 + alpha 0 + spatial_residual. No-graph arms were run for c1 and c2 only (residual does not apply to them; c3 is compared against c1, c4 against c2).

| level | arm | D baseline | c1 (wd 0) | c2 (wd 0, alpha 0) | c3 (wd 0, residual) | c4 (wd 0, alpha 0, residual) |
|---|---|---|---|---|---|---|
| high | HH_hyper (THINK) | +0.074 (0.27) | +0.174 (0.63) | +0.213 (0.77) | +0.271 (0.97) | +0.273 (0.98) |
| high | EH_hyper | +0.090 (0.32) | +0.202 (0.72) | +0.212 (0.76) | +0.266 (0.95) | +0.271 (0.97) |
| high | HH_none | +0.187 (0.67) | +0.252 (0.90) | +0.255 (0.91) | | |
| high | EE_none | +0.226 (0.81) | +0.247 (0.89) | +0.250 (0.90) | | |
| group | HH_hyper (THINK) | +0.038 (0.26) | +0.123 (0.84) | +0.142 (0.97) | +0.128 (0.87) | +0.141 (0.96) |
| group | EH_hyper | +0.053 (0.36) | +0.113 (0.77) | +0.142 (0.97) | +0.108 (0.74) | +0.132 (0.90) |
| group | HH_none | +0.035 (0.24) | +0.046 (0.32) | +0.063 (0.43) | | |
| group | EE_none | +0.047 (0.32) | +0.028 (0.19) | +0.060 (0.41) | | |

Graph arm minus its no-graph counterpart, paired by seed (IC, mean +- s.e., seeds positive):

| level | pair | c1 | c2 | c3 (vs c1 none) | c4 (vs c2 none) |
|---|---|---|---|---|---|
| group | HH_hyper - HH_none | +0.077 +- 0.004 (5/5) | +0.079 +- 0.004 (5/5) | +0.082 +- 0.004 (5/5) | +0.078 +- 0.004 (5/5) |
| group | EH_hyper - EE_none | +0.085 +- 0.005 (5/5) | +0.082 +- 0.004 (5/5) | +0.080 +- 0.006 (5/5) | +0.072 +- 0.003 (5/5) |
| high | HH_hyper - HH_none | -0.078 +- 0.015 (0/5) | -0.042 +- 0.001 (0/5) | +0.019 +- 0.003 (5/5) | +0.018 +- 0.001 (5/5) |
| high | EH_hyper - EE_none | -0.045 +- 0.005 (0/5) | -0.038 +- 0.003 (0/5) | +0.019 +- 0.002 (5/5) | +0.020 +- 0.002 (5/5) |

## 4. Answers

- **Do THINK and EH recover at least 50% of the oracle IC on `high` and `group`?** Yes, with `weight_decay=0` and relative inputs: THINK 63% / 84%, EH 72% / 77% (5 seeds). With the further changes below, 77-98%. Not in level mode.
- **Do graph arms beat no-graph at `group`?** Yes in every stage-2 config, 5/5 seeds, by +0.07 to +0.085 IC. At `high`, no-graph wins under wd 0 (own-lag dominates and DHHAN has no self path: -0.04 to -0.08, 0/5); **a residual self path flips it** (+0.018 to +0.020, 5/5) and takes graph arms to 95-98% of the oracle. (No-graph arms at `group` recover only 19-43%, because the graph-only part of the signal is invisible to them.)
- **Minimal config that learns: relative inputs + `weight_decay=0`** (alpha 1, everything else the repo default). Most effective additions: `alpha=0` (`group` 97%) and `spatial_residual=true` (`high` 97-98%).
- **Fidelity cost.** `weight_decay=0` and `alpha=0` are INFERRED-setting changes: the paper states no optimizer, weight decay, loss or alpha (paper_audit P44; `D_resolved_configs.md`). `relative` inputs are INFERRED (the paper states no input scaling; Phase 1 headline used level inputs with `norm: paper`). `spatial_residual` is a **DEPARTURE** from eq. 15 (the paper's DHHAN output is `exp_o(ReLU(sum alpha log_o(F(z))))` with no self path, B_model.md sec. 2); it is needed only for graph-beats-no-graph at `high`, not for learning. `head_scale` and `input_std` are DEPARTURES that were not needed.

## 5. Consequences for Phase 1 (not re-run here, flagged for the orchestrator)

R5_g2 and every small-scale Phase 1 run used wd 5e-4 with `norm: paper` level inputs (or relative inputs with wd 5e-4), i.e. the collapsed setup: relative inputs with wd 5e-4 reach only 26% (HH) and 32% (EH) of the oracle. Their nulls therefore come from a model that cannot learn even a strong planted signal, and the cause is now identified. The full-NYSE rerun with the fix is running (below).

Caveats: the planted signal is linear in the previous return; 30 epochs; stage-1 factors on the wd0 base are not a full factorial (interactions with alpha 0 and residual are covered only in stage 2); lr/init factors were not tuned per arm; the benchmark shows the pipeline can learn once wd is 0, not that real data contain a learnable signal.

## RESUME HERE (written 2026-10-02, laptop shutting down)

Running on Kaggle: **`tomphamdustry/hypershift-run-r5f`** (preset 10): corrected full-NYSE rerun, config `input_mode=relative weight_decay=0 log_ic=true epochs=100 patience=1000`, arms HH (THINK; base `E1_main/THINK_paperProtocol`) and EH seeds 0-9, EE seeds 0-4, `norm=paper` into `results/R5_f_paper/<arm>/seed_k` and `norm=train` into `results/R5_f_train/<arm>/seed_k`; seeds outermost so a cut-off session leaves paired rows; 3 workers; smoke OK (1-epoch HH 17 s, EH 14 s, EE 11 s on T4, so about 30 min per THINK run: roughly 5.5 h for 50 runs). Also running (launched 2026-10-02 as preset 11, same F config + batch_days=8, micro_batch_days 2 RSR-I / 4 STHGCN): **`tomphamdustry/hypershift-run-r8f`**, R8 baselines RSR_I and STHGCN on full NYSE, `R8_f_train` and `R8_f_paper`, seeds 0-4, seeds outermost (smoke `hypershift-run-r8f-smoke` OK: 18 s / 13 s per epoch-1 incl. warmup). Fetch with `bash kaggle/fetch.sh r8f`; analysis = r5f's script with `R8_f_*` plus the r5f HH/EH rows. As of this writing r5f was still RUNNING (not yet fetched). Finished: `hypershift-run-f8`, `hypershift-run-f9`, `hypershift-run-f-smoke`, `hypershift-run-r5f-smoke`.

```bash
export KAGGLE_USER=tomphamdustry
source kaggle/_kaggle_cli.sh; kg kernels status tomphamdustry/hypershift-run-r5f
bash kaggle/fetch.sh r5f        # downloads kaggle/build/out_r5f/results_r5f.zip when COMPLETE
bash kaggle/merge_results.sh kaggle/build/out_r5f/results_r5f.zip --dry-run   # then without --dry-run
# If merge_results.py hits PermissionError on os.rename (it did with f8): delete stale results/**/*.kaggle_tmp,
# then extract the new R5_f_* exp dirs straight from the zip with python zipfile (skip files that exist).
# Analysis: adapt scripts/r5_g2_analysis.py to results/R5_f_paper and R5_f_train (labels HH, EH, EE);
# history.jsonl now has val_ic, test_ic, test_pred_sd per epoch.
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/f_analysis.py --seeds 5 --prefix f_f2_   # reprints stage 1-2 tables
```

Still to conclude:
1. Fetch, merge and analyse r5f: does THINK beat EH/EE under the corrected optimizer (validation-selected epoch and `test_oracle_sr`), and `norm=paper` vs `norm=train`?
2. Decide whether the small-scale Phase 1 arms (g2, POC `rel_*`, clique, R8 baselines) need a wd-0 rerun. The same wd 5e-4 applies to RSR-I and STHGCN in this repo (a deviation for the baselines, `D_resolved_configs.md` sec. 5).
3. Test `alpha=0` and `spatial_residual` on full NYSE (r5f has wd 0 only).
4. Update the `docs/PHASE1_TRACKER.md` verdict (currently "not reproduced under validation selection") with the F finding that the setup could not learn.
