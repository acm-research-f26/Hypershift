# THINK Study Tracker

*Started 2026-09-29. This is the running record of what has been tested, what is ruled out, and what is still open. Update it whenever a run finishes or a decision is made. Evidence lives in `results/`, `docs/HANDOFF.md` and `docs/POC_PRESENTATION.md`.*

## Phases

- **Phase 1 (current): reproduce THINK.** Rebuild each headline result and the paper's own ablations under the paper's protocol, and alongside it under a leak-free protocol.
- **Phase 2 (if Phase 1 fails): diagnose why.** Candidate causes: overfitting or selection effects, the way relationships are represented (the grouping problem), and the trading strategy (equity curves: does the portfolio make money even though the ranking is wrong, or lose money even though it is right?).

Status words: **DONE**, **PARTIAL**, **RUNNING**, **TODO**, **BLOCKED** (cannot run as specified), **DECISION** (waiting on the user), **ELIMINATED** (ruled out as an explanation), **P2** (deferred to Phase 2).

## Ground rules (frozen)

| Rule | Current state |
|---|---|
| Dataset, universe, splits | RSR NYSE/NASDAQ: train 2013–15, val 2016, test 2017 (T = 1245). Small-scale universe: 309 NYSE stocks (Energy/Utilities + Finance). |
| Matched ablations | Same dates, inputs, seeds and tuning budget for every compared arm (CLAUDE.md). |
| Seeds | Paper: 25. Ours: 10 small scale, 5 full NYSE. **Gap** (see R5). |
| Epoch selection | Always report both the leak-free choice (best validation Sharpe) and the paper-style choice (best test epoch, an upper bound). |
| Uncertainty | Paired Wilcoxon over seeds, Holm correction, stationary block bootstrap over dates (`eval/stats.py`). |
| Pre-registration | Every variant run is listed in this file. |
| Timestamp safety | Windows are leak-free by construction and tested (`test_window_offsets_no_leak`). Default normalization uses training data only. See controls C2 and C9. |

## Phase 1: headline reproduction (paper Table II)

| ID | Task | Paper | Status | Evidence / next step |
|---|---|---|---|---|
| R1 | Twitter tennis, node regression (MSE) | 0.58 | TODO | Not implemented. Research how the paper builds this graph and where to get the data (PyG-Temporal). |
| R2 | Chickenpox (MSE) | 1.09 | TODO | Same as R1. |
| R3 | Windmill (MSE) | 1.05 | TODO | Same as R1. |
| R4 | China stock risk (MSE) | 0.32 | TODO | Check whether the data can be obtained; likely BLOCKED. |
| R5 | NYSE ranking (Sharpe / NDCG) | 1.18 / 0.86 | PARTIAL | Full NYSE with the paper protocol (paper normalization, 100 epochs), **5 seeds, pre-eq.14 attention**: THINK best-test 2.40 ± 0.20, leak-free −0.05. Needs an **eq.14 rerun with 25 seeds** (see Decisions). NDCG 0.86 is not comparable because of the evaluator bug (E3). |
| R6 | TSE ranking | 1.19 / 0.81 | BLOCKED | TSE data is not public. Revisit only if the authors share it. |
| R7 | NASDAQ 3-class movement (F1) | 0.49 | TODO | `scripts/run_clf.py` is implemented (25 seeds × {HH, EH, EE}, macro-F1) but has never been run. The up/down/neutral thresholds and the F1 averaging still need checking against the paper. |
| R8 | Paper baselines (STHGCN, RSR-I) | Table II | TODO | Not implemented. Our Euclidean twins are not the paper's baselines. |
| R9 | Hyperbolicity (Table I) | δ_hg 0.5, δ_rel 0.087 | DONE | δ_hg gap explained by sampling (ELIMINATED as a data difference). δ_rel is unresolved because the paper doesn't define its features. |

## Phase 1: the paper's own ablations

| ID | Ablation | Status | Evidence / next step |
|---|---|---|---|
| A5 | TCONV + DHHAN (Euclidean temporal) | PARTIAL | Full NYSE `R_paperProtocol/EH`, 5 seeds, pre-eq.14: best-test 1.64, leak-free 0.72. Rerun with eq.14 as part of R5. |
| A9 | Geometry 2×2 | PARTIAL | HH, EH and EE exist on full NYSE (5 seeds). **HE (hyperbolic temporal, Euclidean spatial) is missing.** Small scale has only HH and EE. |
| A10 | Attention without distance | TODO | Knob `attn_dist=off` exists. Not run. |
| G1 | Hyperedges vs pairwise clique expansion | PARTIAL | Small scale, 10 seeds: no significant difference, leak-free or relative. The eq.14 rerun is RUNNING. Not done on full NYSE. |
| G2 | Hyperedge decomposition by size (Fig 3) | TODO | Knobs `decompose_mode` / `decompose_size` exist and are tested. Not run. |
| G5 | No hyperedges | PARTIAL | Small scale HH_none / EE_none, 10 seeds. |
| G12 | Hub removal (Fig 3) | TODO | Knob `drop_hub_degree` exists and is tested. Not run. |

## Correctness controls (prerequisite for trusting any result)

| ID | Control | Status | Evidence / next step |
|---|---|---|---|
| C1 | Shuffled training labels collapse to chance | TODO | Knob `shuffle_train_labels` and grid label `E1_main/THINK_shuffled` exist. Never run. |
| C2 | Changing data after time t leaves the prediction at t unchanged | DONE | `tests/test_controls.py` (0898b50): all 12 variants, `level` and `relative`. Features, mask, gt and base price mutated from the target day onward; predictions identical. Non-vacuity check included. |
| C3 | Future hyperedge injection is rejected | P2 | Hypergraphs are static in RSR, so this doesn't apply until time-varying edges exist. |
| C4 | Relabelling stock IDs permutes predictions consistently | DONE | `tests/test_controls.py`: node permutation and hyperedge reordering, all variants. |
| C5 | Batch isolation: dates batched together or separately give identical predictions | DONE | `tests/test_controls.py`: days sit on a separate tensor axis, so no hyperedge can span dates. 1×5, 5×1, reversed and 2+3 batchings match. |
| C6 | An isolated node gives a finite prediction | DONE | `tests/test_controls.py`: an uncovered node passes through the attention unchanged (`torch.where(has_edge, out, u)`), equals its no-graph value, and is finite. Size-1 hyperedges give finite gradients. |
| C7 | Geometry numerics | DONE | `tests/test_poincare.py`, `tests/test_layers.py`, `tests/test_attention.py`. |
| C8 | Recompute metrics from saved predictions | DONE | Fig 1 diagnostics recomputed MSE, NDCG, IC, hit rate and spread from `test_*.npy` and matched `metrics.json` (commit 18abfc7). |
| C9 | Normalization is fitted on training data only | DONE | `norm: train` is the default; `norm: paper` is kept only for paper-protocol runs. |
| C10 | Survivorship: delisted stocks included | BLOCKED | RSR contains only stocks that survived through 2017. Report as a limitation. |

## What has been ruled out or established

| Finding | Status | Evidence |
|---|---|---|
| Implementation broken | ELIMINATED (mostly) | At the best test epoch we reproduce the paper's ordering and exceed its numbers (small scale 2.28, full NYSE 2.40). Every equation was reviewed and 216 tests pass, including correctness controls C2/C4/C5/C6. |
| Paper's NDCG 0.86 shows skill | ELIMINATED | The authors' evaluator scores index numbers on the last day only. 43% of random models score ≥ 0.86 (`scripts/ndcg_bug_demo.py`). |
| δ_hg difference means a different graph | ELIMINATED | It is a sampling effect: computed exactly, δ_hg = 1.5 (`scripts/hyperbolicity_sensitivity.py`). |
| Largest RSR "industry" is a real industry | ELIMINATED | It is the `n/a` bucket (500 stocks). |
| Leak-free THINK beats simpler arms | NOT SUPPORTED (small scale) | No Holm-significant comparison, v1 or v2 (`results/POC_sectors*/summary.md`). |
| Models predict returns | NOT SUPPORTED | MSE ≥ predicting 0, IC ≈ 0, NDCG at random level, predictions collapsed (THINK spread 1.4%). POC_PRESENTATION §4b cause 3. |
| Inputs at the Poincaré ball boundary hurt the hyperbolic model | ESTABLISHED | Radius ~0.95; 15% of inputs past 0.99 in 2017. The relative-input fix gives +1 Sharpe (not significant). |
| Validation and test years disagree | ESTABLISHED | Rank correlation between THINK's per-epoch 2016 and 2017 Sharpe is −0.63. |
| The eq.14 formula error changes conclusions | RUNNING | 80% of stocks are in exactly one hyperedge, so the effect should be small. Rerun queued. |

## Runs in flight

The GPU queue is `scripts/queues/phase1_gpu.sh`, launched detached. Its log is `results/logs/driver.log`.

1. eq.14 reruns: HH_hyper and HH_clique, 10 seeds, v1 and v2 → `results/POC_sectors_eq14/`, `results/POC_sectors_rel_eq14/`.
2. Equal-budget tuning with eq.14 (HH and EE grids, lr × α, 3 seeds, validation only) → `results/POC_sectors_rel_tuned_eq14/`, followed by a tuned 10-seed rerun.
   - `results/POC_sectors_rel_tuned/` is **superseded**: it was started with the pre-eq.14 attention.

## Decisions pending

| Decision | Options | Cost |
|---|---|---|
| Full-NYSE R5 eq.14 rerun | 25 seeds × {HH, EH, EE, HE}, paper protocol, 100 epochs | Measured: HH 21.6 s/epoch → 36 min/seed; EH 29 min; EE 16 min. 25 seeds × 3 arms ≈ **34 GPU-hours**. Adding HE ≈ +12 h. 10 seeds ≈ 14 h. |

## Blockers

- **Push to `origin` (acm-research-f26/Hypershift) returns 403.** Account `OhamjDung` has read-only access. Needs write access or a fork.
- **Smart App Control** intermittently blocks venv DLLs (scipy/sklearn). A process launched through WMI was blocked consistently; `Start-Process` works. The queue retries each step 3 times.

## Log

- 2026-09-29: Found every GPU run stopped with no traceback (killed at session end). The full-NYSE runs and the tuning runs used the pre-eq.14 attention. Restarted the queue with a fresh eq.14 tuning variant.
- 2026-09-29: Correctness controls C2, C4, C5 and C6 added and passing for all 12 variants (0898b50). No leak or bug found.
