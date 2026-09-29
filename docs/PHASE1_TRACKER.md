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
| R1 | Twitter tennis, node regression (MSE) | 0.58 | TODO (order 3) | Data obtainable (PyG-T JSON; Table I matches `rg17`). Paper's baselines (~2.05) are about 5× worse than a constant predictor (0.42), so the setup is unclear. Est. 10–16 eng-h. See `docs/phase1/R_feasibility.md`. |
| R2 | Chickenpox (MSE) | 1.09 | DONE (paper not matched) | `docs/phase1/R2_chickenpox.md` (7b9c388), 10 seeds, untuned, both protocols agree within 0.005. THINK 0.956, EE 0.886, pairwise 0.823, **no relations 0.733**, AR(4) 0.725, mean 1.047. Our THINK beats the paper's 1.09. The paper's 1.09 (and its baselines, 1.11–1.14) is worse than predicting the mean. **Adding relations makes it worse, and hyperbolic is worse than Euclidean.** Nothing beats a linear AR(4). |
| R3 | Windmill (MSE) | 1.05 | TODO (order 2) | Data obtainable (47 MB). Stored series has lag-1 autocorrelation ≈ 0, and the mean predictor scores 1.02 *(not yet independently checked)*. The graph is complete, so the hyperedge cut the paper never states is decisive. Est. 6–10 eng-h. |
| R4 | China stock risk (MSE) | 0.32 | BLOCKED | No public dataset matches Table I (85 nodes, 1293 steps). The cited [35] is a US 10-K text paper; the CSE dataset [22] has 91 stocks over 2 years. |
| R5 | NYSE ranking (Sharpe / NDCG) | 1.18 / 0.86 | PARTIAL | Full NYSE with the paper protocol (paper normalization, 100 epochs), **5 seeds, pre-eq.14 attention**: THINK best-test 2.40 ± 0.20, leak-free −0.05. **Plan chosen: eq.14 rerun, 25 seeds THINK (HH) and EH, 10 seeds EE and HE** → `results/R5_eq14/` (queue 3, waiting on queue 2). The eq.14 change does not alter small-scale conclusions (see the ruled-out table), so the old numbers are expected to hold up. NDCG 0.86 is not comparable because of the evaluator bug (E3). |
| R6 | TSE ranking | 1.19 / 0.81 | BLOCKED | TSE data is not public. Revisit only if the authors share it. |
| R7 | NASDAQ 3-class movement (F1) | 0.49 | TODO (order 4; queued at the end of queue 2, not started) | Thresholds resolved: the STHGCN code uses tertiles of pooled training returns, which matches our `clf.py`. Open: lookback (their 50, ours 16) and macro vs micro F1 (report both). Chance macro-F1 is 0.33. Est. 3–6 eng-h, 6–12 GPU-h. |
| R8 | Paper baselines (STHGCN, RSR-I) | Table II | TODO (order 5) | RSR-I: public TF1 code, 8–12 eng-h. STHGCN: repo unrunnable (hard-coded 423 nodes, dead data link), so reimplement in PyTorch, 12–20 eng-h. |
| R9 | Hyperbolicity (Table I) | δ_hg 0.5, δ_rel 0.087 | DONE | δ_hg gap explained by sampling (ELIMINATED as a data difference). δ_rel is unresolved because the paper doesn't define its features. |

## Phase 1: the paper's own ablations

| ID | Ablation | Status | Evidence / next step |
|---|---|---|---|
| A5 | TCONV + DHHAN (Euclidean temporal) | PARTIAL | Full NYSE `R_paperProtocol/EH`, 5 seeds, pre-eq.14: best-test 1.64, leak-free 0.72. Rerun with eq.14 as part of R5. |
| A9 | Geometry 2×2 | PARTIAL | HH, EH and EE exist on full NYSE (5 seeds). **HE (hyperbolic temporal, Euclidean spatial) is missing.** Small scale has only HH and EE. |
| A10 | Attention without distance | DONE (small scale) | `results/POC_sectors_{,rel_}A10_nodist/`, HH, eq.14, 10 seeds, vs `HH_hyper` in `POC_sectors_{,rel_}eq14` paired by seed. **Removing the distance term changes nothing.** Level inputs: leak-free −0.59 vs −0.30 (diff −0.29, Wilcoxon p 0.43, bootstrap CI [−0.45, +0.07]); best-test 2.24 vs 2.19 (p 0.32). Relative inputs: +0.70 vs +0.57 (diff +0.14, p 1.00, CI [−0.85, +0.44]); best-test 2.10 vs 2.04 (p 0.50). Not run on full NYSE. |
| G1 | Hyperedges vs pairwise clique expansion | DONE (small scale) | Small scale, 10 seeds, exact eq.14 (`POC_sectors{,_rel}_eq14`): hyperbolic hyperedges − pairwise = −0.18 (level, Holm p 1.00) and +0.21 (relative, Holm p 1.00), both NO EVIDENCE. Best-test epoch: 2.19 vs 1.27 (level), 2.04 vs 1.98 (relative). Same with tuned settings (+0.12, Holm p 0.49). Pre-eq.14 gave the same picture. Full NYSE not run: clique arms are ~50× slower (P2). |
| G2 | Hyperedge decomposition by size (Fig 3) | RUNNING | Queue 2. `large_first` 30 is at seed 2 of 0–4 (`results/POC_sectors_G2_decomp_large30/`); `large_first` 15 and `small_first` 5 follow. |
| G5 | No hyperedges | PARTIAL | Small scale HH_none / EE_none, 10 seeds. |
| G12 | Hub removal (Fig 3) | TODO (queued) | Queue 2, after G2: degree 12, 8, 5. Not started. |

## Correctness controls (prerequisite for trusting any result)

| ID | Control | Status | Evidence / next step |
|---|---|---|---|
| C1 | Shuffled training labels collapse to chance | DONE (passes, with a caveat) | `results/POC_sectors_C1_shuffled/`, HH and EE, 5 seeds, level inputs, eq.14. **IC is zero**: HH −0.016 ± 0.006, EE +0.006 ± 0.004 (real labels: −0.010 and −0.007). Leak-free Sharpe HH −0.48 ± 0.80, EE +0.97 ± 0.44. Those are not on the 0.34 random-5 line, but the yardstick is wrong: the models pick a nearly fixed portfolio (HH picks only 14–30 distinct stocks over the year, and its five most-picked stocks are each picked on 72–237 of the 237 days), and a fixed random 5-stock portfolio has Sharpe 0.62 ± 0.92 (2000 draws), which contains both values. **Best-test-epoch Sharpe with shuffled labels is 2.23 (HH) and 2.14 (EE), the same as with real labels (2.19, 2.08).** The paper-protocol score carries no information about the labels. |
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
| THINK's relation/hyperbolic layers help on non-stock data | NOT SUPPORTED (R2) | Chickenpox: no relations < pairwise < Euclidean hyperedges < THINK in MSE (lower is better). The best arm only ties AR(4). |
| Inputs at the Poincaré ball boundary hurt the hyperbolic model | ESTABLISHED | Radius ~0.95; 15% of inputs past 0.99 in 2017. The relative-input fix gives +1 Sharpe (not significant). |
| Validation and test years disagree | ESTABLISHED | Rank correlation between THINK's per-epoch 2016 and 2017 Sharpe is −0.63. |
| The eq.14 formula error changes conclusions | ELIMINATED | Rerun done, 10 seeds, v1 and v2 (`POC_sectors{,_rel}_eq14/summary.md`). Leak-free THINK −0.56 → −0.30 (v1) and +0.48 → +0.57 (v2); best-test 2.28 → 2.19 and 2.13 → 2.04; pairwise −0.38 → −0.13 and +0.59 → +0.36. Paired Wilcoxon old vs new p ≥ 0.25 everywhere. **No Holm verdict changed: all NO EVIDENCE before and after.** |
| Shuffled labels still give a high best-test Sharpe | ESTABLISHED (C1) | 2.2 with random labels vs 2.2 with real labels (5 seeds). The best-test-epoch number is selection on the test year, not skill. |
| Attention distance term matters (A10) | NOT SUPPORTED | Removing it moves leak-free Sharpe by −0.29 (level) and +0.14 (relative); both p > 0.4. |
| Equal-budget tuning rescues hyperbolic or hyperedges | NOT SUPPORTED | Tuned relative-input arms (`POC_sectors_rel_tuned_eq14`, 10 seeds, same grid per geometry): hyperbolic vs Euclidean +1.10, bootstrap CI [+0.21, +2.02] but Holm p 0.15; interaction +1.07, Holm p 0.059. All NO EVIDENCE. Tuned vs untuned per arm: no change is significant (all p ≥ 0.1). |

## Runs in flight

Log: `results/logs/driver.log`. Status as of 2026-09-29 ~9:00.

1. **Queue 1** (`scripts/queues/phase1_gpu.sh`): DONE (7:36). eq.14 reruns, eq.14 tuning (HH lr 1e-3, α 10; EE lr 3e-3, α 10, both chosen at the top of the α grid) and the tuned 10-seed rerun `results/POC_sectors_rel_tuned_eq14/`. `POC_sectors_rel_tuned/` is superseded (pre-eq.14).
2. **Queue 2** (`scripts/queues/phase1_gpu_2.sh`), started 7:37, on 309 stocks with eq.14:
   - C1 shuffled: DONE. A10 (level and relative): DONE.
   - G2 decomposition (`large_first` 30 and 15, `small_first` 5; HH, 5 seeds): RUNNING, `large_first` 30 at seed 2.
   - G12 hub removal (degree 12, 8, 5; HH, 5 seeds): waiting.
   - R7 NASDAQ 3-class (25 seeds × HH, EH, EE; macro and micro F1) → `results/E11_clf/`: waiting.
   - Levels were rescaled to the 309-stock graph (largest edge 47, max degree 27); see the comments in the script.
3. **Queue 3** (`scripts/queues/phase1_gpu_3.sh`) waits for "phase1 queue 2 done", then runs the full-NYSE paper protocol with eq.14 → `results/R5_eq14/`: 25 seeds THINK (HH) and EH, 10 seeds EE and HE, 100 epochs, two parallel workers. It doesn't run `aggregate.py`.
4. **R8 baselines**: another agent is building the baseline queue (code changes in `src/hypershift`). Nothing is running yet.

## Decisions pending

| Decision | Options | Cost |
|---|---|---|
| Non-stock tasks R1–R3: worth doing? **In progress** (R2 done; R3 and R1 next; another agent is on the PyG-T loaders). Original options: | The paper's numbers sit at or near a constant predictor. Protocol options: leak-free split vs the PyG-T convention (run both?), windmill hyperedge cut (threshold vs top-k), tennis target (log1p vs raw) | ~25–35 eng-h, <10 GPU-h in total |
| Full-NYSE R5 eq.14 rerun | **Decided:** 25 seeds HH and EH, 10 seeds EE and HE (queue 3) | HH 21.6 s/epoch → 36 min/seed; EH 29 min; EE 16 min. Roughly 25 × (36 + 29) min + 20 × 16 min ≈ 32 GPU-hours (two workers share the GPU, so wall time may differ). |

## Blockers

- **Push to `origin` (acm-research-f26/Hypershift) returns 403.** Account `OhamjDung` has read-only access. Needs write access or a fork.
- **Smart App Control** intermittently blocks venv DLLs (scipy/sklearn). A process launched through WMI was blocked consistently; `Start-Process` works. The queue retries each step 3 times.

## Log

- 2026-09-29: Found every GPU run stopped with no traceback (killed at session end). The full-NYSE runs and the tuning runs used the pre-eq.14 attention. Restarted the queue with a fresh eq.14 tuning variant.
- 2026-09-29: Correctness controls C2, C4, C5 and C6 added and passing for all 12 variants (0898b50). No leak or bug found.
- 2026-09-29: R-series feasibility researched (`docs/phase1/R_feasibility.md`, c327be5). R4 blocked; R1–R3 obtainable but near-trivial targets; R7 thresholds resolved.
- 2026-09-29: Added `poc_sectors --set` overrides and queue 2 (e35c6cb); launched it waiting on queue 1.
- 2026-09-29: R2 chickenpox done (7b9c388). The paper's number is worse than a mean predictor; in our runs relations and hyperbolic geometry both hurt MSE; AR(4) is unbeaten.
- 2026-09-29: eq.14 reruns, tuning and tuned rerun finished. eq.14 did not change any verdict (all NO EVIDENCE). Tuning does not rescue hyperbolic. C1 and A10 done: shuffled labels still give best-test Sharpe 2.2, so the paper-protocol number is selection, not skill; the distance term has no effect. G2 running; G12 and R7 waiting; queue 3 (R5_eq14, 25 seeds) waits on queue 2.
