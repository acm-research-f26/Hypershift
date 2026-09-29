# R3: Windmill Output (Large) (THINK Table II, paper MSE 1.05)

Code: `src/hypershift/data/pygt.py` (`load_windmill`, `topk_hypergraph`, `quantile_hypergraph`, `topk_pairs`), `scripts/run_pygt.py --dataset windmill`, `scripts/summarize_pygt.py`, `tests/test_pygt.py`. Results: `results/R3_windmill/<protocol>/<label>/seed_k/metrics.json` (force-added). CPU only, fixed hyperparameters, **no tuning**, **5 seeds** per arm (not 10; see cost caps).

Paper facts used (`docs/paper/icdm22-think.pdf`): Table I p.850 (WMill: 17,472 timesteps, 319 nodes, delta_hg 1.0, delta_rel 0.025); Table II p.852; Sec. IV-A p.851 (single-time-step node regression, ref [20] PyG-T); Sec. IV-B p.852 (MSE); Appendix B p.854 (neighbourhood hyperedges merged by Sorensen-Dice, threshold unspecified).

## Results (test MSE on per-node z-scored output, mean +- std over 5 seeds)

| Arm | `pygt` protocol | `leakfree` protocol |
|---|---|---|
| THINK (hyp/hyp, top-5 neighbourhood hyperedges, 249 edges) | 1.0212 +- 0.0016 | 1.0232 +- 0.0012 |
| THINK, quantile hyperedges (top 10% of weights, 283 edges) | 1.0212 +- 0.0016 | 1.0233 +- 0.0015 |
| EE (Euclid/Euclid, top-5 hyperedges) | 1.0225 +- 0.0020 | 1.0246 +- 0.0022 |
| EH (Euclid temporal conv + hyperbolic hypergraph attention = paper's TCONV+DHHAN, top-5 hyperedges) | 1.0219 +- 0.0014 | 1.0243 +- 0.0009 |
| THINK, structure none | 1.0236 +- 0.0006 | 1.0289 +- 0.0032 |
| THINK, pairwise/clique (top-5 pairs, 999 edges) | 1.0212 +- 0.0012 | 1.0234 +- 0.0011 |
| Baseline: train mean | 1.0202 | 1.0215 |
| Baseline: persistence | 2.0270 | 2.0283 |
| Baseline: AR(8) pooled | 1.0203 | 1.0216 |
| Baseline: AR(8) per node | 1.0206 | 1.0219 |

**Paired THINK vs EH (paper's comparison), 5 seeds (same caps as the other arms), EH - THINK (positive = THINK better):**

| Protocol | THINK | EH | mean diff | Wilcoxon p (two-sided) | seeds EH better / THINK better |
|---|---|---|---|---|---|
| `pygt` | 1.0212 | 1.0219 | +0.0007 | 0.81 | 2 / 3 |
| `leakfree` | 1.0232 | 1.0243 | +0.0010 | 0.125 | 1 / 4 |

THINK is nominally ahead in the paper's direction but the gap (0.001) is noise-level (with 5 seeds the smallest attainable two-sided exact p is 0.0625); both sit at the mean predictor. No evidence either way. Regenerate with `scripts/compare_eh.py`.

Paper (Table II, p.852): THINK 1.05, TCONV+DHHAN 1.08, STHGCN 1.19, EGCN-H 1.21, RSR-I 1.23, ST-TGCN 1.24, DyGrAE 1.24, TGCN 1.27, DCRNN 1.28, EGCN-O 1.36, GConvGRU 1.38.

## Findings

- The series (in stored row order) has essentially no autocorrelation: persistence is 2.03 (= 2 x variance) and AR(8) equals the mean predictor (1.020). No model can beat about 1.02, and none does. All THINK arms land at 1.021-1.029, i.e. **at, not below, the mean predictor** (0.001-0.009 above); best epochs are 3-7 of 15, so the models learn a near-constant output. Our THINK (1.021) is slightly better than the paper's 1.05, and the paper's baselines (1.19-1.38) are all worse than a constant.
- THINK does not beat the mean or AR baselines (it ties them). R3 is a weak test: it measures "not worse than a constant", not skill.
- Relations and geometry: THINK ~ clique ~ quantile (identical to 4 digits under `pygt`) < EE (+0.0013) < none (+0.002 to +0.006). Paired by seed, hyperedge THINK beats none in 4/5 (pygt) and 5/5 (leakfree) seeds and beats EE in 4/5 and 3/5, but the differences (0.001-0.006) are at noise level with 5 seeds and no tuning. Not evidence that relations or hyperbolic geometry help.
- The protocols agree to within 0.002-0.006; the full-series per-node z-score look-ahead is immaterial.

## Choices (all INFERRED (not in paper) unless noted)

1. Dataset: PyG-T `WindmillOutputLargeDatasetLoader` (`windmilllarge.py`): 319 nodes x 17,472 steps, matching Table I (paper p.850). JSON from the Box link in the loader (47 MB, `data/raw/pygt/windmill_output.json`, git-ignored). Edges: complete weighted graph (319^2 incl. self-loops).
2. Lags 8, horizon 1 (PyG-T default), single channel; `seq=8`, kernel 2. 17,464 windows. The paper gives none.
3. `pygt` protocol: PyG-T per-node z-score over the full series (+1e-10; look-ahead), `temporal_signal_split(0.8)` (val carved from the last 10% of train windows). `leakfree`: chronological 70/10/20 on target time, per-node z-score from the train period only (70/10/20 is also the ratio in the PyG-T index-batching loader).
4. Hyperedges: the complete graph leaves "neighbours" undefined, and the paper's Dice merge threshold is unspecified (App. B, p.854), so no merge was applied. Primary: node + its **top-5 out-neighbours by edge weight** (self-loop excluded), 249 hyperedges after dedupe. Secondary arm: node + all neighbours with weight in the top 10% of all non-self weights (283 hyperedges, larger). Pairwise arm = the top-5 edges as 2-node hyperedges (999).
5. Training as R2 (MSE, Adam lr 5e-3, batch 32, hidden 16, no tuning) **with CPU cost caps**: max 15 epochs, patience 5, 512 randomly sampled training windows per epoch (not all ~12.6k), epoch selection on every 4th validation window, test evaluated once at the selected epoch (no per-epoch test oracle). Models are therefore lightly trained; given the no-signal series this is unlikely to matter. 5 seeds instead of 10.
6. MSE = mean over (window, node) of squared error in z-scored units.
