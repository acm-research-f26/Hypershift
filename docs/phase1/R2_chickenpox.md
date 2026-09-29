# R2: Hungary Chickenpox (THINK Table II, paper MSE 1.09)

Code: `src/hypershift/data/pygt.py`, `scripts/run_pygt.py`, `tests/test_pygt.py`. Results: `results/R2_chickenpox/<protocol>/<arm>/seed_k/metrics.json` (force-added; history/config too are local). CPU only, 10 seeds per arm (~10-40 s per run), fixed hyperparameters, **no tuning**.

## Results (test MSE, mean ± std over 10 seeds, next-week node regression)

| Arm | `pygt` protocol | `leakfree` protocol |
|---|---|---|
| THINK (hyp/hyp, neighbourhood hyperedges) | 0.956 ± 0.007 | 0.957 ± 0.010 |
| EE (Euclid/Euclid, neighbourhood hyperedges) | 0.886 ± 0.003 | 0.890 ± 0.006 |
| EH (Euclid temporal conv + hyperbolic hypergraph attention = paper's TCONV+DHHAN) | 0.969 ± 0.009 | 0.964 ± 0.008 |
| THINK, structure none (temporal conv only) | 0.733 ± 0.006 | 0.738 ± 0.007 |
| THINK, pairwise/clique (41 undirected 2-node edges) | 0.823 ± 0.017 | 0.821 ± 0.010 |
| Baseline: train mean | 1.047 | 1.060 |
| Baseline: persistence y_t = y_{t-1} | 3.013 | 3.054 |
| Baseline: AR(4), pooled lstsq (+intercept) | 0.725 | 0.737 |
| Baseline: AR(4), per node | 0.734 | 0.748 |

**Paired THINK vs EH (paper's comparison; Sec. V.A p.852, Table II), 10 seeds, EH - THINK (positive = THINK better):**

| Protocol | THINK | EH | mean diff | Wilcoxon p (two-sided) | seeds EH better / THINK better |
|---|---|---|---|---|---|
| `pygt` | 0.9561 | 0.9686 | +0.0124 | 0.037 | 1 / 9 |
| `leakfree` | 0.9571 | 0.9640 | +0.0069 | 0.084 | 2 / 8 |

THINK is better than EH in the paper's direction (paper: 1.09 vs TCONV+DHHAN, Table II), by a small margin: significant at 0.05 under `pygt`, not under `leakfree`. Uncorrected, untuned, one configuration. Regenerate with `scripts/compare_eh.py`.

Paper (Table II): THINK 1.09, baselines 1.11-1.14. (The paper's numbers are worse than the train-mean predictor; ours are below it.)

Test MSE of the best test epoch (diagnostic, not selection): pygt THINK 0.931, EE 0.877, none 0.719, clique 0.796.

## Findings

- All THINK arms beat the mean predictor (1.05) and persistence (3.0) under both protocols. **None beats the linear AR(4)** (0.725 / 0.737): the best arm, structure-none, ties it (0.733 vs 0.725; 0.738 vs 0.737).
- Adding the spatial layer makes things worse here: none (0.73) < clique (0.82) < EE (0.89) < THINK (0.96). Hyperbolic is worse than Euclidean by about 0.07. This is a single fixed configuration with no per-arm tuning, so it is an observation, not a verdict (CLAUDE.md: never tune one arm only, and none were tuned).
- The two protocols agree to within ~0.005: the paper protocol's look-ahead in standardization (full-series z-score) is immaterial at this scale.
- The paper's 1.09 is not a meaningful skill target (worse than a constant); our THINK reaches 0.96, in the same region but better than the constant.

## Inferred choices (the paper specifies none of these)

1. Data: the PyG-T JSON `FX` [521, 20] is **already** globally standardized (mean ~0, std 1 over the whole series, including the test period). Raw counts are not stored. Edges: 102 directed, including self-loops.
2. Lags = 4 (PyG-T default), horizon 1, single channel. Window w predicts time w+4: 517 windows. THINK's temporal conv needs `seq % kernel == 0` and `tconv2` kernel `seq/kernel`, so `seq=4, kernel=2` (n=2, K=2) as R_feasibility suggested. No adjustment needed.
3. Hypergraph: the paper's Sorensen-Dice merge ([A854] Sec. B: pairs are merged "until no two pairs had an SCD score lower than a threshold") is legible but gives no threshold value, and the stopping rule as printed reads inverted, so it was **not** applied. **This is a deviation from a stated step**, not the resolution of a garbled sentence; the hypergraph the paper actually used is unknown. Used one neighbourhood hyperedge per node = {v} + out-neighbours (20 edges, no duplicates after dedupe). Pairwise arm = the original edges as 2-node hyperedges, self-loops dropped and direction merged (41 edges).
4. Protocol `pygt`: PyG-T `temporal_signal_split(train_ratio=0.8)` over the 517 windows (train 413 / test 104), stored standardization. PyG-T has no validation set; the last 10% of the train windows (41) are carved out for early stopping (train 372 / val 41 / test 104).
5. Protocol `leakfree`: chronological 70/10/20 on target time (365 / 52 / 104 target steps, windows may use earlier inputs), re-standardized with a global scalar mean/std of the train period only. Since the stored series is already an affine transform of raw counts this is an affine change; it removes the full-series statistics but cannot restore raw scale.
6. Training: MSE loss only, Adam lr 5e-3, batch 32 windows, hidden 16, max 100 epochs, patience 15 on validation MSE, best-val weights restored. Seeds 0-9. Hyperparameters were set once and not tuned (no test-set peeking beyond a single 1-seed smoke run per arm).
7. MSE = mean over (window, node) of squared error, in standardized units. Val MSE is much lower than test (0.4 vs 0.95): the validation weeks are a low-variance stretch.
8. Baselines are fitted on the same train windows as the models (AR(4) via `numpy.lstsq` with intercept), evaluated on the same test windows. The `mean` baseline is the mean of train targets (not exactly 0 under `pygt` because the series is standardized on the full length).
