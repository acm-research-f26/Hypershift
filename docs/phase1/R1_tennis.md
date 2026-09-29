# R1: Twitter tennis (DTT) (THINK Table II, paper MSE 0.58)

Code: `src/hypershift/data/pygt.py` (`load_tennis`, `make_windows_tennis`, `tennis_neighbourhood`, `tennis_pairs`), `scripts/run_pygt.py --dataset tennis`, `scripts/summarize_pygt.py`, `tests/test_pygt.py`. Results: `results/R1_tennis/<protocol>/<label>/seed_k/metrics.json` (force-added). CPU only, fixed hyperparameters, **no tuning**. Same pipeline as R2 (`R2_chickenpox.md`).

Paper facts used (`docs/paper/icdm22-think.pdf`): Table I p.850 (DTT: 120 timesteps, 1,000 nodes, delta_hg 1.0, delta_rel "-"); Table II p.852; Sec. IV-A p.851 (node regression = forecast each node's value for a single time step); Sec. IV-B p.852 (MSE); Appendix B p.854 (hyperedges from node neighbourhoods, merged by Sorensen-Dice, threshold unspecified); refs [18] Beres et al. and [20] PyG-Temporal.

## Results (test MSE on log1p mention counts, mean +- std over seeds)

| Arm | `pygt` protocol | `leakfree` protocol |
|---|---|---|
| THINK (hyp/hyp, neighbourhood hyperedges), 10 seeds | 0.405 +- 0.001 | 0.407 +- 0.003 |
| EE (Euclid/Euclid, neighbourhood hyperedges), 10 seeds | 0.329 +- 0.018 | 0.354 +- 0.024 |
| EH (Euclid temporal conv + hyperbolic hypergraph attention = paper's TCONV+DHHAN, neighbourhood hyperedges), 10 seeds | 0.384 +- 0.023 | 0.391 +- 0.019 |
| THINK, structure none (temporal conv only), 10 seeds | 0.284 +- 0.002 | 0.285 +- 0.002 |
| THINK, pairwise/clique (~11k 2-node edges), 5 seeds | 0.357 +- 0.003 | 0.361 +- 0.002 |
| Baseline: global train mean | 0.420 | 0.420 |
| Baseline: per-node train mean | 0.323 | 0.328 |
| Baseline: persistence (previous snapshot's target) | 0.233 | 0.233 |
| Baseline: AR(4) pooled on previous targets | 0.194 | 0.194 |
| Baseline: AR(4) per node | 0.208 | 0.213 |

Paper (Table II, p.852): THINK 0.58, TCONV+DHHAN 0.61, STHGCN 1.03, GConvGRU 2.05, EGCN-O 2.06, DCRNN 2.05, TGCN 2.04, ST-TGCN 2.04, DyGrAE 2.03, EGCN-H 2.04 (RSR-I not run).

Secondary, raw counts (`pygt` protocol, 3 seeds, MSE in count units): mean 13054, per-node mean 11737, persistence 16017, AR(4) 13898; THINK 13048, EE 12422, none 13029. Counts are heavy-tailed (max 13,196); all models collapse to about the mean, so this variant carries no information. log1p is primary.

Caveat: our "EE" makes both temporal and spatial layers Euclidean; the paper's Euclidean variant (TCONV+DHHAN, Sec. V.A p.852, Table II, Fig. 3 caption p.853) is Euclidean temporal conv + hyperbolic hypergraph attention, which is our `EH` arm (same hyperedges, hyperparameters and seeds 0-9 as THINK; run afterwards).

**Paired THINK vs EH (the paper's comparison), 10 seeds, test MSE, EH - THINK (negative = EH better):**

| Protocol | THINK | EH | mean diff | Wilcoxon p (two-sided) | seeds EH better / THINK better |
|---|---|---|---|---|---|
| `pygt` | 0.4049 | 0.3840 | -0.0209 | 0.0098 | 9 / 1 |
| `leakfree` | 0.4069 | 0.3912 | -0.0157 | 0.0137 | 8 / 2 |

The paper reports THINK 0.58 better than TCONV+DHHAN 0.61 (Table II, p.852). Here the sign is reversed: EH is better than THINK, consistently. EH's best epoch is ~93 (pygt) / ~88 (leakfree) of a 100-epoch cap, so it is still improving when stopped (same cap as every arm). Regenerate with `scripts/compare_eh.py`.

Test MSE of the best test epoch (diagnostic): pygt THINK 0.404, EE 0.328, none 0.283, clique 0.356.

## Findings

- THINK (0.405) beats only the global train mean (0.420), marginally. It loses to the per-node mean (0.32), persistence (0.23) and AR(4) (0.19). The AR and persistence baselines see previous **labels**; the neural models see only the structural features, so they are not like-for-like. The fairer comparison is the per-node mean (0.32): the best neural arm (structure none, 0.284) beats it, THINK does not.
- Relations hurt here: none 0.284 < EE 0.33 < clique 0.36 < THINK 0.405. Paired by seed, every seed of none, EE and clique beats THINK (10/10, 10/10, 5/5). Hyperbolic is worse than Euclidean by about 0.08 (pygt) / 0.05 (leakfree). Against the paper's actual ablation (EH), THINK is also worse: EH 0.384 < THINK 0.405 (9/10 and 8/10 seeds, p 0.01), so the paper's THINK > TCONV+DHHAN ordering is not reproduced on tennis. One fixed configuration, no per-arm tuning: an observation, not a verdict.
- The two protocols agree to about 0.02 (pointwise features and target, so they differ only in split and train-only edges). Both use the same 24 test windows.
- Paper scale is not reproduced: the paper's THINK 0.58 and baselines near 2.05 sit far above the constant predictor (0.42) here. The paper's setup (target transform, split, features) evidently differs; our absolute numbers are not comparable to it.

## Choices

Stated by the paper: dataset and size (Table I), next-step node regression, MSE. Everything below is **INFERRED (not in paper)** unless noted.

1. Event `rg17` (Roland-Garros 2017): matches Table I (120 snapshots, 1000 nodes; `uo17` has 112). Data: PyG-T repo `dataset/twitter_tennis_rg17.json`.
2. Target = PyG-T loader convention `log(1 + y_json[min(t+1, T-1)])` (checked in `twitter_tennis.py`: the loader applies `np.log(1.0 + y)`; the JSON stores raw counts). Raw counts are reported as a secondary.
3. Features: PyG-T `feature_mode="encoded"` (5 one-hot bins of capped log-degree + 11 one-hot bins of transitivity = 16 channels). No scaling in either protocol.
4. Windows: 4 consecutive snapshots as lookback (`seq=4`, kernel 2), predicting the loader target of the last snapshot; 117 windows. The paper gives no lookback.
5. Hypergraph is **static**: union of directed mention edges over the training period only (`pygt`: up to the last train window; `leakfree`: snapshots < 84), one hyperedge per node = {v} + out-neighbours (848 / 831 hyperedges after dedupe; singletons dropped). The paper's Sorensen-Dice merge (App. B, p.854: "merged them until no two pairs had an SCD score lower than a threshold") was **not applied**: the threshold is unspecified and the literal rule is degenerate. Per-snapshot hypergraphs (eq. 16 allows them) were not built. Clique arm = the union edges as 2-node hyperedges, self-loops dropped, direction merged (10.7-11.9k).
6. Protocol `pygt`: PyG-T `temporal_signal_split(0.8)` over 117 windows (train 84 / val 9 carved from the train windows / test 24). Protocol `leakfree`: chronological 70/10/20 on the last snapshot (train < snapshot 84, val < 96, same 24 test windows).
7. Training identical to R2: MSE loss, Adam lr 5e-3, batch 32, hidden 16, kernel 2, max 100 epochs, patience 15 on val MSE, best-val weights, seeds 0-9. **Exception:** the clique arm used max 30 epochs / patience 8 and 5 seeds (~35 s per epoch on CPU), so it is handicapped relative to the others (best epoch ~29 of 30).
8. MSE = mean over (window, node) of squared error in log1p units.
9. Baselines use label history (targets of the previous 1-4 snapshots) as inputs; window 0 lacks history and is dropped from the fit. Persistence = previous snapshot's target.
