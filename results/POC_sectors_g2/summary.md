# THINK proof of concept — 309 NYSE stocks (Energy/Utilities + Finance)

558 hyperedges; test period = 2017 (237 days); Sharpe = mean/std of daily top-5 return x sqrt(252).

| arm | seeds | test Sharpe (mean ± std) | val Sharpe | NDCG@5 | best-test-epoch Sharpe (paper protocol) | seed-ensemble Sharpe | seed-ensemble NDCG@5 |
|---|---|---|---|---|---|---|---|
| HH_hyper | 10 | -0.681 ± 1.219 | 1.784 | 0.540 | 2.513 ± 0.421 | -1.267 | 0.536 |
| HH_clique | 10 | -0.542 ± 0.433 | 1.650 | 0.543 | 1.817 ± 0.560 | -1.096 | 0.538 |
| HH_none | 10 | 0.279 ± 0.350 | 1.757 | 0.550 | 1.108 ± 0.261 | 0.276 | 0.562 |
| EE_hyper | 10 | -0.154 ± 0.879 | 1.746 | 0.544 | 2.134 ± 0.862 | -1.013 | 0.537 |
| EE_clique | 10 | -0.531 ± 0.267 | 1.783 | 0.545 | 1.199 ± 0.601 | -0.692 | 0.542 |
| EE_none | 10 | 0.203 ± 0.583 | 1.565 | 0.551 | 1.184 ± 0.599 | -0.452 | 0.545 |
| EH_hyper | 10 | -0.268 ± 1.121 | 1.809 | 0.543 | 2.100 ± 0.976 | 1.991 | 0.558 |
| EH_clique | 10 | 0.158 ± 0.841 | 1.803 | 0.549 | 1.392 ± 0.672 | 1.035 | 0.555 |
| baseline: market | - | 0.752 | - | - | - | - | - |
| baseline: random | - | 0.338 | - | - | - | - | - |
| baseline: momentum | - | -0.276 | - | - | - | - | - |
| baseline: oracle | - | 43.184 | - | - | - | - | - |

| question | seeds | Sharpe diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|---|
| hyperbolic vs Euclidean (with hyperedges) | 10 | -0.528 | 0.2324 | 1.0000 | [-1.212, -0.228] | NO EVIDENCE |
| hyperedges vs pairwise edges (hyperbolic) | 10 | -0.140 | 0.4316 | 1.0000 | [-1.557, +0.560] | NO EVIDENCE |
| relations vs none (hyperbolic) | 10 | -0.961 | 0.0840 | 0.7559 | [-2.226, -0.390] | NO EVIDENCE |
| hyperedges vs pairwise edges (Euclidean) | 10 | +0.377 | 0.4922 | 1.0000 | [-0.623, +1.164] | NO EVIDENCE |
| relations vs none (Euclidean) | 10 | -0.357 | 0.3750 | 1.0000 | [-1.558, +0.448] | NO EVIDENCE |
| hyperbolic vs Euclidean temporal conv (paper ablation, hyperedges) | 10 | -0.414 | 0.1934 | 1.0000 | [-0.678, +0.029] | NO EVIDENCE |
| hyperbolic vs Euclidean temporal conv (paper ablation, pairwise edges) | 10 | -0.700 | 0.0020 | 0.0195 | [-1.379, +0.246] | NO EVIDENCE |
| hyperedges vs pairwise edges (Euclidean temporal conv) | 10 | -0.426 | 0.2754 | 1.0000 | [-1.638, +0.234] | NO EVIDENCE |
| hyperbolic vs Euclidean hypergraph attention (Euclidean temporal conv) | 10 | -0.114 | 0.5566 | 1.0000 | [-0.819, +0.043] | NO EVIDENCE |
| interaction: (hyperedge gain in hyperbolic) − (in Euclidean) | 10 | -0.517 | 0.3223 | 1.0000 | [-1.600, +0.004] | NO EVIDENCE |

Verdicts: STRONG = Holm p < 0.01 and CI excludes 0; SEED-ROBUST ONLY = Holm p < 0.01 but CI includes 0 (consistent across seeds, within market noise); NO EVIDENCE otherwise. With n seeds the smallest possible Wilcoxon p is 2/2^n.
