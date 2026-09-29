# THINK proof of concept — 309 NYSE stocks (Energy/Utilities + Finance)

558 hyperedges; test period = 2017 (237 days); Sharpe = mean/std of daily top-5 return x sqrt(252).

| arm | seeds | test Sharpe (mean ± std) | val Sharpe | NDCG@5 | best-test-epoch Sharpe (paper protocol) | seed-ensemble Sharpe | seed-ensemble NDCG@5 |
|---|---|---|---|---|---|---|---|
| HH_hyper | 10 | 0.007 ± 0.781 | 1.189 | 0.551 | 1.924 ± 0.000 | -0.370 | 0.549 |
| HH_clique | 10 | 0.463 ± 0.458 | 1.305 | 0.552 | 1.924 ± 0.000 | 0.164 | 0.556 |
| HH_none | 10 | 0.484 ± 0.370 | 1.696 | 0.556 | 0.948 ± 0.296 | 0.546 | 0.557 |
| EE_hyper | 10 | -0.431 ± 0.280 | 1.831 | 0.546 | 1.709 ± 0.313 | -0.344 | 0.542 |
| EE_clique | 10 | 0.009 ± 0.513 | 1.683 | 0.551 | 0.988 ± 0.257 | 0.916 | 0.556 |
| EE_none | 10 | 0.061 ± 0.288 | 1.874 | 0.552 | 1.032 ± 0.263 | 0.655 | 0.558 |
| EH_hyper | 10 | -0.297 ± 0.249 | 1.723 | 0.547 | 1.629 ± 0.230 | -0.906 | 0.544 |
| EH_clique | 10 | 0.023 ± 0.478 | 1.581 | 0.552 | 1.064 ± 0.585 | 0.160 | 0.551 |
| baseline: market | - | 0.752 | - | - | - | - | - |
| baseline: random | - | 0.338 | - | - | - | - | - |
| baseline: momentum | - | -0.276 | - | - | - | - | - |
| baseline: oracle | - | 43.184 | - | - | - | - | - |

| question | seeds | Sharpe diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|---|
| hyperbolic vs Euclidean (with hyperedges) | 10 | +0.438 | 0.1055 | 0.7383 | [-0.226, +1.110] | NO EVIDENCE |
| hyperedges vs pairwise edges (hyperbolic) | 10 | -0.456 | 0.0488 | 0.3906 | [-1.579, +0.562] | NO EVIDENCE |
| relations vs none (hyperbolic) | 10 | -0.477 | 0.1602 | 0.7383 | [-1.971, +0.542] | NO EVIDENCE |
| hyperedges vs pairwise edges (Euclidean) | 10 | -0.439 | 0.0137 | 0.1367 | [-1.624, +0.485] | NO EVIDENCE |
| relations vs none (Euclidean) | 10 | -0.492 | 0.0195 | 0.1758 | [-1.980, +0.723] | NO EVIDENCE |
| hyperbolic vs Euclidean temporal conv (paper ablation, hyperedges) | 10 | +0.305 | 0.3223 | 0.8262 | [-0.235, +0.826] | NO EVIDENCE |
| hyperbolic vs Euclidean temporal conv (paper ablation, pairwise edges) | 10 | +0.439 | 0.1055 | 0.7383 | [-0.250, +1.082] | NO EVIDENCE |
| hyperedges vs pairwise edges (Euclidean temporal conv) | 10 | -0.321 | 0.2754 | 0.8262 | [-1.369, +0.623] | NO EVIDENCE |
| hyperbolic vs Euclidean hypergraph attention (Euclidean temporal conv) | 10 | +0.133 | 0.1055 | 0.7383 | [-0.354, +0.691] | NO EVIDENCE |
| interaction: (hyperedge gain in hyperbolic) − (in Euclidean) | 10 | -0.016 | 0.9219 | 0.9219 | [-0.707, +0.749] | NO EVIDENCE |

Verdicts: STRONG = Holm p < 0.01 and CI excludes 0; SEED-ROBUST ONLY = Holm p < 0.01 but CI includes 0 (consistent across seeds, within market noise); NO EVIDENCE otherwise. With n seeds the smallest possible Wilcoxon p is 2/2^n.
