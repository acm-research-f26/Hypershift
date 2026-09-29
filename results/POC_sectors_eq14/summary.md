# THINK proof of concept — 309 NYSE stocks (Energy/Utilities + Finance)

73 hyperedges; test period = 2017 (237 days); Sharpe = mean/std of daily top-5 return x sqrt(252).

| arm | seeds | test Sharpe (mean ± std) | val Sharpe | NDCG@5 | best-test-epoch Sharpe (paper protocol) | seed-ensemble Sharpe | seed-ensemble NDCG@5 |
|---|---|---|---|---|---|---|---|
| HH_hyper | 10 | -0.300 ± 1.429 | 1.741 | 0.543 | 2.190 ± 0.533 | -1.004 | 0.539 |
| HH_clique | 10 | -0.125 ± 0.502 | 1.767 | 0.546 | 1.272 ± 0.778 | -0.793 | 0.541 |
| HH_none | 10 | 0.279 ± 0.350 | 1.757 | 0.550 | 1.108 ± 0.261 | 0.276 | 0.562 |
| EE_hyper | 10 | 0.316 ± 0.812 | 1.835 | 0.547 | 2.075 ± 0.470 | -0.458 | 0.542 |
| EE_clique | 10 | -0.253 ± 0.504 | 1.703 | 0.546 | 1.161 ± 0.572 | -0.682 | 0.541 |
| EE_none | 10 | 0.203 ± 0.583 | 1.565 | 0.551 | 1.184 ± 0.599 | -0.452 | 0.545 |
| baseline: market | - | 0.752 | - | - | - | - | - |
| baseline: random | - | 0.338 | - | - | - | - | - |
| baseline: momentum | - | -0.276 | - | - | - | - | - |
| baseline: oracle | - | 43.184 | - | - | - | - | - |

| question | seeds | Sharpe diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|---|
| hyperbolic vs Euclidean (with hyperedges) | 10 | -0.616 | 0.1309 | 0.7852 | [-1.473, -0.321] | NO EVIDENCE |
| hyperedges vs pairwise edges (hyperbolic) | 10 | -0.175 | 0.5566 | 1.0000 | [-1.275, +0.565] | NO EVIDENCE |
| relations vs none (hyperbolic) | 10 | -0.579 | 0.1309 | 0.7852 | [-1.894, +0.147] | NO EVIDENCE |
| hyperedges vs pairwise edges (Euclidean) | 10 | +0.569 | 0.2324 | 0.9297 | [-0.097, +1.483] | NO EVIDENCE |
| relations vs none (Euclidean) | 10 | +0.113 | 0.8457 | 1.0000 | [-0.838, +0.941] | NO EVIDENCE |
| interaction: (hyperedge gain in hyperbolic) − (in Euclidean) | 10 | -0.745 | 0.2754 | 0.9297 | [-1.782, -0.292] | NO EVIDENCE |

Verdicts: STRONG = Holm p < 0.01 and CI excludes 0; SEED-ROBUST ONLY = Holm p < 0.01 but CI includes 0 (consistent across seeds, within market noise); NO EVIDENCE otherwise. With n seeds the smallest possible Wilcoxon p is 2/2^n.
