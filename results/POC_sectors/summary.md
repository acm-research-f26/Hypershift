# THINK proof of concept — 309 NYSE stocks (Energy/Utilities + Finance)

73 hyperedges; test period = 2017 (237 days); Sharpe = mean/std of daily top-5 return x sqrt(252).

| arm | seeds | test Sharpe (mean ± std) | val Sharpe | NDCG@5 |
|---|---|---|---|---|
| HH_hyper | 10 | -0.556 ± 0.859 | 1.557 | 0.542 |
| HH_clique | 10 | -0.383 ± 0.746 | 1.610 | 0.544 |
| HH_none | 10 | 0.279 ± 0.350 | 1.757 | 0.550 |
| EE_hyper | 10 | 0.316 ± 0.812 | 1.835 | 0.547 |
| EE_clique | 10 | -0.253 ± 0.504 | 1.703 | 0.546 |
| EE_none | 10 | 0.203 ± 0.583 | 1.565 | 0.551 |
| baseline: market | - | 0.752 | - | - |
| baseline: random | - | 0.338 | - | - |
| baseline: momentum | - | -0.276 | - | - |
| baseline: oracle | - | 43.184 | - | - |

| question | seeds | Sharpe diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|---|
| hyperbolic vs Euclidean (with hyperedges) | 10 | -0.872 | 0.0371 | 0.1855 | [-1.508, -0.436] | NO EVIDENCE |
| hyperedges vs pairwise edges (hyperbolic) | 10 | -0.173 | 0.9219 | 1.0000 | [-1.223, +0.899] | NO EVIDENCE |
| relations vs none (hyperbolic) | 10 | -0.835 | 0.0273 | 0.1641 | [-2.059, +0.170] | NO EVIDENCE |
| hyperedges vs pairwise edges (Euclidean) | 10 | +0.569 | 0.2324 | 0.6973 | [-0.097, +1.483] | NO EVIDENCE |
| relations vs none (Euclidean) | 10 | +0.113 | 0.8457 | 1.0000 | [-0.838, +0.941] | NO EVIDENCE |
| interaction: (hyperedge gain in hyperbolic) − (in Euclidean) | 10 | -0.743 | 0.1309 | 0.5234 | [-1.547, -0.156] | NO EVIDENCE |

Verdicts: STRONG = Holm p < 0.01 and CI excludes 0; SEED-ROBUST ONLY = Holm p < 0.01 but CI includes 0 (consistent across seeds, within market noise); NO EVIDENCE otherwise. With n seeds the smallest possible Wilcoxon p is 2/2^n.
