# THINK proof of concept — 309 NYSE stocks (Energy/Utilities + Finance)

73 hyperedges; test period = 2017 (237 days); Sharpe = mean/std of daily top-5 return x sqrt(252).

| arm | seeds | test Sharpe (mean ± std) | val Sharpe | NDCG@5 | best-test-epoch Sharpe (paper protocol) | seed-ensemble Sharpe | seed-ensemble NDCG@5 |
|---|---|---|---|---|---|---|---|
| HH_hyper | 10 | 0.480 ± 1.281 | 1.329 | 0.549 | 2.134 ± 0.371 | -0.275 | 0.547 |
| HH_clique | 10 | 0.585 ± 0.953 | 1.319 | 0.552 | 2.021 ± 0.276 | 0.750 | 0.559 |
| HH_none | 10 | 0.484 ± 0.370 | 1.696 | 0.556 | 0.948 ± 0.296 | 0.546 | 0.557 |
| EE_hyper | 10 | -0.128 ± 0.587 | 1.875 | 0.545 | 1.803 ± 0.531 | 1.142 | 0.548 |
| EE_clique | 10 | 0.072 ± 0.378 | 1.720 | 0.551 | 1.285 ± 0.409 | 0.987 | 0.556 |
| EE_none | 10 | 0.061 ± 0.288 | 1.874 | 0.552 | 1.032 ± 0.263 | 0.655 | 0.558 |
| baseline: market | - | 0.752 | - | - | - | - | - |
| baseline: random | - | 0.338 | - | - | - | - | - |
| baseline: momentum | - | -0.276 | - | - | - | - | - |
| baseline: oracle | - | 43.184 | - | - | - | - | - |

| question | seeds | Sharpe diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|---|
| hyperbolic vs Euclidean (with hyperedges) | 10 | +0.608 | 0.2324 | 1.0000 | [-0.451, +1.599] | NO EVIDENCE |
| hyperedges vs pairwise edges (hyperbolic) | 10 | -0.106 | 1.0000 | 1.0000 | [-1.147, +1.311] | NO EVIDENCE |
| relations vs none (hyperbolic) | 10 | -0.005 | 1.0000 | 1.0000 | [-1.706, +1.185] | NO EVIDENCE |
| hyperedges vs pairwise edges (Euclidean) | 10 | -0.200 | 0.6250 | 1.0000 | [-1.260, +0.762] | NO EVIDENCE |
| relations vs none (Euclidean) | 10 | -0.190 | 0.2754 | 1.0000 | [-1.581, +1.013] | NO EVIDENCE |
| interaction: (hyperedge gain in hyperbolic) − (in Euclidean) | 10 | +0.095 | 0.8457 | 1.0000 | [-0.672, +1.292] | NO EVIDENCE |

Verdicts: STRONG = Holm p < 0.01 and CI excludes 0; SEED-ROBUST ONLY = Holm p < 0.01 but CI includes 0 (consistent across seeds, within market noise); NO EVIDENCE otherwise. With n seeds the smallest possible Wilcoxon p is 2/2^n.
