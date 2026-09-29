# THINK proof of concept — 309 NYSE stocks (Energy/Utilities + Finance)

73 hyperedges; test period = 2017 (237 days); Sharpe = mean/std of daily top-5 return x sqrt(252).

| arm | seeds | test Sharpe (mean ± std) | val Sharpe | NDCG@5 | best-test-epoch Sharpe (paper protocol) | seed-ensemble Sharpe | seed-ensemble NDCG@5 |
|---|---|---|---|---|---|---|---|
| HH_hyper | 10 | 0.694 ± 1.003 | 1.263 | 0.550 | 2.068 ± 0.307 | 1.201 | 0.552 |
| HH_clique | 10 | 0.577 ± 0.821 | 1.298 | 0.554 | 1.926 ± 0.006 | 0.477 | 0.554 |
| HH_none | 10 | 0.130 ± 0.619 | 1.712 | 0.553 | 1.304 ± 0.310 | -0.142 | 0.553 |
| EE_hyper | 10 | -0.404 ± 0.477 | 1.814 | 0.544 | 2.016 ± 0.265 | -0.555 | 0.544 |
| EE_clique | 10 | 0.544 ± 0.676 | 1.627 | 0.552 | 1.840 ± 0.388 | 1.175 | 0.562 |
| EE_none | 10 | 0.066 ± 0.522 | 1.928 | 0.553 | 1.124 ± 0.262 | 0.486 | 0.561 |
| baseline: market | - | 0.752 | - | - | - | - | - |
| baseline: random | - | 0.338 | - | - | - | - | - |
| baseline: momentum | - | -0.276 | - | - | - | - | - |
| baseline: oracle | - | 43.184 | - | - | - | - | - |

| question | seeds | Sharpe diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|---|
| hyperbolic vs Euclidean (with hyperedges) | 10 | +1.099 | 0.0371 | 0.1484 | [+0.213, +2.017] | NO EVIDENCE |
| hyperedges vs pairwise edges (hyperbolic) | 10 | +0.117 | 0.4922 | 0.4922 | [-0.734, +1.176] | NO EVIDENCE |
| relations vs none (hyperbolic) | 10 | +0.564 | 0.1934 | 0.3867 | [-0.524, +1.529] | NO EVIDENCE |
| hyperedges vs pairwise edges (Euclidean) | 10 | -0.948 | 0.0137 | 0.0684 | [-2.412, +0.258] | NO EVIDENCE |
| relations vs none (Euclidean) | 10 | -0.470 | 0.0488 | 0.1484 | [-2.009, +0.796] | NO EVIDENCE |
| interaction: (hyperedge gain in hyperbolic) − (in Euclidean) | 10 | +1.065 | 0.0098 | 0.0586 | [+0.046, +2.544] | NO EVIDENCE |

Verdicts: STRONG = Holm p < 0.01 and CI excludes 0; SEED-ROBUST ONLY = Holm p < 0.01 but CI includes 0 (consistent across seeds, within market noise); NO EVIDENCE otherwise. With n seeds the smallest possible Wilcoxon p is 2/2^n.
