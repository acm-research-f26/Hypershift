# THINK proof of concept — 309 NYSE stocks (Energy/Utilities + Finance)

558 hyperedges; test period = 2017 (237 days); Sharpe = mean/std of daily top-5 return x sqrt(252).

| arm | seeds | test Sharpe (mean ± std) | val Sharpe | NDCG@5 | best-test-epoch Sharpe (paper protocol) | seed-ensemble Sharpe | seed-ensemble NDCG@5 |
|---|---|---|---|---|---|---|---|
| HH_hyper | 10 | 0.069 ± 0.643 | 1.872 | 0.549 | 1.318 ± 0.479 | 0.125 | 0.552 |
| HH_clique | 10 | 0.022 ± 0.945 | 1.707 | 0.549 | 1.562 ± 0.491 | -0.295 | 0.546 |
| HH_none | 10 | 0.520 ± 0.526 | 1.729 | 0.556 | 1.293 ± 0.262 | 1.202 | 0.569 |
| EE_hyper | 10 | -0.243 ± 0.491 | 1.821 | 0.547 | 1.506 ± 0.186 | -0.633 | 0.541 |
| EE_clique | 10 | 0.008 ± 0.639 | 1.749 | 0.550 | 1.150 ± 0.540 | 0.411 | 0.555 |
| EE_none | 10 | 0.215 ± 0.469 | 1.657 | 0.553 | 1.247 ± 0.421 | 0.505 | 0.559 |
| EH_hyper | 10 | 0.172 ± 0.831 | 1.962 | 0.548 | 1.686 ± 0.281 | 0.237 | 0.551 |
| EH_clique | 10 | 0.342 ± 0.369 | 2.060 | 0.554 | 1.365 ± 0.457 | 0.569 | 0.552 |
| baseline: market | - | 0.752 | - | - | - | - | - |
| baseline: random | - | 0.338 | - | - | - | - | - |
| baseline: momentum | - | -0.276 | - | - | - | - | - |
| baseline: oracle | - | 43.184 | - | - | - | - | - |

| question | seeds | Sharpe diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|---|
| hyperbolic vs Euclidean (with hyperedges) | 10 | +0.312 | 0.3223 | 1.0000 | [-0.401, +0.930] | NO EVIDENCE |
| hyperedges vs pairwise edges (hyperbolic) | 10 | +0.047 | 0.9219 | 1.0000 | [-0.977, +1.182] | NO EVIDENCE |
| relations vs none (hyperbolic) | 10 | -0.451 | 0.1055 | 0.9492 | [-1.597, +0.261] | NO EVIDENCE |
| hyperedges vs pairwise edges (Euclidean) | 10 | -0.251 | 0.1602 | 1.0000 | [-1.313, +0.745] | NO EVIDENCE |
| relations vs none (Euclidean) | 10 | -0.459 | 0.0488 | 0.4883 | [-2.014, +0.770] | NO EVIDENCE |
| hyperbolic vs Euclidean temporal conv (paper ablation, hyperedges) | 10 | -0.102 | 0.9219 | 1.0000 | [-0.553, +0.721] | NO EVIDENCE |
| hyperbolic vs Euclidean temporal conv (paper ablation, pairwise edges) | 10 | -0.320 | 0.3750 | 1.0000 | [-1.361, +0.055] | NO EVIDENCE |
| hyperedges vs pairwise edges (Euclidean temporal conv) | 10 | -0.170 | 0.4922 | 1.0000 | [-1.882, +0.477] | NO EVIDENCE |
| hyperbolic vs Euclidean hypergraph attention (Euclidean temporal conv) | 10 | +0.415 | 0.2754 | 1.0000 | [-0.447, +0.795] | NO EVIDENCE |
| interaction: (hyperedge gain in hyperbolic) − (in Euclidean) | 10 | +0.298 | 0.6250 | 1.0000 | [-0.442, +1.223] | NO EVIDENCE |

Verdicts: STRONG = Holm p < 0.01 and CI excludes 0; SEED-ROBUST ONLY = Holm p < 0.01 but CI includes 0 (consistent across seeds, within market noise); NO EVIDENCE otherwise. With n seeds the smallest possible Wilcoxon p is 2/2^n.
