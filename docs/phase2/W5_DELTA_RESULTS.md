# W5 results: do any computation settings reproduce THINK Table I (delta_hg, delta_rel) on our data?

Spec (predeclared, committed before computing, `4403a10`): `W5_DELTA_SPEC.md`. Code: `scripts/w5_delta_grid.py`, `scripts/w5_delta_analysis.py`, new functions in `src/hypershift/geometry/hyperbolicity.py`, tests in `tests/test_hyperbolicity.py`. Full tables (every cell): `W5_DELTA_RESULTS_tables.md`; machine-readable summary `W5_DELTA_RESULTS.json`; raw per-draw data `results/w5_delta/*.jsonl` (git-ignored).
Paper targets (p.850 Table I): NYSE delta_hg 0.5, delta_rel 0.087; NASDAQ delta_hg 1.0, delta_rel 0.107. Tolerances: +-0.25 (delta_hg), +-0.01 (delta_rel).

## Grid actually run (adaptations stated)
- delta_hg: 2 markets x 5 graphs (v2 App. B build, industry only, wiki only, v2 without the n/a bucket, v1 old star-for-every-relation) x s in 1..5 x {sampled "dist" (full-graph distances), sampled "induced" (recompute s-walk distance on the induced sub-hypergraph)} x k in {10,20,30,50,100,200,500} x {single base point, all base points (k<=100 only)} + the full largest connected component (24 random single bases) = 870 cells (437 NYSE, 433 NASDAQ). 200 draws for k<=100, **50 draws for k=200,500** (budget). Disconnected rule: largest component. Exact all-bases delta (every node a base point) run for v2, s=1.
- delta_rel: 20 feature/norm/subset-size combos per market x {500,1000,1500,2000,all} x {with replacement (Khrulkov convention), without} = 140 cells per market, **20 subsets for with-replacement; without-replacement only for m<=1000 with 10 subsets** (budget). The 1-D features (close or one MA at a single day) lie on a line, so delta_rel = 0 exactly (measured 2e-7), they were measured once and then dropped from the grid. NASDAQ has 1026 stocks, so 1500/2000/all coincide there. A previous 80-task attempt was killed and reduced (the machine was shared and RAM-limited); graph building happens once in a parent process because each worker loading the 2.4 GB relation tensor ran out of memory.
- Not done: TSE/CSE (no data); the Khrulkov sample-with-replacement convention is not applied to delta_hg (the paper gives no sampling).

## Key numbers
**delta_hg on the paper-faithful graph (v2, s=1; NYSE 1737 nodes, 4350 edges, LCC 1530; NASDAQ 1026 nodes, 1066 edges, LCC 601):**
| | NYSE (paper 0.5) | NASDAQ (paper 1.0) |
|---|---|---|
| exact, all base points | **1.5** | **1.5** |
| full LCC, single base: min / median over bases | 1.0 / 1.5 (24 bases); over all 1530 bases min 1.0, median 1.5 | min 1.0, median 1.0 (all 601 bases) |
| sampled "dist" k=10,20,30 (single base, median) | 0.5 (share of draws =0.5: 0.58, 0.85, 0.79) | 0.5 (0.03, 0.14, 0.23) |
| k=50 / 100 / 200 / 500 | 1.0 / 1.0 / 1.0 / 1.0 | 0.5 / 1.0 / 1.0 / 1.0 |
| "induced" k=500 (single base) | 1.0 | 1.0 |

So the full-graph value reproduces NASDAQ only for some base points (its median 1.0, exact 1.5) and never NYSE (no base point gives 0.5, min 1.0). NYSE 0.5 appears only for samples of at most 30 nodes (dist) and NASDAQ 1.0 only for samples of at least 100 (dist), so one sample size does not give 0.5 and 1.0 in the right order except at k=30 with all-base points (NYSE 0.5, share 0.61; NASDAQ 1.0, share 0.52).
- s>=2 is not a measurement of the 1737-node graph: the s-adjacency graph collapses (v2 LCC at s=2: NYSE 19 nodes, NASDAQ 10; s>=3: 1 node). Only s=1 on v2/v2_no_na/v1_old is meaningful; v1_old (the pre-fix graph) keeps LCC 229/82 at s=2 and 68-119 / 12-33 nodes at s=3-5, where delta=0.5 at every k is a small-component artefact.
- Industry-only graph: disjoint cliques, delta = 0 everywhere (degenerate); wiki-only at s=1: LCC 253 / 106, full-LCC single-base median 1.0 on both markets.

**delta_rel (Khrulkov convention, mean of 2 delta/diam over subsets):** ranges 0.045 to 0.41 (NYSE) and 0.033 to 0.38 (NASDAQ) over the grid. By feature, all-stocks subset (m = all):
- close-price series (train or full period, any normalisation) and the flattened 5-feature series: 0.20 to 0.41 (way above target).
- daily returns: NYSE 0.16 / 0.18 (train / full), NASDAQ 0.26 / 0.26.
- last-day 5-feature vector / 16-day level window under the paper (full-series-max) normalisation: NYSE 0.106 / 0.101, NASDAQ 0.088 / 0.110 (full period last day); at the last train day NYSE 0.070 / 0.153, NASDAQ 0.081 / 0.164. Under train normalisation: 0.04 to 0.19.
- Closest single cells: NYSE feat5 last-series-day, paper norm, m=500: 0.0850; NASDAQ win16 level last-series-day, paper norm, m=500 without replacement 0.1047. The only (feature, norm, m, replacement) cell within +-0.01 for both markets is **win16_level_fu, paper norm, m=500, with replacement: NYSE 0.0934, NASDAQ 0.1028**. It is not stable under the subset size: at m=1000/1500/all NYSE moves to 0.099/0.107/0.101 (misses by 0.012 to 0.020), so the match depends on m=500 and on the feature-day choice.

## Joint-match verdict
Counts of cells within tolerance (full list in the tables file):
| | NYSE alone | NASDAQ alone | both (same setting) | expected for both if independent |
|---|---|---|---|---|
| delta_hg, all cells | 93 of 437 | 53 of 433 | 14 of 429 | 11.5 |
| delta_hg, non-degenerate cells (LCC >= 200 in both markets, k < LCC) | 21 | 22 | 1 of 78 | 5.9 |
| delta_rel | 4 of 140 | 6 of 140 | 1 of 140 | 0.17 |

- The 14 delta_hg joint cells: 13 sit on the old (wrong) graph at s=3-5 or s=2 with tiny components, or on tiny samples; the single non-degenerate one is v2, s=1, "dist", k=30, all base points. The joint delta_hg count (14 vs 11.5 expected) and the non-degenerate one (1 vs 5.9 expected) are what chance gives.
- Predeclared rule: a single setting must match all four numbers. Formally the grid contains one delta_rel setting that matches both markets (win16_level_fu, paper norm, m=500, with replacement) and one non-degenerate delta_hg setting that matches both (v2, s=1, dist, k=30, all bases), so the rule is **technically satisfied by one pairing**. We do **not** treat that as a reproduction: 870+280 cells were scanned; each half is at or below the number of matches chance would give, and each half is a single fragile cell next to cells with different k or m that miss. The delta_rel joint cell sits at the same feature family (last-day levels, with the paper's full-series-max normalisation, i.e. look-ahead) as the near misses, which is the one non-chance-looking pattern, but its NYSE value at m>=1000 is outside tolerance.
- Interpretation: no setting reproduces Table I robustly on our data. The exact delta_hg of the App. B graph is 1.5 for both markets (paper 0.5, 1.0), the paper's 0.5 for NYSE needs a sample of at most about 30 nodes (undocumented in the paper), and delta_rel matches only for last-day level features of the data files as shipped (paper normalisation) at m=500. Either their hypergraph differs from App. B as we built it, or their measurement (sampling size, feature definition, normalisation) is undocumented; the grid cannot tell which. Conclusion (honest): the paper's Table I hyperbolicity numbers are not reproduced; they are compatible with small-sample estimates of our graph (delta_hg) and with one undocumented feature choice (delta_rel), neither identifiable.
