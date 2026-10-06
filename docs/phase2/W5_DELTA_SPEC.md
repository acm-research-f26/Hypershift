# W5 spec: which settings (if any) reproduce THINK Table I hyperbolicity on our data

Predeclared 2026-10-05, before any grid value is computed. Code reused: `src/hypershift/geometry/hyperbolicity.py`,
`scripts/hyperbolicity.py`, `scripts/hyperbolicity_sensitivity.py`. Prior findings (R9, handoff 2026-09-28 sec. 6):
exact delta_hg on our NYSE graph 1.5; 30-node samples give <=0.5 in 90% of draws; delta_rel 0.16-0.40 by feature, none 0.087.

## Paper targets and definitions (docs/paper/icdm22-think.pdf)
- Table I (p.850): NYSE 1,245 days x 1,737 nodes, delta_hg 0.5, delta_rel 0.087; NASDAQ 1,245 x 1,026, delta_hg 1.0, delta_rel 0.107. TSE/CSE data unavailable, skipped.
- delta_hg: p.850 eq. 2 (minimal delta>0 such that (x,z)_w >= min((x,y)_w,(y,z)_w) - delta for all x,y,z,w), distance = s-walk distance of Algorithm 1 (p.854): s_adj[v][v']=1 iff |H[v] and H[v']| >= s (number of shared hyperedges), then Dijkstra on that graph. **s is not stated** (UNKNOWN).
- delta_rel: p.850 text + Appendix A p.854 eq. 18-19: Gromov product on Euclidean distances between "temporal node features" ([23] per text); delta_rel = 2 delta / diam(W) following [36] Khrulkov. **Which temporal features, normalisation, sample size: not stated** (UNKNOWN). Khrulkov's public code (`hyptorch/delta.py`, fetched 2026-10-05): per try, `idx = np.random.choice(len, batch_size)` (WITH replacement), distance matrix, single base point (index 0), `2*delta/diam`; aggregate mean (and std) of the per-try ratios over `n_tries=10`, `batch_size=1500` defaults. Our primary delta_rel convention follows this; without-replacement is a variant.
- Number of nodes: nothing in the paper states a sampling for delta_hg; sampling is one of the unknowns we scan.

## Grid
**delta_hg** (per market NYSE, NASDAQ; Gromov delta with base point = first sampled node, "single-base", unless noted):
- graph variant G in {v2 (industry+wiki, App. B, cache v2); industry only; wiki only; v2 without the n/a bucket (edges of size>=400 inside the no-industry set); v1 old graph (star for every wiki relation channel, rebuilt)}.
- s in {1,2,3,4,5}.
- node sampling k in {10,20,30,50,100,200,500, full}. Sampled cells: 200 draws for k<=100, 50 draws for k=200,500 (compute adaptation; stated), full cell = full LCC.
- sampling scheme: (a) "dist": sample k nodes from the LCC of the full-graph s-distance matrix, use full-graph distances; (b) "induced": sample k nodes, induce the sub-hypergraph (edges restricted, size>=2), recompute s-distance, take its LCC (a plausible reading if the authors subsampled the data).
- base-point rule: single-base (Khrulkov) vs all-bases (exact 4-point) for k<=100; for full graph, single-base over 64 random bases (distribution reported) and all-bases over those 64 (lower bound on exact), plus true all-1737-bases exact for G=v2, s=1 if time permits.
- disconnected components rule (UNKNOWN in paper): largest connected component (LCC) of the s-adjacency graph; isolated/other-component nodes dropped. Reported with LCC size per (G,s,market).
- Cell value = median over draws (deterministic cells: the value); also report share of draws equal to target.

**delta_rel** (NYSE, NASDAQ):
- features X in: close (norm.) at last train day; close at last day of the series; each of MA5/MA10/MA20/MA30 at last train day; 5-feature vector at last train day; 5-feature vector at last series day; 16-day input window ending last train day flattened, level and relative (apply_input_mode); same two at last series day; daily returns over train (days 1..valid_index-1) and over the full period; per-stock normalised close series over train and over the full period; 5-feature series flattened over train and full. (Missing cells follow the repo loader: FILL=1.1, returns 0.)
- norm in {train (renormalize_train), paper (full-series max, look-ahead)}; relative/returns features are norm-independent and run once.
- subset size m in {500,1000,1500,2000,all}, capped at N (NASDAQ N=1026: 1500, 2000 = all); >=20 subsets (30 used), Khrulkov convention (with replacement, single base 0, mean of per-subset 2*delta/diam); the without-replacement variant is run alongside. For m = all the subsets are the full set with different base points.

## Match tolerance and success rule
- delta_hg cell matches a target if |value - target| <= 0.25 (the half-step; delta on integer-distance graphs is a multiple of 0.5, so this means equality).
- delta_rel cell matches if |mean over subsets - target| <= 0.01.
- A **setting** = (delta_hg part: G, s, scheme, k, base rule) x (delta_rel part: feature, norm, m, replacement convention). **Success = one single setting, identical for NYSE and NASDAQ, that matches all four numbers (NYSE and NASDAQ, delta_hg and delta_rel)**. Because the two procedures are independent, the joint count is (# delta_hg settings matching both markets) x (# delta_rel settings matching both markets); both factors must be >= 1.
- Chance accounting: report, for each market alone and for both jointly, the number of delta_hg cells and delta_rel cells matching, out of the total cells scanned (multiplicity stated). A match counts as evidence for a setting only if the matching cells are few relative to the grid for that market alone and non-empty jointly; a lone cell among hundreds matching one market is chance-level. No tolerance changes after seeing results.
- Interpretation if none: their graph or feature set differs from App. B as we built it, OR the measurement is undocumented. If one: record it and its siblings.
