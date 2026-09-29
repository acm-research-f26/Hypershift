# Fig. 3 degree reconciliation (U6)

Question: which graph construction and degree definition reproduces the paper's Fig. 3 axes? Fig. 3b (hub removal, NYSE) x-axis: node degree 31, 28, 22, 16, 2 (p853, Fig. 3b). Fig. 3a x-axis, "Hyperedge Degree": 500, 15, 9, 5, 3 (p853, Fig. 3a).

Analysis only, CPU, no changes to `src/`. Full NYSE (N = 1737) and NASDAQ (N = 1026) from `data/raw/rsr/data`, using the `hypergraph.py` builders from `de20f8e` (`industry_hyperedges`, `wiki_first_order_channels`, `wiki_hyperedges`, `canonical`). Scripts were run from the scratchpad; the numbers below are reproducible with those builders.

## Paper facts that constrain the answer

- p853 Sec. VI-C: "We identify hubs of the scale-free network by sorting the nodes in decreasing order of their degree and we successively remove the corresponding hubs' hyperedges." No definition of node degree, no schedule.
- [A854] = p854, Appendix B: industry hyperedges plus Wiki hyperedges; first-order = star ("a hyperedge of a source stock and a set of target stocks"); second-order X -R2-> Z <-R3- Y "is pairwise in nature". It does not say the pairs are merged per Z.
- p854 Algorithm 1 ("s-walk distance") and eq. 18-19 (Gromov product, delta hyperbolicity) define no node degree. Algorithm 1 defines s-adjacency, vertices sharing at least s hyperedges (`length(e ∩ ne) >= s`), used for delta_hg. I report the s = 2 count below only as the closest thing the paper defines; it is not a stated degree.
- Fig. 3a's axis label is "Hyperedge Degree" and the largest tick is 500, so "degree" of a hyperedge is its size there (p853). In every full-NYSE variant that keeps the industry channel, max size is 500 (the n/a bucket).

## Table: NYSE (max node degree; top-10 in the second table)

Definitions: **inc** = number of incident hyperedges; **inc>=3** = incident hyperedges of size >= 3; **nbr** = distinct neighbours; **s2** = vertices sharing >= 2 hyperedges (Algorithm 1 adjacency, s = 2).

| Variant | #edges | max size | inc max | inc>=3 max | nbr max | s2 max |
|---|---|---|---|---|---|---|
| (a) v2 (industry + 3 first-order stars + second-order pairs) | 4350 | 500 | 114 | 5 | 596 | 14 |
| (b) old all-star (industry + a star per wiki channel) | 312 | 500 | 37 | 34 | 601 | 113 |
| (c) industry only | 107 | 500 | 1 | 1 | 499 | 0 |
| (d) industry + first-order stars only | 129 | 500 | 19 | 5 | 500 | 2 |
| (e) v2 without the n/a bucket | 4349 | 112 | 114 | 5 | 181 | 14 |
| (f) v2 pairs merged per intermediate entity Z | not computable | | | | | |
| extra: wiki part of v2 only | 4243 | 3 | 113 | 4 | 113 | 2 |
| extra: (d) without n/a | 128 | 112 | 19 | 5 | 111 | 2 |

Note (c): each stock is in exactly one industry (n/a included), so industry-only incident degree is 1 for every stock.

Top-10 values, NYSE, "inc" definition:

| Variant | top-10 inc |
|---|---|
| (a) | 114, 101, 100, 98, 96, 96, 96, 96, 96, 93 |
| (b) | 37, 27, 25, 24, 23, 21, 21, 20, 20, 20 |
| (d) | 19, 6, 3, 2, 2, 2, 2, 2, 2, 2 |
| (e) | 114, 101, 99, 98, 96, 96, 96, 96, 96, 93 |

Top-10 for the other definitions (NYSE): (b) inc>=3: 34, 24, 19, 19, 19, 18, 17, 15, 15, 15; (b) s2: 113, 104, 99, 98, 98, 97, 89, 88, 83, 83; (a) s2: 14 ten times; (a) inc>=3: 5, 3, 2, 2, 2, 2, 2, 1, 1, 1; (a) nbr: 596, 575, 543, 518, 499 (x6).

## Table: NASDAQ (secondary check; paper has no NASDAQ Fig. 3 numbers, footnote 2 on p853 says "similar trends in other datasets")

| Variant | #edges | max size | inc max | inc>=3 max | nbr max | s2 max |
|---|---|---|---|---|---|---|
| (a) v2 | 1066 | 156 | 55 | 1 | 155 | 9 |
| (b) old all-star | 162 | 156 | 17 | 14 | 155 | 49 |
| (c) industry only | 96 | 156 | 1 | 1 | 155 | 0 |
| (d) industry + first-order stars | 101 | 156 | 3 | 1 | 155 | 1 |
| (e) v2 without n/a | 1065 | 132 | 55 | 1 | 134 | 9 |

NASDAQ top-10 "inc" in (a): 55, 54, 51, 49, 46, 45, 45, 42, 41, 40.

## Are the paper's ticks quantiles of the degree distribution?

Only tested on the degree the code uses ("inc"), on v2 NYSE, over stocks with degree > 0 (1720 of 1737):

- Percentiles 90 / 94 / 95 / 96 / 99 / 100 = 11 / 27 / 30.1 / 74 / 85 / 114. The 95th percentile is about 30 and the 94th is 27, near the paper's 31 and 28. The distribution jumps from 30 to 74 between the 95th and 96th percentile (there are 86 stocks of degree >= 31, none of exactly 31), so ticks at 31, 28, 22, 16 would sit inside a single 5%-wide band, while the paper's SR drops steadily across them. I treat this as a coincidence, not a match: nothing else is consistent with it (max degree would be 114, not 31, and a threshold of 2 removes 15% of stocks).
- Old all-star (b), "inc": 95th percentile 5, max 37. No quantile gives 31 with sensible spacing.
- No published tick spacing rule exists (9-10 points, 5 labels), so a quantile reading cannot be tested further from the PDF alone.

Degree at the paper's tick positions, v2 NYSE "inc": stocks with degree >= 31 / 28 / 22 / 16 / 2 are 86 / 93 / 116 / 159 / 263 (5.0% / 5.4% / 6.7% / 9.2% / 15.1%). Stocks with exactly that degree: 0 / 6 / 6 / 1 / 14.

## Variant (f): merge second-order pairs per shared entity Z

Not computable from the data in the repo. `<M>_connections.json` maps `qid_i -> qid_j -> [property paths]`, e.g. `Q806693 -> Q16131515 -> [["P31","P31"],["P414","P414"]]` in NYSE. A second-order path lists only the two properties (R2, R3); the entity Z is not stored. `selected_wiki_connections.csv` lists only property-pair names (e.g. `P361_P361`). Recovering Z needs a fresh Wikidata query (network, out of scope here; and the paper does not say it groups by Z, so this variant is a guess about the authors' method).

## Verdict

No combination reproduces the paper's axes. Looking for max node degree near 31 together with max hyperedge size 500:

- Max size 500 holds in (a), (b), (c), (d) on NYSE. None has max degree near 31: the maxima are 114, 37 (34 counting only edges of size >= 3), 1 and 19.
- Nearest by number: (b) old all-star, 37 incident hyperedges (34 if only edges of size >= 3 are counted, 113 by s2). That is the closest max to 31 and has max size 500, but it is off by 6 (3 for the >= 3 count), and it is the construction the audit found contradicts [A854] Sec. B (second-order should be pairs). It does not reproduce 31, so I do not call it a match.
- (d) (first-order stars only) has max degree 19, too small; the paper's graph must have more relations than that.
- (e) removes the size-500 bucket, so it loses the Fig. 3a start of 500 and does not change max degree (114).
- Neither Algorithm 1 nor eq. 18-19 define a node degree, so that path is closed.

Why the answer is UNKNOWN from the PDF: node degree is never defined, the hub-removal schedule is not given, Fig. 3b's labelled ticks (5) are fewer than its points, and the y-axis is clipped (U5). The hypergraph (industry + first-order stars + second-order pairs) is the [A854] Sec. B reading, and v2 follows it; the paper's 31 may come from a different second-order handling (for example per-Z merging, variant (f), untestable here), a different Wikidata snapshot than RSR's `NYSE_wiki_relation.npy`, or a degree counted differently. Left for Phase 2 (per user instruction).

## Cites

p853 Fig. 3a/3b and Sec. VI-C "Impact of Hyperbolic Learning"; p854 Appendix B "Hypergraph Construction" and Algorithm 1; p854 eq. 18-19 (Gromov product, delta).
