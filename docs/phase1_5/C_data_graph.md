# Phase 1.5 C: data and graph fidelity audit

Scope: full NYSE (N = 1737) and NASDAQ (N = 1026) from `data/raw/rsr/data`, CPU only. Paper source is `docs/paper/icdm22-think.pdf`: Table I (p850), Fig. 3 (p853), Appendix B (p854). Numbers below are programmatically checked in `tests/test_phase15_data.py` (11 tests; full suite 269 passed). The independent rebuild lives in that test file (`_rebuild`, `_wiki_raw`) and does not import `hypergraph.py`.

## 1. Data audit

**Shapes and what is dropped.**
- NYSE: 1737 tickers x 1245 rows x 6 columns (index + 5 features). Table I (p850) lists NYSE 1,245 timesteps / 1,737 nodes and NASDAQ 1,245 / 1,026. Both match.
- `data/raw/rsr/data/2013-01-01/` holds 1769 NYSE files, 32 more than the ticker list (AOI, APH, CAA, ..., VRX). They are not in the ticker file, the `_wiki.csv` or the industry JSON, so they are unused. NASDAQ: 1048 files, 22 unused.
- NASDAQ raw files have 1246 rows; the loader drops the last (2017-12-11). That row is **not all missing** (CLAUDE.md says it is): 474 of 1026 stocks are missing, 552 have real values. Dropping it is needed to reach T = 1245, which Table I states. Cost: for 24 NASDAQ stocks the "paper" scale (max close over 1246 rows) is attained on the dropped row, so their max close is 0.966 to 0.9999 instead of 1. Irrelevant for the default `norm: train`.
- Every file has a sequential index column 0..T-1 (checked). No NaN or inf.

**Ticker order.** Order is the ticker file order (`NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv`). It equals the row order of `NYSE_wiki.csv` (checked, identical list), and the industry JSON covers exactly the same 1737 tickers once each. Industry and wiki relation matrices are indexed in that order (the independent rebuild of section 2 reproduces the cached graph from JSON by ticker, which would fail if the orders differed).

**Date vector.** `NYSE_aver_line_dates.csv` has 1274 dates (2012-11-19 .. 2017-12-08), strictly increasing, no weekends, max gap 4 days. 1274 - 1245 = **29**: the file's row 0 is `dates[29]`. This matches the MA30 warm-up (29 prior days). External anchor (my own recollection, not in repo): AAPL's largest early drop in the data is -12.3507% at row 15, which maps to 2013-01-24 only with offset 29 (known AAPL -12.35% earnings day). Splits fall on `dates[29+756]` = 2016-01-04 (val start) and `dates[29+1008]` = 2017-01-03 (test start). NASDAQ: 1275 dates, last = 2017-12-11 (the dropped row). Nothing in the repo loads the date vector (`timestamps=None` for RSR); row-to-date is only by this offset rule.

**Features exactly as stored.** Columns are `[idx, MA5, MA10, MA20, MA30, close]`. Verified: for every row t >= w with a full window of observed closes, MAw equals the trailing mean of the last w closes (NYSE, every 11th stock, 300k rows per MA; max abs error < 1e-5, median 2e-7). So the column order is as the loader assumes. Each stock is divided by its full-series max close ("paper" norm): max close = 1.000000 for all 1737 NYSE stocks. Loader's `norm: train` rescales by the train-period max close (`renormalize_train`). MAs of the first ~40 rows (max row 40 in NYSE) can exceed 1 (74 NYSE cells > 1, max 1.054) because they average the 29 pre-sample days, which are not in the file.

**Return target.** `gt[t] = (close[t] - close[t-1]) / close[t-1]` when both days are observed, else 0 (`rsr.py` `parse_eod`). Recomputed independently from raw closes: matches to 1e-6 for all 1737 x 1245. Scale-invariant, so identical under paper and train norm. Extremes: min -0.495, max +0.982, 8 days with |ret| > 0.5 (NYSE); 15 in NASDAQ. Not winsorised.

**Missing values.** Missing marker -1234 appears in all five columns together, or only in the close-keyed pattern below:
- NYSE: **1389 stock-days** missing close of 2,162,565 (0.064%); 114 stocks have any (max 70 days for one stock); no stock is fully missing; no day has more than 36 stocks missing. Breakdown: 656 leading (36 stocks, pre-listing), 34 trailing (4 stocks), 699 interior (85 stocks).
- NASDAQ (after dropping the last row): 1728 of 1,277,370 (0.135%); 164 stocks.
- Pattern of missingness: 1058 NYSE cells have all five values missing; **331 have close missing but real MAs** (NASDAQ 514). The loader keys the mask on the close only and replaces every missing cell with FILL = 1.1. So on those 331 days the close input is 1.1 while MAs are real; no day has the close present and MA missing.
- **Fabricated returns.** `gt` is stored as 0 on 349 NYSE "orphan" stock-days (observed today, missing yesterday; NASDAQ 539) and `mask` is 1 there (checked: `mask` alone does not protect them). The training/eval path is safe: `gather_batch` builds the target mask as the `min` of `mask` over all input days **and** the target day, and the orphan day's predecessor is the last input day, so these targets get mask 0 (checked on real data). The same `min` also masks any stock with a missing day inside the window, which is stricter than a per-target-day mask (stocks with an interior gap are excluded from the loss for the next `seq` days). Whether the original RSR/STHAN code does the same is not stated in the paper; UNKNOWN, not checked.
- Leak-through risk worth knowing: a masked stock still enters the graph. Its 1.1 fill values are inputs to temporal convolution and are aggregated into its hyperedge neighbours. Affected share is small (0.06% NYSE stock-days), so impact is low.

## 2. Graph audit (Appendix B, p854)

Paper text (p854): industry hyperedges plus Wiki hyperedges; first-order: "a hyperedge of a source stock and a set of target stocks related to it via the same Wikidata relation"; second-order: X -R2-> Z <-R3- Y, "pairwise in nature".

**Raw files.** `NYSE_wiki.csv` gives each stock a Wikidata QID; **1140 of 1737 NYSE stocks (65.6%) have QID `unknown`** (NASDAQ 514 of 1026). `NYSE_connections.json` has 597 source QIDs, all real. `selected_wiki_connections.csv` lists 57 path types (e.g. `P361_P361`, `P127`). `NYSE_connections.json` holds 59 distinct path types, 29 of length 1 and the rest length 2; 27 of them are not in the selected list (`P31_P31`, `P414_P414`, `P17_P17`, `P1454_P1454`, ... together ~740k qid pairs, mostly "instance of / stock exchange / country") and are excluded from the npy and from my rebuild. The 32 non-self channels of `NYSE_wiki_relation.npy` each equal the ordered pair set of a selected json type (all 32 matched; two types with a single identical pair each, `P1830_P127`/`P1830_P749`, map to two channels). So npy and json agree.

**First-order stars: per source x relation type, not one merged star.**
- Code: `hypergraph.py` lines 118-123 loop over wiki channels `k`; for a first-order channel it emits `for i in nonzero(a.sum(axis=1)>0): out.append((i, *targets of i in channel k))`. A star is keyed by (source i, channel k). There is no `any(axis=2)` merge across channels.
- Independent rebuild: keys `(source, property)` from `connections.json` paths of length 1.
- NYSE: 23 stars, 23 distinct sources, so none has two relation types; all are `P127`/`P749`/`P355`; size 2 (19) or 3 (4). Examples: `HSBC -P127-> {BLK, JPM}` -> edge {HSBC, BLK, JPM}; `WMT -P127-> {STT}` -> {WMT, STT}. NASDAQ: 9 stars from 7 sources; GOOGL has two stars (`P355` -> GOOG and `P155` -> GOOG) that are different (source, relation) keys but identical node sets, so they collapse to one edge by deduplication. A merged-per-source star would differ only when a source has distinct target sets under distinct relations, which does not occur in NYSE.
- Live check of direction (external, today's Wikidata, not the 2019 snapshot): `wd:Q483551 (WMT) wdt:P127 wd:Q2037125 (STT)` holds, so first-order is X -R-> Y with X the source. Matches the npy convention `rel[i, j, k] = 1` for source i.

**Second-order: X -R2-> Z <-R3- Y.**
- Both `rel` and `connections.json` store only the ordered path-type `(R2, R3)` per ordered stock pair; **Z is not stored** anywhere in the repo (`selected_wiki_connections.csv` has only type names and counts).
- Consequences: (a) a pair (i, j) in channel `(R2, R3)` means "there is at least one Z"; it can arise from several different Z with no way to tell. For example `P361_P361` on NYSE (part of) gives 2709 pairs in only 3 connected components; if each is a clique it is one Z (e.g. a shared index/group) each. (b) Pairs from different Z are merged into the same pair edge. This is harmless for the pairwise-in-nature reading of the paper but prevents building per-Z hyperedges.
- Directions and the "wrong pattern" question. Evidence that the stored pairs follow converging X -> Z <- Y and not a chain X -> Z -> Y: all 16 (NYSE) and 26 (NASDAQ) asymmetric types have an exact transposed partner type `(R3, R2)` whose pair set is the transpose, and all same-property types are symmetric (checked). A chain pattern would not give transposed pair sets in general. This does **not** separate converging (X -> Z <- Y) from diverging (X <- Z -> Y), which are both transposition-symmetric. Live Wikidata spot check (external, today's snapshot): for the `P127_P355` pair (SKX, PNC), `SKX -P127-> Z` and `PNC -P355-> Z` both hold with Z = BlackRock (Q219635) (converging); the reversed form `Z -P355-> PNC` returns nothing. One pair, one type, 2026 snapshot: weak evidence only. So whether chain or diverging paths leak in cannot be excluded from the repo files alone. The excluded high-volume types are all same-property types (P31, P414, P17...), so no hub via instance-of leaks in.
- Types included in NYSE: 29 second-order types; top by pairs: `P361_P361` 5418, `P452_P452` 2332, `P127_P127` 570, `P1056_P1056` 108.

**Duplicates, singletons, isolated stocks (NYSE).**
- Raw edge records: 130 industry, 23 first-order stars, 9112 second-order pair records. Ordered pair incidences by type = 8738, which collapse to 4222 unique unordered pairs; 220 unordered pairs sit in 2 or more channels (max 5).
- Dropped by `canonical`: 23 singleton industries (size 1; 130 industries -> 107 with size >= 2; all non-"n/a"), 4890 duplicate second-order records, 2 duplicate stars. NASDAQ: 17 singleton industries, 1196 + 5 duplicates.
- Final NYSE graph: 4350 edges = 107 industry + 21 stars + 4222 pairs (after dedupe; label by first occurrence). 4250 of 4350 edges have size 2.
- No-edge stocks: 17 NYSE (e.g. AGM, COO, UPS, VAR), all singleton-industry stocks with no wiki edge; NASDAQ 15. Stocks with only the industry edge (degree 1): 1457. Stocks with any wiki-derived edge: 266 of 1737 (15%); NASDAQ 135 of 1026.

**Cardinality vs Fig. 3a (axis 500, 15, 9, 5, 3, p853).** NYSE v2 edge sizes: max 500 (the `n/a` no-industry bucket, as in CLAUDE.md), then 112, 50, 47, 44, 43, 34, ... Counts of edges with size >= 15 / 9 / 5 / 3: 24 / 51 / 78 / 100; size exactly 3: 14, 4: 8, 2: 4250. The 500 tick is reproduced; the lower ticks are consistent with an edge-size threshold axis but the paper gives no schedule, so this is not a match test. Note decomposition touches only 100 of 4350 edges. NASDAQ max size is 156 (n/a), so Fig. 3 is NYSE-specific.

**Node degree vs Fig. 3b (axis 31, 28, 22, 16, 2).** Recap of `docs/phase1/fig3_degree_reconcile.md` (U6), re-verified by the independent rebuild: v2 NYSE max incident-hyperedge degree **114** (top 114, 101, 100, 98, 96, ...), 86 stocks with degree >= 31, 17 with degree 0, 1457 with degree 1. Paper axis starts at 31. No variant reproduced 31; nearest was the old all-star build (37), which contradicts App. B. New exploratory variant tried here (INFERRED, not in paper; per-Z merging approximated by merging each connected component of a symmetric second-order channel into one hyperedge): clique components only (26 of 43 are cliques): max degree still 114; all components merged: 274 edges, max degree 22. Neither gives 31; the true per-Z grouping is not recoverable from the repo.

**Independent rebuild diff.** Script rebuilds from raw files only: industry from `*_industry_ticker.json`; wiki from `*_connections.json` + `*_wiki.csv` filtered by `selected_wiki_connections.csv`; stars per (source, property); second-order as one unordered pair per ordered pair; dedupe; size >= 2. Compared with the cached `hypergraph_cache/*_industry-wiki.json` (version 2, built from the npy by `hypergraph.py`):

| Market | cached edges | independent edges | matched | only in cache | only in independent |
|---|---|---|---|---|---|
| NYSE | 4350 | 4350 | 4350 | 0 | 0 |
| NASDAQ | 1066 | 1066 | 1066 | 0 | 0 |

Incidence totals agree (NYSE 10,204 node-edge incidences; NASDAQ 2,949). Zero edge-level differences. Caveat: the rebuild shares the convention "`connections.json[qa][qb]` path means qa is the source/X", which is validated only by the two spot checks above.

## 3. Unresolved differences, ranked by likely impact

1. **Second-order Z grouping unknown.** Pairs are all the repo can build; the paper may group per Z or use another structure. This is the most likely driver of max degree 114 versus the paper's 31 (Fig. 3b) and affects the "hyperedge vs pairwise" claim since 98% of v2 edges are size 2. Not recoverable without a fresh Wikidata query; even then the paper does not say it groups by Z.
2. **Wikidata coverage and snapshot.** 65.6% of NYSE stocks have no QID, only 15% of stocks have a wiki edge; the paper's Wikidata snapshot/date is unstated (checked only the PDF pages given). A different or fuller mapping would change both hyperedge counts and degrees.
3. **Pattern direction not fully verifiable.** Chain excluded by transpose structure; converging vs diverging only spot-checked (one pair) on a 2026 snapshot.
4. **Hub-removal and decomposition schedule** (Fig. 3a/3b ticks 15, 9, 5, 3 and 28, 22, 16, 2) not defined in the paper, so the x-axis cannot be matched even if the graph were right (carried from U6).
5. **Input handling of missing days.** FILL = 1.1 fed into the model for missing closes (and into neighbours), 331 NYSE days with real MAs but filled close, and the `min`-window target mask being stricter than a per-day mask; unverified against the authors' code.
6. **Paper-norm look-ahead and the dropped NASDAQ row**: documented, no effect on the default `norm: train` run, but under `norm: paper` the NASDAQ scale of 24 stocks uses the dropped row.
7. **Minor**: CLAUDE.md statement "last NASDAQ row is all missing" is wrong (474/1026 missing); MAs of the first 40 rows include 29 unseen warm-up days (can exceed 1); date vector is not loaded by the code (offset 29 rule only); 23 singleton NYSE industries and the 32 NYSE (22 NASDAQ) files not in the ticker list are unused.
