# Independent THINK hyperbolicity audit

**Conclusion: the published table is not fully independently verified.** I retrieved original public source data and performed independent calculations, including exact small/medium metric evaluations and inspectable lower bounds for larger metrics. But the released material does not identify the exact processed incidence matrices and feature tensors behind the reported numbers. TSE and CSE remain uncomputed because I could not obtain the matching complete inputs. Differences below are conditional on declared reconstructions; they are not proof that the authors calculated their own inputs incorrectly.

## Reported values and audit status

The reported columns below come from Table I of [THINK](https://tylersnetwork.github.io/papers/icdm22-think.pdf). Graph delta and feature relative delta are different quantities. Our feature comparisons use one full temporal trajectory per node; that representation is an explicit assumption, not a recovered THINK setting.

| Dataset | Reported graph delta | Reported feature relative delta | Independent result on declared source reconstruction |
|---|---:|---:|---|
| CPox | 1.5 | 0.190 | Original county graph: exact 1.5. Unmerged neighborhood hypergraph: exact 1.0. Published FX trajectory relative delta: exact 0.199416. |
| WMill | 1.0 | 0.025 | All positive source connections give a complete graph: exact 0. Published trajectory relative delta: exact 0.257989. |
| NYSE | 0.5 | 0.087 | Reconstructed s=1 hypergraph has 549 components; no finite global metric. Component delta bounds [1.5, 3.0]. Feature relative delta at least 0.376174. |
| NASDAQ | 1.0 | 0.107 | Reconstructed s=1 hypergraph has 220 components; no finite global metric. Component delta bounds [1.5, 3.5]. Feature relative delta at least 0.351499. |
| DTT | 1.0 | Not reported | All 120 snapshots inspected for each public RG17/UO17 variant. Maximum component-delta bounds: RG17 [1.5, 2.0]; UO17 [1.5, 2.0]. No global disconnected-graph delta substituted. |
| TSE | 1.5 | 0.074 | Not computed: matching 95-stock, 1,159-step panel and incidence matrix unavailable. |
| CSE | 1.5 | 0.176 | Not computed: matching 85-stock, 1,293-step panel and incidence matrix unavailable. |

A sampled maximum is always displayed as a lower bound unless it reaches a proven upper bound. The stock lower bounds already exceed the reported values under our feature representation, but **the representations have not been shown to be the same**. That prevents a valid claim that the paper is numerically wrong.

## Calculation method, without a forecasting model

No GNN is trained or used in these calculations. Hyperbolicity is calculated directly from a chosen distance matrix.

In plain language, a hyperedge is a group, such as several companies in one industry. Hyperbolicity asks how the resulting distances compare with distances in a branching tree. It does not measure forecasting accuracy. A low value alone does not establish a useful financial hierarchy: even a complete graph has zero hyperbolicity when considering only its vertices. Whether a hyperbolic model helps must still be tested on future held-out outcomes against Euclidean hypergraphs and models with no graph.

For graph geometry, H records which nodes belong to which hyperedges. The number of shared groups for nodes i,j is the (i,j) entry of H H-transpose. We connect distinct nodes if that count is at least s. Breadth-first search then gives the number of steps in the shortest unweighted path. Components are handled separately; infinity is never replaced by zero.

For feature geometry, each node is represented by a declared vector. We calculate every Euclidean distance using d(i,j) = sqrt(sum_k (x[i,k]-x[j,k])^2). The implementation uses a Gram matrix to avoid allocating a huge node-by-node-by-feature tensor.

For every four distinct nodes A,B,C,D, form S1=d(A,B)+d(C,D), S2=d(A,C)+d(B,D), and S3=d(A,D)+d(B,C). Sort the three sums. Half the gap between the largest and middle sum is that quadruple's contribution. Delta is the largest contribution over all quadruples. Relative delta is 2*delta/diameter, with diameter the largest pairwise distance in the metric. See the independent [SageMath definition](https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/hyperbolicity.html).

For at most 350 points we enumerate every distinct quadruple exactly once. Above that size we use 2,000,000 reproducible random draws, recording the best actual witness and the general diameter/2 upper bound. Hitting that upper bound certifies the maximum. Otherwise the result remains a bound. Exactness here concerns the supplied finite metric, not identity with the paper's missing processed dataset.

The tests independently evaluate the Gromov-product inequality over all ordered quadruples of an eight-point example and confirm the same result as the four-point method. Additional checks cover a path, a clique, a four-cycle, disconnected graphs, scale invariance and the distinction between bounds and exact values.

## CPox: what matches, and what does not

The [public source](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/ea40a6a396b6688a94d7482d9d5fd288eaa2cb3b/dataset/chickenpox.json) has 20 counties and **521** FX observations, versus 522 time steps in the reported table. FX is already standardized: per-county means are approximately zero and standard deviations approximately one. It must not be described as unprocessed case counts. The artifact filename raw_trajectories means the published FX values without further changes.

I symmetrized the provided county adjacency and removed self-loops. That graph gives 1.5. Its agreement with the table is a **numeric match on the original graph**, not verification of the merged hypergraph construction.

### County graph: exhaustive witness

Assign A, B, C and D respectively to: **BARANYA, BEKES, KOMAROM, ZALA**.

| Distance | Value |
|---|---:|
| d(A,B) | 3.000000000 |
| d(A,C) | 3.000000000 |
| d(A,D) | 2.000000000 |
| d(B,C) | 3.000000000 |
| d(B,D) | 5.000000000 |
| d(C,D) | 2.000000000 |

- S1 = d(A,B)+d(C,D) = 5.000000000
- S2 = d(A,C)+d(B,D) = 8.000000000
- S3 = d(A,D)+d(B,C) = 5.000000000
- Quadruple contribution = (8.000000000 - 5.000000000) / 2 = **1.500000000**.

Diameter of this metric/component: 6.000000000. Normalized witness value: 2 × 1.500000000 / 6.000000000 = **0.500000000**.

Method: exhaustive four-point. Evaluated 4,845 quadruples/draws out of 4,845 possible distinct unordered quadruples.

The calculation certifies this as the maximum for this specified metric, to floating-point precision.

![County graph and the four-point calculation](runs/think_audit/chickenpox_calculation.png)

### Hyperedge-construction sensitivity

| Construction | Exact delta |
|---|---:|
| Original county graph | 1.5 |
| Closed neighborhoods, no merging, s=1 | 1.0 |
| Closed neighborhoods, no merging, s=2 | 1.0 |
| Greedy union of most similar pairs with Dice >=0.5 | 0.5 |
| Greedy union of most similar pairs with Dice >=0.75 | 1.0 |
| Literal low-similarity interpretation: merge Dice <0.5 | 0.0 |

These variants expose sensitivity to missing settings. Thresholds were declared as reconstruction examples, not tuned until a match appeared. Similarity, merging direction, union rule, tie order and s must all be fixed to replicate a result.

### Published FX trajectories: exhaustive feature calculation

Assign A, B, C and D respectively to: **BACS, FEJER, KOMAROM, ZALA**.

| Distance | Value |
|---|---:|
| d(A,B) | 29.270448690 |
| d(A,C) | 28.837405405 |
| d(A,D) | 33.903762949 |
| d(B,C) | 31.917246150 |
| d(B,D) | 29.879825213 |
| d(C,D) | 29.789612860 |

- S1 = d(A,B)+d(C,D) = 59.060061550
- S2 = d(A,C)+d(B,D) = 58.717230618
- S3 = d(A,D)+d(B,C) = 65.821009099
- Quadruple contribution = (65.821009099 - 59.060061550) / 2 = **3.380473775**.

Diameter of this metric/component: 33.903762949. Normalized witness value: 2 × 3.380473775 / 33.903762949 = **0.199415845**.

Method: exhaustive four-point. Evaluated 4,845 quadruples/draws out of 4,845 possible distinct unordered quadruples.

The calculation certifies this as the maximum for this specified metric, to floating-point precision.

Re-standardizing the published FX trajectories gives the same answer to floating-point precision. The difference from 0.190 is not explained by merely applying that standardization again. Different lag windows, point definitions or subsampling could still explain it.

## WMill: the supplied adjacency is dense

The [official loader](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/ea40a6a396b6688a94d7482d9d5fd288eaa2cb3b/torch_geometric_temporal/dataset/windmilllarge.py) links the downloaded 17,472-by-319 series. Its edge list has 101,761 entries, exactly 319 squared; every supplied weight is positive. Keeping every positive connection yields a complete graph. Every closed neighborhood then contains every node.

For any four distinct nodes, all six unweighted distances are 1. Thus S1=S2=S3=2 and delta=(2-2)/2=**0**. Merging identical universal neighborhoods does not change the s=1 result. The paper's value 1.0 therefore requires additional choices beyond this literal positive-edge construction.

As a sensitivity check, keeping only weights greater than 0.5 before creating closed neighborhoods yields four components with sizes 289, 26, 3 and 1; their maximum delta is exactly 2.0. This threshold is our declared example, not an asserted paper setting.

### Published hourly trajectories: exhaustive feature calculation

Assign A, B, C and D respectively to: **91, 96, 106, 219**.

| Distance | Value |
|---|---:|
| d(A,B) | 42.453931962 |
| d(A,C) | 35.667798786 |
| d(A,D) | 38.170685875 |
| d(B,C) | 63.047419190 |
| d(B,D) | 47.177306296 |
| d(C,D) | 40.058979386 |

- S1 = d(A,B)+d(C,D) = 82.512911348
- S2 = d(A,C)+d(B,D) = 82.845105082
- S3 = d(A,D)+d(B,C) = 101.218105065
- Quadruple contribution = (101.218105065 - 82.845105082) / 2 = **9.186499991**.

Diameter of this metric/component: 71.216227066. Normalized witness value: 2 × 9.186499991 / 71.216227066 = **0.257988955**.

Method: exhaustive four-point. Evaluated 423,402,001 quadruples/draws out of 423,402,001 possible distinct unordered quadruples.

The calculation certifies this as the maximum for this specified metric, to floating-point precision.

Standardizing each turbine trajectory separately changes relative delta to **0.345133**. Neither full-trajectory variant reproduces 0.025. That demonstrates the importance of defining temporal features precisely.

## NYSE: source universe recovered; processed hypergraph not recovered

The downloaded panel has 1,737 stocks, 1,245 time steps and five features per time. Counts match the reported universe. Sources are the [original RSR repository](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) linked by the [STHAN-SR release](https://github.com/midas-research/sthan-sr-aaai).

Our reconstructed groups use the supplied named industries, excluding the missing-industry category n/a. For the supplied selected Wikidata paths, first-order links form source-plus-target groups by relation; second-order links form pairs. Identical memberships are deduplicated. These are inspectable assumptions; the exact THINK relation snapshot, filtering and duplicate policy remain unknown.

At s=1 there are 549 components, with the largest containing 1034 nodes. Consequently there is no finite global shortest-path metric. The following is a witness inside a component, not a global-delta replacement.

### Reconstructed relationship metric: witness and bound

Assign A, B, C and D respectively to: **GSK, M, TDG, TNH**.

| Distance | Value |
|---|---:|
| d(A,B) | 2.000000000 |
| d(A,C) | 3.000000000 |
| d(A,D) | 2.000000000 |
| d(B,C) | 2.000000000 |
| d(B,D) | 4.000000000 |
| d(C,D) | 2.000000000 |

- S1 = d(A,B)+d(C,D) = 4.000000000
- S2 = d(A,C)+d(B,D) = 7.000000000
- S3 = d(A,D)+d(B,C) = 4.000000000
- Quadruple contribution = (7.000000000 - 4.000000000) / 2 = **1.500000000**.

Diameter of this metric/component: 6.000000000. Normalized witness value: 2 × 1.500000000 / 6.000000000 = **0.500000000**.

Method: sampled lower bound. Evaluated 2,000,000 quadruples/draws out of 47,353,052,626 possible distinct unordered quadruples.

This is a **lower bound**, not an exact maximum: delta lies in [1.500000000, 3.000000000]. The upper bound uses diameter/2. Sampling uses seed 20260928, with replacement across draws and four distinct vertices within a draw.

At s=2, the same memberships instead yield 1597 components, with exact maximum component delta 1.0. This is another sensitivity calculation.

For feature geometry, I follow the cited RSR loader's five-feature inputs: NASDAQ drops the final source row; sentinel -1234 values are replaced by 1.1 (5,621 replacements here). I then flatten each stock's full feature history into one point. This deliberately reproduces that published loader convention, but does not establish that THINK used these exact feature vectors for its geometry table.

### Full processed feature trajectories: sampled lower bound

Assign A, B, C and D respectively to: **BSX, GDOT, RNR-C, SRF**.

| Distance | Value |
|---|---:|
| d(A,B) | 20.558416533 |
| d(A,C) | 32.978213896 |
| d(A,D) | 45.904828607 |
| d(B,C) | 47.548021699 |
| d(B,D) | 35.648484394 |
| d(C,D) | 51.749163594 |

- S1 = d(A,B)+d(C,D) = 72.307580127
- S2 = d(A,C)+d(B,D) = 68.626698290
- S3 = d(A,D)+d(B,C) = 93.452850306
- Quadruple contribution = (93.452850306 - 72.307580127) / 2 = **10.572635089**.

Diameter of this metric/component: 56.211345952. Normalized witness value: 2 × 10.572635089 / 56.211345952 = **0.376174415**.

Method: sampled lower bound. Evaluated 2,000,000 quadruples/draws out of 377,995,709,070 possible distinct unordered quadruples.

This is a **lower bound**, not an exact maximum: delta lies in [10.572635089, 28.105672976]. The upper bound uses diameter/2. Sampling uses seed 20260928, with replacement across draws and four distinct vertices within a draw.

After separately standardizing each per-stock feature trajectory, the relative-delta lower bound becomes 0.527680. This alternative is also recorded, not selected as the supposed original.

## NASDAQ: source universe recovered; processed hypergraph not recovered

The downloaded panel has 1,026 stocks, 1,245 time steps and five features per time. Counts match the reported universe. Sources are the [original RSR repository](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) linked by the [STHAN-SR release](https://github.com/midas-research/sthan-sr-aaai).

Our reconstructed groups use the supplied named industries, excluding the missing-industry category n/a. For the supplied selected Wikidata paths, first-order links form source-plus-target groups by relation; second-order links form pairs. Identical memberships are deduplicated. These are inspectable assumptions; the exact THINK relation snapshot, filtering and duplicate policy remain unknown.

At s=1 there are 220 components, with the largest containing 601 nodes. Consequently there is no finite global shortest-path metric. The following is a witness inside a component, not a global-delta replacement.

### Reconstructed relationship metric: witness and bound

Assign A, B, C and D respectively to: **JCOM, JOUT, NWBI, TROW**.

| Distance | Value |
|---|---:|
| d(A,B) | 4.000000000 |
| d(A,C) | 5.000000000 |
| d(A,D) | 5.000000000 |
| d(B,C) | 5.000000000 |
| d(B,D) | 2.000000000 |
| d(C,D) | 3.000000000 |

- S1 = d(A,B)+d(C,D) = 7.000000000
- S2 = d(A,C)+d(B,D) = 7.000000000
- S3 = d(A,D)+d(B,C) = 10.000000000
- Quadruple contribution = (10.000000000 - 7.000000000) / 2 = **1.500000000**.

Diameter of this metric/component: 7.000000000. Normalized witness value: 2 × 1.500000000 / 7.000000000 = **0.428571429**.

Method: sampled lower bound. Evaluated 2,000,000 quadruples/draws out of 5,381,985,050 possible distinct unordered quadruples.

This is a **lower bound**, not an exact maximum: delta lies in [1.500000000, 3.500000000]. The upper bound uses diameter/2. Sampling uses seed 20260928, with replacement across draws and four distinct vertices within a draw.

At s=2, the same memberships instead yield 967 components, with exact maximum component delta 0.5. This is another sensitivity calculation.

For feature geometry, I follow the cited RSR loader's five-feature inputs: NASDAQ drops the final source row; sentinel -1234 values are replaced by 1.1 (6,584 replacements here). I then flatten each stock's full feature history into one point. This deliberately reproduces that published loader convention, but does not establish that THINK used these exact feature vectors for its geometry table.

### Full processed feature trajectories: sampled lower bound

Assign A, B, C and D respectively to: **FDEF, FOSL, SHY, TTWO**.

| Distance | Value |
|---|---:|
| d(A,B) | 38.434524550 |
| d(A,C) | 31.071230235 |
| d(A,D) | 27.363260834 |
| d(B,C) | 45.015168485 |
| d(B,D) | 40.538543205 |
| d(C,D) | 56.309546331 |

- S1 = d(A,B)+d(C,D) = 94.744070882
- S2 = d(A,C)+d(B,D) = 71.609773440
- S3 = d(A,D)+d(B,C) = 72.378429319
- Quadruple contribution = (94.744070882 - 72.378429319) / 2 = **11.182820781**.

Diameter of this metric/component: 63.629398598. Normalized witness value: 2 × 11.182820781 / 63.629398598 = **0.351498553**.

Method: sampled lower bound. Evaluated 2,000,000 quadruples/draws out of 45,902,419,200 possible distinct unordered quadruples.

This is a **lower bound**, not an exact maximum: delta lies in [11.182820781, 31.814699299]. The upper bound uses diameter/2. Sampling uses seed 20260928, with replacement across draws and four distinct vertices within a draw.

After separately standardizing each per-stock feature trajectory, the relative-delta lower bound becomes 0.516205. This alternative is also recorded, not selected as the supposed original.

## DTT: dynamic and disconnected

The public PyTorch Geometric Temporal release contains both [Roland-Garros RG17](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/ea40a6a396b6688a94d7482d9d5fd288eaa2cb3b/dataset/twitter_tennis_rg17.json) and [US Open UO17](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/ea40a6a396b6688a94d7482d9d5fd288eaa2cb3b/dataset/twitter_tennis_uo17.json). I inspected all 120 timestamps in each variant, retaining the full 1,000-node universe, symmetrizing mentions and creating unmerged closed-neighborhood hyperedges with s=1. No feature relative-delta number is reported for DTT in THINK.

RG17: 120/120 snapshots are disconnected. The maximum component delta across snapshots lies in [1.5, 2.0]. All component calculations exact: False. This time aggregation is our explicit audit summary, not an identified aggregation from the paper.

### RG17: largest found witness, snapshot 30

Assign A, B, C and D respectively to: **6, 27, 119, 319**.

| Distance | Value |
|---|---:|
| d(A,B) | 2.000000000 |
| d(A,C) | 2.000000000 |
| d(A,D) | 4.000000000 |
| d(B,C) | 3.000000000 |
| d(B,D) | 2.000000000 |
| d(C,D) | 2.000000000 |

- S1 = d(A,B)+d(C,D) = 4.000000000
- S2 = d(A,C)+d(B,D) = 4.000000000
- S3 = d(A,D)+d(B,C) = 7.000000000
- Quadruple contribution = (7.000000000 - 4.000000000) / 2 = **1.500000000**.

Diameter of this metric/component: 6.000000000. Normalized witness value: 2 × 1.500000000 / 6.000000000 = **0.500000000**.

Method: exhaustive four-point. Evaluated 377,396,635 quadruples/draws out of 377,396,635 possible distinct unordered quadruples.

The calculation certifies this as the maximum for this specified metric, to floating-point precision.

UO17: 112/112 snapshots are disconnected. The maximum component delta across snapshots lies in [1.5, 2.0]. All component calculations exact: False. This time aggregation is our explicit audit summary, not an identified aggregation from the paper.

### UO17: largest found witness, snapshot 44

Assign A, B, C and D respectively to: **6, 160, 328, 697**.

| Distance | Value |
|---|---:|
| d(A,B) | 2.000000000 |
| d(A,C) | 2.000000000 |
| d(A,D) | 3.000000000 |
| d(B,C) | 4.000000000 |
| d(B,D) | 2.000000000 |
| d(C,D) | 2.000000000 |

- S1 = d(A,B)+d(C,D) = 4.000000000
- S2 = d(A,C)+d(B,D) = 4.000000000
- S3 = d(A,D)+d(B,C) = 7.000000000
- Quadruple contribution = (7.000000000 - 4.000000000) / 2 = **1.500000000**.

Diameter of this metric/component: 4.000000000. Normalized witness value: 2 × 1.500000000 / 4.000000000 = **0.750000000**.

Method: exhaustive four-point. Evaluated 276,821,545 quadruples/draws out of 276,821,545 possible distinct unordered quadruples.

The calculation certifies this as the maximum for this specified metric, to floating-point precision.

Per-snapshot component counts, bounds and witnesses are saved in the DTT JSON files. Components smaller than four nodes have zero vertex-metric delta and are summarized by their sizes.

## TSE: not computed

The [cited source repository](https://github.com/liweitj47/overnight-stock-movement-prediction) links a Baidu archive described as original news data. That link could not be retrieved through the available browsing interface. Its repository does not supply the matching 95-stock panel and THINK incidence matrix. The STHAN-SR release contains TSE loading code but no complete matching files; its [TSE dataset issue](https://github.com/midas-research/sthan-sr-aaai/issues/11) also provides no resolution in the inspected page. No witness or numerical estimate is invented for this dataset.

## CSE: not computed

The [cited source paper](https://arxiv.org/pdf/1805.07979) links historical Baidu archives for news and social data; those links could not be retrieved through the browsing interface. Following related authors' releases through [HYPHEN](https://github.com/gtfintechlab/HYPHEN-ACL) to [FAST](https://github.com/midas-research/fast-eacl) identifies related data and sample files, but not a verified complete 85-node, 1,293-step THINK panel with incidence matrix. The cited source discusses a different original sampling period/universe, so the precise filtering/history behind THINK remains unresolved. No surrogate current-market download is labeled as the original CSE dataset.

## What would permit a conclusive verification?

The [THINK repository](https://github.com/shivamag125/ICDM22-THINK) returned GitHub API status 409, "Git Repository is empty," during this audit. To independently check the exact table, we need:

1. The incidence matrix for each static dataset, and each dynamic snapshot, with node order and file hashes.
2. The actual temporal feature tensor used for relative delta: what constitutes a point, lag length, date range and scaling.
3. The s value, edge direction/weight handling, duplicate policy and treatment of disconnected nodes.
4. The neighborhood-merging threshold, merge direction and update/tie rules.
5. The exact-versus-sampled calculation procedure, including seeds/sample sizes and DTT time aggregation.

Without these, I can verify the mathematics on explicit inputs and identify unresolved discrepancies, but cannot responsibly certify or reject the entire published table. A numerical match produced by trying many undocumented settings would not resolve this problem.

## Reproduce and inspect

```powershell
.\.venv\Scripts\python.exe fetch_think_sources.py
.\.venv\Scripts\python.exe fetch_stock_audit.py
.\.venv\Scripts\python.exe -m unittest test_audit -v
.\.venv\Scripts\python.exe run_think_audit.py --section public
.\.venv\Scripts\python.exe run_think_audit.py --section stocks
.\.venv\Scripts\python.exe report_think_audit.py
```

The public calculation can take several minutes because it includes exhaustive evaluations and all Twitter snapshots. Large stock metrics return declared bounds. Downloads use commit-pinned sources where available and retain their hashes. The compressed relation archive expands to several GB if fully extracted; our script reads only its small JSON/text members, not its huge dense NumPy arrays.

A separate pre-publication revision check confirms that the downloaded Chickenpox and both Twitter files are byte-for-byte identical to their latest GitHub revisions before October 2022. The old Windmill loader pointed to graphmining.ai, which failed DNS resolution during this audit; we used the current official loader's Box mirror and cannot prove byte identity with that unavailable historical URL. See historical_revision_check.json and windmill_legacy_url_check.json.

- `data/think_audit/sources.json`: source URLs and SHA256 values.
- `data/think_audit/stock_sources.json`: provenance for every NYSE/NASDAQ series.
- `runs/think_audit/*.json`: calculations, six witness distances, pair sums, bounds and assumptions.
- `runs/think_audit/*_reconstructed_hyperedges.json`: inspectable US-stock group memberships.
- `hyperbolicity_audit.py`: independent geometry implementation.
- `run_think_audit.py`: reconstruction and preprocessing choices.
- `HOURLY_FORECASTING.md`: the separate hourly-data and live-inference explanation.
