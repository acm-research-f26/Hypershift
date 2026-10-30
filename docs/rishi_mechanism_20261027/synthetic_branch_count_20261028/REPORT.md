# Synthetic active-count sensitivity by network branch

In this fixed toy, the dynamic graph branch accounts for the observed change in the focal output when additional slots become active. Across all 18 seed/input/population combinations, changing from two to eight active nodes left the focal encoder and static wavelet representations exactly unchanged, while the focal dynamic representation changed. This is a property of this pair-graph configuration and random network initialization, not evidence that the trained financial model uses helpful peer information.

Protocol f146a0e and code 4070df1 were frozen before this run. There are 72 cells: three seeds, three synthetic inputs, diverse or replicated peers, and four active counts. The focal node's static graph pair is always nodes 0 and 1, which are active in every case; new active nodes belong to other pairs. The following ranges are across nine seed/input cells per population, for eight active nodes relative to two. Representation changes are L2 norms in internal synthetic units; score changes are signed.

| Population | Encoder change | Dynamic change | Wavelet change | Fused change | Score change |
|---|---:|---:|---:|---:|---:|
| Diverse peers | 0 | 0.2015–0.2650 | 0 | 0.0672–0.0883 | -0.02381 to +0.02101 |
| Replicated peers | 0 | 0.1949–0.2827 | 0 | 0.0650–0.0942 | -0.02337 to +0.02005 |

All 72 reconstructed full forecasts match the actual adapter exactly on active nodes; the separately reconstructed dynamic aggregation matches its module exactly, all stored components are finite, and incidence row sums differ from one by at most 1.20e-7. An independent join confirms all 72 focal score changes match the previous full-path member-count batch exactly. Three focused tests passed. `CELLS.csv` contains every case and `RESULTS.json` input hashes. The temporary launcher copy of the implementation is byte-identical to organized source.

The dynamic layer sums incidence-weighted representations before its edge mixing, activation and residual normalization. This experiment identifies sensitivity of that branch to the added peers collectively, but does not distinguish incidence changes from aggregation or later nonlinearities; it also does not test other graph topologies, trained weights, real membership events or returns. The original-device numerical gate, financial retraining decision and prospective restrictions remain unchanged.
