# Phase 1 feasibility: R1-R4, R7, R8 (THINK Table II, non-ranking rows and baselines)

*Written 2026-09-29. Research only: nothing here was implemented. Probes were CPU-only and small (<50 MB), kept outside the repo.*

Labels: **[V]** = verified by reading the source or running a probe; **[I]** = inferred, needs a decision or a check.

Sources:
- **Paper**: Agarwal, Sawhney et al., ICDM 2022, https://tylersnetwork.github.io/papers/icdm22-think.pdf. Six pages plus a one-page appendix (proceedings pp. 849-854). Hyperparameters, splits, lookbacks and scaling are **not stated anywhere** in it [V].
- **Official code**: https://github.com/shivamag125/ICDM22-THINK (empty, per CLAUDE.md).
- **PyG-Temporal (PyG-T)**: https://github.com/benedekrozemberczki/pytorch_geometric_temporal (loader code and data).
- **STHGCN code**: https://github.com/midas-research/sthgcn-icdm
- **RSR code**: https://github.com/fulifeng/Temporal_Relational_Stock_Ranking

## Summary table

| ID | Task (paper value) | Obtainable now? | Spec completeness | Est. effort | Recommendation |
|---|---|---|---|---|---|
| R2 | Chickenpox MSE (1.09) | **Yes** [V]. 220 KB JSON on GitHub raw, no torch-geometric-temporal needed | Low: data and loader defaults known; lags, split, hypergraph threshold and MSE averaging inferred | 6-10 h eng, <0.5 GPU-h (CPU fine, 20 nodes) | **Do first.** Report next to a mean predictor (1.047) and an AR(4) (0.72) |
| R3 | Windmill MSE (1.05) | **Yes** [V]. 47 MB JSON via the Box link in the PyG-T loader (graphmining.ai mirror is dead) | Low: as R2, and the graph is complete so the hyperedge threshold is unspecified and decisive | 6-10 h eng (after R2), ~2-4 GPU-h [I] | **Do second.** Series has ~zero autocorrelation: any model lands near 1.02 |
| R1 | Twitter tennis MSE (0.58) | **Yes** [V]. ~2 MB JSON per event on GitHub raw; Table I matches `rg17` | Low-medium: snapshots and target are in the loader; feature mode, split, window, scale inferred | 10-16 h eng, ~2-4 GPU-h [I] | **Do third.** Paper baselines (2.05) are 5x worse than a constant predictor (0.42): the paper's scale is doubtful |
| R7 | NASDAQ 3-class F1 (0.49) | **Yes** (RSR data already local) | Medium: STHGCN repo shows tertile labels and macro+micro F1 [V]; the paper text does not | 3-6 h eng, ~6-12 GPU-h for 25 seeds x {HH,EH,EE} [I] | **Do fourth.** Code exists, never run. Fix the mismatches below first |
| R8 | STHGCN and RSR-I baselines | **Code yes, data partly.** RSR: code public, our RSR data already local. STHGCN: repo unfinished, data link 404 | RSR-I: high. STHGCN: medium (architecture readable, protocol not) | RSR-I 8-12 h eng, ~3 GPU-h; STHGCN 12-20 h eng, ~5-8 GPU-h [I] | **Do fifth.** Reimplement in PyTorch on our data; do not port the TF1 / legacy code |
| R4 | China stock risk MSE (0.32) | **No.** No public dataset with this shape found | Very low: dataset and target both undefined | n/a (weeks for a proxy) | **BLOCKED**, like R6 TSE. Proxy only if the user wants it |

## Biggest surprises

1. **The PyG-T windmill series has essentially zero temporal autocorrelation as stored** (lag-1 -0.002, lag-24 -0.004 [V]), and chickenpox has lag-1 autocorrelation -0.51 [V]. On standardized targets the mean predictor scores 1.02 (windmill) and 1.047 (chickenpox). The paper's THINK numbers (1.05, 1.09) are at or *above* the mean predictor, and all listed baselines are worse. R2 and R3 test "not worse than a constant", not skill.
2. **The tennis baselines (~2.05) are 5x worse than a constant predictor** (0.42 on `rg17`, 80/20 split) [V]. The paper's DTT scale is not the PyG-T default scale, or the baselines were run badly. THINK's 0.58 is also above the constant.
3. **The STHGCN repo answers the R7 threshold question** that our plan (Task 18) calls "not published": the default `label_proportion = [1,1,1]` means tertiles of pooled training returns, and the evaluator reports both macro and micro F1 [V]. Our `clf.py` already does tertiles and macro-F1. But that repo is an S&P500 pipeline (423 stocks, 1652 days, lookback 50), not the RSR NASDAQ data.
4. **The STHGCN repo does not run as shipped**: 423 nodes and lookback 50 hard-coded, undefined variables in `evaluator.py`, data on a Google Drive link that now ends in 404 [V].
5. **The windmill graph is complete**: 101,761 edges = 319^2, all ordered pairs plus self-loops [V]. "Neighbourhood hyperedges" only make sense after an edge-weight cut the paper does not give.
6. **R4's cited source is a text-regression paper on US 10-K filings** (Kogan et al. 2009), and the CSE data source [22] (78 CSI100 + 13 HK stocks, 2015-2016) does not match Table I (85 nodes, 1293 timesteps).

---

## R2: Chickenpox (paper MSE 1.09, baselines 1.11-1.14)

**1. Data.**
- Paper cites [19] Rozemberczki et al. 2021 (arXiv:2102.08100) and [20] PyG-T (arXiv:2104.07788) [V].
- Direct download, no torch-geometric-temporal needed: `https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json` (220,340 bytes) [V, downloaded].
- Contents [V]: keys `edges` (102 directed edges, 20 counties), `node_ids`, `FX` shape (521, 20). `FX` is already standardized over the whole series (mean ~0, std 1, max 8.87).
- Table I: 522 timesteps, 20 nodes, δ_hg 1.5, δ_rel 0.190. Off by one from 521 rows; immaterial.

**2. How THINK is applied.**
- Hypergraph (Appendix B, p. 854) [V]: for each node v take neighbours N(v), form {(v, N(v))}, then merge pairs by Sorensen-Dice coefficient (SCD) "until no two pairs had an SCD score lower than a threshold", following [37] (Sun et al., WSDM 2021). The sentence is garbled: merging until no pair is below a threshold would merge everything. Most likely intended: merge while SCD is above a threshold [I]. The threshold is not given [V]. With 20 nodes there are at most 20 hyperedges before merging.
- Features, lookback, horizon, split, scaling: **not in the paper** [V].
- PyG-T loader defaults [V, from source]: `lags=4`, target = value at the next week, one channel, chronological `temporal_signal_split(train_ratio=0.8)` in the examples.
- THINK's temporal conv needs tau = n*K; with 4 lags that means K=2, n=2 [I].
- MSE: per (node, snapshot) squared error averaged [I]. `FX` is globally standardized, so MSE is in units of variance.

**3. Calibration probes** [V; my own numpy code; 80/20 chronological split, lags 4]:

| Predictor | Test MSE |
|---|---|
| Training-mean | 1.047 |
| Persistence (y_t = y_{t-1}) | 3.02 (lag-1 autocorrelation -0.51) |
| Global linear AR(4) | **0.72** |

So 1.09 is *worse than the constant predictor* and much worse than a trivial AR(4). A faithful target is "about 1.05-1.15"; a good model will beat it comfortably.

**4. Unspecified, to infer:** lags, horizon, split ratios (no validation described), epochs, whether `FX` was used as-is (it is full-series standardized), SCD threshold and direction, other node features, MSE averaging.

**5. Effort.** 6-10 engineering hours: a small data module (windows [N,T,C], static hypergraph from the SCD procedure), a regression head/loss, reuse `models/think.py` with `out_dim=1`. Compute <0.5 GPU-h (CPU fine). Run 25 seeds x {HH, EH, EE} plus the mean/AR references.

---

## R3: Windmill (paper MSE 1.05, baselines 1.19-1.38)

**1. Data.**
- Paper cites [20] PyG-T. The dataset is `WindmillOutputLarge` (319 windmills, 17,472 hourly steps); Table I's 17,472 and 319 match exactly [V].
- The loader downloads `https://anl.app.box.com/shared/static/wgwb75lt3ty3pv5a15y9bilx1mjhcq59` to `windmill_output.json`. It is **not** in the GitHub repo's dataset folder. The older mirror `https://graphmining.ai/temporal_datasets/windmill_output.json` did not resolve (DNS failure) [V].
- A HEAD request on the Box URL returns 404, but a normal GET redirects to `public.boxcloud.com` and returns **47,159,335 bytes** [V, downloaded to a temp folder]. Any fetch script must use GET with `-L`. The link could rot; keep a local copy in `data/raw/` (git-ignored).
- Contents [V]: keys `block` (17472 x 319), `edges` (101,761), `weights` (101,761), `time_periods`. Three of 319 nodes have zero variance.
- The graph is complete: 101,761 = 319^2, all ordered pairs including self-loops; weights encode proximity [V].

**2. How THINK is applied.**
- Hypergraph: same neighbourhood + Dice recipe as R2 [V]. On a complete graph N(v) is all nodes unless edges are cut by weight or top-k. **The paper gives no cut** [V]. This is the largest free parameter of R3 [I].
- PyG-T defaults [V]: `lags=8`, per-node z-score over the full series (+1e-10 in the denominator; a look-ahead like `norm: paper`), target = next hour, one channel; 0.8 chronological split in the examples. About 17,464 windows, ~3,493 in the test split.
- MSE in standardized units [I].

**3. Calibration probes** [V; lags 8, 80/20]:

| Predictor | Test MSE |
|---|---|
| Training-mean | 1.020 |
| Persistence | 2.03 (= 2 x variance, i.e. no autocorrelation) |
| Global linear AR(8) | 1.020 |

In the row order stored, the series behaves like i.i.d. noise. Either the JSON's row order is not chronological or there is no temporal signal at these lags. Either way **no model can beat about 1.02**, so the paper's 1.05 is "predict the mean, plus a little variance", and its baselines (1.19-1.38) are worse than a constant. R3 is a weak test of THINK. I did not investigate the row order further (the JSON has no timestamps).

**4. Unspecified, to infer:** lags, split, hyperedge weight cut or top-k, node feature (single channel), scaling (full-series z-score leaks the future), epochs and batch size, MSE averaging.

**5. Effort.** 6-10 engineering hours after R2 (reuse the data module; add the weight-cut option and a 47 MB download script). Compute est. 2-4 GPU-h for 25 seeds x 3 arms [I, extrapolated, not measured]. No clique arms needed.

---

## R1: Twitter tennis, DTT (paper MSE 0.58, baselines 2.03-2.06)

**1. Data.**
- Paper cites [18] Beres et al. 2019 (Applied Network Science), "Node embeddings in dynamic graphs" [V]. The PyG-T loader `TwitterTennisDatasetLoader` takes `event_id` in {`rg17`, `uo17`} (Roland-Garros 2017, US Open 2017).
- Direct downloads [V, downloaded]: `.../pytorch_geometric_temporal/master/dataset/twitter_tennis_rg17.json` (1,982,751 B) and `twitter_tennis_uo17.json` (1,936,663 B). (The loader points at a `ferencberes/...developer` branch; the main repo copy works.)
- Table I: 120 timesteps, 1000 nodes, δ_hg 1.0. `rg17` has 120 snapshots and `uo17` has 112 [V], so the paper used **rg17** (or a variant).
- Each snapshot has `index, edges, weights, y, X`: about 89 edges among 1000 nodes at t=0 (sparse), `X` is (1000, 2) raw degree and transitivity, `y` is (1000,) next-snapshot mention counts [V].

**2. How THINK is applied.**
- Hypergraph: neighbourhood + Dice merge (Appendix B) [V]. The graph is dynamic, so one hypergraph per snapshot [I] (the paper says only that DTT is "dynamic"). DHHAN is applied per snapshot (eq. 16), so this is compatible, but our `Hypergraph` class is static; a per-snapshot list is needed.
- Loader defaults [V]: `feature_mode="encoded"` = 5 one-hot bins of log-degree + 11 one-hot bins of transitivity = 16 features (or raw 2, or identity); `target_offset=1`; target `y = log(1 + y)`; all 1000 nodes.
- The natural task is one snapshot in, one out, so THINK's temporal conv has no obvious lookback. The paper gives none [V]; a window of k snapshots is an inference [I].

**3. Calibration probes** [V; `rg17`, target `log1p(y)` at offset 1, chronological 80/20, 24 test snapshots]:

| Predictor | Test MSE (uo17 in brackets) |
|---|---|
| Global training-mean | 0.420 (0.536) |
| Per-node training-mean | 0.307 (0.326) |
| Persistence | 0.240 (0.228) |

The paper's baselines (2.03-2.06 for eight of nine) are 5-10x worse than any constant; THINK's 0.58 is worse than a constant. The setup differs (a target without log, another split, or mis-run baselines). Near-identical 2.04-2.06 across very different architectures suggests a shared failure such as a collapsed output on a differently-scaled target [I]. **Do not treat 0.58 as a reproducible target** until the setup is pinned; report our result against the constant and persistence references.

**4. Unspecified, to infer:** rg17 vs uo17, feature mode, target transform, offset, split (80/20 leaves only 24 test snapshots), temporal window, Dice threshold, per-snapshot hypergraphs, MSE averaging.

**5. Effort.** 10-16 engineering hours (most of the three PyG-T tasks: per-snapshot hypergraphs, sparse graph with many isolated nodes, which touches control C6). Compute est. 2-4 GPU-h [I].

---

## R7: NASDAQ 3-class movement (paper F1: THINK 0.49; TCONV+DHHAN 0.44; STHGCN 0.40; RSR-I 0.38)

**1. Data.** Table I row NASDQ: 1245 timesteps, 1026 nodes, identical to RSR NASDAQ, already local [V]. The paper cites [2] STHGCN for the task; the Table II caption says "Clf denotes stock movement classification on NASDAQ" [V].

**2. Task definition in the paper.** Sec. IV-A [V]: "stock movement classification [2] (on NASDAQ) with targets being stock prices going up, going down, or staying neutral". Sec. IV-B [V]: "We evaluate THINK on node classification using F1-score". **No thresholds and no averaging are named.**

**3. What the STHGCN repo shows** [V; `src/config.py`, `dataset.py`, `evaluator.py`]:
- `label_proportion` default `[1,1,1]`: class thresholds are the 1/3 and 2/3 quantiles of pooled training returns (all stocks, all days of the training phase). This is what our `tertile_thresholds` does.
- `evaluator.get_f1` returns both macro and micro F1. Which one is headlined is not stated. Early stopping in the repo uses accuracy by default.
- That repo is a HATS-derived S&P500 pipeline (`data_type='snp500'`, 1652 days, lookback 50, 423 stocks hard-coded, rolling phases: test 100, dev 50, train 3 x 100). It does not describe the RSR NASDAQ 1245 x 1026 setup, so how the authors ran NASDAQ is unknown [I].

**4. Comparison with our code** (`src/hypershift/train/clf.py`, `scripts/run_clf.py`):

| Item | Ours | STHGCN repo / paper | Mismatch? |
|---|---|---|---|
| Thresholds | Pooled training-return tertiles (`np.quantile(g[m>0.5], [1/3, 2/3])`) | Pooled training tertiles | No [V] |
| Classes | 0/1/2 via `np.digitize`; "neutral" = middle third | Same | No |
| F1 averaging | Macro over all valid stock-days pooled across the split | Macro and micro, per evaluation day in the repo; aggregation incomplete | **Ambiguous.** Report macro, micro and accuracy [I] |
| Lookback | `cfg.seq` (16 in RSR configs) | 50 in the repo (S&P) | Different; unspecified for THINK |
| Protocol | RSR: train 2013-15, val 2016, test 2017 | Rolling 100-day phases | Different; RSR split is implied by Table I [I] |
| Epoch selection | Best validation macro-F1, patience | Best validation accuracy | Minor |
| Loss | Masked cross-entropy on 3 outputs | Cross-entropy | No |
| Head | `THINK(out_dim=3)` | n/a | THINK's classification head is unspecified [I] |
| Class balance | Tertiles fixed on train; val/test shares can drift (2017 was low-volatility) | n/a | Check and report class shares [I] |
| Baselines | Only our HH/EH/EE | STHGCN 0.40, RSR-I 0.38 | **Missing** (needs R8) |
| Features | 5 RSR features, `norm: train` | Repo used the `return` feature only | Different |

- The plan (Task 18) says STHGCN thresholds are unpublished. That is now partly outdated: the code default is tertiles [V].
- Chance macro-F1 with three balanced classes is 0.33. The paper's 0.38-0.49 range is only 0.05-0.16 above chance, so seed noise will matter more than geometry. Plan on all 25 seeds.

**5. Unspecified, to infer:** thresholds actually used on RSR NASDAQ, averaging, lookback, validation protocol, class-imbalance handling.

**6. Effort.** 3-6 engineering hours (add micro-F1, accuracy, class shares and a chance-level line; `run_clf.py` exists). Compute is an estimate from the measured full-NYSE HH 21.6 s/epoch, scaled down for 1026 stocks: roughly 25 seeds x 3 arms x 6-12 min = **6-12 GPU-h** [I, not measured]. Try 10 seeds first.

---

## R8: Baselines, STHGCN and RSR-I

THINK's Table II baselines: GConvGRU, EGCN-O, DCRNN, TGCN, ST-TGCN, DyGrAE (PyG-T style, no hypergraph), RSR-I, EGCN-H, STHGCN. This item covers RSR-I and STHGCN (the stock-task baselines), as asked.

### RSR-I

- **Specs** [V; `training/relation_rank_lstm.py`]:
  - (a) A pretrained sequential embedding from `rank_lstm.py` (LSTM on the 5 features).
  - (b) A relational layer with edge weight `<feature_i, feature_j> * rel_weight[i,j]` (flag `--inner_prod 1`), where `rel_weight = leaky_relu(dense(multi-hot relation))`, masked softmax over neighbours, then `concat[feature, propagated]`, then a leaky-relu dense to a scalar prediction.
  - The explicit variant (RSR-E, default flag 0) uses `head_weight + tail_weight + rel_weight` instead.
  - The inner-product variant is the "RSR-I" of THINK's table [I: mapping from the RSR paper's I/E naming, not re-read in the paper text].
  - README run lines: NASDAQ `-l 16 -u 64`, NYSE `-l 8 -u 32`.
- **Loss** [V]: masked MSE on return ratio plus alpha x pairwise ranking hinge, the same objective our `train/loss.py` implements.
- **Language**: TensorFlow 1.x (Python 3.6); not runnable in a current env without a TF1 shim [V].
- **What it takes on our data:**
  1. Reimplement in PyTorch on our RSR arrays (LSTM + inner-product relational layer).
  2. Relation input: RSR uses multi-hot `[N,N,K]` tensors. NYSE industry is `[1737,1737,108]` float = about 1.3 GB, Wiki `[1737,1737,33]`; on a 4 GB GPU compute `rel_weight` from a sparse edge list or in chunks [I].
  3. Match THINK's protocol: same splits, same 25 seeds, same evaluator (SR, correct NDCG). Report both validation-selected and best-test-epoch numbers, as for our THINK.
- **Effort**: 8-12 engineering hours; about 3 GPU-h for 25 seeds on NYSE [I, an LSTM plus one attention layer should be cheaper than our HH; not measured].

### STHGCN

- **Paper**: Sawhney, Agarwal, Wadhwa, Shah, ICDM 2020 (https://ieeexplore.ieee.org/abstract/document/9338303). I could not read the text (IEEE returned HTTP 418). Per search snippets it is "a gated temporal convolution over hypergraphs", evaluated on S&P500 stocks traded in NASDAQ and NYSE over 12 phases [V, snippet].
- **Code** [V]: https://github.com/midas-research/sthgcn-icdm (Python 3.6, PyTorch, PyG, networkx). `HGNN.py` has a `TimeBlock` (`relu(conv1 + sigmoid(conv2) + conv3)`), an `STGCNBlock`, and an `HGNN` class using per-stock GRUs and two `HGNN_conv` layers. `hypergraph_utils.py` is the standard HGNN `generate_G_from_H`. `dataset.py` is HATS-derived (phases, tertile labels).
- **Not runnable as shipped** [V]:
  - `evaluator.py` uses undefined `pred_li`/`true_li` and hard-codes a reshape to `(1,423,50,1)`; `HGNN.forward` hard-codes 423 GRUs and lookback 50.
  - `dataset.py` needs `ordered_ticker.pkl`, `adj_mat.pkl`, `rel_num.pkl` and `data/price/snp500/processed/*.csv`.
  - `download.sh` points at a Google Drive file (id `1VXMkPmWNRsRCpQVkYQKfgR15l5_XVrWO`); the redirect ends in HTTP 404, so the data is gone [V].
  - How the incidence matrix H is built for stocks is not documented in the repo.
- **What it takes on our data**: implement as gated temporal conv (as `TimeBlock`) -> HGNN spatial conv (`G = Dv^-1/2 H W De^-1 H^T Dv^-1/2`) on our existing `Hypergraph` incidence -> gated temporal conv -> head. Train with our ranking loss for SR, and cross-entropy for R7. This is close to `structure=hyper, spatial=euc` with a gated temporal conv and no attention. Widths and depth for RSR are not given, so it is best-effort [I].
- **Effort**: 12-20 engineering hours; est. 5-8 GPU-h for 25 seeds on NYSE ranking, similar order again for NASDAQ classification [I, not measured].

**Interpretation note**: the paper's baseline numbers (RSR-I 1.05, STHGCN 1.10 NYSE SR) come from the authors' unknown protocol. Our re-runs under our protocol will not reproduce them exactly, and the paper's ~1e-3 spreads are not realistic (plan Part 0.3).

---

## R4: China stock exchange risk (paper MSE 0.32, baselines 0.37-0.61)

**1. Dataset.**
- Table I row CSE: 1293 timesteps, 85 nodes, δ_hg 1.5, δ_rel 0.176; cites [22] Huang, Zhang, Zhang, Zhang, "A tensor-based sub-mode coordinate algorithm for stock prediction", IEEE DSC 2018 (arXiv:1805.07979) [V].
- That paper uses **78 CSI 100 A-share stocks + 13 HK stocks over 2015-2016**, with news and social sentiment plus quantitative data [V, via arXiv summary]. No code or data release was found. 91 stocks and ~2 years do not match 85 nodes and 1293 timesteps (~5 years of daily bars), so the THINK data is not simply that of [22].
- Not public as far as I can find: neither the THINK paper nor [22] gives a download link [V/I].

**2. What "risk" means.**
- Sec. IV-A [V]: "risk forecasting [35] on the Chinese stock exchange (CSE)"; [35] is Kogan et al., "Predicting risk from financial reports with regression", NAACL 2009 [V]. That paper predicts **log volatility** (log of the standard deviation of daily returns over a horizon after a filing) from US 10-K text [I, from memory of Kogan et al.; not re-read].
- THINK's version is single-step per-node regression, so the target is plausibly next-period realized volatility from a lookback of price features [I]. Not defined in the paper. An MSE of 0.32 has no meaning without the transform and standardization.

**3. Feasibility.** BLOCKED as an exact reproduction. A proxy is possible, for example free CSI 300 constituent daily bars (baostock / akshare / tushare; not probed [I]) with target = log realized volatility over the next 5-20 days. That would be our own task, and a comparison with 0.32 would be meaningless. Record R4 as BLOCKED (same status as R6 TSE) unless the authors share data. A proxy costs about 10-16 engineering hours plus data cleaning; only worth it if the user wants "does hyperbolic help on volatility" as a separate question.

**4. Unspecified:** everything (universe, dates, features, target formula, horizon, scaling, split; RSR-style industry/Wiki hyperedges do not exist for Chinese stocks).

---

## Consolidated list of what the paper leaves unspecified

Across R1-R4, R7, R8 the paper gives none of the following [V: read the whole PDF]:

1. Hyperparameters of any kind (hidden size, kernel K, lookback, epochs, learning rate, batch size, optimizer).
2. Train/validation/test splits for DTT, CPox, WMill, CSE, and whether model selection used validation or test.
3. Feature scaling for the non-stock tasks. PyG-T loaders standardize over the full series, which leaks the future.
4. Lookback and horizon for the regression tasks (PyG-T defaults 4, 8 and 1 are only a guess).
5. The Dice merge threshold, the direction of the inequality, and how neighbourhoods are defined on weighted or complete graphs (windmill).
6. Whether the DTT hypergraph is rebuilt per snapshot.
7. The DTT target transform (log1p in the loader) and the MSE scale.
8. Up/down/neutral thresholds and F1 averaging for R7, the NASDAQ lookback, and whether the setup is RSR's split rather than STHGCN's rolling phases.
9. The CSE dataset and the definition of "risk".
10. How the baselines were configured and tuned (same budget? unknown).
11. The 25-run count is stated, but the reported ±1e-3 spreads are implausibly tight for these tasks, so the p<0.01 claims cannot be reproduced from them.

## Recommended order

1. **R2 chickenpox**: smallest; validates the regression data module, hypergraph builder and MSE evaluation; gives the mean/AR references.
2. **R3 windmill**: reuses R2; needs the weight-cut decision; expect about 1.02-1.05 regardless.
3. **R1 tennis**: needs per-snapshot hypergraphs; pin the target scale first using the calibration table.
4. **R7 NASDAQ classification**: code exists; add micro-F1, accuracy, class shares and chance level; run on the GPU queue when a slot opens.
5. **R8 RSR-I then STHGCN**: needed to compare against the paper's baseline rows in R5 and R7; RSR-I first since it is fully specified.
6. **R4**: mark BLOCKED; revisit only if data is shared.

Total: roughly 45-75 engineering hours for R1-R3, R7 and R8, and about 20-40 GPU-hours (estimated, mostly R7 and R8) on the RTX 3050. The PyG-T tasks are CPU-friendly.

## Decisions needed from the user

- **Regression protocol for R1-R3**: chronological 70/10/20 split with validation-selected epoch and train-only normalization (leak-free, our default), or the PyG-T convention (80/20, full-series standardization, no validation)? Recommend running both, as we do for RSR.
- **Windmill hyperedges**: weight cut or top-k (suggest top-k neighbours with k in {5,10,20}, reported as a sensitivity axis).
- **Tennis target**: log1p (loader default) or raw counts.
- Whether R1-R3 are worth the effort at all: the probes show they are near noise (R3) or have unclear scale (R1). They test "does our THINK run and match the paper's ordering", not hyperbolic geometry.

## Probe log (files kept outside the repo)

- `chickenpox.json` 220,340 B, `twitter_tennis_rg17.json` 1,982,751 B, `twitter_tennis_uo17.json` 1,936,663 B from raw.githubusercontent.com (HTTP 200).
- `windmill_output.json` 47,159,335 B from the Box link with GET (HEAD gave 404; GET redirected to public.boxcloud.com). `graphmining.ai` DNS failed.
- STHGCN `download.sh` Google Drive id: 303 then 404.
- Calibration numbers are from my own numpy code on 80/20 chronological splits with PyG-T default lags. They are references, not the paper's protocol.
