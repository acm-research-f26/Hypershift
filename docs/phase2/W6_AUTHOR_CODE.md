# W6: what do the THINK authors' other papers and code say about how they produced their numbers?

2026-10-06. CPU only. Clones under `external/` (git-ignored). Scripts: `scripts/w6_selection_readings.py` (rescoring saved histories, output `w6_selection_readings.json`), `scripts/w6_authors_delta.py` (authors' delta sampler, output `w6_authors_delta.json`).
Authors verified from `docs/paper/icdm22-think.pdf` p.849: Shivam Agarwal*, Ramit Sawhney*, Megh Thakkar, Preslav Nakov, Jiawei Han, Tyler Derr.

## 1. Papers and code (source: https://shivamag125.github.io/, fetched 2026-10-06)

| Paper | Venue | Code | Commit cloned |
|---|---|---|---|
| THINK | ICDM'22 | github.com/shivamag125/ICDM22-THINK (p.852 fn.1) | **empty repo** (clone warns "empty repository") |
| STHAN-SR (Sawhney, Agarwal, Wadhwa, Derr, Shah) | AAAI'21 | midas-research/sthan-sr-aaai | 05a23fc (2021-06-08) |
| HyperStock-GAT "scale-free nature of stock markets" | WWW'21 | midas-research/hyper-stockgat-www | cbd6f40 (2021-07-11) |
| STHGCN | ICDM'20 | midas-research/sthgcn-icdm | 9f6be48 (W2, data deleted) |
| FAST | EACL'21 | midas-research/fast-eacl | e3e297c (2021-03-26) |
| Quantitative day trading (RL) | NAACL'21 | midas-research/profit-naacl | b90a883 (RL with its own env; best by validation Sharpe, `main.py` L92-94; not comparable) |
| Modeling Financial Uncertainty (curriculum) | UAI'21 | midas-research/finclass-uai | 9d76afc (classification, not read in depth) |
| Others, not stock ranking: MAN-SF EMNLP'20, HYPHEN ACL'22, HypMix EMNLP'21, Hyperbolic T-LSTM SIGIR'21, HyperSteg ICASSP'23, CryptoBubbles NAACL'22 (gtfintechlab/CryptoBubbles-NAACL cloned, not read), MRIL AISTATS'22 | | listed on the page | not read |

Co-author pages (Sawhney, Derr) were not crawled separately; the first author's page already lists all joint code. No later hyperbolic temporal/hypergraph stock code was found there. Code for THINK's other tasks (DTT/CPox/WMill/Risk/Clf) is unpublished.

## 2. Findings from the code (file:line)

Applies to STHAN-SR (`sthan-sr-aaai/training/*`) and HyperStock-GAT (`hyper-stockgat-www/training/*`), which share files nearly verbatim (`evaluator.py` is identical). FAST and STHGCN are noted where they differ.

- **Data.** RSR data (both READMEs: "Download the dataset and follow preprocessing steps from fulifeng/Temporal_Relational_Stock_Ranking"). NYSE 1737 stocks hard-coded (`train_nyse.py` L152, `hgat_nyse.py` L22, L41, L74), NASDAQ 1026 (`train_nasdaq.py` L152). Splits `valid_index=756, test_index=1008` (`train_nyse.py` L95-96), same as ours.
- **Normalisation: full-series max.** `preprocess/eod.py` (about L124-165): `price_max = np.max(selected_EOD[begin_date_row:, 4])` over the whole series including the test years; features `/price_max`; target base price `close/price_max`. So the "full-period (paper) normalisation" is inherited from the shared RSR preprocessing, not a THINK-specific choice (our `norm: paper`).
- **Input and loss.** 5 features per day (4 moving averages and close), level inputs, window `-l` (16 in the README). Loss on `(pred - base_price)/base_price` vs gt return (`train_nyse.py` L42-59): MSE plus alpha times pairwise hinge; alpha 1 (NYSE) and 0.1 (NASDAQ) in the README; HyperStock-GAT default `-a 2` (`config.py` L69).
- **Hyperparameters.** Adam, lr 1e-3, **`weight_decay=5e-4`** (`train_nyse.py` L131-133; `train_hgat_tse.py` L128; HyperStock-GAT README uses `--weight-decay 0.0001`). Epochs: NYSE 100 (`train_nyse.py` L325), NASDAQ 50 (`train_nasdaq.py` L325), TSE 70 (`train_hgat_tse.py` L317). So our Phase 1 default wd 5e-4 is consistent with the authors' habit. The weight-decay collapse found in Phase 1.5 F would then apply to their setup too (INFERRED: whether THINK kept it is unknown).
- **Hypergraph.** The incidence matrix is loaded from `hypergraph_nyse.npy` / `hypergraph_x.npy` (`train_nyse.py` L134); **no script builds it** (preprocess has only `sector_industry.py`, `wikidata.py`, `eod.py`; grep for "hypergraph" in `preprocess/` finds nothing). The STHAN-SR hypergraph construction is therefore undocumented in code, same as STHGCN's `hypergraph.npy`.
- **Model selection: none.** Each epoch trains, then evaluates valid and test and only `print`s both (`train_nyse.py` L217-219, L269-270); no best-model tracking, no checkpoint, no early stopping, `train()` returns None. STHGCN is the same (W2). HyperStock-GAT `train.py` L168 sets `args.patience = args.epochs` and has `args.save` options, but the stock loop (L218-352) also only prints. A reported number was therefore picked from printed logs by an unknown rule: last epoch, best test epoch or best validation epoch are all possible. This is the single biggest unknown.
- **Seed.** `seed = 123456789` hard-coded (`train_nyse.py` L26, L66; FAST `train.py` L18, L26). One run per script call; no seed loop or aggregation code anywhere.
- **Evaluator** (`evaluator.py` L8-63):
  - Sharpe: `(np.mean(sharpe_li5)/np.std(sharpe_li5))*15.87 #To annualize` (L62): top-5 equal-weight daily returns, population std, no risk-free rate, **annualised by 15.87 (= sqrt 252)**. FAST L92-93 is the same with top-1 (and its `sharpe_li5` appends the top-1 return, L73). The THINK paper's printed formula (p.852 Sec. V.B) has Rf and no annualisation, so the printed formula and the code lineage disagree; the code points to annualised values, consistent with the W3 finding that the paper's values appear only under annualisation.
  - NDCG: `performance['ndcg_score_top5'] = ndcg_score(np.array(list(gt_top5))..., np.array(list(pre_top5))...)` inside the per-day loop (L43), on **ticker-index sets** (not relevance scores), overwritten every day, so the reported value is the last test day only. FAST L61 is the same. This is the "STHAN-SR evaluator bug" of Phase 1/W3.
  - IRR/return: `btl5 = 1 + sum of daily top-5 mean returns`, additive, no costs (L16, L52, L60).
  - **The released STHAN-SR/HyperStock-GAT evaluator cannot run:** L61 `sharpe_li = np.array(sharpe_li)` uses a name never defined in the function (`NameError`; grep finds it only on that line). STHGCN's evaluator has a similar unbound-variable crash (W2). So the released code is not what produced the printed numbers; an edited local version was.
- **Hyperbolicity code (HyperStock-GAT only).** `training/utils/hyperbolicity.py` L12-35: `hyperbolicity_sample(G, num_samples=50000)`, 4 random distinct nodes (`np.random.choice(G.nodes(), 4, replace=False)`), unweighted shortest paths, pairs without a path skipped (`except: continue`), per tuple `(s[-1]-s[-2])/2` of the sorted sums, **returns the max over samples** (L35). Input `graph.gpickle` is not in the repo. No delta_rel / feature-hyperbolicity code exists in any repo.
- **Std reporting.** No code computes mean and std across runs. THINK Table II caption says "mean of 25 runs" (p.852); every cell carries ±1e-3 to ±1e-4, including SR and NDCG, and baseline rows (e.g. RSR-I 1.05±1e-3) too. Fig. 2 caption mentions "confidence intervals (over 25 runs)".

## 3. Test: can epoch-reading choices produce small stds? (`w6_selection_readings.py`)

From our saved `history.jsonl` (100 epochs per run, NYSE), seed mean (SD) of test annualised Sharpe, k=5:

| reading | HH R5_f3 (5 seeds) | HH R5_f_paper (10) | EH R5_f_paper (10) | STHGCN R8_f_paper (5) |
|---|---|---|---|---|
| validation-selected (ours) | 2.20 (0.41) | 1.93 (0.60) | 1.15 (0.70) | 1.24 (0.93) |
| LAST epoch (no selection) | 1.82 (0.25) | 1.56 (0.40) | 0.94 (0.65) | 1.34 (0.90) |
| mean over epochs | 1.77 (0.12) | 1.32 (0.08) | 1.25 (0.21) | 1.10 (0.17) |
| test-oracle max (leaky) | 3.07 (0.39) | 2.89 (0.28) | 2.76 (0.38) | 3.24 (0.37) |

The smallest seed SD of any SR reading or group is 0.04 (epoch-mean, R5_f_train EE); the paper's ±1e-3 is 40 to 1000 times smaller, and a standard error over 25 runs would still need an SD of about 0.005. Buggy NDCG at the oracle epoch is 0.95-0.99 (ceiling), above the paper's 0.78-0.86; last-epoch and epoch-mean buggy NDCG is 0.77-0.83 (paper band, as in W3). Correct NDCG@5 has SD 0.001-0.003 at a value of 0.57. **No epoch reading reproduces ±1e-3 on SR.** A mechanism consistent with the evidence (INFERRED, untested): the scripts hard-code one seed, so "25 runs" may be repeats of the same seed whose only variance is GPU non-determinism. The ± might also be a standard error or half-width of another quantity. UNKNOWN.

## 4. Authors' delta code on our data (`w6_authors_delta.py`, CPU)

The authors' sampler (50,000 random 4-tuples, max of (largest - second)/2, disconnected tuples skipped) applied to our s=1 distance matrices (clique-expansion distance), 20 random seeds, nodes drawn from all nodes or from non-isolated nodes. Table I (p.852): NYSE 0.5, NASDAQ 1.0.

| market | graph | share of tuples connected | max over 50k tuples (values seen over 20 seeds) |
|---|---|---|---|
| NYSE | v2 (App. B) | 0.60-0.63 | **1.0** (1.5 in 1 of 20 seeds, non-isolated pool) |
| NYSE | v1_old (star per relation) | 0.60-0.63 | 1.0 |
| NYSE | wiki-only, all-nodes pool | 0.000 (almost none) | {0, 0.5} |
| NASDAQ | v2 | 0.12-0.13 | **1.0** |
| NASDAQ | v1_old | 0.12-0.13 | 1.0 (0.5 in some seeds, all-nodes pool) |
| NASDAQ | wiki-only | 0 to 0.38 | {0, 0.5} / 1.0 |

Result: the authors' own sampler gives **NASDAQ 1.0 (matches Table I)** and **NYSE 1.0, not 0.5 (does not match)**. 0.5 appears only when almost no sampled tuple is connected (wiki-only graph, all-nodes pool). 1.0 is the modal output for sparse graphs under this sampler, so the NASDAQ match is weak evidence. The exact delta (W5) is 1.5 on both. Verdict: NYSE delta_hg NOT reproduced by the authors' sampler; NASDAQ matches at the modal value. Their code does not cover delta_rel.

## 5. Most likely THINK protocol (INFERRED from sibling code; THINK's own code is empty)

| Unspecified detail | Evidence from sibling code | Likelihood | Already tested by us |
|---|---|---|---|
| Sharpe annualised x sqrt252 (15.87), no Rf, k=5, population std | `evaluator.py` L62 (STHAN-SR, HyperStock-GAT), FAST L93 | high | yes: `ours` columns of `protocol_matrix.md` (the printed formula = `unann` also tested) |
| Normalisation = full-series max | `preprocess/eod.py` (RSR lineage) | high | yes: `norm=paper` families R5_f_paper, R8_f_paper |
| NDCG = last-day ticker-index-set NDCG | `evaluator.py` L43, FAST L61 | high | yes: `ndcg_buggy` cells |
| Epoch selection: none coded; hand-picked from printed logs | `train_nyse.py` L217-270 (print only) | high that no rule is coded; which rule used UNKNOWN | oracle and validation yes; last epoch and epoch-mean now (Section 3), no new match |
| Sharpe from best-test epoch | no code | medium | yes: oracle rows (2.7-3.2, above the paper's 1.1-1.2) |
| Level inputs, 5 features, window 16 | README `-l 16` | high | yes (Phase 1 default `level`) |
| wd 5e-4, lr 1e-3, alpha 1 / 0.1, 100 epochs on NYSE | `train_nyse.py` L130-133, L325 | high for STHAN-SR; THINK unknown | yes (Phase 1 default; collapse fixed in 1.5 F) |
| Single hard-coded seed, no aggregation code | `train_nyse.py` L26 | medium for THINK's "25 runs" | **no** |
| Graph: stars per relation vs pairwise second-order | no builder in repo; App. B p.854 only | n/a | App. B graph and v1_old both (W5) |
| delta_hg by 4-tuple sampling, max | HyperStock-GAT `hyperbolicity.py` | medium | now: NASDAQ 1.0 yes, NYSE no (Section 4) |
| delta_rel features and normalisation | none | n/a | W5 (16-day level windows, full-period norm, m=500 matches) |

## 6. Untested routes that could plausibly yield Table II (with cost)

1. **Same-seed repeats ("25 runs" of one hard-coded seed, std from GPU non-determinism only).** Could explain ±1e-3 but not the means. Cost: 5 repeats of one THINK seed on the laptop GPU, about 35 min each (21 s/epoch x 100), about 3 h total; the GPU queue owns the GPU. On CPU deterministic ops give SD exactly 0, so only a GPU test is informative. Low value for the paper's means.
2. **Joint reading at the best-test-Sharpe epoch** (Sharpe and NDCG both read at that epoch, rather than per-metric oracle). CPU, minutes, but only R5_f3 HH/EH have per-epoch predictions saved. Expected: SR near 3, so still above the paper; low payoff.
3. **Authors' exact STHAN-SR recipe** (wd 5e-4, level inputs, 100 epochs, last epoch): effectively the Phase 1 default (R5_f_paper); nothing new.
4. None of these recovers the paper's tiny stds or a stable THINK > EH ordering, so the REPORT_CLOSURE.md conclusion is unchanged. What would move the study: the authors' edited evaluator or training logs, or a reply from the authors (the THINK repo is empty).

## 7. Engineering log
- Shallow-cloned sthan-sr-aaai, hyper-stockgat-www, fast-eacl, profit-naacl, finclass-uai, CryptoBubbles-NAACL, ICDM22-THINK (empty) into `external/`.
- Read evaluator, training loop, loader, model and preprocess for STHAN-SR; config, evaluator and hyperbolicity for HyperStock-GAT; evaluator for FAST. Not read in depth: profit-naacl, finclass-uai, CryptoBubbles.
- All computation CPU (`CUDA_VISIBLE_DEVICES=-1`); no `results/` folder written.
