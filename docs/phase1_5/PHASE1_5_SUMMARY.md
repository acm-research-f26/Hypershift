# Phase 1.5 summary: fidelity audit of the Phase 1 reproduction

*Written 2026-10-02 from the audit notes `A_evaluator.md`, `B_model.md`, `C_data_graph.md`, `D_known_signal.md`, `D_resolved_configs.md`, `E_rsr_original.md`, `F_learnability.md` (same folder) and `docs/PHASE1_TRACKER.md`; updated with `F_p1f_smoke_results.md`. Paper = `docs/paper/icdm22-think.pdf` (pp. 849-854). This file consolidates; it adds no new training results except the tie-break table in section 3 and the p1f smoke readout in section 5.*

## 0. Status in one paragraph

Phase 1 ended with "not reproduced under validation selection" (`docs/PHASE1_TRACKER.md`). Phase 1.5 asked whether that null could be our fault. The evaluator, window alignment, data loading and graph build were audited and found correct (A, C). The model matches the printed equations except for a short list of departures (B). Two things undermine the Phase 1 nulls: `norm: paper` leaks future price scale into inputs (A), and the repo's default optimizer setting, coupled weight decay 5e-4, makes the model unable to learn even a planted signal (D, F). The authors' own RSR-I code shows the same no-skill picture on our evaluator (E), so part of the null is shared across implementations. **The corrected p1f smoke has only seed 0 and is INSUFFICIENT SEEDS. The corrected full-NYSE R5_f rerun gives NO EVIDENCE for THINK vs TConv+DHHAN under both `norm=paper` and `norm=train`; R8_f baseline comparisons have INSUFFICIENT SEEDS.**

Fidelity labels used below: **PAPER** = stated in `docs/paper/icdm22-think.pdf` and implemented as printed; **INFERRED** = the paper is silent, we chose (usually from the STHAN-SR / RSR code); **DEPARTURE** = differs from what the paper prints, or from the authors' code, for a stated reason; **UNKNOWN** = cannot be determined from the PDF.

## 1. Audits A to F

### A. Evaluator, target alignment, normalization (`A_evaluator.md`)
- **Question.** Are Sharpe, NDCG, the day alignment and the normalization correct and leak-free?
- **Finding.** No bug in `metrics.py`, `loop.py`, `rsr.py`. Day t predictions are scored against close(t+1)/close(t)-1; 237 test days; the daily top-5 return was recomputed from raw CSV closes (max abs diff 3.7e-9). Three issues found:
  1. Exact prediction ties are broken by lowest ticker index, and collapsed models tie often (THINK: ties at the top-5 boundary on 31% of test days, section 3). Effect on the headline is small (section 3).
  2. `norm: paper` divides each stock by its full-series max close, so features before the test year carry future scale (confirmed by a synthetic test). A static rank on that factor scores Sharpe 4.37 on the test year, an upper bound on what the leak contains. All R5/R8 full-NYSE runs and small-scale paper-protocol runs used it.
  3. The x15.87 Sharpe annualization is from the STHAN-SR evaluator, not RSR (verified 2026-10-02 on the authors' repos: the RSR `evaluator.py` computes `btl` only; STHAN-SR `evaluator.py` line 62 is `sharpe5 = mean/std*15.87 #To annualize`). The tracker/docs wording was corrected in this housekeeping step.
- **Fidelity.** Sharpe formula: paper writes `E[R_a-R_f]/std[R_a-R_f]`, top-k, no annualization (p852 Sec. IV-B); ours is top-5, x sqrt(252), R_f = 0 (INFERRED from STHAN-SR; k, R_f, annualization UNKNOWN for the paper). NDCG k, gains, relevance: UNKNOWN. Train/val/test split: INFERRED (not in paper; Table I p850 gives T = 1245 only).
- **Effect on verdict.** Leak-free Sharpe moves by about 0.1 under any tie rule: negligible against seed std 0.3-0.76. The `norm: paper` leak favours a good score, so it cannot explain the poor leak-free result; it does mean paper-protocol numbers are not leak-free.

### B. Model vs paper eq. 6-17 (`B_model.md`)
- **Question.** Does the code compute what the paper prints?
- **Finding.** Temporal conv (eq. 9-12), gyromidpoint (eq. 13), eq. 14 score, eq. 15 aggregation, parameter sharing (eq. 16) and the eq. 17 pipeline match the print (PAPER, up to numerical clamps). Departures ranked by likely impact: (1) the operator in eq. 14 (the paper's eq. 7, p850, is not implementable as printed; we use a plain product); (2) a per-node softmax over hyperedges that the paper does not print (p851); (3) static graph for all time slices (eq. 16 allows G_t); (4) isolated nodes pass through; (5) HNN++ unit-norm form of eq. 10. Kernel 4, stride 4, hidden 32, lookback 16, second kernel 4 are INFERRED. Switches `attn_odot` and `attn_norm` were added (defaults unchanged).
- **Effect on verdict.** None measured. The 1-epoch smoke runs of the switches only show they run. Ranks 1-2 are untested at full scale (OPEN).

### C. Data and graph (`C_data_graph.md`)
- **Question.** Do the shipped data and the Appendix B hypergraph (p854) match what we built?
- **Finding.** Shapes match Table I (p850): NYSE 1,245 x 1,737, NASDAQ 1,245 x 1,026. Returns recomputed from raw closes match to 1e-6. An independent rebuild from raw files reproduces the cached graphs exactly (NYSE 4350 edges = 107 industry + 21 stars + 4222 pairs; NASDAQ 1066). 65.6% of NYSE stocks have no Wikidata QID, 15% have a wiki edge. Max node degree is 114 versus the paper's Fig. 3b axis starting at 31 (p853): not reproduced; the per-Z grouping of second-order relations is not recoverable from the repo files (UNKNOWN). Also: the NASDAQ last raw row is not all missing (474 of 1026 listed stocks missing, 552 present); the loader drops it to reach T = 1245; CLAUDE.md corrected.
- **Fidelity.** Graph structure follows App. B text (PAPER) with the Z grouping and Wikidata snapshot UNKNOWN; the missing-value fill 1.1 and window-min target mask are INFERRED from the RSR code, not verified against it.
- **Effect on verdict.** No bug found; graph differences vs the paper remain a possible but unquantified cause.

### D. Known-signal test and resolved configs (`D_known_signal.md`, `D_resolved_configs.md`)
- **Question.** Can the pipeline recover a planted signal (309 stocks, real g2 graph, synthetic returns)? What are the exact Phase 1 settings?
- **Finding.** Null control clean (no manufactured signal). With Phase 1 level inputs, every arm reaches at most 19% of the oracle IC even for a strong signal, while a ridge probe on the same tensors finds it. With relative inputs, no-graph arms recover 24-81%; graph arms lose to no-graph where the signal is own-lag. THINK was the weakest arm and its predictions shrank to 1e-8..1e-12. Resolved configs: all Phase 1 runs used `weight_decay 5e-4` (coupled), `batch_days 8` (93 steps/epoch vs the repos' 740), `grad_clip 1.0`, `target: return` (the repos output a price), alpha 1.
- **Fidelity.** Every hyperparameter above is INFERRED; the paper states none of them (pp. 849-854). Departures from the authors' repos (price target, 1 day/step, weight decay on RSR-I, validation days possibly entering training in STHAN-SR) are listed in `D_resolved_configs.md` sec. 5 and are unquantified.
- **Effect on verdict.** Phase 1 nulls cannot distinguish "no signal in the data" from "this setup cannot learn a 1-2% daily signal".

### E. Authors' original RSR-I code (`E_rsr_original.md`)
- **Question.** Is the "no skill" result shared with the authors' own baseline code?
- **Finding.** Their RSR-I (industry relations, pre-trained LSTM, their data byte-for-byte, TF2-compat run, 5 seeds) scored by our evaluator: Sharpe 0.45 ± 0.46 at their epoch rule, 0.93 ± 0.24 at ours, 1.41 ± 0.27 at the test-selected epoch; hold-all 1.531; NDCG@5 0.563-0.567 vs random 0.5639; IC about 0. No ties (continuous predictions).
- **Fidelity.** A different RSR variant from our R8 (sum-weight, industry only, pretrained embedding; INFERRED which variant the paper calls RSR-I, UNKNOWN). How their LSTM embedding was trained is UNKNOWN. TF2 vs TF1 not bit-checked.
- **Effect on verdict.** The missing advantage is not THINK-specific on this split. It is not a replication of the paper's 1.05 (different Sharpe formula, selection).

### F. Learnability (`F_learnability.md`)
- **Question.** Why did THINK not learn the planted signal, and what config does?
- **Finding.** Root cause (CPU probe plus Kaggle grids): coupled L2 weight decay 5e-4 in Adam at lr 1e-3 is 10-40x the loss gradient, driving the hyperbolic weights to zero (|z| 0.71 to 0.006), prediction sd to 4e-12. `weight_decay=0` alone raises THINK from 26% to 63-84% of the oracle IC and EH from 32% to 72-77% (5 seeds, relative inputs); decoupled decay (AdamW) behaves the same. Level inputs stay unlearnable (<50% in every config tried). Graph arms beat no-graph at `group` in all stage-2 configs (5/5 seeds); at `high` they lose unless a residual self path is added (DEPARTURE from eq. 15). `alpha=0` and `spatial_residual` raise recovery to 77-98%.
- **Fidelity.** `weight_decay=0`, `alpha=0`, `relative` inputs: INFERRED (paper silent). `spatial_residual`, `head_scale`, `input_std`: DEPARTURE (not needed for learning; residual only for graph-beats-none at `high`).
- **Effect on verdict.** The Phase 1 setup (wd 5e-4) was the collapsed one. **Phase 1 nulls are not evidence about THINK's design until the corrected rerun is analysed.** F also shows the benchmark is a necessary check only (linear planted signal, 30 epochs).

## 2. Fidelity table (what is where)

| item | label | source |
|---|---|---|
| Hyperbolic temporal conv, beta-concat, Poincare FC (eq. 9-12) | PAPER (eq. 10 in HNN++ unit-norm form: DEPARTURE, low impact) | p850; B |
| Gyromidpoint, eq. 14 score, eq. 15 aggregation, eq. 17 pipeline | PAPER | p850-851; B |
| `(.)` operator in eq. 14 as plain product | DEPARTURE (printed eq. 7 not implementable) | p850; B |
| Softmax over hyperedges per node | INFERRED / DEPARTURE (paper prints none) | p851; B |
| Static graph, isolated nodes pass through | INFERRED / DEPARTURE | p851; B |
| K, stride, hidden, lookback, lr, epochs, batch, optimizer | INFERRED (paper states none, pp. 849-854) | D_resolved_configs |
| `weight_decay 5e-4` | INFERRED, and shown harmful (F) | D, F |
| `relative` inputs (F fix) | INFERRED | F |
| Loss = masked MSE + alpha ranking hinge | INFERRED (STHAN-SR objective) | D_resolved_configs |
| Industry + Wikidata hyperedges, first-order stars, second-order pairs | PAPER (App. B, p854); Z grouping and snapshot UNKNOWN | C |
| Splits 756 / 1008, T = 1245 | T from paper Table I p850; split INFERRED (RSR code) | A, C |
| Sharpe: top-5, x sqrt(252), R_f = 0, equal weight | INFERRED (STHAN-SR evaluator); paper formula differs (p852 Sec. IV-B) | A |
| NDCG@5, linear gain, shifted relevance | INFERRED / UNKNOWN for the paper | A |
| `norm: paper` (full-series max) | matches shipped files; look-ahead (DEPARTURE from leak-free practice) | A |
| `spatial_residual` | DEPARTURE from eq. 15 | F |

## 3. Random tie-break column (new, this step)

`scripts/tiebreak_report.py` and `hypershift.eval.metrics.evaluate_random_ties` rescore saved test predictions with exact ties broken at random (mean of 20 draws, seed 0). The default evaluator is unchanged; the unit test is `tests/test_metrics.py::test_random_tiebreak_matches_default_without_ties_and_differs_with_ties`. R5_g2 holds the selected-epoch predictions (leak-free), full NYSE, `norm: paper`, level inputs. Tie days = share of test days with an exact tie at the top-5 boundary.

| R5_g2 arm | n | tie days | Sharpe default | Sharpe random ties | mean delta | max abs per-seed delta | NDCG@5 default / random |
|---|---|---|---|---|---|---|---|
| THINK (HH) | 25 | 31% | 0.089 ± 0.300 | -0.005 ± 0.252 | -0.094 | 0.567 | 0.562 / 0.561 |
| EH | 25 | 38% | 0.383 ± 0.758 | 0.389 ± 0.703 | +0.007 | 0.547 | 0.563 / 0.563 |
| EE | 10 | 23% | 0.237 ± 0.752 | 0.316 ± 0.765 | +0.079 | 0.526 | 0.563 / 0.563 |
| HE | 10 | 67% | 0.193 ± 0.449 | 0.109 ± 0.325 | -0.085 | 0.488 | 0.562 / 0.562 |
| EE_none | 10 | 22% | 0.846 ± 0.645 | 0.663 ± 0.419 | -0.183 | 1.519 | 0.565 / 0.565 |
| HH_none | 10 | 0% | 0.830 ± 0.727 | 0.831 ± 0.728 | +0.001 | 0.006 | 0.566 / 0.566 |
| THINK_nodist | 10 | 35% | -0.023 ± 0.353 | -0.107 ± 0.307 | -0.084 | 0.380 | 0.561 / 0.561 |

Reading: random tie-break moves arm means by -0.18 to +0.08 and single seeds by up to 1.5, against seed std 0.3-0.76. NDCG@5 is unchanged to three decimals (sklearn already averages ties). The ordering of arms and the NO EVIDENCE verdicts do not change. The no-graph arms with no ties (HH_none) are unaffected, as expected. These numbers agree with the 100-draw values in `A_evaluator.md` (THINK -0.006 there).

Planted-signal runs (D and F; synthetic returns on the 309-stock universe; Sharpe is dominated by the noise-path market return, so "xSR" = top-5 minus hold is also in the script output). Changes in mean Sharpe, default to random ties:
- D level inputs, collapsed predictions: `high` HH_hyper 3.24 to 2.89 (-0.35; 61% tie days; max per-seed 1.57), EH_hyper 1.29 to 1.18, EE_hyper 1.60 to 1.54; `group` EE_hyper 0.43 to 0.63, EH_hyper 0.08 to 0.25, HH_hyper 0.85 to 0.82.
- D relative inputs (wd 5e-4): changes -0.19 to +0.31; no-graph arms 0 to +0.06.
- F `wd0` (relative): tie days fall to 7-21% (0% with `spatial_residual`), changes -0.22 to +0.18; F level-mode wd0: HH_hyper 8.70 to 8.47, EH_hyper 3.19 to 3.45.
- IC is tie-free for the learning conclusions of D and F (constant-prediction days count as IC 0), so none of the D/F learning conclusions depend on the tie rule.

## 4. Still open

- **r5f (Kaggle `hypershift-run-r5f`, preset 10) and r8f (`hypershift-run-r8f`, preset 11) are fetched and analysed in `F_r5f_results.md`.** They rerun full NYSE with `input_mode=relative weight_decay=0` for THINK/EH/EE (and RSR-I/STHGCN), under both `norm=paper` and `norm=train`.
- Small-scale Phase 1 smoke `p1f` reran seed 0 only with `input_mode=relative weight_decay=0`; the rest of small-scale Phase 1 arms (level-input g2, tuned `rel_tuned_g2`, A10/C1/G2/G12 ablations and R1-R3) were not rerun in preset `p1f` (`F_p1f_smoke_results.md` sec. 3).
- `alpha=0` and `spatial_residual` full-NYSE smoke documents have no finished runs in `F_r5f2_smoke_results.md`, so their result status is PENDING.
- Untested model departures (B ranks 1-2: eq. 14 operator, softmax).
- The paper's Sharpe formula, k, R_f, buy price, split, Fig. 3b degree axis, Wikidata snapshot and Z grouping remain UNKNOWN.
- Differences from the authors' repos that are unquantified: price-ratio target, 1 day per step, validation days entering training (STHAN-SR, by code reading only), clipping, dropout.
- Fidelity results are for synthetic signals and one test year; they show the pipeline can learn once wd is 0, not that real data hold a learnable signal.

## 5. Final verdict

**Current Phase 1.5 verdict: NO EVIDENCE for corrected full-NYSE THINK vs TConv+DHHAN under both norms; INSUFFICIENT SEEDS for corrected R8_f baseline comparisons and the corrected p1f smoke.** `F_r5f_results.md` reports finished full-NYSE seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for HH (THINK) and EH (TConv+DHHAN), and [0, 1, 2, 3, 4] for EE, RSR-I and STHGCN. `F_p1f_smoke_results.md` reports only one finished paired seed, so no allowed verdict stronger than INSUFFICIENT SEEDS is supported. Finished p1f seeds: seed 0 for `HH_hyper`, `HH_clique`, `EH_hyper`, `EE_none`, `rsr_i`, `HH` and `EH`. `F_r5f2_smoke_results.md` reports no finished `alpha=0` or `spatial_residual` smoke runs.

Paper anchor: Table II p. 852, Sec. V.A reports NYSE stock ranking THINK 1.18 vs TCONV+DHHAN 1.14 Sharpe and NASDAQ Clf F1 THINK 0.49 vs TCONV+DHHAN 0.44; Sec. IV-B p. 852 defines Sharpe as `E[R_a - R_f] / std[R_a - R_f]`. The corrected runs use our evaluator, so levels are not like-for-like with the paper.

| scope | norm | validation-selected epoch | `test_oracle_sr` / best-test | finished seeds | verdict |
|---|---|---|---|---|---|
| p1f 309-stock THINK vs TConv+DHHAN, hyperedges (`HH_hyper - EH_hyper`) | UNKNOWN in result document | `diff leak-free` +1.039; `HH_hyper` 1.245 ± 0.000, `EH_hyper` 0.206 ± 0.000 | result document labels this `best-test diff` +1.039; `HH_hyper` 1.245 ± 0.000, `EH_hyper` 0.206 ± 0.000 | seed 0 only | INSUFFICIENT SEEDS |
| p1f 309-stock hyperedges vs pairwise (`HH_hyper - HH_clique`) | UNKNOWN in result document | `diff leak-free` -0.129; `HH_hyper` 1.245 ± 0.000, `HH_clique` 1.374 ± 0.000 | result document labels this `best-test diff` -0.129; `HH_hyper` 1.245 ± 0.000, `HH_clique` 1.374 ± 0.000 | seed 0 only | INSUFFICIENT SEEDS |
| p1f R8 small baseline, `rsr_i minus HH_hyper` | UNKNOWN in result document | +0.392; `rsr_i` 1.637 ± 0.000, `HH_hyper` 1.245 ± 0.000 | UNKNOWN in result document | seed 0 only | INSUFFICIENT SEEDS |
| p1f R8 small baseline, `rsr_i minus EH_hyper` | UNKNOWN in result document | +1.430; `rsr_i` 1.637 ± 0.000, `EH_hyper` 0.206 ± 0.000 | UNKNOWN in result document | seed 0 only | INSUFFICIENT SEEDS |
| p1f NASDAQ Clf `HH - EH` macro-F1 | UNKNOWN in result document | `mean diff` -0.0073; `HH` 0.3441 ± 0.0000, `EH` 0.3513 ± 0.0000 | UNKNOWN in result document | seed 0 only | INSUFFICIENT SEEDS |
| p1f NASDAQ Clf `HH - EH` micro-F1 | UNKNOWN in result document | `mean diff` +0.0203; `HH` 0.3817 ± 0.0000, `EH` 0.3615 ± 0.0000 | UNKNOWN in result document | seed 0 only | INSUFFICIENT SEEDS |
| corrected full NYSE R5_f, THINK vs TConv+DHHAN | norm=paper | `diff (a-b) leak-free` +0.783; HH 1.929 ± 0.565, EH 1.146 ± 0.661 | `best-test diff` +0.134; HH 2.891 ± 0.263, EH 2.757 ± 0.357 | HH/EH seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | NO EVIDENCE |
| corrected full NYSE R5_f, THINK vs TConv+DHHAN | norm=train | `diff (a-b) leak-free` +0.252; HH 1.687 ± 0.316, EH 1.436 ± 0.562 | `best-test diff` +0.186; HH 2.925 ± 0.128, EH 2.739 ± 0.271 | HH/EH seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | NO EVIDENCE |
| corrected full NYSE R5_f, THINK vs EE | norm=paper | `diff (a-b) leak-free` +0.913; HH 1.929 ± 0.565, EE 1.045 ± 0.222 | `best-test diff` +0.442; HH 2.891 ± 0.263, EE 2.558 ± 0.181 | HH seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; EE seeds [0, 1, 2, 3, 4] | INSUFFICIENT SEEDS |
| corrected full NYSE R5_f, THINK vs EE | norm=train | `diff (a-b) leak-free` +0.997; HH 1.687 ± 0.316, EE 0.788 ± 0.581 | `best-test diff` +0.248; HH 2.925 ± 0.128, EE 2.660 ± 0.240 | HH seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; EE seeds [0, 1, 2, 3, 4] | INSUFFICIENT SEEDS |
| corrected full NYSE R8_f baselines vs THINK/EH | norm=paper | HH 1.929 ± 0.565; EH 1.146 ± 0.661; RSR-I 1.020 ± 0.287; STHGCN 1.240 ± 0.827 | HH 2.891 ± 0.263; EH 2.757 ± 0.357; RSR-I 2.543 ± 0.178; STHGCN 3.243 ± 0.332 | HH/EH seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; RSR-I/STHGCN seeds [0, 1, 2, 3, 4] | INSUFFICIENT SEEDS |
| corrected full NYSE R8_f baselines vs THINK/EH | norm=train | HH 1.687 ± 0.316; EH 1.436 ± 0.562; RSR-I 0.981 ± 0.397; STHGCN 1.264 ± 1.039 | HH 2.925 ± 0.128; EH 2.739 ± 0.271; RSR-I 2.686 ± 0.209; STHGCN 2.958 ± 0.232 | HH/EH seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; RSR-I/STHGCN seeds [0, 1, 2, 3, 4] | INSUFFICIENT SEEDS |
| corrected full NYSE `alpha=0` smoke | norm=train | PENDING | PENDING | no finished runs | PENDING |
| corrected full NYSE `spatial_residual` smoke | norm=train | PENDING | PENDING | no finished runs | PENDING |

Reading: corrected full-NYSE THINK is higher than TConv+DHHAN in both norms, but the allowed verdict in `F_r5f_results.md` is NO EVIDENCE. Corrected R8_f has only five paired baseline seeds, so its allowed verdicts are INSUFFICIENT SEEDS. The `alpha=0` and `spatial_residual` smoke result document has no finished runs, so those rows stay PENDING.
