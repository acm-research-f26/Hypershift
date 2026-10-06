# Can the THINK paper's results be recreated? Closure report

2026-10-05. Written by the Claude Code orchestrator. Spec: `docs/phase2/SPEC.md`. The headline numbers below were re-checked by the orchestrator from the run files and the worker JSON outputs.

Paper claim (`docs/paper/icdm22-think.pdf`, p. 852, Table II, NYSE): THINK Sharpe 1.18 / NDCG 0.86 > TCONV+DHHAN 1.14 / 0.81 > STHGCN 1.10 / 0.78.

## Final statement

**We cannot recreate the THINK paper's claimed advantage under any leak-free evaluation we could construct. Its headline numbers are reachable only under protocol readings that use the test year, an annualised Sharpe the paper does not print, or a buggy NDCG. Even then, the model ordering is not robust.**

Specifically:
- **Leak-free:** no model (THINK, its Euclidean arm, RSR-I, STHGCN, or the RSR authors' own code) shows ranking skill beyond holding the market. The evidence covers:
  - 2017 (Phases 1, 1.5, 1.5a)
  - 2018-2023 with a frozen model (Phase 1.5b)
  - 2019-2023 with annual retraining (Phase 1.5c)
- **THINK > TCONV+DHHAN is not stable.** Two replications with identical configs flip its direction:

  | Run | THINK ahead | THINK mean | EH mean |
  |---|---|---|---|
  | R5_f2 | 2 of 5 seeds | 1.98 | 2.05 |
  | R5_f3 | 4 of 5 seeds | 2.20 | 1.46 |

  With 10 seeds (R5_f_train), THINK leads 6 of 10, NO EVIDENCE.
- Across 88 protocol combinations, the full ordering THINK > TCONV+DHHAN > STHGCN never holds with both gaps above 2 standard errors.

This is a statement about an inferred-settings reimplementation, run on the authors' public data with their earlier public code. THINK's own code is unpublished (the repository is empty), so we cannot exclude an undisclosed implementation or protocol detail that changes the result. Every route we could construct is listed below.

## Routes tried

| # | Route | Where | Outcome |
|---|---|---|---|
| 1 | Paper protocol as most likely run: best test epoch, full-period normalisation | Phase 1 (`docs/phase1/`), W3 | **Reproduces and exceeds the paper's values and ordering** (THINK 2.40 vs EH 1.64, old setup). This is test-set selection, so leaky |
| 2 | Leak-free: validation-selected epoch | Phase 1, 1.5 F (`F_r5f_results.md`) | Ordering and advantage vanish. Corrected full NYSE THINK vs EH: NO EVIDENCE (10 seeds, both normalisations) |
| 3 | Fix the training collapse (weight decay) and the inputs (relative) | Phase 1.5 F | The model learns a planted signal (63-84% of oracle) but shows no real-data ranking skill |
| 4 | Variants: alpha=0, spatial residual, attention readings, structure (hyper/clique/none), decomposition, universe size | Phases 1 and 1.5 | None gives evidence of an advantage |
| 5 | The 2017 Sharpe of about 2 | Phase 1.5a | Not distinguishable from random daily top-5 baskets in a strong year (high-beta baskets) |
| 6 | Frozen model on 2018-2023 (Alpaca, survivor-free) | Phase 1.5b | THINK 0.43 vs hold-all 0.51. NO EVIDENCE of ranking skill. THINK vs EH: INSUFFICIENT SEEDS |
| 7 | Annual retraining on 2019-2023 | Phase 1.5c | THINK 0.70 / EH 0.51 / hold-all 0.69. NO EVIDENCE of skill. HH vs EH: INSUFFICIENT SEEDS (4 of 5, p 0.31) |
| 8 | Authors' **RSR** code and data, unchanged | Phase 1.5 E, Phase 2 W1 | **Reproduces the RSR paper's own metrics** (IRR 1.052 vs paper 1.06; MRR 0.045 vs 0.0451; MSE 2.27e-4 vs 2.26e-4; Table 6 p. 15). The same predictions under our leak-free Sharpe show no skill (0.45-0.93 vs hold-all 1.53) |
| 9 | Authors' **STHGCN** code | Phase 2 W2 | **Not runnable:** data deleted (Drive 404, no mirror), required files unpublished, evaluator crashes, and the code scores the test set every epoch with no selection rule. Paper values UNKNOWN (paywalled) |
| 10 | Every reading of the paper's evaluation (selection × normalisation × Sharpe formula × k × NDCG) | Phase 2 W3 (`protocol_matrix.md`) | See next section |

## What the protocol matrix shows (W3, 88 cells)

- **Sharpe values.** The paper's 1.18 range appears only with an **annualised** Sharpe. The formula the paper prints (p. 852, Sec. IV-B, no √252) gives about 0.06-0.19 for every model. So the paper's printed formula and its reported values are hard to reconcile; the annualisation is INFERRED.
- **NDCG.** A correct NDCG@5 is about 0.57 for every model, including random (0.564). The paper's 0.86 is reached only by the STHAN-SR evaluator bug (ticker-index sets, last day only). Under that bug, 43% of random models score at least 0.86 (Phase 1, `scripts/ndcg_bug_demo.py`).
- **Ordering.** THINK > TCONV+DHHAN > STHGCN holds on means in 22 of 88 cells, and in 0 of 88 with both gaps above 2 SE. The best cell is R5_f3, k=5, validation, where THINK vs EH has z 2.05 but EH vs STHGCN has z 0.33. R5_f3's identical-config twin R5_f2 reverses THINK vs EH, so that cell is noise.
- **Value matches.** Six cells "match" all three paper Sharpe values only because our seed SD (0.3-1.5) is the tolerance. The paper reports SDs of about 0.005, which is implausibly small next to the seed-to-seed spread we measure.

## Hyperbolicity check (W5, `docs/phase2/W5_DELTA_RESULTS.md`)

The question: does our data or graph differ from the paper's (Table I: NYSE δ_hg 0.5 / δ_rel 0.087; NASDAQ 1.0 / 0.107)? We searched 1,150 predeclared settings.

- **δ_rel is reproducible.** Using the 16-day level input windows with the paper's full-period scaling and 500-point Khrulkov subsets gives NYSE 0.093 and NASDAQ 0.103. Only 0.17 joint matches were expected by chance, so this probably is the authors' measurement, and it is consistent with the price data being the same.
- **δ_hg is not reproducible by one method.**
  - Computed exactly on our App. B graph, both markets give 1.5.
  - NYSE 0.5 appears only in samples of 30 nodes or fewer; NASDAQ 1.0 only in samples of 100 or more.
  - Joint matches (14 of 429) occur at the chance rate (11.5 expected), and 13 of the 14 use the old star-for-every-relation graph or tiny connected components.

  Their graph may therefore differ from their App. B text, or they used an undocumented sampling scheme. This cannot be settled without them.
- **Implication for the closure: none.** δ is a descriptive statistic and never enters the model. The star-graph variant was already run in Phase 1, with the same no-skill outcome under leak-free evaluation.

## The authors' other code (W6, `docs/phase2/W6_AUTHOR_CODE.md`)

The claims below were checked by the orchestrator in `external/sthan-sr-aaai/training/`. The STHAN-SR (AAAI'21) and HyperStock-GAT (WWW'21) repositories show the protocol these authors used before THINK:
- **No epoch selection.** Validation and test are printed every epoch and nothing is saved (`train_nyse.py:213-270`).
- **Hard-coded seed 123456789** (`train_nyse.py:26,66`), one run per call, and no aggregation code.
- **Training settings:** weight decay 5e-4 (`:133`), lr 1e-3, 100 epochs.
- **Annualised Sharpe:** `mean/std * 15.87`, top-5, no risk-free rate (`evaluator.py:62`). The shipped evaluator crashes on an undefined `sharpe_li` (`:61`).
- **NDCG** uses the index-set, last-day bug.
- **Price scaling** divides by the full-series max (`preprocess/eod.py:132`).
- **No hypergraph builder.** `hypergraph_nyse.npy` is loaded from an unpublished file.
- **Their δ sampler** (HyperStock-GAT `hyperbolicity.py`, 50k random 4-tuples) gives NASDAQ 1.0, which matches, and NYSE 1.0, where the paper reports 0.5.

Every one of these readings is already covered by Phase 1 or W3. The most likely THINK protocol (full-series max, weight decay 5e-4, annualised Sharpe, buggy NDCG, epoch picked from the test-printing logs) is the one that reproduces Table II in our Phase 1 route 1.

**The reported ±1e-3 standard deviations are not explained by any training protocol.** Our R5_f2 and R5_f3 runs are same-seed repeats (seeds 0-4 both times, identical configs), yet per-seed Sharpe differs by up to 1.6 (EH seed 1: 2.11 vs 0.52). Repeating a seed on GPU gives a spread of about ±0.6, not ±0.001. The paper's ± must therefore describe something other than run-to-run spread; what exactly is UNKNOWN.

## The lineage pattern (RSR → STHGCN → THINK)

- The RSR authors' code reproduces its paper on the paper's own metric: top-1 cumulative return, which is very noisy (seeds 1.16 to 2.75).
- The same predictions carry no ranking skill under a leak-free top-5 Sharpe, IC or correct NDCG.
- The STHGCN code evaluates the test set every epoch with no selection rule.

Together with Phase 1 route 1, this points to the published numbers in this line of work coming from noisy metrics and test-informed epoch selection, not from stable predictive signal. That last step is an inference; the authors' selection rule is UNKNOWN (the THINK paper does not state it).

## Remaining UNKNOWNs (cannot be settled without the authors)

- THINK source code (repository empty, p. 852 footnote 1)
- epoch or model selection rule
- the exact Sharpe definition: k, R_f, annualisation
- hyperparameters (none listed)
- the eq. 7 ⊙ operator (ill-formed as printed) and whether attention uses a softmax
- the STHGCN dataset and that paper's values

An email to the authors asking for the selection rule and the Sharpe definition would settle the largest of these.

## Evidence index

`docs/PHASE1_TRACKER.md`, `docs/phase1/paper_audit.md`, `docs/phase1_5/PHASE1_5_SUMMARY.md` (A-F), `docs/phase1_5a/REPORT_2017.md`, `docs/phase1_5b/REPORT_POST2017.md`, `docs/phase1_5c/REPORT_WF.md`, `docs/phase2/{protocol_matrix.md, rsr_paper_compare_table.md, W2_STHGCN.md}`.
