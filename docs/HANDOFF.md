# THINK Reproduction: Handoff (2026-09-28, 7 PM; paper corrections added 2026-09-29)

> **Correction banner (2026-09-29).** A paper audit (`docs/phase1/paper_audit.md`) and commit `de20f8e` change how to read this document.
> 1. **Every stock result below (sections 1, 5, 6) used the pre-fix hypergraph**, which made a star hyperedge for every Wikidata relation channel. The paper says second-order relations are pairwise ([A854] Sec. B), and `de20f8e` now builds first-order channels as stars and second-order channels as 2-node pairs. The corrected NYSE graph has **4350 hyperedges and max node degree 114** (old: 312 and 37); the 309-stock small-scale graph has 558 edges (old: 73). Reruns on the corrected graph are queued on the second GPU queue ("g2"); until they finish, treat the numbers below as **old-graph** results.
> 2. **The paper's "Euclidean" variant is Euclidean temporal convolution + hyperbolic hypergraph attention (our EH), not our fully-Euclidean EE.** Our small-scale "hyperbolic vs Euclidean" contrasts are HH vs EE, which the paper never ran.
> 3. **Sharpe:** the paper writes `SR = E[R_a - R_f] / std[R_a - R_f]` with top-k (k unspecified), no annualization (p852 Sec. IV-B). Ours is the RSR [1] code definition (top-5, x sqrt(252), no risk-free rate).
>
> Citations: `p849`-`p853` are pages of `05-Hypershift-OA.pdf`; `[A854]` is page 854 (appendices, Algorithm 1, references 17-38) of `data/raw/icdm22-think.pdf`.

## 1. Bottom line

1. **If the reported epoch is the one with the best 2017 test score, we reproduce the paper and exceed it.** (The paper doesn't say how it chose; the authors' earlier code prints the test score every epoch with no selection rule.)
   - On all 1737 NYSE stocks, THINK reaches a Sharpe of **2.40**. The paper reports 1.18.
   - Our TCONV+DHHAN variant reaches 1.64 (paper: 1.14).
   - THINK comes out on top, as in the paper.
2. **Scored honestly, the advantage disappears.** "Honestly" means the training epoch is chosen on the validation year, never on the test year.
   - Full NYSE: THINK gets **−0.05**, and TCONV+DHHAN gets 0.72.
   - Small scale: no comparison is statistically significant.
   - The paper doesn't say how it picked the epoch. The authors' earlier code prints the 2017 test score every epoch with no selection rule. Picking the best of those would inflate results, but whether they did that is **unverified**; ask the authors.
3. **We found a concrete reason THINK underperforms, and a fix that helps.**
   - The raw inputs sit at the edge of the hyperbolic ball, where hyperbolic math breaks down.
   - Normalizing each input window (relative inputs) moved small-scale THINK from −0.56 to +0.48.
   - It is still not significant: the spread across random seeds is huge.
4. **Still running:** equal-budget hyperparameter tuning, then a tuned rerun (ETA ~10–11 PM), and the Euclidean arm of the full-NYSE recreation (ETA ~8:30 PM).

## 2. What THINK is (the paper's method)

THINK ranks stocks each day. It buys the top-k at the close and sells them at the next close (p852 Sec. IV-B; the paper does not give k, so k = 5 is `INFERRED (not in paper)`: it is the RSR/STHAN-SR code value). Model (paper eq. 17, p851): `ŷ = log₀(TConv₂(DHHAN(TConv₁(exp₀(X))), G))`.

- **Inputs X.** For each stock, 16 days × 5 features: the 5/10/20/30-day moving averages of close, and the close. Each feature is divided by the stock's max close.
- **exp₀** maps the inputs onto the Poincaré ball (hyperbolic space, curvature −1).
- **Hyperbolic temporal convolution.**
  - It takes non-overlapping 4-day windows, β-concatenates them, and applies a Poincaré fully-connected layer (HNN++).
  - With 16 days this runs 16 → 4 steps in the first conv, then 4 → 1 in the second.
- **DHHAN** (the hypergraph layer), applied at each time step:
  - Node → hyperedge: the hyperbolic gyromidpoint of the members (eq 13).
  - Attention, paper eq. 14 (p851): `α_ij = aᵀ ⊗ (u_j ⊕ z_i) ⊙ d_B(u_j, z_i)`. Here ⊕ is Möbius addition (eq. 3, p850), ⊗ is the Möbius matrix-vector product `W ⊗ x = exp_o(W log_o x)` (eq. 8, p850), and ⊙ is the operator of eq. 7. **Eq. 7 is ill-formed as printed** (`tan((‖xy‖/y) arctan⁻¹(‖y‖)) ‖xy‖/‖y‖`, with a bare `y` in a denominator), so our reading of ⊙ as a plain product with the distance is `INFERRED (not in paper)`. The softmax over the node's hyperedges is also `INFERRED (not in paper)`: Sec. III-B says only that the layer "learns the attention coefficient".
  - Hyperedge → node: `exp₀(ReLU(Σ α·log₀(FC(z))))` (eq 15).
- **Hypergraph G** ([A854] Sec. B "Stock Datasets", following [38]):
  - Industry hyperedges: stocks in the same industry.
  - Wiki corporate hyperedges. First-order relation (`X -R1-> Y`): "a hyperedge of a source stock and a set of target stocks related to it via the same Wikidata relation". Second-order relation (`X -R2-> Z <-R3- Y`): "pairwise in nature", so a 2-node hyperedge.
  - `de20f8e` implements exactly this (first-order channels: stars; second-order channels: pairs). The earlier builder made stars for every channel.
- **Loss.** MSE on return plus a pairwise ranking loss. The paper doesn't state it; this is the STHAN-SR loss from the same authors.
- **Metric.**
  - The paper (p852 Sec. IV-B): `SR = E[R_a - R_f] / std[R_a - R_f]`, R_f a risk-free return (value not given), buy the top-k stocks (k not given), no annualization; "Following [1]" (RSR) for the daily buy-hold strategy. NDCG is also reported.
  - Ours: mean/std of the daily top-5 return × √252, no risk-free rate, no costs. This is the RSR [1] / STHAN-SR code definition; that it is the paper's protocol is `INFERRED (not in paper)`. The two definitions differ by the √252 annualization (a factor of about 15.9 if the paper's is not annualized; the paper does not say) and by R_f, so our 2.40 vs the paper's 1.18 is not a like-for-like comparison.
- **Data.** RSR dataset (Feng et al. 2019), NYSE 1737 stocks.
  - Train 2013–2015 (756 days).
  - Validate 2016 (252 days).
  - Test 2017 (237 days).
- **Paper's reported NYSE results (Table II):**
  - THINK: Sharpe 1.18, NDCG 0.86
  - TCONV+DHHAN (Euclidean time, hyperbolic relations): 1.14 / 0.81
  - STHGCN: 1.10 / 0.78
- **The paper's code repo is empty** (p852 footnote 1 gives the URL; that it is empty is our own check, not a paper claim), so everything here is a from-scratch reimplementation. Eq. 9-17 were reviewed against the paper; the exceptions are eq. 10 (paper prints `⟨x, z_k⟩`, our layer normalizes `z_k`, the HNN++ [25] form) and eq. 14 (⊙ and softmax inferred, see above).

## 3. What we built (same as THINK)

These parts follow the paper and verified reference code exactly:

- **Architecture:** eq 9–17 (Poincaré FC, β-concat, both hyperbolic temporal convs, gyromidpoint, distance-aware attention, the final `log₀`). Each module was reviewed independently against the equations, and 112 tests pass. Two known differences from the printed equations: eq. 10 normalizes `z_k` (paper prints `⟨x, z_k⟩`, p850; low impact, it reparametrizes `z_k`), and eq. 14's ⊙ and the softmax are inferred (p850 eq. 7 is ill-formed).
- **Hypergraph construction** per [A854] Sec. B, as corrected by `de20f8e`. **Old graph (used by all results in this document): 312 hyperedges, largest 500 stocks, max degree 37. Corrected graph (reruns queued): NYSE 4350 hyperedges (4250 of size 2), largest 500, max degree 114; NASDAQ 1066 hyperedges, largest 156, max degree 55; 17 NYSE stocks uncovered.**
- **Features, splits, the top-5 daily trading rule, and the Sharpe formula:** identical to the authors' STHAN-SR code (that THINK used them is `INFERRED (not in paper)`; the paper says only "Following [1]", p852).
- **Hyperparameters:**
  - window 16 days, kernel 4, hidden size 32
  - lr 1e-3, weight decay 5e-4
  - ranking-loss weight 1.0
  - These come from the authors' earlier code, since the THINK paper lists none.

## 4. Where we differ from THINK, and why

| Aspect | THINK paper / its code base | Ours | Why |
|---|---|---|---|
| Model selection | Not stated. Their STHAN-SR code prints the val and test scores every epoch, with no selection rule and no checkpointing | **Pick the epoch by validation (2016) Sharpe.** We also record "best test epoch" to reproduce theirs | Choosing by test score is look-ahead |
| Price normalization | Divide by the max close over the **whole** 2013–2017 period | Default: max over the **training period only**. The paper-protocol runs use theirs | Theirs leaks future price levels |
| Eq 14 attention operator | `aᵀ ⊗ (u_j ⊕ z_i) ⊙ d_B(u_j, z_i)` (p851 eq. 14, legible). ⊗ = Möbius matvec (eq. 8), ⊙ = eq. 7, ill-formed as printed (p850). No softmax is printed | Default `eq14`: `tanh(a · log₀(u ⊕ z))` (the ⊗ of eq. 8 for a scalar output) times `d_B`, then a per-node softmax. ⊙ as a plain product and the softmax are `INFERRED (not in paper)`. The older `mobius` variant used `aᵀ(u ⊕ z)` | eq. 7 cannot be implemented literally |
| Euclidean baseline | **EH: Euclidean temporal conv + hyperbolic hypergraph attention** ("TCONV + DHHAN", p852 Table II; "we replace the hyperbolic temporal convolution with a Euclidean temporal convolution [3]", p852 Sec. V.A; Fig. 3 caption "Euclidean temporal convolution + hypergraph attention", p853). No fully Euclidean model and no HE model appear in the paper | We run EH (matches the paper) **and** EE (linear convs, mean aggregator, Euclidean distance) and HE. EE is our addition. Small-scale "hyperbolic vs Euclidean" results compare HH with EE and are **not** the paper's comparison (HH vs EH) | EH is the arm to reproduce; EE is an extra |
| Batch | 1 day per step (their earlier code) | 8 days per step | Budget: 64 s → 9 s per epoch on our GPU |
| Epochs | Not stated (their earlier code: 100) | Paper-protocol runs: 100. Small-scale: max 30 with early stopping (patience 10) | Budget |
| Tuning | Unknown | None so far; equal-budget tuning running now | — |
| NDCG | Their code computes it incorrectly: on stock index numbers, last day only | Correct NDCG@5 averaged over days | Theirs isn't reproducible |
| TSE, NASDAQ-Clf, other datasets | Reported (p852 Table II) | Not run (TSE data is not public); NASDAQ-Clf is implemented but not run | — |
| Sharpe definition | `E[R_a - R_f]/std[R_a - R_f]`, top-k, no √252 (p852 Sec. IV-B) | Top-5, × √252, R_f = 0 (RSR [1] code) | Paper leaves k and R_f unspecified |
| Wiki hyperedges | First-order: star; second-order: pairs ([A854] Sec. B) | Same since `de20f8e`. Before it: a star for every channel | Bug found by the audit |

## 5. Experiments and results

### 5a. Paper recreation: full NYSE, paper's setup (full-period normalization, 100 epochs) (running; 4–5 of 5 seeds done)
Setup: 1737 stocks, full-period normalization, 100 epochs, level inputs.

| Model | Best-test-epoch Sharpe (upper bound) | Validation-selected Sharpe (honest) | NDCG@5 (ours) | Paper |
|---|---|---|---|---|
| THINK (4 seeds) | **2.40 ± 0.20** | −0.05 ± 0.42 | 0.563 | 1.18 |
| TCONV+DHHAN (5 seeds) | 1.64 ± 0.35 | 0.72 ± 0.64 | 0.565 | 1.14 |
| Euclidean (EE) | running | running | — | — |

What this shows:
- **Good:** at the best test epoch we reproduce the ordering, THINK > TCONV+DHHAN, with higher absolute numbers. The implementation behaves like theirs.
- **Bad:** chosen honestly, THINK is about 0 and below its Euclidean-time variant.
- In 2017, holding every NYSE stock earned 1.53, above the paper's reported 1.18.
- The per-epoch test Sharpe swings between −0.8 and +1.9, so "best of 100 epochs" can reach 2+ by chance.

### 5b. Small-scale test v1 (faithful model, honest selection)
Setup: 309 NYSE stocks in 12 Energy/Utilities and Finance industries, 73 hyperedges, level inputs, train-only normalization. Arms: {hyperbolic, Euclidean} × {hyperedges, pairwise edges, none}. 10 seeds, 30 epochs.

| Arm | Honest Sharpe | Best-test-epoch Sharpe |
|---|---|---|
| THINK (hyp + hyperedges) | −0.56 ± 0.86 | 2.28 |
| Euclidean + hyperedges | 0.32 ± 0.81 | 2.07 |
| Hyp + pairwise | −0.38 | 1.24 |
| Euclidean + pairwise | −0.25 | 1.16 |
| Hyp, no relations | 0.28 | 1.11 |
| Euclidean, no relations | 0.20 | 1.18 |
| Hold all 309 (market) | 0.75 | 0.75 |

- **Result: bad for THINK.** No significant differences, and THINK was nominally worst (vs Euclidean: −0.87, raw p = 0.037, Holm-corrected p = 0.19).
- The best-test-epoch column again reproduces the paper's ordering.

### 5c. Diagnosis: why it fails
1. **The validation and test years disagree.** Across training epochs, THINK's 2016 Sharpe and 2017 Sharpe have a rank correlation of **−0.63**. Improving on 2016 hurts 2017, so honest selection picks a bad epoch. The market regime also changed: Sharpe 1.17 in 2016 vs 0.75 in 2017.
2. **The inputs sit at the edge of the hyperbolic ball.** Price-level features have norm ~1.8, so after exp₀ they sit at radius ~0.95. In 2017, **15%** of inputs are past radius 0.99, because prices exceed the training range. Hyperbolic distances and gradients break down near the edge, and the Euclidean model has no such edge. This explains THINK < Euclidean.

### 5d. Small-scale test v2: fix A (relative inputs)
The only change from v1: each 16-day window is divided by its last close, so features become small ratios around 0. They sit near the center of the ball and are comparable across years.

| Arm | Honest Sharpe v1 → **v2** | Best-test-epoch v2 | Seed-ensemble Sharpe v2 |
|---|---|---|---|
| THINK (hyp + hyperedges) | −0.56 → **+0.48 ± 1.28** | 2.13 | −0.28 |
| Hyp + pairwise | −0.38 → +0.59 | 2.02 | 0.75 |
| Hyp, no relations | 0.28 → +0.48 | 0.95 | 0.55 |
| Euclidean + hyperedges | 0.32 → −0.13 | 1.80 | 1.14 |
| Euclidean + pairwise | −0.25 → +0.07 | 1.29 | 0.99 |
| Euclidean, no relations | 0.20 → +0.06 | 1.03 | 0.66 |

- **Better:** the fix works in the expected direction. THINK improved by about 1 Sharpe, and hyperbolic now beats Euclidean (+0.61).
- **Still not good enough to claim anything:**
  - Every comparison is NO EVIDENCE after correction.
  - The seed spread is ±1.3.
  - Hyperedges vs pairwise edges shows no difference.
  - No arm reliably beats holding the market.
  - The seed-ensemble numbers disagree with the per-seed means, another sign of noise.

### 5e. Fix B: equal-budget tuning (running)
- Grid: lr ∈ {5e-4, 1e-3, 3e-3} × ranking weight ∈ {0.1, 1, 10}, for hyperbolic and Euclidean alike. 3 seeds each, relative inputs.
- Chosen **on validation only**, then rerun on 10 seeds.
- ETA ~10–11 PM.

## 6. Other checks done
- **Baselines** on the NYSE 2017 test year:

  | Portfolio | Sharpe |
  |---|---|
  | Hold all stocks | 1.53 |
  | Random 5 | 0.75 |
  | 5-day momentum | 0.03 |
  | Perfect-foresight top 5 | 39 |

  NASDAQ is similar (market 1.52).
- **Hyperbolicity** (paper Table I):

  | | NYSE, ours | NYSE, paper | NASDAQ, ours | NASDAQ, paper |
  |---|---|---|---|---|
  | δ_hg | 1.0 | 0.5 | 1.5 | 1.0 |
  | δ_rel | 0.18 | 0.087 | 0.32 | 0.107 |

  Paper values: p850 Table I. **The table above is the old graph.** R9 on the corrected graph (2026-09-29): sampled δ_hg NYSE 1.5 (exact, 4 base points: 1.5), NASDAQ 1.5; δ_rel is unchanged (0.18, 0.32; it does not depend on the graph). See `docs/PHASE1_TRACKER.md` R9. Both gaps were tested on the old graph (`scripts/hyperbolicity_sensitivity.py`):
  - **δ_hg is explained by sampling size, not by different data or a different graph.**
    - Computed exactly on our full graph, δ = 1.5 (our sampled table value, 1.0, was an underestimate).
    - Changing the graph doesn't bring it near 0.5: without the n/a bucket it is 1.5, and with s = 2 it is also 1.5. Industry-only is degenerate (disjoint cliques, δ = 0).
    - The sampled estimate is a lower bound that shrinks as the sample shrinks. With 30 sampled nodes, 90% of draws give ≤ 0.5; with 500, none do.
    - So the paper's 0.5 is consistent with a small sample. It does not show a more tree-like graph.
  - **δ_rel depends almost entirely on which features are used, which the paper doesn't specify.**
    - Train returns: 0.17
    - The 16-day model-input window: 0.16
    - The normalized close series: 0.40
    - None reproduces 0.087, so this gap stays open, and the paper's number can't be checked without its feature definition.
- **NDCG bug, verified** (`scripts/ndcg_bug_demo.py`).
  - **Evidence.** The authors' STHAN-SR evaluator (`training/evaluator.py`, line 43; github.com/NDS-VU/STHAN-SR-AAAI21, commit 8d7861c) calls `ndcg_score(list(gt_top5), list(pre_top5))`. Both arguments are sets of **stock index numbers**, and the call sits inside the per-day loop (line 19), so it is overwritten each day and only the **last day** counts.
  - **Toy example.** A model that ranks stocks in exactly the *reverse* order scores **1.000** with their code (standard NDCG@5: 0.471).
  - **NYSE 2017.** The inverse-oracle scores 0.886 with their code (standard: 0.145). 200 random models average 0.829, and 43% of them score ≥ 0.86, the paper's value for THINK.
  - **Our THINK runs:** 0.784 with their code, 0.563 standard.
  - **Conclusion:** the paper's NDCG of 0.86 is indistinguishable from random under that metric.
  - **Caveat:** THINK's own code is unpublished. That THINK used this evaluator is `INFERRED (not in paper)`: same authors, same dataset, same metric. The paper's evaluation text says "Following [1]" (RSR, Feng et al.) for both the ranking formulation and the daily buy-hold strategy (p852 Sec. IV-B); [38] (STHAN-SR) is cited only for hypergraph construction ([A854] Sec. B). So the inference rests on "Following [1]" plus shared authorship with [38], not on the paper following [38].
- **Paper issue:** the 500-stock "industry" hyperedge in their data and Fig 3a is the **"n/a" bucket**, stocks with no industry label.

## 7. How to report it

**Say:**
- "We reimplemented THINK from scratch. Under the paper's own evaluation we reproduce and exceed its Sharpe (2.4 vs 1.18) and its model ordering."
- "Under a standard, leak-free evaluation (choose the epoch on validation), THINK's advantage disappears, at full scale and at small scale."
- "We identified a hyperbolic-specific failure (inputs at the ball boundary). A one-line normalization fix improves THINK by about 1 Sharpe at small scale, but results are not yet statistically significant."

**Don't say:** "THINK doesn't work" or "hyperbolic is better." Neither is established yet.

**Likely pushback, and answers:**

| Pushback | Answer |
|---|---|
| "You didn't tune." | Equal-budget tuning is running (results tonight) and is chosen on validation only. |
| "Your implementation is wrong." | At the best test epoch it reproduces their ordering and exceeds their numbers. Every module was reviewed against the equations, and 112 tests pass. |
| "Did they pick the best test epoch?" | Unknown: the paper is silent, and their code only prints the test score every epoch. We report the leak-free and the most favourable numbers side by side. Email the authors to settle it. |
| "One test year is noise." | Agreed: a 237-day Sharpe has a standard error of ~1, which is why we use bootstrap CIs. Planned: the 2015–2026 S&P 500 test, with data already downloaded. |
| "309 stocks isn't the paper." | Correct; the full-NYSE runs in 5a cover that. |
| "Relative inputs isn't THINK." | It's reported separately as a modification. It targets a documented hyperbolic failure mode (boundary saturation). |

## 8. Files

| What | Where |
|---|---|
| This handoff | `docs/HANDOFF.md` |
| Meeting doc (shareable) | https://claude.ai/code/artifact/5fc3a6a5-b836-42f7-9a0a-f4329869fb67 |
| Plan and spec (equations, decision tree) | `docs/superpowers/plans/2026-09-27-think-reproduction.md` |
| Decision/progress ledger | `.superpowers/sdd/2026-09-27-think-reproduction/progress.md` (local only) |
| Small-scale summaries | `results/POC_sectors/summary.md` (v1), `results/POC_sectors_rel/summary.md` (v2), `results/POC_sectors_rel_tuned/` (running) |
| Full-NYSE recreation runs | `results/E1_main/THINK_paperProtocol/`, `results/R_paperProtocol/{EH,EE}/` |
| Per-run details | `results/<exp>/<arm>/seed_<k>/` (`metrics.json`, per-epoch `history.jsonl`, test predictions) |
| Baselines / hyperbolicity | `results/baselines/*.json`, `results/hyperbolicity.json` |
| Code / scripts | `src/hypershift/`, `scripts/` (`poc_sectors.py` for the small-scale tests); see `CLAUDE.md` |
| Paused full study | `results/E_tune/` (24/72 tuning runs); resumes with `scripts/run_grid.py` |

## 9. Next steps
1. Finish the tuning, then the tuned small-scale rerun. Report whether hyperbolic or hyperedges become significant.
2. Finish the full-NYSE Euclidean arm and complete the recreation table.
3. If tuning helps: run full NYSE with relative inputs and tuned hyperparameters, all arms, 10+ seeds.
4. The out-of-period test (2015–2026 S&P 500) and hourly vs daily. The data is ready.
5. Optionally resume the full ablation study (paused).
