# THINK Reproduction: Handoff (2026-09-28, 7 PM)

## 1. Bottom line

1. **Scored the way the paper's code scores, we reproduce the paper, and exceed it.**
   - On all 1737 NYSE stocks, THINK reaches a Sharpe of **2.40**. The paper reports 1.18.
   - Our TCONV+DHHAN variant reaches 1.64 (paper: 1.14).
   - THINK comes out on top, as in the paper.
2. **Scored honestly, the advantage disappears.** "Honestly" means the training epoch is chosen on the validation year, never on the test year.
   - Full NYSE: THINK gets **−0.05**, and TCONV+DHHAN gets 0.72.
   - Small scale: no comparison is statistically significant.
   - The paper's protocol picks the best of 100 epochs by *test* score, which inflates results.
3. **We found a concrete reason THINK underperforms, and a fix that helps.**
   - The raw inputs sit at the edge of the hyperbolic ball, where hyperbolic math breaks down.
   - Normalizing each input window (relative inputs) moved small-scale THINK from −0.56 to +0.48.
   - It is still not significant: the spread across random seeds is huge.
4. **Still running:** equal-budget hyperparameter tuning, then a tuned rerun (ETA ~10–11 PM), and the Euclidean arm of the full-NYSE recreation (ETA ~8:30 PM).

## 2. What THINK is (the paper's method)

THINK ranks stocks each day. It buys the top 5 at the close and sells them at the next close. Model (paper eq. 17): `ŷ = log₀(TConv₂(DHHAN(TConv₁(exp₀(X))), G))`.

- **Inputs X.** For each stock, 16 days × 5 features: the 5/10/20/30-day moving averages of close, and the close. Each feature is divided by the stock's max close.
- **exp₀** maps the inputs onto the Poincaré ball (hyperbolic space, curvature −1).
- **Hyperbolic temporal convolution.**
  - It takes non-overlapping 4-day windows, β-concatenates them, and applies a Poincaré fully-connected layer (HNN++).
  - With 16 days this runs 16 → 4 steps in the first conv, then 4 → 1 in the second.
- **DHHAN** (the hypergraph layer), applied at each time step:
  - Node → hyperedge: the hyperbolic gyromidpoint of the members (eq 13).
  - Attention: α = softmax over the node's hyperedges of aᵀ(u ⊕ z)·d(u, z), a distance-aware attention (eq 14).
  - Hyperedge → node: `exp₀(ReLU(Σ α·log₀(FC(z))))` (eq 15).
- **Hypergraph G** (paper appendix B):
  - One hyperedge per industry.
  - Wikidata company-relation hyperedges: a company plus all companies linked to it by the same relation.
- **Loss.** MSE on return plus a pairwise ranking loss. The paper doesn't state it; this is the STHAN-SR loss from the same authors.
- **Metric.** Sharpe = mean/std of the daily top-5 return × √252, with no risk-free rate and no costs, per the authors' code. NDCG is also reported.
- **Data.** RSR dataset (Feng et al. 2019), NYSE 1737 stocks.
  - Train 2013–2015 (756 days).
  - Validate 2016 (252 days).
  - Test 2017 (237 days).
- **Paper's reported NYSE results (Table II):**
  - THINK: Sharpe 1.18, NDCG 0.86
  - TCONV+DHHAN (Euclidean time, hyperbolic relations): 1.14 / 0.81
  - STHGCN: 1.10 / 0.78
- **The paper's code repo is empty**, so everything here is a from-scratch reimplementation, checked equation by equation.

## 3. What we built (same as THINK)

These parts follow the paper and verified reference code exactly:

- **Architecture:** eq 9–17 (Poincaré FC, β-concat, both hyperbolic temporal convs, gyromidpoint, distance-aware attention, the final `log₀`). Each module was reviewed independently against the equations, and 112 tests pass.
- **Hypergraph construction** per appendix B. On NYSE: 312 hyperedges, largest 500 stocks, max stocks-per-node degree 37.
- **Features, splits, the top-5 daily trading rule, and the Sharpe formula:** identical to the authors' STHAN-SR code.
- **Hyperparameters:**
  - window 16 days, kernel 4, hidden size 32
  - lr 1e-3, weight decay 5e-4
  - ranking-loss weight 1.0
  - These come from the authors' earlier code, since the THINK paper lists none.

## 4. Where we differ from THINK, and why

| Aspect | THINK paper / its code base | Ours | Why |
|---|---|---|---|
| Model selection | Evaluates test every epoch, reports the best (implied by the authors' STHAN-SR code) | **Pick the epoch by validation (2016) Sharpe.** We also record "best test epoch" to reproduce theirs | Choosing by test score is look-ahead |
| Price normalization | Divide by the max close over the **whole** 2013–2017 period | Default: max over the **training period only**. The paper-protocol runs use theirs | Theirs leaks future price levels |
| Eq 14 attention operator | Symbol lost in the PDF | Möbius addition ⊕ × distance (literal reading) | Ambiguous source |
| Euclidean baseline | Not fully specified | Same structure with linear convs, a mean aggregator and Euclidean distance | Closest twin |
| Batch | 1 day per step (their earlier code) | 8 days per step | Budget: 64 s → 9 s per epoch on our GPU |
| Epochs | Not stated (their earlier code: 100) | Paper-protocol runs: 100. Small-scale: max 30 with early stopping (patience 10) | Budget |
| Tuning | Unknown | None so far; equal-budget tuning running now | — |
| NDCG | Their code computes it incorrectly: on stock index numbers, last day only | Correct NDCG@5 averaged over days | Theirs isn't reproducible |
| TSE, NASDAQ-Clf, other datasets | Reported | Not run (TSE data is not public); NASDAQ-Clf is implemented but not run | — |

## 5. Experiments and results

### 5a. Paper recreation: full NYSE, paper's protocol (running; 4–5 of 5 seeds done)
Setup: 1737 stocks, full-period normalization, 100 epochs, level inputs.

| Model | Best-test-epoch Sharpe (paper's way) | Validation-selected Sharpe (honest) | NDCG@5 (ours) | Paper |
|---|---|---|---|---|
| THINK (4 seeds) | **2.40 ± 0.20** | −0.05 ± 0.42 | 0.563 | 1.18 |
| TCONV+DHHAN (5 seeds) | 1.64 ± 0.35 | 0.72 ± 0.64 | 0.565 | 1.14 |
| Euclidean (EE) | running | running | — | — |

What this shows:
- **Good:** under the paper's protocol we reproduce the ordering, THINK > TCONV+DHHAN, with higher absolute numbers. The implementation behaves like theirs.
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
- The paper-protocol column again reproduces the paper's ordering.

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

  δ_hg matches within tolerance. δ_rel is about 2× the paper's. Both gaps were then tested (`scripts/hyperbolicity_sensitivity.py`):
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
| "Your implementation is wrong." | Under the paper's protocol it reproduces their numbers and ordering. Every module was reviewed against the equations, and 112 tests pass. |
| "Everyone picks the best epoch." | It uses the test year to choose the model. We report both numbers side by side. |
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
