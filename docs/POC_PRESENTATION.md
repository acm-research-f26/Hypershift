# THINK: Small-Scale Reproduction (309 NYSE stocks)

*Team update, 2026-09-28. Scope: the small-scale test only. The full-scale study is paused.*

## 1. One-slide summary

- We reimplemented **THINK** (Temporal Hypergraph Hyperbolic Network, ICDM 2022) from scratch. The authors' code repo is empty.
- On a 309-stock NYSE subset, **if the reported epoch is the one with the best 2017 (test) score, we reproduce the paper's story. The authors' earlier code allows that choice: it prints the test score every epoch and has no selection rule**:
  - THINK comes out best, at Sharpe **2.28**.
  - Hyperedges beat pairwise edges, which beat no relations.
- **Scored leak-free (epoch picked on the validation year), the advantage disappears.** THINK gets −0.56, no comparison is statistically significant, and nothing beats simply holding the 309 stocks (0.75).
- *Correction:* THINK's attention (eq. 14) used a close but not exact formula. It is now fixed, and the two affected arms are being rerun (§4d). All other findings are unaffected.
- We found a hyperbolic-specific cause: **inputs sit at the edge of the hyperbolic ball.** A one-line fix (relative inputs) lifts THINK to +0.48 and puts hyperbolic ahead of Euclidean. It is still not significant.

## 2. What THINK does (the paper's method)

1. **Data:** RSR NYSE daily prices, 2013–2017.
   - Per stock-day features: the 5/10/20/30-day moving averages of close, and the close, each divided by the stock's max price.
   - The model sees 16 days × 5 features and predicts the next day's return.
2. **Hypergraph** (paper appendix B):
   - one hyperedge per industry;
   - plus Wikidata company-relation hyperedges (a company and all companies linked to it by the same relation).
3. **Model** (paper eq. 17): `ŷ = log₀(TConv₂(DHHAN(TConv₁(exp₀(X))), G))`
   - `exp₀` maps the features onto the Poincaré ball (hyperbolic space).
   - A hyperbolic temporal convolution (β-concatenation + Poincaré fully-connected layer) runs 16 days → 4 steps → 1.
   - DHHAN (distance-aware hyperbolic hypergraph attention) mixes information between stocks that share a hyperedge.
4. **Trading rule and metric:**
   - Each test day, buy the top 5 predicted stocks at the close and sell at the next close.
   - Sharpe = mean / std of daily returns × √252, with no risk-free rate and no costs.
5. **Paper's NYSE numbers:** THINK Sharpe **1.18**, TCONV+DHHAN 1.14, STHGCN 1.10.

## 3. What we did in the small-scale test

| Step | THINK paper | Our small-scale test | Same? |
|---|---|---|---|
| Dataset | RSR NYSE, 1737 stocks | RSR NYSE, **309 stocks** (12 industries: Energy/Utilities + Finance) | **Different** (subset) |
| Train / val / test | 2013–15 / 2016 / 2017 | same | Same |
| Features | MA5, MA10, MA20, MA30, close; 16-day window | same | Same |
| Price normalization | ÷ max over **all** years (includes the test year) | ÷ max over **training years only** | **Different** (theirs peeks at the future) |
| Hyperedges | industry + Wikidata relations | same construction, restricted to the 309 stocks (73 hyperedges) | Same method, smaller graph |
| Hyperbolic temporal conv | eq. 9–12 | eq. 9–12, kernel 4 | Same |
| DHHAN | eq. 13–15 | eq. 13–15 | Same (see the next row) |
| Attention formula (eq. 14) | `α = aᵀ ⊗ (u ⊕ z) ⊙ d(u, z)` | **Results in §4 used** `aᵀ·(u ⊕ z)·d(u, z)`: a plain dot product instead of the Möbius product ⊗ (`tanh(aᵀ·log₀(·))`). **Now corrected to the exact formula; THINK arms rerunning (§4d)** | **Was slightly different, now fixed** |
| Loss | not stated | MSE + pairwise ranking loss (their earlier STHAN-SR code) | Same as their code |
| Hyperparameters | not stated | window 16, hidden 32, lr 1e-3, weight decay 5e-4, ranking weight 1 (their earlier code) | Same as their code |
| Batch / epochs | 1 day per step; ~100 epochs (earlier code) | **8 days** per step; **max 30 epochs**, early-stopped (~17 in practice) | **Different** (compute budget) |
| Hyperparameter tuning | unknown | **none** | **Different** |
| Epoch selection | **not stated in the paper**. Their earlier STHAN-SR code prints both the 2016 and 2017 scores every epoch, with no selection rule | **both** reported: best test epoch, and epoch chosen on validation | Both shown |
| Trading rule and Sharpe | top-5 daily, mean/std × √252 | same | Same |
| NDCG | computed incorrectly in their code (see §7) | standard NDCG@5 | **Different** (fixed) |
| Seeds | 25 runs ("mean of 25 runs") | 10 per arm | Fewer |

**What the differences mean, in plain terms:**

- **Price normalization.** Stock prices vary wildly (one stock trades at \$5, another at \$500), so each stock's prices are divided by that stock's **highest price**, putting every stock on a 0–1 scale.
  - The dataset the paper uses takes that highest price **from all five years, including 2017**, the year we test on. That quietly tells the model something about the future: a stock whose 2016 prices sit far below 1 must rise later.
  - We divide by the highest price **from 2013–2015 only**, so nothing from the test period leaks in.
- **Batch and epochs.** A *batch* is how many trading days the model learns from before each update: their earlier code used 1 day, we used 8, which is about 7× faster. An *epoch* is one full pass over the training years: they used about 100, we capped at 30 and stopped early once the validation score hadn't improved for 10 epochs. In short, we trained less, to fit the compute budget.
- **Hyperparameter tuning.** Learning rate, ranking-loss weight and the like were not tuned. We used the values from the authors' earlier code. Tuning, equal for all arms and chosen on 2016 only, is built but not yet run.
- **NDCG.** A second score the paper reports (0.86 for THINK). It measures whether the model's top-5 picks are the stocks that *actually* went up most. We **did** compute it: the standard NDCG@5 in the §4 tables, about 0.55. The authors' code computes it incorrectly (§7): it scores stock ID numbers instead of returns, and uses only the last test day. With their code, a model that ranks stocks in exactly the reverse order still gets 1.000. So their 0.86 can't be compared with ours or trusted, and we report the standard version.
- **Seeds.** A neural net starts from random numbers, so training the same model twice gives different results. Each "seed" is one full training run with different starting randomness. The paper averages 25 runs; we ran 10 per model. That's fewer, but it is enough for our significance tests (the smallest possible p-value with 10 seeds is 0.002).

**Ablation arms (each is 10 seeds):**
- THINK: hyperbolic + hyperedges.
- Same model with pairwise edges: each hyperedge split into all its stock pairs.
- Same model with no relations.
- All three again with Euclidean layers instead of hyperbolic ones.
- Comparison baselines: hold all 309 stocks (market), random 5 stocks, 5-day momentum.

## 4. Results

### 4a. Faithful THINK (v1): the protocol decides the result

![Fig 1](figures/fig1_protocol_gap.png)

| Model | Best test epoch (upper bound) | Epoch picked on validation (leak-free) |
|---|---|---|
| **THINK (hyp + hyperedges)** | **2.28 ± 0.58** | −0.56 ± 0.86 |
| Euclidean + hyperedges | 2.07 ± 0.47 | 0.32 ± 0.81 |
| Hyperbolic + pairwise | 1.24 ± 0.91 | −0.38 ± 0.75 |
| Euclidean + pairwise | 1.16 ± 0.57 | −0.25 ± 0.50 |
| Hyperbolic, no relations | 1.11 ± 0.26 | 0.28 ± 0.35 |
| Euclidean, no relations | 1.18 ± 0.60 | 0.20 ± 0.58 |
| *Baseline:* hold all 309 stocks | 0.75 | 0.75 |
| *Baseline:* random 5 stocks | 0.34 | 0.34 |
| *Baseline:* 5-day momentum | −0.28 | −0.28 |

- **Left column (best test epoch, the most favourable choice the authors' setup allows):** the paper's full ordering reproduces. THINK is best, hyperedges > pairwise > none, and hyperbolic ≥ Euclidean.
- **Right column (leak-free):** the ordering vanishes. Statistical tests (paired Wilcoxon over seeds, Holm correction, bootstrap CI) find **no significant difference** anywhere. THINK vs Euclidean is −0.87 Sharpe, with p = 0.037 before correction and 0.19 after.

### 4b. Why it breaks: two measured causes

![Fig 3](figures/fig3_val_vs_test.png)

1. **Validation and test years disagree.** Over training, THINK's 2016 Sharpe and 2017 Sharpe have a rank correlation of **−0.63**: what helps 2016 hurts 2017.
   - So an epoch picked on 2016 is a bad one for 2017, while picking on 2017 itself (possible in the authors' setup) looks great.
   - The market also changed: Sharpe 1.17 in 2016 vs 0.75 in 2017.
2. **The inputs sit at the edge of the hyperbolic ball.** Price-level features have norm ~1.8, so after `exp₀` they land at radius **~0.95** (the edge is 1.0).
   - In 2017, **15%** of inputs are past 0.99, because prices exceed the training range.
   - Hyperbolic distances and gradients break down near the edge. The Euclidean model has no edge, which is why THINK trails its Euclidean twin.

### 4c. Fix (v2): relative inputs

The one change: each 16-day window is divided by its last close, so features become small ratios near 0. That keeps them at the center of the ball and comparable across years. Everything else is identical to v1.

![Fig 2](figures/fig2_fix_effect.png)

| Model | v1 leak-free | **v2 leak-free (fix)** | v2 best test epoch |
|---|---|---|---|
| **THINK** | −0.56 | **+0.48 ± 1.28** | 2.13 |
| Hyperbolic + pairwise | −0.38 | +0.59 | 2.02 |
| Hyperbolic, no relations | 0.28 | +0.48 | 0.95 |
| Euclidean + hyperedges | 0.32 | −0.13 | 1.80 |
| Euclidean + pairwise | −0.25 | +0.07 | 1.29 |
| Euclidean, no relations | 0.20 | +0.06 | 1.03 |

- **Better:**
  - THINK improves by about 1 Sharpe.
  - Hyperbolic now beats Euclidean (THINK vs Euclidean + hyperedges: +0.61).
  - At its best test epoch, THINK is again the top model.
- **Not yet enough to claim anything:**
  - The seed-to-seed spread is huge (±1.3).
  - After correction, every comparison is "no evidence".
  - Hyperedges vs pairwise edges shows no difference.
  - No model reliably beats holding the market.

### 4d. Correction: exact eq. 14 attention (rerun in progress)

- **What happened.** The results above used a close but not exact form of the attention formula: `aᵀ·(u ⊕ z)` instead of the paper's `aᵀ ⊗ (u ⊕ z) = tanh(aᵀ·log₀(u ⊕ z))`. Before this, only the text-extracted PDF was available, and its ⊗ symbol was lost; the exact formula has since been confirmed from the PDF. The code is now corrected.
- **How much it can matter.** Attention decides how a stock *weights* its hyperedges. **80% of the 309 stocks are in exactly one hyperedge**, and their weight is 100% whatever the formula. So the change only affects the ~20% of stocks in several hyperedges (mainly big banks and oil majors).
- **What is affected:** only the two arms that use this attention, *THINK (hyp + hyperedges)* and *hyp + pairwise*. Both are being rerun with 10 seeds each, faithful (v1) and relative-input (v2).
- **What is unaffected:**
  - the Euclidean and no-relation arms, and the baselines;
  - the epoch-choice gap, which appears in the Euclidean arms too (2.07 vs 0.32);
  - the ball-boundary diagnosis, which is measured on the inputs;
  - the 2016-vs-2017 regime change, and the NDCG and n/a-bucket findings.
- Results: `results/POC_sectors_eq14/summary.md` (v1) and `results/POC_sectors_rel_eq14/summary.md` (v2).

## 5. Scientific assessment

- **Reproduction:** our implementation behaves like THINK. At the best test epoch it gives the paper's ordering, and the same happens on the full NYSE, where THINK scores 2.40 vs the paper's 1.18.
- **Claim under test:** "hyperbolic space and hyperedges improve stock ranking." At small scale, with leak-free evaluation, **not supported yet.** The faithful model is nominally worst, and the fixed model is nominally better but not significant.
- **What the gap suggests:** THINK's reported advantage **could** come from how the epoch was chosen. This is an open question, not a finding. The paper doesn't say how it chose, and the authors' earlier code prints the test score every epoch with no rule. Our leak-free score (−0.56 small scale, ≈0 full NYSE) and best-test score (2.28 / 2.40) bracket the paper's 1.18. The paper also reports THINK as 1.18 **± 0.004** over 25 runs, while our seeds vary by ±0.2 to ±0.9. That spread is unusually tight and worth asking the authors about. The per-epoch test Sharpe swings between about −0.8 and +1.9, so the best of many epochs looks strong even when the model isn't.
- **Limits of this test:**
  - 309 stocks, not 1737
  - one test year (a 237-day Sharpe has a standard error of about 1)
  - no tuning
  - 30 epochs
  - 10 seeds

## 6. Likely questions

| Question | Answer |
|---|---|
| Is the implementation right? | Every equation was checked in independent reviews, and 112 automated tests pass. At the best test epoch it reproduces the paper's ordering and exceeds its numbers. |
| You didn't tune it. | True. Equal-budget tuning (same grid for hyperbolic and Euclidean, chosen on validation only) is built and partly run; results next. |
| Did the authors pick the best test epoch? | Unknown: the paper doesn't say. Their earlier code prints the test score every epoch with no selection rule. We show the leak-free and the most favourable scores so the range is visible. Next step: email the authors. |
| Why only 309 stocks? | The full study takes about a week on our GPU. The full-NYSE check (THINK 2.40 at its best test epoch) is in the handoff. |
| Is relative input still THINK? | It is reported as a modification, separately. It targets a known hyperbolic failure mode (points at the ball boundary). |

## 7. Other findings about the paper

- **The NDCG number (0.86) is uninformative.**
  - The authors' earlier evaluator scores NDCG on stock *index numbers*, and only on the last test day.
  - With that code, a model that ranks stocks in exactly the reverse order scores **1.000** on a toy example and 0.886 on NYSE 2017.
  - 43% of random models score ≥ 0.86.
  - Reproduce with `scripts/ndcg_bug_demo.py`. THINK's own code is unpublished, so we infer that it used this evaluator.
- **The largest "industry" hyperedge in their data (500 stocks) is actually the "n/a" group**, stocks with no industry label. Excluded here.
- **Hyperbolicity:** the paper's δ = 0.5 for NYSE is consistent with estimating it from a small sample. Computed exactly, we get 1.5.

## 8. Files

| File | What it is |
|---|---|
| `docs/POC_PRESENTATION.md` | This document |
| `docs/figures/fig1_protocol_gap.png`, `fig2_fix_effect.png`, `fig3_val_vs_test.png` | The figures above (made by `scripts/poc_figures.py`) |
| `results/POC_sectors/summary.md` | v1 (faithful) full statistics table: Wilcoxon, Holm, bootstrap CIs |
| `results/POC_sectors_rel/summary.md` | v2 (relative inputs) full statistics, incl. seed-ensemble and paper-protocol columns |
| `results/POC_sectors*/<arm>/seed_<k>/` | Per-run raw output: `metrics.json`, per-epoch `history.jsonl`, test predictions |
| `scripts/poc_sectors.py` | Runs and summarizes the small-scale test (`run`, `tune`, `summarize`) |
| `src/hypershift/` | The THINK implementation (models, geometry, data, training, metrics, stats) |
| `docs/HANDOFF.md` | Full handoff, including the full-NYSE recreation |
| `docs/superpowers/plans/2026-09-27-think-reproduction.md` | Full plan, with every THINK equation (Part 0) |

**Reproduce the small-scale test** from the repo root, with `.venv` installed:
```bash
.venv/Scripts/python.exe scripts/poc_sectors.py run --seeds 0-9                                      # v1 faithful
.venv/Scripts/python.exe scripts/poc_sectors.py run --variant rel --input-mode relative --seeds 0-9  # v2 fix
.venv/Scripts/python.exe scripts/poc_sectors.py summarize [--variant rel]
.venv/Scripts/python.exe scripts/poc_figures.py
```
