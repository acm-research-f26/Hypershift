# THINK: Small-Scale Reproduction (309 NYSE stocks)

*Team update, 2026-09-28. Scope: the small-scale test only. The full-scale study is paused.*

## 1. One-slide summary

- We reimplemented **THINK** (Temporal Hypergraph Hyperbolic Network, ICDM 2022) from scratch. The authors' code repo is empty.
- On a 309-stock NYSE subset, **scored the way the paper's code scores (best test epoch), we reproduce the paper's story**:
  - THINK comes out best, at Sharpe **2.28**.
  - Hyperedges beat pairwise edges, which beat no relations.
- **Scored leak-free (epoch picked on the validation year), the advantage disappears.** THINK gets −0.56, no comparison is statistically significant, and nothing beats simply holding the 309 stocks (0.75).
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
| Attention formula (eq. 14) | `aᵀ(u ? z)·d(u, z)`; the symbol is lost in the PDF | Möbius addition ⊕ (literal reading) | Best guess |
| Loss | not stated | MSE + pairwise ranking loss (their earlier STHAN-SR code) | Same as their code |
| Hyperparameters | not stated | window 16, hidden 32, lr 1e-3, weight decay 5e-4, ranking weight 1 (their earlier code) | Same as their code |
| Batch / epochs | 1 day per step; ~100 epochs (earlier code) | **8 days** per step; **max 30 epochs**, early-stopped (~17 in practice) | **Different** (compute budget) |
| Hyperparameter tuning | unknown | **none** | **Different** |
| Epoch selection | best **test** epoch (inferred from their earlier code) | **both** reported: best test epoch, and epoch chosen on validation | Both shown |
| Trading rule and Sharpe | top-5 daily, mean/std × √252 | same | Same |
| NDCG | computed incorrectly in their code (see §7) | standard NDCG@5 | **Different** (fixed) |
| Seeds | 25 | 10 per arm | Fewer |

**Ablation arms (each is 10 seeds):**
- THINK: hyperbolic + hyperedges.
- Same model with pairwise edges: each hyperedge split into all its stock pairs.
- Same model with no relations.
- All three again with Euclidean layers instead of hyperbolic ones.
- Comparison baselines: hold all 309 stocks (market), random 5 stocks, 5-day momentum.

## 4. Results

### 4a. Faithful THINK (v1): the protocol decides the result

![Fig 1](figures/fig1_protocol_gap.png)

| Model | Best test epoch (paper's way) | Epoch picked on validation (leak-free) |
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

- **Left column (the paper's protocol):** the paper's full ordering reproduces. THINK is best, hyperedges > pairwise > none, and hyperbolic ≥ Euclidean.
- **Right column (leak-free):** the ordering vanishes. Statistical tests (paired Wilcoxon over seeds, Holm correction, bootstrap CI) find **no significant difference** anywhere. THINK vs Euclidean is −0.87 Sharpe, with p = 0.037 before correction and 0.19 after.

### 4b. Why it breaks: two measured causes

![Fig 3](figures/fig3_val_vs_test.png)

1. **Validation and test years disagree.** Over training, THINK's 2016 Sharpe and 2017 Sharpe have a rank correlation of **−0.63**: what helps 2016 hurts 2017.
   - So an epoch picked on 2016 is a bad one for 2017, while picking on 2017 itself (the paper's way) looks great.
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
  - Under the paper's protocol, THINK is again the top model.
- **Not yet enough to claim anything:**
  - The seed-to-seed spread is huge (±1.3).
  - After correction, every comparison is "no evidence".
  - Hyperedges vs pairwise edges shows no difference.
  - No model reliably beats holding the market.

## 5. Scientific assessment

- **Reproduction:** our implementation behaves like THINK. Under the paper's protocol it gives the paper's ordering, and the same happens on the full NYSE, where THINK scores 2.40 vs the paper's 1.18.
- **Claim under test:** "hyperbolic space and hyperedges improve stock ranking." At small scale, with leak-free evaluation, **not supported yet.** The faithful model is nominally worst, and the fixed model is nominally better but not significant.
- **What the gap suggests:** most of THINK's reported advantage comes from **choosing the epoch on the test year**. The per-epoch test Sharpe swings between about −0.8 and +1.9, so the best of many epochs looks strong even when the model isn't.
- **Limits of this test:**
  - 309 stocks, not 1737
  - one test year (a 237-day Sharpe has a standard error of about 1)
  - no tuning
  - 30 epochs
  - 10 seeds

## 6. Likely questions

| Question | Answer |
|---|---|
| Is the implementation right? | Every equation was checked in independent reviews, and 112 automated tests pass. Under the paper's protocol it reproduces the paper's ordering and exceeds its numbers. |
| You didn't tune it. | True. Equal-budget tuning (same grid for hyperbolic and Euclidean, chosen on validation only) is built and partly run; results next. |
| Isn't picking the best epoch standard? | It picks the model using the test answers. We show both numbers so the difference is visible. |
| Why only 309 stocks? | The full study takes about a week on our GPU. The full-NYSE paper-protocol check (THINK 2.40) is in the handoff. |
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
