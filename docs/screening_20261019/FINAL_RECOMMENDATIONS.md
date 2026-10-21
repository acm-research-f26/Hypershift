# Final recommendations — HyperShift research screen (2026-10-21)

**Status: exploratory.** These recommendations rest on these documents (plus `X_PAIRWISE_STAR_REPORT.md`, Study X):
- `SCREENING_REPORT.md`: v3 screen, 31 arms, 171 fits;
- `DIAGNOSTIC_REPORT.md`: failure analysis, budget probe and platform control;
- `DEEP_ABLATIONS.md`: 150 deep fits, 10 seeds × 3 folds.

Every year used has been inspected before in this repository, so none of this is confirmatory.

## Bottom line

**No method is recommended for adoption.** All 10 directions were screened, plus the MASTER- and MDGNN-inspired arms (J was blocked). Nothing beats the common baseline after correction. The baseline itself cannot be separated from a no-graph model, a per-stock MLP, 16-day reversal or Ridge (pooled RankIC 0.006–0.011). The two arms that passed the screen gate are both NO EVIDENCE in the predeclared deep test. So is the exploratory hybrid.

## What the evidence supports

| # | Recommendation | Evidence |
|---|---|---|
| 1 | **Do not adopt any arm.** In particular, do not add second graph stages (F), masked pretraining (H), listwise loss mixing (C), or clique/star graph expansion (Study X) on the strength of this screen. Do not trade within-group pairs either. | Study X: clique Holm 0.890; star Holm 0.146, and star only reaches the no-graph level (star − none −0.0001). Pairs trading is NO EVIDENCE for every variant and does not beat random groups (`X_PAIRWISE_STAR_REPORT.md`). Deep test: two_residual Holm p 0.077, 2023 fold negative. pretrain Holm p 0.184. hybrid exploratory p 0.100. A random hypergraph gets +0.0028, the same size as pretrain and hybrid. |
| 2 | **Treat the bottleneck as signal, not optimisation.** More budget makes the fit better but not the ranking. Any next step should change *inputs or targets*, not add architecture. | Budget probe: training loss falls 3–4× further, but the selected epoch stays at 5.5–7. Predictions correlate 0.8–0.9 with the market move. Factor-neutral RankIC is ≤ 0.009 for every arm. |
| 3 | **Strengthen the parent before testing any child.** Tune the baseline's budget with the same tuning budget for every arm (project convention). Report no-graph, MLP and reversal as standing bars. | static gains +0.011 under the budget probe (2022 +0.023). Against that parent, two_residual goes from +0.009 to −0.0048. |
| 4 | **Screen with ≥ 5 seeds and a built-in null arm.** The 3-seed 'Promising' gate is too fragile to promote anything. Include `random_graph` (or a seed-shuffled graph) in every screen, so that "beats parent" is measured against a structure-free perturbation. | Re-pairing on Kaggle static flips 4 labels, including the null arm. hybrid's per-fold seed pattern reverses between v3 and deep, with 7 more seeds and a rerun parent (2023: 1/3 → 9/10 positive; 2022: 2/3 → 3/10). The parent rerun alone flips 2 of hybrid's 6 original seed deltas. |
| 5 | **Pin numerics for paired comparisons.** Run every arm of a comparison in the same compute session, or on the same machine. Log the CPU model (`lscpu`) in each Kaggle kernel. | Laptop reruns are bit-identical. Kaggle-vs-Kaggle reruns are identical for some kernels and not others (static seeds 7/19 differ by up to 0.006 RankIC). |
| 6 | **Direction A was never tested with working learned edges.** Fix the soft-assignment collapse before drawing any conclusion about learned hyperedges. Options: temperature or annealing, an entropy or balance penalty, prototype initialisation from correlation clusters. | Assignment entropy = log 16 (uniform) in every residual_soft and soft fit. |
| 7 | **If factor-residual targets (B) are retested, predeclare the factor-neutral RankIC as the primary metric.** Raw RankIC penalises B by construction. | B loses −0.017 to −0.026 raw. Neutral RankIC is mixed (better in 2022, worse in 2023). |
| 8 | **Combinations: do not run.** The brief allows them only after single effects are measured, and none is established. | — |
| 9 | **Keep 2024+ reserved.** Spend it only on a candidate that first passes a predeclared deep test on 2015–2023. | 2024+ was never scored. |

## Decisions needed from the user

- **J (Kronos foundation-model features).** Kronos was pretrained on data to June 2024, so a test on 2022/23 is contaminated. A clean test must evaluate from 2024-07 onward, which consumes the reserved period. Options:
  - (a) keep J blocked (the current state);
  - (b) authorise a 2024-07+ evaluation for J only, with the frozen v3 baseline;
  - (c) use a model whose pretraining ends before the evaluation window, if one is available and free to download.
- **Whether to continue at all on this panel.** With pooled RankIC around 0.01 for every baseline, and survivorship bias in the data, the expected value of more architecture screening is low. The most informative next step is recommendation 2: richer inputs, such as volume and point-in-time fundamentals where free sources exist, and neutral targets. That step should get a fresh predeclared plan.

## If one candidate must be retested

Retest only **F two_residual**. It is the only arm that clearly exceeds the random-graph null, by more than 2× (+0.0065 vs +0.0028; 22/30 seeds positive). This comparison is descriptive: the null arm was not session-matched to its parent. Preconditions, all predeclared:
1. Tune the budget for static and two_residual alike.
2. Use ≥ 3 folds, adding 2021, which requires a hashed-source edit and a new config identity.
3. Use 10 seeds, one compute session per pair, and random_graph in the family.
4. Apply the same STRONG / SEED-ROBUST ONLY / NO EVIDENCE rule.

The prior expectation is low: under a better-trained static, the budget probe already shows −0.0048.

## Not transferable

- These results come from 462 current S&P 500 stocks, 2015–2023, with yfinance OHLC-derived features, h5 targets and hidden 16.
- They do not transfer automatically to Rishi's `feat-rs` model or data, which remain unavailable. They also do not transfer to the THINK reproduction in Phase 1/2, which uses different data and protocol.
- Every method here is a simplified adaptation, not a reproduction: master ≠ MASTER, soft/rolling ≠ MDGNN, ListNet ≠ LapSum, pooled fusion ≠ TimeMixer, masked reconstruction ≠ CI-STHPAN, adaptive normalisation ≠ DoubleAdapt.
