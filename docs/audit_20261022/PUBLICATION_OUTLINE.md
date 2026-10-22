# Publication outline — HyperShift methodological and negative-results findings

**Working title:** *Why graph and machine-learning stock rankers look skilful: leakage, style exposure and underpowered comparisons in a reproduction of THINK and simple price-based benchmarks.*

**Type:** a methods and negative-results paper (e.g. a reproducibility track, or a finance/ML methods venue). All empirical claims are exploratory unless prospectively confirmed. State this in the abstract.

## 1. Contributions (each backed by a repository artefact)

1. **Reproduction of THINK (ICDM 2022).** The paper's ranking advantage is not reproduced under validation selection or leak-free evaluation. Its 2017 Sharpe of about 2 is matched by random high-beta baskets.
   - Artefacts: `docs/PHASE1_TRACKER.md`, `docs/phase2/REPORT_CLOSURE.md`, `docs/phase1_5a/REPORT_2017.md`.
2. **A weight-decay collapse in stacked hyperbolic layers.** Planted-signal recovery is 26% of oracle with coupled weight decay, against 63–84% with the fix.
   - Artefact: `docs/phase1_5/F_learnability.md`.
3. **Two data leaks, each with a mechanism and a size:**
   - full-series normalisation;
   - dividend back-adjustment in vendor "adjusted" prices. The factor depends on future prices, and the future-only part predicts 5-day returns at IC +0.031, more than twice any model (A5).
4. **Metric pitfalls:**
   - "factor-neutral" RankIC (beta and sector removed) rewards style exposure: a style-only learner scores about twice the full model;
   - linear or quintile style-residual RankIC is not style-free (synthetic proof, A12).
5. **Underpowered model comparisons.** A standard gradient-boosting configuration cannot exploit a planted incremental feature of standalone IC 0.046. Feature-ablation nulls built on it are therefore uninformative (A12). Nested-model noise penalties make matched increments negative under the null (Clark & West 2007).
6. **The economics are style.** Buffered, beta-hedged neutral books are positive net of linear costs. Style-only books do as well or better than the model; turnover control, not model choice, is the economic lever (A8, A13).
7. **Process.** Predeclared plans, freeze manifests with code hashes, synthetic validation of evaluation procedures, an append-only audit trail, and errata. Two of our own errors are documented as lessons: a confounded placebo comparison, and a committed SyntaxError.

## 2. Structure

1. **Introduction:** the promise of graph and ML stock ranking; why claims fail to replicate.
2. **Data and setup:**
   - RSR NYSE/NASDAQ;
   - the S&P 500 current-constituent panel, with its survivorship stated;
   - the evaluation protocol: walk-forward folds, purging, stationary bootstrap, Holm.
3. **Reproducing THINK:** results, the collapse, and the leaks (normalisation; test-epoch selection).
4. **Simple benchmarks (Study FB):** reversal, Ridge, LightGBM, MLP; targets and horizons.
5. **Leakage through vendor adjustment:** the mechanism, a decomposition by trailing yield, and the corrected results.
6. **What does "neutral" skill measure?**
   - style exposure (neutral-target and style-only learners);
   - the failure of residualised metrics (synthetic S/I);
   - the matched 2×2.
7. **Statistical power and learner power:**
   - per-contrast MDE;
   - prospective power (about 7%);
   - the synthetic learner-power probe;
   - the nested-model penalty.
8. **Economics:** buffered neutral books, costs, null distribution, style-only books.
9. **Recommendations for practitioners:**
   - a truncation-invariance test for vendor data;
   - co-report beta/sector-neutral and matched style-only controls;
   - validate every evaluation procedure on synthetic S/I data before real use;
   - per-contrast power before freezing;
   - prospective tests sized for realistic effects.
10. **Limitations:** survivorship, current sectors, one market, development years already inspected, linear costs.
11. **Conclusion.**

## 3. Figures and tables to produce (from saved outputs; no new experiments)

1. THINK vs EH vs no-graph under the paper protocol and leak-free protocols (Phase 2 tables).
2. Leak-term decomposition (A5): trailing yield vs future-only IC.
3. Target ladder and style-only learner (A4, A4b, A11).
4. Synthetic S/I panel: residualised metrics vs matched increment vs learner power (A12).
5. Power: MDE by contrast and prospective days (A1, A9).
6. Cost curves of buffered vs unbuffered books (A8, A13).

## 4. Before submission

- **Run R1** (a validated nested-model test). Otherwise the paper must state that "beyond style" is untestable with the procedures shown.
- **Fresh-context review** of every number against the A-files.
- **Release** code, freeze manifests and saved predictions.
- **Report v1/v1-LF honestly,** if scored by then.
