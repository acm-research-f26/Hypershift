# F. Final publication recommendation

## Is the thesis supported?

**Proposed thesis:** "Financial machine-learning benchmarks can systematically misattribute performance to sophisticated architectures when experimental protocols, optimisation settings, market exposures and statistical power are not adequately controlled."

**Verdict: partly supported. Narrow it.**

- **What the evidence supports.** In *one* reproduction case and *one* simple-model benchmark, each named mechanism did produce, or could have produced, a misleading conclusion, and each is shown with a controlled diagnostic. The mechanisms:
  - protocol [T5];
  - optimisation [T2];
  - leakage [L1, L2];
  - exposure [S1–S3];
  - power [P1, M2].
- **What it does not support.** "Systematically" across the field. We examined one model lineage (RSR → STHGCN/STHAN-SR → THINK) and our own benchmark.

**Recommended thesis:** *"In a reproduction of a hypergraph stock ranker and a matched simple-model benchmark, apparent skill was attributable to identifiable mechanisms (test-informed selection, an optimisation failure, leakage through vendor-adjusted price levels, exposure to known characteristics, and comparisons lacking statistical and learner power). We quantify each and give inexpensive diagnostics. After controlling for them, the available evidence cannot identify stock-specific information beyond characteristics and simple reversal."*

## Contributions, assessed

| Contribution | Novelty | Strength | Keep? |
|---|---|---|---|
| Vendor-adjustment leakage via future prices in level features, with an ex-ante/future decomposition and its size | **Strongest original finding.** Practitioners know adjusted prices are "look-ahead" in a vague sense; the quantified predictive channel and the decomposition are rarely shown | Algebraic mechanism; recomputed (R) | **Lead contribution** |
| Residualised metrics do not isolate information beyond characteristics for nonlinear learners; matched characteristics-only learner as control | Moderate: known in principle, rarely demonstrated in a benchmark | Synthetic plus real placebo | Keep, framed as demonstration plus control |
| Learner power as a precondition for feature-ablation claims | Moderate; a useful warning | One synthetic design: needs strengthening | Keep **only after** a small simulation extension (below), or present as a preliminary observation |
| Weight-decay collapse in stacked Poincaré layers | Low–moderate (AdamW is known) | Clean planted-signal evidence | Keep, brief |
| THINK non-reproduction under leak-free protocols | Useful to the field; reviewers will probe the implementation gap | Many seeds and nulls, one test year plus out-of-sample years | Keep, with careful wording |
| Economics (buffering, costs) | Not novel (Novy-Marx & Velikov, 2016) | Descriptive | Short section only |

## Venue and format

**A combined reproducibility and methodology paper.** Suitable targets, ordered by fit:
1. a reproducibility or ML-evaluation track (e.g. a datasets-and-benchmarks or reproducibility venue, or ML-for-finance workshops), emphasising diagnostics;
2. a quantitative-finance methods journal, emphasising the leakage decomposition and metric control, with the THINK reproduction as a case study;
3. alternatively, a short focused note on the vendor-adjustment leakage alone. This is the fastest route and has the clearest novelty.

**Recommended route:** a short leakage note first (it is self-contained), then the combined paper.

## Ready for external review?

**Not yet.** Required before submission:
1. **Learner-power simulation extension** (methods only, no market data): more replicates and seeds, a signal-to-noise grid, two or three learner configurations, and reported power curves. Without it, contribution C3 must be cut to a one-paragraph caveat. This is the only additional computation we recommend, and it uses no market data.
2. **Plotting scripts** for Figs 1–7 from saved outputs (`FIGURES_TABLES.md`).
3. **Licence check and release plan** for data (`REPRODUCIBILITY.md`). Do not publish raw Yahoo or Alpaca prices.
4. **Verify the secondary-only reference** (Lo & MacKinlay, 1990), or drop it.
5. **Independent fresh-context review** of the manuscript against `CLAIMS_REGISTER.md` and `CLAIMS_VERIFICATION.md`, ideally by someone outside the project. The project's own audit found and fixed several of its errors; an outside check is still needed.
6. **Optional, cheap:** a style-factor attribution regression of the characteristics-only books on saved returns (supports or removes E2's interpretation).
7. **Optional:** a query to the THINK authors (draft exists). Report any answer.

## Exclude from the paper

- Any claim of stock-specific predictive skill.
- Any claim that returns are unpredictable, or that features carry no information.
- Any statement of what the THINK authors actually did.
- Investability or attribution claims for any portfolio.
- Results from 2024+ (Kronos zero-shot) except as a one-line footnote. That study was separate and is not needed for the thesis.
- Exact numbers that are doc-only and not regenerable from local files (Phase 2 protocol-matrix cells): present them as tabulated in the project report, clearly sourced.
