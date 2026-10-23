# Manuscript changes: draft 1 → draft 2 (2026-10-22)

Draft 1 is preserved in git (commit `6c09a6c`, `docs/publication/MANUSCRIPT.md`). Every substantive change is listed here.

| # | Location | Draft 1 | Draft 2 | Reason |
|---|---|---|---|---|
| 1 | §6, abstract | "Returns computed from adjusted prices are correct total returns, because F cancels in ratios" | Derivation: factors for ex-dates **after** the window cancel; factors **inside** the window remain, and they are what make the ratio a total return (reinvestment at the theoretical ex-price P(e−1) − D) | Mathematical error (review item 1); `MATH_ADJUSTMENT.md` §2 |
| 2 | §6, abstract, register L1 | "The future-only part of the adjustment factor (residual on trailing yield) predicts … +0.031" | The residual is a mixture of ex-ante information not spanned by the proxy and ex-post information. It is reported only as an association. The future-price channel is shown by a construction-based split: L_path +0.094 [+0.086, +0.101]; L_div +0.010 (CI includes 0); L +0.019 | Over-interpretation (review item 2); new descriptive decomposition on data already on disk (`LEAK_MECHANISM.md`) |
| 3 | Throughout | "leak-free" (THINK evaluation, refits, v1-LF) | "validation-selected" (selection), with normalisation stated per run; "adjustment-corrected" (features); per-experiment status table in `LEAKAGE_AUDIT.md` | Review item 3. The original reproduction used full-series normalisation; all universes are survivor-conditioned |
| 4 | §1 C1, §2 | C1 framed as a new leakage channel | C1 framed as a **quantified demonstration** of a mechanism partly known among practitioners. §2 adds practitioner sources, Ljungqvist et al. (2009) and Brown et al. (1992) | Prior-art check (review item 4); `PRIOR_ART.md` |
| 5 | §3 | Data description | Adds adjustment conventions per dataset (RSR unknown; Alpaca split-only; yfinance backward dividend+split; unadjusted download for the analysis) | Review item 3 |
| 6 | §5.2–5.3 | "leak-free Sharpe" | "validation-selected Sharpe", stating "full-series normalisation" for the original setup | Review item 3 |
| 7 | §8.2 | "negative (−0.002 to −0.003)" | "zero or negative (−0.003 to 0.000; two of three replicates CI below zero)" | Accuracy (A12 values) |
| 8 | §12 | Diagnostics list | Adds the **adjustment-term-as-score** test | Follows from §6 |
| 9 | References | Lo & MacKinlay flagged as secondary-only | Verified: *RFS* 3(2), 175–205. Added Ljungqvist, Malloy & Marston (2009) and one practitioner source, clearly labelled | Reference check |
| 10 | Header | — | Points to `APPENDIX.md` (definitions) and `MANUSCRIPT_CHANGES.md` | Review item 6 |

**Unchanged on purpose:** all verified numbers from draft 1; the THINK wording ("can be reached by", never "was obtained by"); the economic caveats; the decision framing (INCONCLUSIVE on incremental information, STOP as a resource decision).
