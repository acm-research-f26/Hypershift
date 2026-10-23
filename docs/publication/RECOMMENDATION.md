# Publication recommendation and submission readiness (revision 2, 2026-10-22)

Revision 1 is preserved in git (commit `1561e2a`). This revision follows the mathematical, prior-art and leakage audits (`MATH_ADJUSTMENT.md`, `PRIOR_ART.md`, `LEAKAGE_AUDIT.md`).

## Option A vs Option B

| Criterion | **A: short note on adjusted-price look-ahead** | **B: combined reproducibility + methods paper** |
|---|---|---|
| Novelty | **Modest.** Dependence on future distributions is known in practitioner sources. New: an explicit future-price derivation, a quantified association, and a benchmark effect | Moderate. The combination of mechanisms, quantified in one case with diagnostics, plus a careful reproduction |
| Evidential strength | Mechanism: MATH (certain). Size: one vendor, one universe, one fetch date, inspected years. The strongest number (L_path +0.094) is partly mechanical by construction, which supports the mechanism but makes the number unsurprising. Benchmark effect about 0.005 | Mixed: T1–T3 recomputed; T4–T8 report-level; learner-power claims rest on one synthetic design |
| Coherence | High | High if organised by mechanism; risk of sprawl |
| Main reviewer risks | "Known issue"; "obviously mechanical"; "single dataset" | Implementation gap to THINK; single test year; synthetic learner power; exploratory status |
| Remaining preparation | Low: figures exist; needs a second-vendor or fetch-date check to be convincing | Medium: learner-power simulation (or cut C3), economics re-execution for figures, licence clean-up, outside review |

**Recommendation: Option B.**
- **Option A alone is thin.** Once prior practitioner awareness is credited, the remaining novelty is a quantification in one dataset. Its most striking number is close to mechanical.
- **The mechanisms reinforce each other in B.** They are the paper's real lesson: optimisation failure, protocol, adjusted-level look-ahead, characteristic exposure and power. Section 6 becomes one well-derived mechanism among five.
- **If a short output is wanted quickly,** publish the adjustment material as a technical note or preprint section, not as a standalone peer-reviewed claim of novelty.

**Venue fit for B:** an ML reproducibility or evaluation track, or an ML-in-finance methods venue.

## Readiness checklist

| # | Item | Status |
|---|---|---|
| 1 | Correct the adjusted-return mathematics | **Done** (draft 2 §6; `MATH_ADJUSTMENT.md`) |
| 2 | Reinterpret the trailing-yield decomposition; add a construction-based decomposition | **Done** (L2b; `LEAK_MECHANISM.md`) |
| 3 | Audit every "leak-free" usage | **Done** (`LEAKAGE_AUDIT.md`; terminology replaced) |
| 4 | Prior-art check | **Done** (limited search; `PRIOR_ART.md`) |
| 5 | Claims register with evidential status | **Done** (revision 2) |
| 6 | Figures and tables from saved outputs | **Done** for Figs 1–5 / Tabs 1–5 (`scripts/pubfig.py`). Economics and synthetic figures need re-execution (listed in REPRODUCE) |
| 7 | Environment record | **Done** (`REPRODUCE.md`, `requirements-frozen.txt`) |
| 8 | Clean-clone reproduction | **Not done:** procedure written but not executed |
| 9 | Release audit | **Done.** Two blockers found: the THINK PDF and the AGPL-derived `test_gt.npy` arrays in git |
| 10 | References | **Done:** all verified, including Lo & MacKinlay (1990) |
| 11 | Technical appendix | **Done** (`APPENDIX.md`) |
| 12 | Learner-power contribution (C3) | **Open.** Either run a methods-only simulation (more replicates, signal-to-noise grid, 2–3 learner configurations, no market data) or cut C3 to a one-paragraph caveat. Draft 2 currently presents it as preliminary |
| 13 | Independent fresh-context review | **Open.** All review so far has been by the authoring process. Two of our own errors were found late (a confounded placebo comparison; coverage bounds). In this revision we also caught a mathematical misstatement and over-interpreted terminology |
| 14 | Unexplained mismatch (EH planted IC, earlier report) | Documented; not used in the manuscript; trace or retract in that report |

## Claims to exclude

- Stock-specific predictive skill.
- Unpredictability of returns.
- What the THINK authors did.
- A "future-only" component identified by regression residual.
- Investability or factor attribution of any book.
- Any "leak-free" label.
- Equivalence from non-significant differences.

## Judgment

**READY AFTER SPECIFIED CORRECTIONS.**

The manuscript (draft 2) is scientifically consistent with the verified evidence and suitable to share with an independent reviewer now. It should not go to a journal until items **8, 9, 12 and 13** are resolved:
- **8:** clean-clone run;
- **9:** remove the THINK PDF from the release, and resolve AGPL compliance for the RSR-derived arrays;
- **12:** the learner-power simulation, or the cut;
- **13:** independent review.

None of these requires new financial experiments or access to unused periods.
