# Publication synthesis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the HyperShift record into a publication-ready package:
- a claims register,
- an independent recomputation of the headline numbers from saved outputs,
- a manuscript,
- a figures and tables specification,
- a reviewer-risk assessment,
- a reproducibility checklist,
- a final recommendation.

There are **no new experiments**.

**Architecture:** One verification script, `scripts/pub_verify.py`, recomputes every headline number that local files allow and compares it with the value the documents cite. Its output is `docs/publication/CLAIMS_VERIFICATION.md`. The other deliverables are documents in `docs/publication/`. Literature is verified by web search.

**Tech Stack:** numpy and json for the recomputations; WebSearch for references.

**Spec:** the user prompt "HyperShift — Publication-Ready Scientific Synthesis and Final Claims Audit" (sections 1–8 and deliverables A–F). An abridged copy goes in `docs/publication/REQUEST.md`.

## Global Constraints

- **No new runs or data.** No new model fits, backtests or sweeps. No 2004–2012 prices. No post-freeze outcomes. Frozen files stay untouched (`test_freeze_hashes_unchanged`).
- **Every published number traces to a file:** either a saved output that the script recomputes, or a cited doc line labelled "doc-only (not recomputable locally)".
- **Wording rules:**
  - Never claim returns are unpredictable.
  - Never claim the original authors' procedures were identified.
  - Never describe residual style predictability as leakage.
  - Clark–West applies to MSPE; any RankIC use is an unvalidated adaptation.
  - The 43–58% coverage figure is a symbol-match count.
  - The style-only books' returns are attributed to style premia only as a hypothesis.
- **Commits:** explicit `git add` paths only, never push; HANDOFF stays uncommitted.

## Review Focus

1. **A cited number with no recomputable source.** Mark it "doc-only" and never present it as verified (`test_check_marks`).
2. **Tolerance.** Rounding in docs (±0.5 in the last printed digit) must not count as a mismatch, and a real mismatch must be flagged (`test_check_marks`).
3. **Missing result folders.** The script reports them as NOT FOUND and does not crash.
4. **Manuscript numbers that drift from the claims register.** Every manuscript number cites its register ID.
5. **References.** Every reference has been verified by search; anything unverified is left out.

---

### Task 1: Verification script

**Files:**
- Create: `scripts/pub_verify.py`, holding `check(claimed: float, recomputed: float|None, decimals: int) -> str`, which returns 'MATCH', 'MISMATCH' or 'NOT FOUND'.
- Create: `docs/publication/CLAIMS_VERIFICATION.md`.
- Test: `tests/test_fbaudit.py::test_check_marks`.

**Recomputation targets (claimed values are taken from the docs):**

| Group | Quantities | Source |
|---|---|---|
| THINK 2017, R5_g2 | leak-free Sharpe of THINK (`THINK_paperProtocol` folder, 25 seeds, claimed 0.089) and EH (0.383); hold-all 1.531 | from test arrays |
| Corrected R5_f | HH and EH leak-free means for `norm=train` (1.687, 1.436) and `norm=paper` (1.929, 1.146) | |
| Weight-decay planted signal | HH_hyper `high`/relative IC/oracle: D baseline 0.27 vs wd0 c1 0.63 (oracle 0.279) | `known_signal_high_relative`, `f_f2_c1_high_relative` |
| FB | lgbm residual − raw (+0.0140), recomputed from predictions | |
| Audit | corrected residual − raw (+0.0089); leak-term IC (+0.0178) | from `data/audit` |
| A11 | full − style (−0.0017, −0.0011), from the matched predictions | |
| A13 | style/residual book net Sharpe +1.11. Doc-only, since recomputing it needs the simulator; the script states that | |
| Synthetic | A12 learner probe. Doc-only, flagged as "re-runnable via scripts/closure_synth*.py" | |

### Task 2: Claims register (A)

`docs/publication/CLAIMS_REGISTER.md` has one row per claim with these fields:
- ID;
- claim (publication wording);
- class: reproduced / supported-in-config / suggestive / compromised / not adequately tested;
- source experiment and files;
- configuration;
- data and period;
- n;
- metric;
- effect with uncertainty;
- control;
- verification status (from Task 1);
- limitations;
- words to avoid.

### Task 3: Literature verification and manuscript (B)

- Verify every reference by search.
- Write `docs/publication/MANUSCRIPT.md` with sections 1–14 as specified. Numbers cite register IDs.

### Task 4: Figures/tables spec (C), reviewer risks (D), reproducibility checklist (E), recommendation (F)

Files, all in `docs/publication/`:
- `FIGURES_TABLES.md`: each item gives its source files and the command that regenerates it.
- `REVIEWER_RISKS.md`
- `REPRODUCIBILITY.md`
- `RECOMMENDATION.md`

Then update `CLAUDE.md`, update HANDOFF (left uncommitted), run the full suite, and commit.
