# Matched-control verification and research closure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the confounded style-only placebo comparison with a matched 2×2 design (features × target), check on synthetic data that the evaluation can tell style-only from incremental information, fix and quantify the v1-LF eligibility-mask discrepancy, reassess the economics with paired comparisons, and issue the final research decision.

**Architecture:**
- New module `src/hypershift/fbaudit/matched.py`: nested feature inputs, the encompassing test, and the synthetic generator.
- Scripts `scripts/closure_*.py` write `docs/audit_20261022/A11`–`A13`, then `FINAL_DECISION.md` and `PUBLICATION_OUTLINE.md`.
- FB freeze hashes and v1-LF artifacts are preserved. Any v1-LF code change is appended as a superseding hash row.

**Tech Stack:** numpy, scipy, lightgbm 4.7, existing fbaudit modules.

**Spec:** the user prompt "HyperShift — Final Matched-Control Verification and Research Closure", sections 1–6. An abridged copy goes in `docs/audit_20261022/REQUEST_CLOSURE.md`.

## Global Constraints

- **Out of bounds:**
  - no 2004–2012 prices;
  - no post-freeze outcomes;
  - no architecture or hyperparameter search.
- **Data:** dev 2019–2023 only (inspected, exploratory), with leak-free inputs (`corrected_fb`).
- **Fixed learner:** LightGBM `LGBM_PARAMS`, 50 rounds (`fit_fixed`), seeds {7, 19, 42}, the FB training window (`fb.train`) and evaluation dates (`fb.evaluate`).
- **Commits:** explicit `git add` paths only. Never push. HANDOFF stays uncommitted.

### Predeclared definitions and thresholds

- **M1. Nested inputs** (both feature sets end in the same sector one-hot):
  - `full` = cs_z(all 20 FB features, leak-free) + sector one-hot;
  - `style` = cs_z(beta, log_dvol_20, ret_60, vol_60) + sector one-hot.
  - Targets: `residual` (FB residual target, leak-free regressor) and `neutral` (FB evaluation target).
- **M2. Primary metric:** FB neutral RankIC, the question-A metric.
  - **Question B estimand:** the paired daily difference full − style within each target, averaged over the 3 seeds, pooled as the mean of the 5 fold means, with a stationary bootstrap (block 10, B 5000).
  - **"Increment detected":** 95% CI excludes 0 **and** ≥ 4/5 folds share the sign.
  - **"Increment bounded below δ = 0.005":** the 90% CI lies inside (−0.005, +0.005). Otherwise **unresolved**. Equivalence is never claimed from a non-significant result alone.
  - **Interaction:** (full − style)_residual − (full − style)_neutral, same CI rule.
- **M3. Encompassing (conditional predictive value).** Per date, residualise full's prediction ranks on style's prediction ranks (OLS with intercept), then take the Spearman of that residual against the target. It is pooled like M2, and reported next to M2 as a second view of question B.
- **M4. Synthetic validation.** 400 stocks, 10 sectors, 4 persistent styles (AR(1), φ = 0.98) and 6 other features, with 750 training and 500 evaluation dates. Two scenarios:
  - **S** (returns are nonlinear functions of styles only);
  - **I** (S plus a planted term in a non-style feature, b = 0.15 × the style signal SD).

  **Pass** requires both:
  - in S, the matched increment CI includes 0 (no false increment) while the style-only model's LIN and NL residual-metric scores are > 0, which reproduces the metric failure;
  - in I, the increment CI excludes 0 (> 0).

  Encompassing must behave the same way.
- **M5. Economics.** Using the A8 book (`weekly_buffer`, close1 and open1, 5 bp and 10 bp), compute the paired daily net-return difference of the model book minus each of: the RR book, the raw-neutralised book, and the style/neutral seed-7 book. Report the annualised mean difference, its 95% CI (stationary bootstrap, block 10), and per-year net Sharpe. "Advantage" means the CI excludes 0.

## Review Focus

1. Feature nesting broken: the style columns must be the identical arrays inside both input sets (`test_nested_inputs`).
2. Encompassing on a date where the style prediction is constant: no crash, residual = the full ranks (`test_encompass_constant_base`).
3. The synthetic generator must be deterministic per seed (`test_synth_deterministic`).
4. v1-LF neutral and y_ok must be recomputed with the post-LF eligibility (`test_lf_neutral_uses_post_lf_mask`).
5. The superseding hash row must match the final file, using the LF-normalised hash (`test_v1lf_hashes_match_addendum`, existing).

---

### Task 1: Matched 2×2 on dev (A11)

**Files:**
- Create: `src/hypershift/fbaudit/matched.py` (`nested_inputs(fb, signals, which) -> [S,N,F]`, `encompass(full, base, y, mask) -> [S]`) and `scripts/closure_matched.py`
- Test: `tests/test_fbaudit.py` (`test_nested_inputs`, `test_encompass_constant_base`)

Steps:
- [ ] TDD the two tests.
- [ ] Run the script: 4 cells × 3 seeds × 5 folds = 60 fixed-round fits, saved to `results/FBAUDIT_20261022/matched/`.
- [ ] Write A11, which also records the verified original configuration. A11 contains:
  - the absolute cells;
  - the within-target increments and the interaction (M2) with per-year and per-seed values;
  - encompassing (M3);
  - the cells − RR;
  - LIN and NL scores, descriptive.
- [ ] Commit.

### Task 2: Synthetic validation of the metrics (A12)

**Files:** `matched.py` (`synth_panel(scenario, seed) -> dict`), `scripts/closure_synth.py`. Test: `test_synth_deterministic`.

- [ ] TDD, run, and write A12 with the M4 verdict.
- [ ] Commit.

### Task 3: v1-LF eligibility-mask fix and coverage wording (A13 part 1)

**Files:**
- Modify: `src/hypershift/fbaudit/prospective_lf.py` (`make_leakfree` recomputes `y_ok`-masked `neutral` when eligibility changes), plus an appended superseding hash row in `PROSPECTIVE_ADDENDUM.md`.
- Test: `test_lf_neutral_uses_post_lf_mask`.

Steps:
- [ ] TDD the fix.
- [ ] Quantify the effect on dev: emulate the LF dropping with the audit's raw download, then report the max |Δ neutral RankIC| of the corrected model (old vs new mask) and the number of dropped stock-days.
- [ ] Classify the change.
- [ ] Append an erratum on "at most" vs "lower bound" to `BC_REVIEW.md`.
- [ ] Commit.

### Task 4: Paired economics (A13 part 2)

**Files:** `scripts/closure_econ.py`.

- [ ] Run M5 (uses `book.simulate_book`; 5 arms × 2 executions × 5 folds).
- [ ] Write A13 part 2.
- [ ] Commit.

### Task 5: Final decision and publication outline

**Files:**
- Create: `docs/audit_20261022/FINAL_DECISION.md` (deliverables 1–8) and `PUBLICATION_OUTLINE.md` (deliverable 9).
- Append correction pointers to `BC_REVIEW.md` and `AUDIT_REPORT.md`; update `CLAUDE.md`; update HANDOFF, uncommitted.

Steps:
- [ ] Write both documents.
- [ ] Run the full suite.
- [ ] Commit.
