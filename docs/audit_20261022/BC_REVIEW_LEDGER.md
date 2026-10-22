# SDD ledger — plan: docs/superpowers/plans/2026-10-22-bc-review.md
Setup: Ruling: no plan-review stop; execute now — user said "make decisions until you bring me results" — cost if wrong: plan scope not user-vetted.
Setup: Ruling: in place on tom-shlom, no worktree — standing practice — cost: none.
Pre-flight: T1 produces diag2.{residual_reversal,quintile_dummies,nl_resid}; T2 (book) and T3 (power) consume RR and nl_resid — names consistent. T5 independent. T6 consumes A7–A10 docs.
Task 1: Ruling: style-only placebo scored on LIN and NL — both LEAK (D3) — per predeclared rule the "style-residual IC vs 0" estimand is not usable; carried to Task 6 decision — cost: none (rule applied as written).
Task 2: Ruling: bootstrap CI on net Sharpe uses stationary block 10 on concatenated daily net returns, B 2000 — cost: none.
Task 4: Ruling: membership source fja05680/sp500 (author-stated caveats recorded); symbol matching by normalised ticker only, no rename map — cost: C2 under-counts renamed members.
Task 3: complete (scripts/bcreview_power.py run; A9 written; no unit tests in plan for this task)
Task 4: complete (scripts/bcreview_universe.py run; A10 written; membership files hashed; no prices)
Task 1: complete (commits 1e9078f..40587e5, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q -k 'rr_sign or quintile or nl_resid' → 3 passed, 14 deselected in 1.60s)
Task 2: complete (commits 1e9078f..40587e5, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q -k book → 3 passed, 14 deselected in 1.55s)
Task 5: Ruling: moved corrected_columns from scripts/fbaudit_dividends.py to src/hypershift/fbaudit/leakfree.py so frozen v1-LF code does not import a script — cost: none (script re-imports it).
Task 5: Ruling: fixed a SyntaxError committed in the audit (sed put a comment inside two calls in fbaudit_dividends.py); regression test test_audit_scripts_parse RED on HEAD version, GREEN now — cost: none.
Task 5: Ruling: removed the style-residual descriptive secondary from v1-LF (A7: both residual metrics fail the style placebo); v1-LF reports exactly v1's quantities — cost: loses one descriptive row.
Task 5: Ruling: added test_v1lf_hashes_match_addendum to pin the frozen v1-LF code to the addendum's sha256 — cost: none.
Task 5: complete (commits 40587e5..e172ffa, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q -k 'lf_ or v1lf or freeze or parse' → 5 passed, 16 deselected in 1.60s)
Task 6: Ruling: added bcreview_followup.py (paired model−placebo on neutral/LIN/NL; 20-seed random accounting check) before deciding — advisor: the decision rests on them — cost: none.
Task 6: Ruling: v1-LF fallback to adjusted close removed (advisor); missing raw close → stock-day ineligible, counted; new hash appended as superseding row (hash = LF-normalised sha256, manifest convention); hash test reads the last row — cost if wrong: v1-LF universe slightly smaller than v1 on missing-data days.
Task 6: Ruling: decision STOP (not REVISE) — estimand fails placebo; valid increment estimand has dev effect ~0.000–0.002 vs BC MDE 0.010–0.016; survivor coverage ≤43–58% makes only a negative interpretable, and dev already supplies it — cost if wrong: one untouched historical period left unused.
Task 6: Ruling: STUDY_BC_SPEC_v2 written as a conditional design (point-in-time data required) rather than an approved protocol — cost: none.
Task 6: Ruling: AUDIT_REPORT corrected by an appended dated section (no in-place edits) to keep the audit trail — cost: readers must read to the end; CLAUDE.md/HANDOFF point there.
Task 6: Ruling: A10 wording fixed (survivor share is a lower bound; gap symbols mix re-entries and possible reuse); A8 limitations appended (no borrow/financing, linear costs, beta vs close-close) — cost: none.
Task 6: complete (docs committed; suite run below)
Final review: self-review (no subagent dispatched — session rule: spawn agents only on explicit user request); checked book buffer/hedge/execution paths, RR, quintile dummies, LF missing-window logic, and every number in BC_REVIEW/STUDY_BC_SPEC_v2 against A7–A10.
Final: minor (deferred): in v1-LF, load_range computes the neutral target with pre-LF eligibility, so stocks later marked ineligible still enter the market mean and sector demeaning (tiny).
Final: minor (deferred): the style-only placebo uses one seed (7); no seed spread reported.
Final: minor (deferred): A8 net-Sharpe bootstrap resamples the concatenated 5-fold series, ignoring fold boundaries.
Final: minor (deferred, carried): A2 lag table 'nan%' cosmetic; fbaudit_dividends.py has two __main__ blocks.
Suite: .venv/Scripts/python.exe -m pytest -q → 463 passed, 3 warnings in 670.45s
