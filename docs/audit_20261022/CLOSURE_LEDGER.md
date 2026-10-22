# SDD ledger — plan: docs/superpowers/plans/2026-10-22-matched-closure.md
Setup: Ruling: execute without plan-review stop (user: make decisions until results) — cost: plan not user-vetted.
Setup: Verified original placebo config (scripts/bcreview_style.py): LightGBM fit_fixed 50 rounds, seed 7, target 'neutral', inputs cs_z(beta, log_dvol_20 [leak-free], ret_60, vol_60) + sector one-hot. Full model compared against it: corrected early-stopped FB residual LightGBM (rounds chosen on validation: varies by fold), target 'residual', 20 FB features cs_z, NO sector one-hot, seed 7. => confounded by target, rounds, and non-nested features.
Pre-flight: T1 produces matched.nested_inputs/encompass consumed by T2 (synthetic) and T4 (style book) — consistent. T3 independent. T5 consumes A11–A13.
Task 3: Ruling: v1-LF neutral recomputed with post-LF mask using exactly load_range's call (_neutralise(y.T, (y_ok&eligible).T, beta.T, sectors)); classified class (1) pre-data implementation fix, new superseding hash row; no new protocol version — cost: none (effect −0.00026 pooled on dev).
Task 3: Ruling: coverage erratum appended to BC_REVIEW (append-only); "at most" was wrong, measured share is a lower bound on coverage — cost: none.
Task 2: Ruling: synthetic M4 FAILED for both matched increment (negative in S and I: dilution/capacity confound at 50 rounds) and encompassing (false positive in S); adding ONE dimension-matched control (extra features permuted within date inside bins of the style-only prediction) and re-validating on synthetic before any real-data use — cost: 1 extra synthetic run + 30 real fits if it passes.
Task 2: Ruling: permutation control also NOT VALID; learner-power probe shows the FB LightGBM (50 rounds) cannot exploit a planted standalone-IC 0.046 signal; stop adding procedures (user: no large metric search); question B declared unanswerable with current learner/toolkit; valid design described for future work — cost if wrong: an answerable B left unanswered.
Task 4: Ruling: econ arms extended with matched full/residual, style/residual, style/neutral (seed 7) so the economic comparison is matched too — cost: none.
Task 1: complete (commits 6e2c7b7..6f7c7e2, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q -k 'nested or encompass or synth_det or permute_within or post_lf or lf_ or missing_raw or v1lf or freeze or parse' → 11 passed, 16 deselected in 1.78s)
Task 2: complete (commits 6e2c7b7..6f7c7e2, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q -k 'nested or encompass or synth_det or permute_within or post_lf or lf_ or missing_raw or v1lf or freeze or parse' → 11 passed, 16 deselected in 1.86s)
Task 3: complete (commits 6e2c7b7..6f7c7e2, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q -k 'nested or encompass or synth_det or permute_within or post_lf or lf_ or missing_raw or v1lf or freeze or parse' → 11 passed, 16 deselected in 1.55s)
Task 4: complete (commits 6e2c7b7..6f7c7e2, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q -k 'nested or encompass or synth_det or permute_within or post_lf or lf_ or missing_raw or v1lf or freeze or parse' → 11 passed, 16 deselected in 1.62s)
Task 5: Ruling: two-level verdict (question B INCONCLUSIVE; resource STOP) per advisor — cost: none.
Task 5: Ruling: coverage figure re-described as a symbol-match count, not a bound (renames lower it; reuse BR and pre-buyout symbols KMI/HCA/HLT/DG raise it); Erratum 2 appended to BC_REVIEW and AUDIT_REPORT — cost: none.
Task 5: Ruling: FB feature/model-class nulls reclassified "untested with a learner shown to have power" (scoped to 3 replicates + 1-seed probe at easier SNR) — cost if wrong: overstates learner weakness slightly.
Task 5: Ruling: style-only book result fenced as post hoc / style premia, not a lead — cost: none.
Task 5: complete (docs FINAL_DECISION, PUBLICATION_OUTLINE, errata, CLAUDE.md; HANDOFF uncommitted)
Final review: self-review (no subagent dispatched — session rule); checked A11/A12/A13 numbers against FINAL_DECISION; verified Clark & West (2007) citation.
Final: minor (deferred): economic arms (A13) use seed 7 only for matched cells.
Final: minor (deferred): synthetic SNR (style IC ~0.08) differs from real data (~0.01–0.03); only 3 replicates.
Final: minor (deferred, carried): A2 'nan%' display; two __main__ blocks in fbaudit_dividends.py; A8/A13 bootstrap ignores fold boundaries.
Suite: .venv/Scripts/python.exe -m pytest -q → 468 passed, 3 warnings in 672.48s
