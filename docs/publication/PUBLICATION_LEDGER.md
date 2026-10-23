# SDD ledger — plan: docs/superpowers/plans/2026-10-22-publication.md
Setup: Ruling: execute without plan-review stop (user: make decisions until results) — cost: plan not user-vetted.
Pre-flight: T1 verification statuses consumed by T2 register; T2 IDs consumed by T3 manuscript and T4 docs — consistent.
Task 1: Ruling: hold-all recompute uses ddof=0 to match the repo evaluator (eval/metrics.sharpe); first run with ddof=1 gave 1.5278 vs cited 1.531 (convention, not data) — cost: none.
Task 1: complete (commits c0bab00..823bbec, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q -k check_marks → 1 passed, 27 deselected in 1.94s)
Task 2: complete (CLAIMS_REGISTER.md; docs only)
Task 3: complete (MANUSCRIPT.md; references verified by search 2026-10-22; Lo & MacKinlay 1990 flagged secondary-only)
Task 3: Ruling: thesis narrowed from "systematically" (field-wide) to one reproduction case + one benchmark — evidence covers one model lineage — cost: none.
Task 3: Ruling: lead contribution = vendor-adjustment leakage decomposition (strongest novelty, algebraic + recomputed) — cost if wrong: framing choice only.
Task 4: Ruling: recommend one methods-only simulation extension (learner power) before submission; no market data — user said document missing evidence rather than run it — cost: C3 must be cut if not done.
Task 4: complete (FIGURES_TABLES, REVIEWER_RISKS, REPRODUCIBILITY, RECOMMENDATION; CLAUDE.md; HANDOFF uncommitted)
Final review: self-review (no subagent dispatched — session rule); manuscript numbers checked against CLAIMS_REGISTER and CLAIMS_VERIFICATION; references verified by search except Lo & MacKinlay (flagged).
Final: minor (deferred): pubfig_* plotting scripts not written (spec only).
Final: minor (deferred): environment lock (pip freeze, hardware) not yet recorded.
Suite: .venv/Scripts/python.exe -m pytest -q → 469 passed, 3 warnings in 730.09s

## Revision 2 (2026-10-22)

- Ruling: added a construction-based decomposition (L_div / L_path) on data already on disk. It is analysis, not a new experiment or period, and it is needed to replace an over-interpreted regression residual. Cost: none.
- Ruling: recommendation switched from "short note first" to Option B (combined paper), after the prior-art check showed the core mechanism is known to practitioners. Cost: framing.
- Ruling: EH planted-IC mismatch (earlier report 0.090 vs files 0.098) recorded, not hidden. The manuscript does not use the value.
- Ruling: an event-count bug in pub_leak_mechanism (summed across folds) was fixed before publication of the numbers.
- Release blockers found: the THINK PDF (IEEE) and RSR-derived test_gt.npy arrays (AGPL) are tracked in git.
- Suite: 469 passed (642 s).
- Final review: self-review (no subagent dispatched; session rule).
