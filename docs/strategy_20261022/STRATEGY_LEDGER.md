# SDD ledger — plan: docs/superpowers/plans/2026-10-22-strategy.md
Setup: Ruling: execute without plan-review stop (user: make decisions until results) — cost: plan not user-vetted.
Setup: Ruling: candidate PRICE3 and all rules predeclared in the plan commit before any strategy number computed — cost: none.
Pre-flight: T1 produces signals/engine/factors consumed by T2 (explore) and T3 (holdout); T4 docs consume T2/T3 outputs.
Task 1: complete (commits fdca968..8a3de4c, tests: .venv/Scripts/python.exe -m pytest tests/test_strategy.py -q → 10 passed in 0.28s)
Task 2: Ruling: PRICE3-LO NOT GO and PRICE3-LS NOT GO per predeclared rule; per spec 'reuse an existing saved candidate', added strategy_style_book.py to re-test the style-only LightGBM book (A13) in the WP-A engine with borrow + FF attribution, with a reading rule stated before numbers — cost: one post-hoc diagnostic on inspected years.
Task 2: Ruling: added STYLE_ROBUST checks (cost/worst-year metrics, seeds 19/42 + 3-seed rank average, PIT-membership eligibility, leg decomposition) before any freeze — advisor; cost: none.
Task 2: Ruling: style-only book judged NOT CREDIBLE on current data — PIT-membership filter halves the edge (seed-7 LS 0.80→0.36, alpha t 1.42→0.19; 3-seed LS 1.19→0.61, alpha t 2.36→0.87) and the short leg loses in beaten-down volatile names (survivorship signature); remaining edge inseparable from missing-deletion bias — cost if wrong: a genuine weekly characteristic-interaction edge is deferred until survivorship-free data.
Task 3: Ruling: pseudo-holdout 2024-01..2026-08 NOT used — PRICE3 failed its gate and the style book's residual edge is confounded by survivorship that a 2.7-year kill-test (IR SE ~0.6) cannot resolve; preserve the window for a candidate re-tested on survivorship-free data — cost: one fewer quasi-OOS look now.
Task 3: Ruling: instead freeze a prospective paper-trading spec for the style book (survivorship-free going forward by construction: universe fixed at the start), labelled implementation check + slow evidence, no capital — cost: monitoring effort.
Task 2: complete (commits 8a3de4c..b0db354, tests: .venv/Scripts/python.exe -m pytest tests/test_strategy.py -q → 11 passed in 0.23s)
Task 3: Ruling: refactored LS weight construction into engine.ls_weights (used by run and paper trading); all earlier tests unchanged and green — cost: none.
Task 3: complete (commits b0db354..3220d70, tests: .venv/Scripts/python.exe -m pytest tests/test_strategy.py -q → 14 passed in 6.19s)
Task 4: Ruling: deliverables A/B/F/G consolidated into STRATEGY_REPORT.md (C, D, E as separate files) — fewer files, same content — cost: none.
Task 4: Ruling: corrected the EXPLORE.md capacity sentence in STRATEGY_REPORT (13% of ADV at $100M is material) rather than regenerating the evidence file — cost: none.
Task 4: complete (docs NEW_DATA, EVAL_DESIGN, IMPLEMENTATION, STRATEGY_REPORT, REQUEST; CLAUDE.md; HANDOFF uncommitted)
Final review: self-review (no subagent dispatched — session rule); checked engine accounting via tests (open execution, costs, borrow, trace), decision-rule application, and report numbers against EXPLORE/STYLE_* docs.
Final: minor (deferred): EXPLORE.md capacity uses adjusted-price dollar volume and includes stale zero-volume names (1st-percentile ADV $8.3M likely understated).
Final: minor (deferred): style-book scores use matched-run predictions that were trained with FB's train window (skips year Y-1); the paper-trading spec trains on all matured labels (documented difference).
Final: minor (deferred): pit_eligible matches by ticker only (renames unmatched).
Suite: .venv/Scripts/python.exe -m pytest -q → 483 passed, 3 warnings in 846.19s
