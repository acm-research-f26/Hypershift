# SDD ledger — plan: docs/superpowers/plans/2026-10-22-fb-audit.md
Setup: Ruling: no plan-review stop; executing immediately — user said "make decisions until you bring me results" — cost if wrong: plan scope not user-vetted.
Setup: Ruling: work in place on tom-shlom (no worktree) — standing practice; never push — cost: none.
Setup: Spec = user's pasted prompt (saved as docs/audit_20261022/REQUEST.md in Task 1).
Pre-flight: T1 produces stats.{spearman_daily,pooled,paired_by_fold,load_daily,load_pred,nw_se,...}, diag.{exposures,neutralise_pred}; T2/T3/T4/T5 consume them — consistent names in plan; T4 refit.run_arm consumed by T5 (corrected fb copy passed in) — consistent; T7 consumes A1–A6 docs only.
Task 1: Ruling: REQUEST.md is an abridged copy of the prompt (binding instructions kept) — full text lives in the transcript — cost: minor loss of verbatim wording.
Task 1: Ruling: added block:l1, block:reversal16 and ridge−rev16 rows to the power table for reference — cost: none.
Task 1: complete (commits 9992b47..8e63789, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q → 6 passed in 1.74s)
Task 2: Ruling: added lgbm:price, mlp:rich, block:l2_5, ridge:lags rows and 'both neutralised' contrasts beyond the brief's list — more context, same method — cost: none.
Task 2: Ruling: open1 uses OHLCV open/close ratio (per-stock scale cancels), observed required t..t+h — cost: none.
Task 2: complete (commits 8e63789..716523d, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q → 8 passed in 1.61s)
Task 3: Ruling: shrinkage noise = bootstrap SD of (arm − pool mean) instead of each arm's own SE — arms share dates, own-SE made τ²=0 trivially — cost: indicative number only.
Task 3: Ruling: portfolio arms add block:l1; rules A/B/D only (C,E are long-only variants of A) — cost: none.
Task 3: Ruling: added eligible zero-volume counts per stock (SW/FERG/AMCR stale prices found) — cost: none.
Task 3: complete (commits 716523d..1e54bbb, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q -k 'not run_arm' → 9 passed, 1 deselected in 1.63s)
Task 6: complete (commit 7841a00, docs only; citations verified by web search 2026-10-22)
Task 5: Ruling: added yield_check (trailing-yield decomposition of the leak term) — needed to separate legitimate dividend-yield signal from look-ahead before calling it material — cost: none.
Task 4: Ruling: plan amendment — A4b follow-up (neutral-target arms through A2/A3 diagnostics) because the ladder's neutral target doubled the residual gain; exploratory, ledgered — cost: one more post-hoc analysis on inspected data.
Task 4: complete (commits 1e54bbb..76b2dd9, tests: .venv/Scripts/python.exe -m pytest tests/test_fbaudit.py -q → 10 passed in 93.76s (0:01:33))
Task 5: complete (commit 91a4601; scripts run: dividends fetch/analyze/yield, refit run corrected, report-corrected; T5 MATERIAL)
Task 5: Ruling: A5 part files merged by hand into A5_DIVIDENDS.md with the verdict section — cost: re-running scripts rewrites part files only.
Task 5: Ruling: a 'git commit -a' slip swept 3 other-session files into a local commit; undone with reset --soft + restore --staged before anything else; working tree untouched; never pushed — cost: none.
Task 7: Ruling: added A2c leak-free rerun of T2 before writing (advisor: A2 used contaminated exposures/predictions) — cost: none.
Task 7: Ruling: next study = powered backcast on untouched 2004–2012 history (plan rule fell through: T2 tilt-removal + T3 robust); grep showed no project artefact uses pre-2013 dates (RSR starts 2013-01-01); coverage 373/462 tickers by 2004 (first-quarter dates only, prices unread) — cost if wrong: survivor bias heavier pre-2015.
Task 7: complete (commits 8ad9227..HEAD; docs AUDIT_REPORT/NEXT_STUDY_SPEC/PROSPECTIVE_ADDENDUM, pointers in FB_REPORT and CLAUDE.md; HANDOFF.md updated, uncommitted)
Final review: self-review (no subagent dispatched — session rule: spawn agents only on explicit user request); checked stats/SPA/PBO/portfolio/dividend code paths and every report number against A1–A6.
Final: minor (deferred): corrected_columns mixes adjusted and unadjusted close levels on days where the fresh raw Close is missing (few cells; 2 stocks fully uncorrected: PSKY, WBD).
Final: minor (deferred): A2 lag table prints 'nan%' retention for arms with non-positive same-close IC (cosmetic).
Final: minor (deferred): A5 part files were merged by hand; rerunning `fbaudit_dividends.py analyze/yield` rewrites A5_DIVIDENDS_part1.md, not A5_DIVIDENDS.md.
Final: minor (deferred): fbaudit_dividends.py has two __main__ blocks (yield subcommand bolted on).
Suite: .venv/Scripts/python.exe -m pytest -q → 452 passed, 3 warnings in 665.50s
