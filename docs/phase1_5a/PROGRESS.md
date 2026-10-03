# Phase 1.5a progress

Plan: `docs/superpowers/plans/2026-10-03-phase1-5a-sharpe2-forensics.md`. Execution order: batch 1 = T1, T2, T2B, T5 (ends with Gate A); batch 2 = T3, T4, T6, T7, T8 (trimmed per the Gate A outcome).

Whoever finishes a task (Claude worker or Codex fallback) appends one line: `T<id> done <YYYY-MM-DD HH:MM> by <claude|codex>: <one-line result>`. If blocked: `T<id> BLOCKED: <reason>`. After T8 completes, create the empty file `docs/phase1_5a/DONE`.

## Log
T1 done 2026-10-03 17:44 by claude: forensics module + inventory; 25 run-seeds verified (daily atol 1e-7, SR 1e-5), 6 unit + 5 artifact tests pass
T2 done 2026-10-03 17:48 by claude: mechanism stage full (R_TIE=1000, 200 index perms); THINK node-permutation equivariance holds for all 4 temporal/spatial combos; primary random-tie median SR 1.78/1.79/2.07/2.14/2.17 vs hold-all 1.53; 14 unit tests pass
