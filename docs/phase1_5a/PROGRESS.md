# Phase 1.5a progress

Plan: `docs/superpowers/plans/2026-10-03-phase1-5a-sharpe2-forensics.md`. Execution order: batch 1 = T1, T2, T2B, T5 (ends with Gate A); batch 2 = T3, T4, T6, T7, T8 (trimmed per the Gate A outcome).

Whoever finishes a task (Claude worker or Codex fallback) appends one line: `T<id> done <YYYY-MM-DD HH:MM> by <claude|codex>: <one-line result>`. If blocked: `T<id> BLOCKED: <reason>`. After T8 completes, create the empty file `docs/phase1_5a/DONE`.

## Log
T1 done 2026-10-03 17:44 by claude: forensics module + inventory; 25 run-seeds verified (daily atol 1e-7, SR 1e-5), 6 unit + 5 artifact tests pass
T2 done 2026-10-03 17:48 by claude: mechanism stage full (R_TIE=1000, 200 index perms); THINK node-permutation equivariance holds for all 4 temporal/spatial combos; primary random-tie median SR 1.78/1.79/2.07/2.14/2.17 vs hold-all 1.53; 14 unit tests pass
T2B done 2026-10-03 17:51 by claude: proxy stage; primary score = mean-reversion tilt (ret20 rho -0.2..-0.38, ma30_rel +0.2..0.39), R2 0.09-0.21; fitted SR 0.33-1.76, resid SR 1.83-2.29 (model 1.88-2.14); 16 unit tests pass
T5 done 2026-10-03 17:56 by claude: nulls full (B=10000/2000, 5000 boot); family p F1 0.120 F2 0.081 F3 0.641 F4 0.142 (Holm 0.32-0.64); Gate A: outcome 2 holds, 1 and 3 do not; REPORT_2017.md Gate A written; 19+ tests pass. Batch 1 complete, batch 2 not started.
