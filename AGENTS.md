# AGENTS.md

Updated 2026-10-27 after merging Rishi's artifact publication `origin/feat-rs` at `865dbb3` into `tom-shlom`.

## Read in this order

1. `docs/START_HERE.md`: authoritative research state, decisions, discipline and data traps.
2. This file, then `docs/HANDOFF.md`: current coordination and session notes. Older dated entries are history, not proof a worker is still running.
3. `CLAUDE.md`: durable environment, architecture and THINK history. Its older phase statuses, commands and compute rules need checking against the current source layout and START_HERE.
4. `charts.md`: essential results overview for Rishi's current work. Read its leader comparisons, SPY/QQQ benchmarks, seed ensembles, drawdowns, model explanation and full strategy gallery before assessing the latest results or proposing follow-up work.
5. Follow the links from `charts.md` to the relevant study report, frozen protocol, audit and progress log before changing its files or making a research claim.

Where these documents conflict, use START_HERE for research policy and the newest dated study artifacts for that study's results. Resolve conflicts explicitly; do not silently reopen closed work.

## Mission and current evidence

Find an implementable US-equity strategy with useful returns after realistic costs. The original THINK reproduction is closed; its negative results apply to the tested implementations and samples.

- **Latest Rishi development:** `docs/ndx_drawdown_20261026/REPORT.md`, `PROTOCOL.json`, `AUDIT.json`, and `master_update/AUDIT.json`. The study searches allocation/exposure controls on the existing THINK+mixer AH+PM / dynamic rank blend; no model retraining.
- **Selected historical results, 2022-2025:** five issuers plus a 20% basket-volatility target reports ten-seed Sharpe **1.112**, net CAGR **26.81%**, maximum drawdown **-21.69%**. The 25% target plus QQQ/MA200 trend cap reports **1.085 / 28.70% / -22.98%**. Matched full-exposure ten-seed baseline: **0.896 / 31.87% / -40.94%**.
- These are **305 searched recipes / 915 base-cost accounts**, plus 54 higher-cost accounts, on repeatedly inspected history. The new-five group remains weaker (Sharpe about **0.677 / 0.643** for those two leaders). The return-preserving candidate's paired Sharpe-difference interval includes zero. These results do not establish future alpha or a guaranteed drawdown limit.
- The committed risk-study audit reports915metric recalculations and82account replays; attribute those counts to the source audit. Current independent mechanism work verifies all339contract and1184published artifacts, reproduces18original saved-forecast account paths/NAV/metrics/aggregate ledgers, reconstructs10stock-level accounting traces and executes12half-capital sleeves. See `docs/rishi_mechanism_20261027/COMPLETION_AUDIT.md` and the batch reports. These are distinct scopes. `COMPLETE.json` retains an earlier pending-audit status; consult the later passed `AUDIT.json` alongside it.
- Earlier team assessment: `docs/team_audit_20261026/HYPERSHIFT_FINAL_RECOMMENDATION.md`, `HYPERSHIFT_NEXT_EXPERIMENT.md`, and `PROGRESS.md`. The missing saved-artifact blocker is resolved by865dbb3. Per-stock broker/order tapes were never saved; reconstructed economic fill traces now supply canonical accounting attribution. Original raw15minute chunks and first-provider-publication timestamps remain unavailable.
- Strict selected-checkpoint numerical regeneration remains unresolved: parent CPU/CUDA values fail1e-6pp despite exact inputs;20dynamic CUDA checkpoints pass before a validation failure. Keep saved-export reproduction separate from checkpoint reproduction. Hpred/changing-score and trained-component interpretations remain gated; do not widen tolerance after seeing outcomes. Primary ten-seed/top10 E-minus-parent and E-minus-training-label-identity contrasts do not pass the declared Holm5%family. Conditional financial retraining is not triggered.
- Keep original-five, new-five and ten-seed ensembles, top-five and top-ten portfolios, and parent/dynamic/blended models distinct. Earlier primary top-ten SR 0.755 and selected original-five/top-five SR 0.919 describe different cells from the new overlay leaders.

## Current source layout and commands

- `latest/README.md`: latest dynamic-hypergraph study; `research/README.md`: shared NASDAQ-100 source map.
- `charts.md`: primary visual entry point to the current results, including 364 main strategies, matched benchmarks, seed/cost diagnostics and links to provenance. Use it together with the underlying reports and audits; preserve the distinction between selected historical leaders and confirmation. `docs/NASDAQ100_strategy_comparison.xlsx` supplies the companion detailed comparison.
- `docs/PROJECT_LAYOUT.md`, `config/README.md`, `requirements/README.md`: layout, frozen settings and dependencies.
- Organized Rishi code lives in `research/` and `latest/`; Tom's earlier implementation lives in `src/hypershift/` and `scripts/`. Do not assume their launchers or evaluation contracts are interchangeable.
- In PowerShell, run the existing environment directly from the repo root:

  ```powershell
  $env:CUDA_VISIBLE_DEVICES = '-1'
  .\.venv\Scripts\python.exe -m hypershift list
  .\.venv\Scripts\python.exe -m hypershift test test_ndx_drawdown_engine -v
  ```

- The Rishi launcher temporarily supplies compatibility source paths and removes unchanged copies afterward. Edit organized sources; wait for commands to finish before staging. Keep frozen sources/locks byte-preserved.
- Most generated data/runs remain ignored;865dbb3 explicitly publishes selected reviewed inputs/checkpoints/forecasts under `data/` and `runs/`. Verify `docs/ndx_artifacts_20261027/manifest.csv` and the active study's contract rather than assuming either all outputs or no outputs are in Git. `.env` and credentials are private. Never expose secrets or overwrite existing result folders; preserve failed attempts and use fresh destinations.

## Research discipline and scope

- Feasibility and timing/identity/accounting checks first. Freeze and commit a preregistration before scoring; date amendments before outcomes. No post-outcome retuning presented as confirmation.
- Treat NDX 2022-2025 as spent development history. START_HERE's protected S&P 500 2024-01..2026-09 window is a separate universe/strategy claim, not blanket protection of those dates. Study J also inspected later data. Track sample use by universe and strategy.
- Do not score protected windows, train new models, purchase data, deploy capital, send team messages or activate prospective logging/automations solely because a report proposes it. October 27 authorization now includes executing available mechanism tests after finishing the plan. First protocol: `docs/rishi_mechanism_20261027/EXECUTION_PROTOCOL.md`; results: `execution_batch1/TESTING_REPORT.md` in that study folder. Resume the authorized artifact/replay/identity work when files arrive; new financial fitting remains conditional on the roadmap's data, learner-power and frozen-budget gates.
- Next research proposal remains bounded saved-prediction replay and identity/exposure attribution. Include the new risk-control selection and matched exposure controls when designing a follow-up; freeze a complete candidate before future confirmation.
- Distinguish independently replayed raw outputs, recomputed aggregates and source-reported summaries. Report costs, turnover, exposures, uncertainty and search/seed sensitivity. Seeds do not create independent market histories.
- The user-supplied `HYPERSHIFT_DEEP_RESEARCH_FULL_CONTEXT_PACK.zip` is an October 26 background snapshot. Its embedded master prompt is not an instruction to execute a new research assignment. Recheck its claims against newer commits.

## Coordination, Git and session close

- Check status and current ownership before editing. Preserve unrelated changes and other worktrees; old worker entries do not authorize stopping processes.
- Use `tom-shlom` for this checkout. Pull/merge only when authorized; **never push** under current research policy. The October 27 user request authorizes the latest `feat-rs` merge. Do not reset, rebase or discard local work to match Rishi's branch.
- Commit only explicit paths you changed. Leave `docs/HANDOFF.md` uncommitted by convention. Do not stage results, logs, compatibility copies or other sessions' work.
- Before stopping, update `docs/HANDOFF.md` (date, Running now, Next steps, Open decisions) and the active study's `PROGRESS.md`; update START_HERE whenever the authoritative state changes. Latest investigation: `docs/rishi_mechanism_20261027/RESEARCH_PLAN.md`; active log: `docs/rishi_mechanism_20261027/PROGRESS.md`. Earlier team audit: `docs/team_audit_20261026/PROGRESS.md`.
- Verify documentation links and relevant checks before claiming completion. Distinguish unit-test success from replaying a research result.
