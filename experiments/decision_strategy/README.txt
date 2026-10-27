Decision-strategy research (bounded historical study)

Run from the Hypershift repository using .venv-experiment\Scripts\python.exe.
All raw data, licensed quotes, checkpoints and result bundles remain outside Git.

Commands:
  python -m experiments.decision_strategy.acquire_metadata
  python -c "from experiments.decision_strategy.metadata import finalize; from experiments.decision_strategy.config import STUDY; finalize(STUDY/'decision_strategy_execution_2026-10-27')"
  python -m experiments.decision_strategy.runner engineering
  python -m experiments.decision_strategy.runner freeze
  python -m experiments.decision_strategy.runner proxies
  python -m experiments.decision_strategy.runner quotes

Runner output defaults to decision_strategy_execution_2026-10-27_v7. Metadata
acquisition is preserved in the original decision_strategy_execution_2026-10-27
input directory. V1's interrupted receipt and faulty train-threshold diagnostic
are preserved; v2 repaired these before development comparisons. V3 repairs
an independently detected idle-mark/missing-open accounting error after the
first5 cash-only v2 trials, preserving the full failed active-control tape.
V4 releases expired unfilled basket slots and applies known liquidity capacity
before the economic hurdle. Interrupted versions are retained, not pooled into
valid policy comparisons. Signal thresholds and qualification remain unchanged.
V4 was superseded before execution by v5's pre-run units/control audit:
arithmetic dollar-P&L labels, actual D03 residual exits and declared control
feature restrictions. Coarse group construction and90th-percentile rules stay
fixed. No version chose a cross-recipe development shortlist before these fixes.
V6 adds a fail-closed finite-cost guard. It retains v5, reuses19 completed
receipts only with an exact single-line execution-delta and consumed-decision
equivalence proof, and leaves fits, recipes, activity and risk criteria fixed.
V7 removes industry breadth from the D04 stock-only control before that control
has any development trial. Only that model refits on train60. The other11 models
and their verified tapes carry forward with explicit branch-equivalence proofs.

Current continuation and reporting commands:
  python -m experiments.decision_strategy_v7
  python -m experiments.decision_strategy.runner quotes
  python -m experiments.decision_strategy_independent --output <absolute v7 output directory>
  python -m experiments.decision_strategy_delivery attribute
  python -m experiments.decision_strategy_delivery dispositions
  python -m experiments.decision_strategy_delivery report
  python -m experiments.decision_strategy_delivery verify

The two-worker orchestrator runs exactly315 jobs. Both development cohorts
finish before selection.json freezes; only then can the later cohort open.
The queue reuses the original parallel worker and keeps independent recipes
balanced across two CPUs. Never run both orchestrators simultaneously.
After development selection has frozen, the current later-only continuation is
python -m experiments.decision_strategy_later. It repairs an operational
single-cohort string/sequence error before any later prices were opened. The
original controller source and failure receipt are preserved; strategy bytes,
fits,selection and the105 later jobs are unchanged. Use runner proxies with
--output for a full replay of a populated frozen artifact directory.
Post-freeze reporting helpers never fit a strategy or alter a decision. Exact v7 source
and dependency bytes, including line endings, are retained with the artifacts.
An empty diagnostic set means zero strategy quote replays and unmeasured
coverage, regardless of the separate training-spread acquisition cache.

Do not overwrite a frozen study after changing strategy source. Preserve the
original directory and version correctness repairs explicitly. Classifications
are pre-2025 reconstructed business industry groups, not historical certified
GICS. The user-mandated250-stock universe remains retrospectively selected.

The protected-date manifest is checked before prices or forecasts are opened.
Only the three exposed historical cohorts are authorized. There is no live
brokerage endpoint, model training, data purchase or future collection here.

Every stock leg executes sequentially and records its resulting cash and risk.
Snapshots cannot certify unseen intraminute price or liquidity paths. Missing
hedges, partial fills, closing failures and constraint breaches remain failures.
The3% borrow rate and historical locate availability are explicit assumptions.

Data and execution choices, finite trial budgets, activity requirements and
conditional-stage prerequisites are recorded in protocol.json before outcomes.
Negative and cash-only results are valid research outcomes, not qualification.
