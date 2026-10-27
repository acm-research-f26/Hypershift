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

Runner output defaults to decision_strategy_execution_2026-10-27_v5. Metadata
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
