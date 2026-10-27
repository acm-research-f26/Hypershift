# Actual dynamic-network synthetic recovery gate

October 27, 2026. Protocol frozen in commit `c1ad3b3` before outcomes. All **18 fits completed**, 800 updates each, in **327.48 seconds** total. The first20-update resource smoke projected349.2seconds before any held-out evaluation; the recipe was not changed.

## Results

Held-out R² relative to a zero predictor, ranges across fixed seeds7/19/42:

| Planted target | Full pinned dynamic adapter | Separately trained temporal-only path |
|---|---:|---:|
| Own-stock scalar | 0.965–0.977 | 0.967–0.983 |
| Paired other-stock scalar | **0.880–0.891** | -0.040 to -0.037 |
| Fixed per-stock identity preference | **0.763–0.806** | -0.044 to -0.003 |

All intended-recovery screens (R²≥.5 in all three seeds) passed for the full model; temporal-only passed own-stock recovery. The restricted path did not recover targets unavailable from a stock's own history. Its small negative values mean it performed slightly worse than the zero predictor; they are not negative financial returns. No impossible-signal leakage warning was triggered.

These are genuinely trained synthetic models, unlike the earlier random-initialization capacity probes. Train and held-out dates were independently generated. Labels were not financial data, and none of Rishi's original financial checkpoints was used.

## What changes in the research plan

The current imported implementation has now demonstrated that it can learn useful peer information **when it is planted and accessible through its supplied relational structure**, as well as persistent stock-ID preferences. A future financial ablation cannot dismiss it as a learner that has never shown relational recovery. Equally, a good financial account cannot by itself distinguish relational information from identity preference: both are representable and learnable in this implementation.

The full model includes both known pair wavelet snapshots and learned dynamic incidence, plus node-indexed filters. The peer task does **not** isolate the learned-dynamic path; fixed pair relations may supply the signal. The identity result does **not** show that the original financial model selected persistent winners. Neither result establishes financial alpha, model superiority under realistic costs or a need to deploy capital.

Temporal-only is a separately trained restriction with identical shared-module initialization and optimizer recipe; it bypasses graph branches and has fewer active parameters. Graph dropout also changes subsequent random-number consumption. This is a capacity/negative-control screen, not a parameter/optimization-noise-matched causal comparison of architectures.

## Coverage and remaining work

Completed: actual-model own-stock recovery, accessible peer recovery, fixed-ID recovery, independent held-out generator and impossible-target controls. Existing structural tests separately cover ID reindexing, inactive features, gradients and account/timing mechanics.

Not completed: learned recovery under missingness-only nuisance or changing membership, isolation of learned incidence versus static propagation, original-checkpoint interpretation, training-only financial identity controls, per-stock P&L and financial ablation batches. Those remaining rows stay explicit in `../COMPLETION_AUDIT.md`. The synthetic pass does not bypass the financial artifact/replay or incremental-evidence gates.

Original artifacts remain unavailable. A request contract was derived from committed locks: `../ARTIFACT_CONTRACT_V2.json` enumerates339 original parent/dynamic fit and input files with hashes, including the calendar and shared preprocessing factor dependencies. Account/trade tapes remain additionally required. The earlier335-file subset is retained unchanged, and current live feat-rs results README still identifies large run outputs as ignored.

## Reproducibility and review

Sources: `research/analysis/rishi_synthetic_gate.py`, `tests/test_rishi_synthetic_gate.py`, pinned upstream `source_snapshot/layer.py` and `model.py`. Exact fit values in individual JSON files and `RESULTS.json`; runtime estimate in `RUNTIME_SMOKE.json`. Existing batch output is never overwritten.

Independent read-only review approved the generator, task observability, protocol adherence, code path and interpretation scope. It independently verified all335 entries of the initial artifact subset against their cited locks. Its minor observation: timeout checks occur between fits, so a future run can overrun the cap by one fit; this run completed well within15minutes. No alternate recipe was chosen after seeing outcomes.

The synthetic data/metric tests were first observed failing because the new module did not yet exist, then passed after implementation. The combined focused test log is `TEST_LOG.txt`. The full goal remains active, not complete.
