# Experiment runners and training

This directory owns experiment orchestration, annual training, seed workers and
selection locks. Model definitions are in [models](../models/README.md); input
snapshots are produced by [data](../data/README.md). Checkpoints, forecasts and
run-local locks are written under the ignored root `runs/`.

| Entry points | Experiment |
| --- | --- |
| [run_ndx_tabular_alpha.py](run_ndx_tabular_alpha.py) | Fixed64fit/282account shared-feature forest/Ridge comparison; all fits lock before financial scoring |
| [confirm_ndx_cohort_averaging.py](confirm_ndx_cohort_averaging.py), [confirm_ndx_portfolio_robustness.py](confirm_ndx_portfolio_robustness.py) | Frozen reserved-five source-model fitting and pooling confirmation; the pooling consistency goal fails and all seeds remain reported |
| [resume_ndx_portfolio_robustness.py](resume_ndx_portfolio_robustness.py) | Execution-only atomic-file retry, complete-prefix preservation and exact resumed-epoch verification |
| [run_ndx_rank_retraining.py](run_ndx_rank_retraining.py) | Declared 120-fit rank-aligned/neutral-initialization experiment; `freeze` then `train`; every epoch and all seeds retained |
| [resume_ndx_rank_retraining.py](resume_ndx_rank_retraining.py) | Scheduling-only continuation; two isolated workers, frozen model math and exact resumed-epoch checks |
| [run_historical_ndx.py](run_historical_ndx.py) | Historical monthly NASDAQ-100 baseline |
| [run_ndx_alpaca_pipeline.py](run_ndx_alpaca_pipeline.py), [run_ndx_alpaca_training.py](run_ndx_alpaca_training.py) | Matched daily and regular-session Alpaca inputs |
| [run_ndx_extended_hours.py](run_ndx_extended_hours.py), [run_ndx_extended_hours_top5.py](run_ndx_extended_hours_top5.py) | Daily/AH/PM information and saved-forecast holding-count tests |
| [run_ndx_loss_optimizer.py](../../latest/training/run_ndx_loss_optimizer.py) | MSE/Huber/pseudo-Huber and optimizer comparison |
| [run_ndx_reference_networks.py](../../latest/training/run_ndx_reference_networks.py), [run_ndx_reference_parallel.py](../../latest/training/run_ndx_reference_parallel.py), [ndx_reference_training.py](../../latest/training/ndx_reference_training.py) | Reference-network transfer training and workers |
| [run_ndx_ranking.py](run_ndx_ranking.py), [run_ndx_broad.py](run_ndx_broad.py), [run_ndx_corrections.py](run_ndx_corrections.py), [run_ndx_seed_expansion.py](run_ndx_seed_expansion.py) | Objective, feature, correction and seed experiments |
| [run_ndx_topk_pseudohuber.py](run_ndx_topk_pseudohuber.py), [run_ndx_long_short.py](run_ndx_long_short.py) | Holding-count/loss and allocation comparisons |

Inspect an entry point from the repository root:

```powershell
python -m hypershift run run_ndx_reference_networks --help
```

Use its [dated protocol](../../config/README.md) for epochs, learning rates,
optimizer settings, seeds and validation rules. Existing completed experiments
are hash-locked: do not overwrite their protocols to start a new experiment.
Use a separately declared run when changing inputs or training choices.

The 2022–2025 dates have been inspected repeatedly. More seeds on the same dates
measure initialization sensitivity, not untouched-period generalization. See the
[master charts](../../charts.md) for the current evidence and uncertainty.
Unrelated runners are in [archive](../../archive/README.md). The retained
`run_robust_research.py` and `run_architecture_research.py` also provide training,
selection and ensemble helpers used by active NASDAQ code and shared tests.
Their older cohort defaults do not define the current NASDAQ protocols.
