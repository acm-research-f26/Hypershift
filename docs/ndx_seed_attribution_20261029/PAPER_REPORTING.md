# Reporting the seed discrepancy in the paper

The 1.731 Sharpe result can be reported as a selected historical best case. It
should not be presented as representative or seed-robust while omitting the
matched 0.648 result. The original seed list being fixed in advance does not
justify hiding a subsequently observed robustness failure.

A defensible summary of the original system is:

> On the repeatedly inspected 2022–2025 development period, the original
> five-seed retention/QQQ-hedged ensemble achieved a net annualized Sharpe ratio
> of 1.731. A disjoint five-seed ensemble achieved 0.648, and their combined
> ten-seed ensemble achieved 1.381. Parameter and account attribution identified
> persistent stock-specific initialization preferences in the 2023 THINK
> forecasts and concentration in both Alphabet share classes. These results
> therefore do not establish initialization-robust forecasting superiority.

The retraining study should be reported in addition to this original result,
including negative outcomes. Its fixed criterion is SR above 1.5 in each of the
three disjoint five-seed groups and both pooled ensembles, rather than selecting a
favorable cohort after evaluating it.

The main tables and appendix should retain:

- The exact seed lists, complete search budget, chronological folds and
  checkpoint-selection rules.
- Every predefined ensemble and individual-seed result under the same portfolio,
  costs, borrowing and execution assumptions.
- Means/medians and spread across training initializations, plus the separately
  traded ensemble account. An ensemble's Sharpe is not the average Sharpe of its
  individual seed accounts.
- Ranking quality and the distinction between checkpoint-selection validation,
  diagnostic validation and the already-inspected development test period.
- Untrained, deterministic and issuer-consolidated controls; fees, turnover,
  exposure and asset-level contribution evidence.
- All failed methods and any implementation repairs or execution amendments.

Seed variation measures sensitivity to training randomness conditional on this
market sample. It does not supply independent market histories or prove future
Sharpe. A new model chosen because it improves 2022–2025 still needs genuinely
untouched evaluation for a confirmation claim.

Bailey and López de Prado's
[Deflated Sharpe Ratio paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
explains how multiple testing and partial reporting inflate perceived backtest
performance. A correction cannot be honestly based only on the final few seeds
while ignoring the earlier architecture, objective, allocation and hedge search.
Any effective-trial adjustment needs explicit assumptions; this work does not
claim a corrected significance result merely from a high selected Sharpe.
