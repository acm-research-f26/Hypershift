# Interpretation of the validation ranking stability experiment

The experiment improved the weaker seed ensemble in several variants, but **did
not achieve approximately 1.7 Sharpe with consistent performance across the
original-five, new-five and ten-seed ensembles**. None of the 24 fixed recipes
passed either the primary or the secondary performance criteria. All 216
cost/borrow accounts and failed variants remain available in the [report](REPORT.md).

| Base-cost recipe | Original-five SR | New-five SR | Ten-seed SR | Original/new gap |
|---|---:|---:|---:|---:|
| Existing five-security retention, 75% hedge cap | 1.731 | 0.648 | 1.381 | 1.084 |
| Validation RankIC weighting, five-security retention, 75% cap | 1.377 | 0.841 | 1.349 | 0.537 |
| Stable validation weights + agreement penalty, smoothly weighted ten securities, 50% cap | 1.230 | 0.958 | 1.250 | 0.272 |

The last recipe has the highest weaker-half Sharpe in this grid. It raises the
new-five Sharpe by 0.311 and narrows the half-gap by approximately 75%, while
lowering the original-five Sharpe by 0.501 and the ten-seed Sharpe by 0.131.
Its original/new/ten annualized returns are 23.49% / 17.66% / 22.95%; the previous
recipe returns 33.11% / 12.31% / 26.35%. This is a tradeoff, not the requested
performance-preserving solution. At doubled trading costs plus 6% borrowing,
its Sharpes fall to 1.067 / 0.778 / 1.087.

The highest ten-seed Sharpe in the grid is 1.393 (stable consensus, five-security
retention, 75% cap), but its original/new halves return 1.273 / 0.584. Selecting
that result solely by the ten-seed number would miss the continued seed problem.

## What the validation metrics suggest

The adaptive methods increase average validation RankIC, partly by construction:
they use that same validation evidence to set the weights. Mean ten-seed
validation RankIC rises from 0.0370 with equal weights to 0.0441 with validation
RankIC weights. This does not translate into a consistently superior account.
Even the baseline new-five ensemble has higher mean validation RankIC than the
original-five ensemble (0.0413 versus 0.0298), despite its much weaker test-period
Sharpe. Prior-year ranking quality alone does not reliably identify the ensemble
with stronger later trading returns in this sample.

Ranking agreement, selection agreement and financial consistency are different
metrics. For five-security retention, validation-IC weighting raises test-period
original/new overall rank correlation from 0.6305 to 0.6593, while the overlap of
actual target weights changes from 42.24% to 41.49%. An agreement penalty also
does not automatically increase overlap between independently composed seed
groups. The full diagnostics disclose these cases.

## Boundary of this experiment

The existing saved exports contain only the final MSE-selected checkpoint's
forecasts. This study evaluates validation-based ensemble weighting, disagreement
penalties and allocation, **not ranking-based checkpoint selection or new model
training**. The validation fold was already used for MSE checkpoint selection,
so it is not an independent calibration/evaluation sample. The 2022–2025 test
history is repeatedly inspected development data.

A distinct future study could refit the models while saving every epoch's
validation forecasts, then compare the existing MSE checkpoint rule with a fixed
ranking-quality/stability rule. That needs a new protocol, a fixed compute budget,
validation separation or purging, and the same complete seed-group reporting.
This experiment does not establish that such retraining would recover 1.7 SR.

The original implementation attempt stopped on a NumPy/pandas diagnostic-array
conversion before any account was scored. Its protocol and source snapshot are
preserved under `runs/ndx_rank_stability_20261029/`; the corrected run is
`runs/ndx_rank_stability_20261029_v2/`. The methods, account grid and success
criteria were unchanged between attempts.
