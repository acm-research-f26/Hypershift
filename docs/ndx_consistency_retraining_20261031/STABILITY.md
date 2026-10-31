# Seed agreement diagnostics

These diagnostics describe all twenty seeds; none selects a seed or a financial strategy. Agreement can increase while Sharpe falls. The financial improvement gate remains unchanged.

| Mixture | Method | Mean pairwise issuer-target Jaccard |
|---|---|---:|
| both | original | 0.534 |
| both | dropout_control_raw | 0.541 |
| both | dropout_control_ema | 0.538 |
| both | consistency_raw | 0.540 |
| both | consistency_ema | 0.539 |
| think | original | 0.865 |
| think | dropout_control_raw | 0.862 |
| think | dropout_control_ema | 0.863 |
| think | consistency_raw | 0.866 |
| think | consistency_ema | 0.864 |
| dynamic | original | 0.433 |
| dynamic | dropout_control_raw | 0.447 |
| dynamic | dropout_control_ema | 0.450 |
| dynamic | consistency_raw | 0.445 |
| dynamic | consistency_ema | 0.451 |

Jaccard is the number of common selected issuers divided by the union, averaged over all 190 seed pairs and all 201 decisions. Share classes are consolidated. These are individual-seed decision targets before cohort averaging, rather than actual drifted holdings or independent market trials.

[Forecast diagnostics](FORECAST_STABILITY.csv) give per-family/year disagreement in eligible-stock percentile ranks and the fraction of forecast variance explained by stock-specific time means. The latter is a descriptive decomposition of predictions within each test year; no future return labels are used and no decomposition is fed into training or allocation. A larger static fraction can reflect less time-varying signal, so reduced disagreement alone is not a success criterion.

[Paired Sharpe intervals](PAIRED_INTERVALS.csv) retain all four primary candidate-versus-original contrasts and the three matched consistency/EMA contrasts, for every seven-group account. They use 2,000 common circular 20-session block draws with seed 20261031. These are conditional, pointwise 95% intervals on already fitted/selected paths, with no correction for this or earlier development searches. Neither overlapping seed ensembles nor these intervals establish unseen-market generalization.
